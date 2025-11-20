# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any
import os
import math
import tempfile  # <-- 新增：保存 ckpt 用到
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, AutoProcessor
from infer import (
    filter_to_gen_allow, generate_scores_argmax, prompt_eval_guards, set_strict_greedy_generation
)

# =========================================================
#                   视觉前缀注入（新实现）
# =========================================================
def _aggregate_tensor_list_inmem(
    tensors: List[torch.Tensor],
    method: str = "ema",
    ema_decay: float = 0.9,
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """
    对若干个形状相同的 tensor 做集成（内存版）。
    支持：
      - mean
      - ema
      - loss_inv（按 1/loss 加权）
    """
    assert len(tensors) > 0
    method = method.lower()

    if method == "mean":
        acc = torch.zeros_like(tensors[0], dtype=torch.float32)
        for t in tensors:
            acc += t.to(torch.float32)
        return acc / float(len(tensors))

    if method == "ema":
        ema = tensors[0].to(torch.float32)
        for t in tensors[1:]:
            ema = ema * ema_decay + t.to(torch.float32) * (1.0 - ema_decay)
        return ema

    if method == "loss_inv":
        if not weights or len(weights) != len(tensors):
            raise ValueError("method='loss_inv' 时需要同长度的 weights 列表")
        w = torch.tensor(weights, dtype=torch.float32)
        w = torch.clamp(w, min=1e-8)
        w = w / w.sum()
        acc = torch.zeros_like(tensors[0], dtype=torch.float32)
        for wi, ti in zip(w, tensors):
            acc += wi * ti.to(torch.float32)
        return acc

    raise ValueError(f"未知集成方式：{method}")

def save_prompt_ckpt(path: str, state: Dict):
    """
    简化版安全保存：
    1) 确保目录存在
    2) 如果 path 是目录，则在目录下生成一个默认文件名
    3) 先写入 path + ".tmp"
    4) 再原子重命名为最终路径
    """
    # 如果用户给的是目录，把它变成目录下的一个标准文件名
    if os.path.isdir(path):
        path = os.path.join(path, "prompt_ckpt_ensemble.pt")

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    tmp_path = path + ".tmp"
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
        print(f"[save] 模型已保存到：{path}")
    except Exception as e:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise RuntimeError(f"保存 checkpoint 失败：{path} -> {e}") from e



def load_prompt_ckpt(path: str, map_location="cpu") -> Dict:
    """
    加载 checkpoint；透传 map_location（'cpu'/'cuda' 或 torch.device）。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到 checkpoint：{path}")
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        raise RuntimeError(f"加载 checkpoint 失败：{path} -> {e}") from e


@dataclass
class VisualPromptCfg:
    n_tokens: int = 8
    mode: str = "post"
    cond_pool: bool = False
    lr: float = 8e-4
    weight_decay: float = 0.01
    dropout_p: float = 0.0
    reg_anchor: float = 1e-3
    reg_ortho: float = 1e-3


class _VisualPrefixCore(nn.Module):
    """
    在 [B, T, D] 序列前拼接 n 个可学习前缀；支持 cond_pool。
    """
    def __init__(
        self,
        n_tokens: int,
        dim: int,
        cond_pool: bool = False,
        dropout_p: float = 0.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.n = int(n_tokens)
        self.dim = int(dim)
        self.cond_pool = bool(cond_pool)
        self.dropout_p = float(dropout_p)

        self.prefix = nn.Parameter(torch.empty(self.n, self.dim, dtype=dtype, device=device))
        nn.init.normal_(self.prefix, mean=0.0, std=0.02)

        if self.cond_pool:
            self.adapter = nn.Sequential(
                nn.Linear(self.dim, self.dim, bias=True, dtype=dtype, device=device),
                nn.Tanh(),
                nn.Linear(self.dim, self.dim, bias=True, dtype=dtype, device=device),
            )
        else:
            self.adapter = None

        self.register_buffer("_init", self.prefix.detach().clone(), persistent=False)

    def forward(self, img_seq: torch.Tensor) -> torch.Tensor:
        if self.n <= 0:
            return img_seq

        if torch.is_floating_point(img_seq) and img_seq.dtype != self.prefix.dtype:
            img_seq = img_seq.to(self.prefix.dtype)

        B, T, D = img_seq.shape
        assert D == self.dim, f"VisualPrefix dim mismatch: got {D}, want {self.dim}"

        eff = self.prefix
        if self.training and self.dropout_p > 0.0:
            with torch.no_grad():
                mask = (torch.rand((self.n, 1), device=eff.device) < self.dropout_p)
            eff = torch.where(mask, self._init, self.prefix)

        pre = eff.unsqueeze(0).expand(B, -1, -1)  # [B, n, D]

        if self.cond_pool and self.adapter is not None:
            pooled = img_seq.mean(dim=1)  # [B, D]
            shift = self.adapter(pooled).unsqueeze(1)
            pre = pre + shift

        if self.training and not hasattr(self, "_dbg_once"):
            print(f"[vp] applying visual prefix: +{self.n} tokens (dropout_p={self.dropout_p}, cond_pool={self.cond_pool})")
            self._dbg_once = True
    
        return torch.cat([pre, img_seq], dim=1)

    def reg_terms(self, anchor_w: float = 0.0, ortho_w: float = 0.0) -> torch.Tensor:
        loss_reg = torch.zeros((), dtype=self.prefix.dtype, device=self.prefix.device)
        if anchor_w > 0.0:
            loss_reg = loss_reg + anchor_w * F.mse_loss(self.prefix, self._init)
        if ortho_w > 0.0 and self.n > 1:
            S = F.normalize(self.prefix, dim=1)
            eye = torch.eye(self.n, device=S.device, dtype=S.dtype)
            ortho = (S @ S.t() - eye).pow(2).mean()
            loss_reg = loss_reg + ortho_w * ortho
        return loss_reg


def _get_by_path(root, path: str):
    cur = root
    for a in path.split("."):
        if not hasattr(cur, a):
            return None
        cur = getattr(cur, a)
    return cur


class VisualPrompt:
    """
    在 projector/merger 输出的视觉序列上插入视觉前缀（通过 forward hook）。
    支持 out 是 Tensor 或 (tuple/list, head 是 Tensor)。
    """
    def __init__(self, model, cfg: VisualPromptCfg):
        self.model = model
        self.cfg = cfg
        self.handle = None
        self._cap_handle = None
        self.core: Optional[_VisualPrefixCore] = None

        hidden = self._infer_llm_hidden_size(model)
        compute_dtype = self._infer_compute_dtype(model)
        device = next(model.parameters()).device

        # 最近一次前向的线索
        self._bthw_hint: Optional[Tuple[int, int]] = None
        self._seg_lens: Optional[List[int]] = None
        self._merge_scale_env: Optional[int] = self._read_merge_scale_env()

        # 捕获 batch / seg_lens（pre-hook）
        def _capture_pre(_m, _in, kwargs):
            B = None
            T_common = None
            seg_lens = None
            if isinstance(kwargs, dict):
                igt = kwargs.get("image_grid_thw", None)
                if torch.is_tensor(igt) and igt.dim() == 2 and igt.size(1) >= 3:
                    B = int(igt.size(0))
                    lens = [int(igt[i, 1].item()) * int(igt[i, 2].item()) for i in range(B)]
                    seg_lens = lens
                    if len(set(lens)) == 1:
                        T_common = lens[0]
                else:
                    pv = kwargs.get("pixel_values", None)
                    if torch.is_tensor(pv):
                        B = int(pv.size(0))
                        if pv.dim() == 5:  # [B, C, F, H, W]
                            seg_lens = [int(pv.size(2))] * B
                            T_common = int(pv.size(2))
                        else:
                            seg_lens = [1] * B
                            T_common = 1
            if B is not None:
                self._bthw_hint = (B, T_common if T_common is not None else -1)
            if seg_lens is not None:
                self._seg_lens = seg_lens

        self._cap_handle = self.model.register_forward_pre_hook(_capture_pre, with_kwargs=True)

        # 选择 hook 点
        proj = self._find_projector(model, hidden)
        print(f"[vp] visual prompt will hook on: {proj.__class__.__name__}")

        core = _VisualPrefixCore(
            n_tokens=self.cfg.n_tokens,
            dim=hidden,
            cond_pool=self.cfg.cond_pool,
            dropout_p=self.cfg.dropout_p,
            dtype=compute_dtype,
            device=device,
        )
        self.core = core
        self._dbg_printed = False

        def _apply_prefix_to_tensor(t: torch.Tensor) -> torch.Tensor:
            # 梯度探针：检查是否有 stop-grad
            if not hasattr(self, "_hook_seen"):
                print(f"[vp] hook fired: Tensor shape={tuple(t.shape)} dtype={t.dtype} device={t.device}")
                try:
                    print(f"[vp] t.requires_grad={t.requires_grad}")
                    if t.requires_grad:
                        t.retain_grad()
                except Exception:
                    pass
                self._hook_seen = True

            if torch.is_floating_point(t) and t.dtype != self.core.prefix.dtype:
                t = t.to(self.core.prefix.dtype)

            # 3D：直接前缀
            if t.dim() == 3 and t.size(-1) == core.dim:
                if not self._dbg_printed:
                    B, T, H = t.shape
                    print(f"[vp] projector BEFORE:  B={B}, T={T}, H={H}")
                y = core(t)
                if not self._dbg_printed:
                    B2, T2, H2 = y.shape
                    print(f"[vp] projector AFTER:   B={B2}, T={T2}, H={H2} (ΔT=+{T2 - T})")
                    self._dbg_printed = True
                return y

            # 2D：按 seg_lens 切块
            if t.dim() == 2 and t.size(-1) == core.dim and self._seg_lens is not None:
                N, H = t.shape
                lens_raw = list(self._seg_lens)
                B = len(lens_raw)
                base = sum(lens_raw)
                r = self._merge_scale_env or self._infer_merge_scale(base, N)
                lens = self._rescale_lens_to_N(lens_raw, N, r)
                if sum(lens) != N:
                    print(f"[vp] WARN: rescaled sum(lens)={sum(lens)} != N={N}; skip once.")
                    return t
                if not self._dbg_printed:
                    print(f"[vp] projector 2D split: N={N} <- raw_sum={base}, r={r}, lens={lens}")
                chunks = []
                off = 0
                for li in lens:
                    xi = t[off:off + li, :].view(1, li, H)
                    yi = core(xi).view(-1, H)
                    chunks.append(yi)
                    off += li
                y = torch.cat(chunks, dim=0)
                if not self._dbg_printed:
                    print(f"[vp] projector AFTER 2D-prefix: {tuple(y.shape)} (ΔT=+{B * core.n})")
                    self._dbg_printed = True
                return y

            # 2D：兜底，根据 (B, T_common) 还原
            if t.dim() == 2 and t.size(-1) == core.dim and self._bthw_hint is not None:
                B, T_common = self._bthw_hint
                if T_common is not None and T_common > 0 and (B * T_common == t.size(0)):
                    if not self._dbg_printed:
                        print(f"[vp] reshape 2D->3D: [B*T,H]={tuple(t.shape)} -> [B,T,H]=({B},{T_common},{core.dim})")
                    x = t.view(B, T_common, core.dim)
                    y = core(x).reshape(-1, core.dim)
                    if not self._dbg_printed:
                        print(f"[vp] flatten back: {tuple(y.shape)} (ΔT=+{core.n})")
                        self._dbg_printed = True
                    return y

            if not hasattr(self, "_hook_warned"):
                print(f"[vp] WARN: unsupported tensor shape {tuple(t.shape)}; skip prefix once.")
                self._hook_warned = True
            return t

        def _post_hook(_module, _in, out):
            if not hasattr(self, "_hook_out_seen"):
                print(f"[vp] raw hook out type: {type(out)}")
                self._hook_out_seen = True

            if torch.is_tensor(out):
                return _apply_prefix_to_tensor(out)

            if isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                head2 = _apply_prefix_to_tensor(out[0])
                if isinstance(out, tuple):
                    return (head2, *out[1:])
                else:
                    out[0] = head2
                    return out

            print("[vp] WARN: non-tensor/empty output from projector; skip once.")
            return out

        # 注册 hook
        self.handle = proj.register_forward_hook(_post_hook)
        print("[vp] visual prompt attached (hook registered).")

    # ---------- utils ----------
    def _infer_llm_hidden_size(self, model) -> int:
        if hasattr(model, "config") and getattr(model.config, "hidden_size", None):
            return int(model.config.hidden_size)
        return int(model.get_input_embeddings().weight.shape[1])

    def _infer_compute_dtype(self, model) -> torch.dtype:
        for p in model.parameters():
            if p.is_floating_point():
                return p.dtype
        return torch.float32

    def _find_projector(self, model, hidden: int):
        hook_path = os.getenv("VP_HOOK_PATH", "").strip()
        if hook_path:
            mod = _get_by_path(model, hook_path)
            if mod is not None:
                print(f"[vp] use VP_HOOK_PATH={hook_path}")
                return mod
            else:
                print(f"[vp] VP_HOOK_PATH={hook_path} not found, fallback to auto search.")
        paths = [
            "multi_modal_projector", "model.multi_modal_projector",
            "visual_projector", "model.visual_projector",
            "mm_projector", "model.mm_projector",
            "image_projector", "model.image_projector",
            "projector", "model.projector",
            "visual.merger", "model.visual.merger",
        ]
        for p in paths:
            mod = _get_by_path(model, p)
            if mod is not None:
                print(f"[vp] found projector by path: {p}")
                return mod
        raise RuntimeError("未找到可用的 projector/merger；请设置 VP_HOOK_PATH")

    def _read_merge_scale_env(self) -> Optional[int]:
        v = os.getenv("VP_MERGE_SCALE", "").strip()
        if not v:
            return None
        try:
            r = int(v)
            return r if r >= 1 else None
        except Exception:
            return None

    def _infer_merge_scale(self, base_sum: int, N: int) -> int:
        if N <= 0 or base_sum <= 0:
            return 1
        r = int(round(base_sum / N))
        return max(1, r)

    def _rescale_lens_to_N(self, lens_raw: List[int], N: int, r: int) -> List[int]:
        if r <= 1:
            lens = list(lens_raw)
            s = sum(lens)
            if s == N:
                return lens
            if s == 0:
                return [N] + [0] * (len(lens) - 1)
            frac = [li / s for li in lens]
            base = [max(0, int(math.floor(f * N))) for f in frac]
            rem = N - sum(base)
            order = sorted(range(len(lens)), key=lambda i: (frac[i] - base[i] / max(1, N)), reverse=True)
            for i in range(rem):
                base[order[i % len(base)]] += 1
            return base
        base = [max(0, li // r) for li in lens_raw]
        rem = N - sum(base)
        if rem < 0:
            return self._rescale_lens_to_N(lens_raw, N, 1)
        scores = [li % r for li in lens_raw]
        order = sorted(range(len(lens_raw)), key=lambda i: scores[i], reverse=True)
        for i in range(rem):
            base[order[i % len(base)]] += 1
        return base

    # ---------- 训练侧 API ----------
    def parameters(self):
        return self.core.parameters() if self.core is not None else []

    def add_reg_loss(self, anchor_w: float, ortho_w: float) -> torch.Tensor:
        if self.core is None:
            dt = self._infer_compute_dtype(self.model)
            dev = next(self.model.parameters()).device
            return torch.zeros((), dtype=dt, device=dev)
        return self.core.reg_terms(anchor_w, ortho_w)

    def state_dict(self) -> Dict[str, Any]:
        """
        保存推理所需的全部可学习参数：
        - prefix（必有）
        - adapter.0.weight/adapter.0.bias/adapter.2.weight/adapter.2.bias（若存在）
        以及 cfg（非张量，仅作记录）
        """
        sd: Dict[str, Any] = {"cfg": dict(self.cfg.__dict__)}
        if self.core is None:
            return sd

        # 1) prefix
        sd["prefix"] = self.core.prefix.detach().cpu()

        # 2) adapter（如果启用 cond_pool 就会存在）
        if getattr(self.core, "adapter", None) is not None:
            ad_state = self.core.adapter.state_dict()
            # 展平到顶层，方便推理侧按键名对齐
            for k, v in ad_state.items():
                # k 形如 "0.weight" / "0.bias" / "2.weight" / "2.bias"
                sd[f"adapter.{k}"] = v.detach().cpu()
        return sd

    def load_state_dict(self, state: Dict[str, Any]):
        """
        从上面 state_dict() 的扁平格式恢复参数。
        同时兼容只含 prefix 的旧 ckpt。
        """
        if self.core is None:
            return

        # 1) prefix
        pre = state.get("prefix", None)
        if isinstance(pre, torch.Tensor) and pre.shape == self.core.prefix.shape:
            self.core.prefix.data.copy_(pre.to(self.core.prefix.device, dtype=self.core.prefix.dtype))

        # 2) adapter（有就加载，没有就跳过）
        if getattr(self.core, "adapter", None) is not None:
            # 收集 adapter.* 开头的键，去掉前缀后交给 Sequential.load_state_dict
            ad_kvs: Dict[str, torch.Tensor] = {}
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and k.startswith("adapter."):
                    ad_kvs[k.replace("adapter.", "", 1)] = v
            if ad_kvs:
                # 严格度放宽，避免形状或缺键报错（比如禁用 cond_pool 的情况）
                self.core.adapter.load_state_dict(ad_kvs, strict=False)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self._cap_handle is not None:
            self._cap_handle.remove()
            self._cap_handle = None


# =========================================================
#                   文本 + 视觉 软提示 Learner
# =========================================================

@dataclass
class TrainCfg:
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    grad_accum: int = 8
    log_every: int = 50
    save_ckpt: str = "prompt_ckpt/final.pt"
    use_fp16: bool = True

    eval_every: int = 200
    ckpt_best: str = "prompt_ckpt/best.pt"
    step_ckpt_dir: Optional[str] = "prompt_ckpt/step_ckpts"
    save_every_step: int = 100
    early_stop_patience: int = 10
    monitor: str = "acc"
    minimize: bool = False

    warmup_steps: int = 200
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    init_prompt: Optional[str] = (
        "You are a helpful assistant that answers by returning ONE word like \"<class>\"."
    )
    visual_sp_dropout: float = 0.2  # 视觉前缀 dropout（训练时）
    sp_dropout: float = 0.0          # 文本软提示 dropout（训练时）
    target_mode: str = "token"       # 生成时启用 JSON 解码辅助
    ensemble_mode: str = "none"


class SoftPromptLearner:
    class _ReplaceRowsParam(nn.Module):
        def __init__(self, soft_idx: torch.Tensor, soft_param: nn.Parameter,
                     soft_init: torch.Tensor, dropout_p: float = 0.0):
            super().__init__()
            self.register_buffer("soft_idx", soft_idx, persistent=False)
            self.soft_param = soft_param
            self.register_buffer("soft_init", soft_init, persistent=False)
            self.dropout_p = float(dropout_p)

        def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
            W2 = base_weight.clone()
            if self.training and self.dropout_p > 0.0:
                with torch.no_grad():
                    mask = (torch.rand((self.soft_param.size(0), 1), device=self.soft_param.device) < self.dropout_p)
                eff = torch.where(mask, self.soft_init, self.soft_param)
            else:
                eff = self.soft_param
            W2.index_copy_(0, self.soft_idx, eff)
            return W2

    def __init__(
        self,
        model: PreTrainedModel,
        processor: AutoProcessor,
        template_variants: List[str],
        demo_provider,
        label_space: List[str],
        train_cfg: TrainCfg,
        device: torch.device,
        use_image: bool = False,
        n_visual_sp: int = 8,
        soft_token_names: Optional[List[str]] = None
    ):
        import torch.nn.utils.parametrize as P

        self.model = model
        self.processor = processor
        self.tok = processor.tokenizer
        self.label_space = list(label_space)

        # 模式开关（env）
        self.text_only = int(os.getenv("TEXT_PROMPT_ONLY", 0))
        self.visual_only = int(os.getenv("VISUAL_PROMPT_ONLY", 0))
        if self.text_only and self.visual_only:
            raise ValueError("TEXT_PROMPT_ONLY 和 VISUAL_PROMPT_ONLY 不能同时为 1")

        self.cfg = train_cfg
        self.device = device
        self.template_variants = template_variants
        self.use_image = use_image
        self.n_visual_sp = n_visual_sp if not self.text_only else 0
        mode = (getattr(self.cfg, "ensemble_mode", "none") or "none").lower()
        if mode in ("ema", "mean", "loss"):
            self._ensemble_mode = mode
            print(f"[ensemble] 启用训练后集成模式：{self._ensemble_mode.upper()}")
        else:
            self._ensemble_mode = "none"
        self._ens_soft_vecs: List[Optional[torch.Tensor]] = []
        self._ens_vp_states: List[Optional[Dict[str, Any]]] = []
        self._ens_losses: List[float] = []
        # step 目录
        if self.cfg.step_ckpt_dir:
            os.makedirs(self.cfg.step_ckpt_dir, exist_ok=True)
            print(f"[init] 按步保存目录已创建：{self.cfg.step_ckpt_dir}")

        # vocab 对齐
        new_vocab = len(self.tok)
        lm_head_rows = self.model.get_output_embeddings().weight.shape[0]
        if lm_head_rows != new_vocab:
            self.model.resize_token_embeddings(new_vocab)
        self.model.config.vocab_size = new_vocab

        # 冻结全模型
        for p in self.model.parameters():
            p.requires_grad_(False)
        print(f"[init] 模型主干网络已冻结，仅优化软提示参数")

        # ---------- 文本软提示 ----------
        self.soft_tokens: List[str] = []
        self.soft_ids: List[int] = []
        self.soft_param: Optional[nn.Parameter] = None
        self.emb: Optional[nn.Parameter] = None
        self.hidden_size: int = 0

        if not self.visual_only:
            if soft_token_names is None:
                cand = [t for t in (self.tok.additional_special_tokens or []) if t.startswith("<soft")]
                def _key(s: str) -> int:
                    s = s.replace("<soft", "").replace(">", "")
                    return int(s) if s.isdigit() else 10**9
                self.soft_tokens = sorted(cand, key=_key)
            else:
                self.soft_tokens = list(soft_token_names)

            if not self.soft_tokens:
                raise RuntimeError("未找到文本软提示 Token，请先添加 <softx> 类特殊 Token 到词表")
            self.soft_ids = self.tok.convert_tokens_to_ids(self.soft_tokens)
            if any(x < 0 for x in self.soft_ids):
                raise RuntimeError(f"部分软 token 未在词表中：{self.soft_tokens} → {self.soft_ids}")

            emb_layer = self.model.get_input_embeddings()
            self.emb: nn.Parameter = emb_layer.weight  # [V, H]
            V, self.hidden_size = self.emb.shape
            self.emb.requires_grad_(False)

            soft_idx = torch.tensor(self.soft_ids, device=self.emb.device, dtype=torch.long)
            self.soft_param = nn.Parameter(self.emb[self.soft_ids].detach().clone())
            print(f"[text-sp] 维度: vocab_size={V} hidden_size={self.hidden_size} 可训练参数: {len(self.soft_ids) * self.hidden_size}")
            print(f"[text-sp] 启用的文本软提示 Token: {self.soft_tokens}")
        else:
            print(f"[text-sp] 禁用文本软提示（仅视觉模式）")

        # ---------- 视觉前缀 ----------
        self.vp: Optional[VisualPrompt] = None
        self.vp_cfg: Optional[VisualPromptCfg] = None
        if not self.text_only and self.use_image and self.n_visual_sp > 0:
            self.vp_cfg = VisualPromptCfg(
                n_tokens=self.n_visual_sp,
                cond_pool=False,
                lr=self.cfg.lr * 0.3,
                weight_decay=self.cfg.weight_decay,
                dropout_p=float(getattr(self.cfg, "visual_sp_dropout", 0.0)),
                reg_anchor=3e-3,
                reg_ortho=3e-3,
            )
            self.vp = VisualPrompt(self.model, self.vp_cfg)
            print(f"[visual-sp] 维度: +{self.vp_cfg.n_tokens} prefix tokens | 通过 projector/merger 前向 hook 进行拼接注入")
            # 关键：训练确认
            if self.vp.core is not None:
                pf = self.vp.core.prefix
                print(f"[visual-sp] trainable prefix shape={tuple(pf.shape)} dtype={pf.dtype} device={pf.device} requires_grad={pf.requires_grad}")
        else:
            print(f"[visual-sp] 禁用视觉前缀（仅文本模式或未启用图像）")

        # ---------- 优化器 ----------
        trainable_params = []
        if (self.soft_param is not None) and (not self.visual_only):
            trainable_params.append({"params": [self.soft_param], "lr": self.cfg.lr})
        if self.vp is not None:
            trainable_params.append({"params": list(self.vp.parameters()), "lr": self.vp_cfg.lr})

        if not trainable_params:
            raise RuntimeError("没有可训练的软提示参数，请检查模式开关和配置")

        self.opt = torch.optim.AdamW(trainable_params, weight_decay=self.cfg.weight_decay)
        print(f"[opt] 优化器初始化完成，可训练参数组数量：{len(trainable_params)}")
        tp = sum(p.numel() for g in self.opt.param_groups for p in g["params"])
        print(f"[opt] total trainable params: {tp}")

        # ---------- AMP ----------
        try:
            self.compute_dtype = next(p.dtype for p in self.model.parameters() if p is not None)
        except StopIteration:
            self.compute_dtype = torch.float32
        cuda_available = torch.cuda.is_available()
        want_amp = self.cfg.use_fp16 and cuda_available

        if want_amp and self.compute_dtype == torch.float16:
            self.amp_dtype = torch.float16
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        elif want_amp and self.compute_dtype == torch.bfloat16:
            self.amp_dtype = torch.bfloat16
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        else:
            self.amp_dtype = torch.float32
            self.use_amp = False
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        # ---------- 其他 ----------
        self.pad_id = self.tok.pad_token_id or getattr(self.tok, "eos_token_id", 0)
        self.scheduler = self._build_scheduler(self.opt, self.cfg.max_steps, self.cfg.warmup_steps)

        self.soft_init: Optional[torch.Tensor] = None
        if (self.soft_param is not None) and (not self.visual_only) and self.cfg.init_prompt:
            self._init_soft_from_gaussian(0, None)
            with torch.no_grad():
                self.soft_init = self.soft_param.detach().clone()

        # 文本软提示参数化（仅文本模式或混合）
        self._row_replacer: Optional[SoftPromptLearner._ReplaceRowsParam] = None
        if (self.soft_param is not None) and (not self.visual_only):
            emb_layer = self.model.get_input_embeddings()
            soft_idx = torch.tensor(self.soft_ids, device=self.emb.device, dtype=torch.long)
            P.register_parametrization(
                emb_layer, "weight",
                SoftPromptLearner._ReplaceRowsParam(
                    soft_idx=soft_idx,
                    soft_param=self.soft_param,
                    soft_init=self.soft_init,
                    dropout_p=0.0
                )
            )
            self._row_replacer = emb_layer.parametrizations.weight[0]

        # 自检
        print(f"\n[init-summary] 训练模式：{'仅文本软提示' if self.text_only else ('仅视觉前缀' if self.visual_only else '文本+视觉')}")
        print(f"[init-summary] 文本软提示：{'启用' if (self.soft_param is not None and not self.visual_only) else '禁用'}")
        print(f"[init-summary] 视觉前缀：{'启用' if self.vp is not None else '禁用'}，Token数：{self.vp_cfg.n_tokens if self.vp is not None else 0}")
        print(f"[init-summary] AMP配置：compute_dtype={self.compute_dtype} amp_dtype={self.amp_dtype} use_amp={self.use_amp}")
        print(f"[init-summary] 保存配置：每隔{self.cfg.save_every_step}步保存中间checkpoint")
        print(f"[init-summary] 中间checkpoint目录：{self.cfg.step_ckpt_dir}")
        print(f"[init-summary] 最终模型路径：{self.cfg.save_ckpt}")
        print(f"[init-summary] 最佳模型路径：{self.cfg.ckpt_best}\n")

        # 日志缓存：把“视觉梯度均值”在累积边界读到后缓存，用于 log_every 对齐
        self._last_vis_grad_mean: float = 0.0
        self._last_txt_grad_mean: float = 0.0

    # ---------- Scheduler ----------
    def _build_scheduler(self, opt, max_steps: int, warmup_steps: int):
        self.base_lr = self.cfg.lr
        self.min_lr = self.cfg.lr * 0.1

        def lr_lambda(step: int):
            if step < warmup_steps and warmup_steps > 0:
                return max(step / max(1, warmup_steps), 1e-8)
            t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            t = min(max(t, 0.0), 1.0)
            return (self.min_lr / self.base_lr) + 0.5 * (1 - self.min_lr / self.base_lr) * (1 + math.cos(math.pi * t))
        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    @torch.no_grad()
    def _init_soft_from_gaussian(self, mean: float = 0.0, std: Optional[float] = None):
        emb_layer = self.model.get_input_embeddings()
        base_table = emb_layer.parametrizations.weight.original if hasattr(emb_layer, "parametrizations") else emb_layer.weight

        if std is None:
            try:
                est = float(base_table.std().item())
            except Exception:
                est = 0.02
            std = est if math.isfinite(est) and est > 1e-6 else 0.02
        self.soft_param.normal_(mean=mean, std=std)
        print(f"[init] soft prompts initialized from Gaussian N({mean:.3f}, {std:.3f}^2)")

    def _cache_for_ensemble(self, step: int, current_loss: float):
        """
        在内存里缓存当前 step 的软提示参数，用于训练结束后的集成。
        只在 ensemble_mode != 'none' 时被调用。
        """
        # 文本软提示
        if (self.soft_param is not None) and (not self.visual_only):
            self._ens_soft_vecs.append(self.soft_param.detach().cpu())
        else:
            self._ens_soft_vecs.append(None)

        # 视觉前缀
        if self.vp is not None:
            self._ens_vp_states.append(self.vp.state_dict())
        else:
            self._ens_vp_states.append(None)

        self._ens_losses.append(float(current_loss))
        print(f"[ensemble] 缓存 step={step} 的软提示参数用于后续集成（loss={current_loss:.6f}）")

    def _save_step_ckpt(self, step: int, current_loss: float, current_lr: float):
        # 如果开启了集成模式：不再写中间 ckpt 到磁盘，只在内存里缓存当前软提示
        if self._ensemble_mode in ("ema", "mean", "loss"):
            self._cache_for_ensemble(step, current_loss)
            return

        # ====== 以下是原有逻辑，不动，保证 ensemble_mode='none' 时行为不变 ======
        if not self.cfg.step_ckpt_dir:
            return
        step_ckpt_path = os.path.join(self.cfg.step_ckpt_dir, f"prompt_ckpt_step_{step:06d}.pt")
        print(f"[save] 保存step={step}的中间checkpoint：{step_ckpt_path}")
        step_save_dict = {
            "text_only": self.text_only,
            "visual_only": self.visual_only,
            "soft_tokens": (self.soft_tokens if (self.soft_param is not None and not self.visual_only) else None),
            "soft_vecs": (self.soft_param.detach().cpu() if (self.soft_param is not None and not self.visual_only) else None),
            "vp_state": (self.vp.state_dict() if self.vp is not None else None),
            "step": step,
            "current_loss": current_loss,
            "current_lr": current_lr,
            "optimizer_state_dict": self.opt.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }
        dirname = os.path.dirname(step_ckpt_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        torch.save(step_save_dict, step_ckpt_path)

    def _save_ensemble_final_ckpt(self) -> Optional[Dict[str, Any]]:
        """
        在训练结束时调用：
          - 如果 ensemble_mode='none'：返回 None
          - 如果 ensemble_mode in {EMA,MEAN,LOSS}：
              对缓存的软提示进行集成，保存到 cfg.save_ckpt，并返回集成结果 dict
        """
        if self._ensemble_mode not in ("ema", "mean", "loss"):
            return None

        if not self._ens_soft_vecs and not self._ens_vp_states:
            print("[ensemble] 未缓存任何软提示参数，跳过集成")
            return None

        # 映射到内部聚合方法
        if self._ensemble_mode == "loss":
            agg_method = "loss_inv"
        else:
            agg_method = self._ensemble_mode  # ema / mean

        ema_decay = 0.9  # 你也可以做成 cfg 或 env

        weights = None
        if agg_method == "loss_inv":
            # 按 1/loss 加权
            losses = [max(l, 1e-8) for l in self._ens_losses]
            weights = [1.0 / l for l in losses]

        # 构造结果字典
        result: Dict[str, Any] = {
            "text_only": self.text_only,
            "visual_only": self.visual_only,
            "soft_tokens": (
                self.soft_tokens if (self.soft_param is not None and not self.visual_only) else None
            ),
            "meta": {
                "ensemble_method": agg_method,
                "ema_decay": ema_decay,
                "num_sources": len(self._ens_losses),
            },
        }

        # ========= 文本 soft prompt 集成 =========
        has_text_sp = (self.soft_param is not None) and (not self.visual_only)
        if has_text_sp:
            soft_tensors = [t for t in self._ens_soft_vecs if t is not None]
            if soft_tensors:
                sp_agg = _aggregate_tensor_list_inmem(
                    soft_tensors,
                    method=agg_method,
                    ema_decay=ema_decay,
                    weights=weights,
                )
                result["soft_vecs"] = sp_agg
                print(f"[ensemble] 文本 soft prompt 集成完成，shape={tuple(sp_agg.shape)}")
            else:
                print("[ensemble] 警告：未找到可用于集成的文本 soft prompt 快照")

        # ========= 视觉前缀 vp_state 集成 =========
        vp_states = [v for v in self._ens_vp_states if v is not None]
        if vp_states:
            first_vp = vp_states[0]
            vp_agg: Dict[str, Any] = {}
            # 非 tensor 字段原样拷贝
            for k, v in first_vp.items():
                if not isinstance(v, torch.Tensor):
                    vp_agg[k] = v
            # tensor 字段逐键集成
            vp_keys = [k for k, v in first_vp.items() if isinstance(v, torch.Tensor)]
            for k in vp_keys:
                tensors_k = []
                for vs in vp_states:
                    if k not in vs:
                        raise RuntimeError(f"[ensemble] 某个 vp_state 缺少键：{k}")
                    tensors_k.append(vs[k])
                vp_agg[k] = _aggregate_tensor_list_inmem(
                    tensors_k,
                    method=agg_method,
                    ema_decay=ema_decay,
                    weights=weights,
                )
            result["vp_state"] = vp_agg
            print(f"[ensemble] 视觉前缀 vp_state 集成完成，keys={list(vp_agg.keys())}")

        out_path = self.cfg.save_ckpt
        save_prompt_ckpt(out_path, result)
        print(f"[ensemble] 集成后的软提示 ckpt 已保存到：{out_path}")

        return result


    @torch.no_grad()
    def eval_like_infer_generation(
        self,
        dev_loader,
        label_space,
        max_new_tokens: int = 32,
    ):
        import time, numpy as np
        from utils import parse_label_from_output

        self.model.eval()
        set_strict_greedy_generation(self.model)

        total, correct = 0, 0
        latencies = []

        labels = list(label_space)
        idx_of = {c: i for i, c in enumerate(labels)}
        C = len(labels)
        cm = np.zeros((C, C), dtype=int)

        row_replacer = getattr(self, "_row_replacer", None)
        vp_core = getattr(getattr(self, "vp", None), "core", None)

        with prompt_eval_guards(row_replacer, vp_core):
            for _, batch in enumerate(dev_loader):
                B = len(batch["gold_label_str"])
                for i in range(B):
                    gold_label = batch["gold_label_str"][i]
                    hf_inputs_full = batch["hf_inputs"]
                    hf_inputs = filter_to_gen_allow(hf_inputs_full, take_index=i)

                    for k, v in list(hf_inputs.items()):
                        if torch.is_tensor(v):
                            hf_inputs[k] = v.to(self.device)

                    t0 = time.time()
                    text_out, _ = generate_scores_argmax(
                        self.model, self.tok, hf_inputs,
                        max_new_tokens=max_new_tokens,
                        decode_clean=False,
                    )
                    latencies.append((time.time() - t0) * 1000.0)

                    pred = parse_label_from_output(text_out, label_space, target_mode=self.cfg.target_mode)

                    correct += int(pred == gold_label)
                    total += 1

                    if gold_label in idx_of and pred in idx_of:
                        cm[idx_of[gold_label], idx_of[pred]] += 1

        # ---------- 计算 acc ----------
        acc = correct / max(total, 1)

        # ---------- 计算 micro-F1 ----------
        tp = np.diag(cm).sum()
        fp = cm.sum(axis=0).sum() - tp
        fn = cm.sum(axis=1).sum() - tp
        micro_f1 = 2 * tp / max(2 * tp + fp + fn, 1)

        # ---------- 计算 macro-F1 & per-class-F1 ----------
        f1_list = []
        per_class_f1 = {}
        for i in range(C):
            tp_i = cm[i, i]
            fp_i = cm[:, i].sum() - tp_i
            fn_i = cm[i, :].sum() - tp_i
            f1_i = 2 * tp_i / max(2 * tp_i + fp_i + fn_i, 1)
            f1_list.append(f1_i)
            per_class_f1[labels[i]] = float(f1_i)

        macro_f1 = float(np.mean(f1_list))

        avg_latency = float(sum(latencies) / max(len(latencies), 1)) if latencies else 0.0

        print(f"\n[eval] acc={acc:.4f} micro_f1={micro_f1:.4f} macro_f1={macro_f1:.4f}")
        print(f"[eval] per-class F1: " + ", ".join(f"{k}:{v:.4f}" for k, v in per_class_f1.items()))

        print("[eval] 混淆矩阵（行=真实标签，列=预测标签）:")
        for r in range(C):
            row = " ".join(f"{cm[r, c]:3d}" for c in range(C))
            print(f"  {labels[r]:>10s} | {row}")
        print()

        self.model.train()

        return acc, macro_f1, micro_f1, per_class_f1, avg_latency




    @torch.no_grad()
    def _apply_prompt_dropout(self, p=0.1):
        if hasattr(self, "_row_replacer") and (self._row_replacer is not None):
            self._row_replacer.dropout_p = float(max(0.0, min(1.0, p)))

    # ---------- 训练主循环 ----------
    def fit(self, loader: DataLoader, dev_loader: Optional[DataLoader]):
        self.model.train()
        step = 0
        best_metric = -float("inf")
        gnorm_cache = 0.0
        early_stop_counter = 0
        print(f"\n[train] 开始训练，总步数：{self.cfg.max_steps}，梯度累积步数：{self.cfg.grad_accum}")
        print(f"[train] 评估间隔：{self.cfg.eval_every}步，早停耐心值：{self.cfg.early_stop_patience}")
        print(f"[train] 训练模式：{'仅文本软提示' if self.text_only else ('仅视觉前缀' if self.visual_only else '文本+视觉')}\n")

        try:
            while step < self.cfg.max_steps:
                for batch in loader:
                    if self.cfg.target_mode == "token":
                        targets = [lbl for lbl in batch["gold_label_str"]]
                    else:
                        def label_to_target_json(label: str) -> str:
                            # 和旧版保持一致：训练时让模型生成 {"label": "<cand>"}
                            return f'{{"label": "{label}"}}'

                        targets = [label_to_target_json(lbl) for lbl in batch["gold_label_str"]]
                    target_ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in targets]
                    input_ids2, attn2, labels, others = self._pack_batch(batch["hf_inputs"], target_ids)

                    # 文本 Prompt Dropout（视觉前缀的 dropout 在其内部）
                    if not self.visual_only:
                        self._apply_prompt_dropout(p=self.cfg.sp_dropout)

                    # 前向
                    with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                        out = self.model(input_ids=input_ids2, attention_mask=attn2, labels=labels, **others)
                        loss = out.loss

                        # 正则项
                        reg_loss = 0.0
                        if (self.soft_param is not None) and (not self.visual_only):
                            lambda_anchor = 1e-3
                            anchor_l2 = F.mse_loss(self.soft_param, self.soft_init)
                            reg_loss += lambda_anchor * anchor_l2

                            lambda_ortho = 1e-3
                            S_text = F.normalize(self.soft_param, dim=1)
                            eye_t = torch.eye(len(self.soft_ids), device=S_text.device)
                            ortho_text = (S_text @ S_text.t() - eye_t).pow(2).mean()
                            reg_loss += lambda_ortho * ortho_text

                        if self.vp is not None:
                            reg_loss = reg_loss + self.vp.add_reg_loss(
                                anchor_w=self.vp_cfg.reg_anchor,
                                ortho_w=self.vp_cfg.reg_ortho
                            )

                        total_loss = loss + reg_loss

                    # 反向与优化（支持 AMP 与梯度累积）
                    if self.scaler.is_enabled():
                        self.scaler.scale(total_loss / max(1, self.cfg.grad_accum)).backward()
                        boundary = ((step + 1) % self.cfg.grad_accum) == 0
                        if boundary:
                            self.scaler.unscale_(self.opt)

                            # 在 zero_grad 之前，采样梯度
                            if (self.soft_param is not None) and (not self.visual_only) and (self.soft_param.grad is not None):
                                self._last_txt_grad_mean = self.soft_param.grad.abs().mean().item()
                            if self.vp is not None and self.vp.core is not None and self.vp.core.prefix.grad is not None:
                                self._last_vis_grad_mean = self.vp.core.prefix.grad.abs().mean().item()

                            if self.cfg.max_grad_norm is not None:
                                params_to_clip = []
                                if (self.soft_param is not None) and (not self.visual_only):
                                    params_to_clip.append(self.soft_param)
                                if self.vp is not None:
                                    params_to_clip += list(self.vp.parameters())
                                if params_to_clip:
                                    gnorm_cache = torch.nn.utils.clip_grad_norm_(params_to_clip, self.cfg.max_grad_norm).item()
                            self.scaler.step(self.opt)
                            self.scaler.update()
                            self.opt.zero_grad(set_to_none=True)
                            self.scheduler.step()
                    else:
                        (total_loss / max(1, self.cfg.grad_accum)).backward()
                        boundary = ((step + 1) % self.cfg.grad_accum) == 0
                        if boundary:
                            if (self.soft_param is not None) and (not self.visual_only) and (self.soft_param.grad is not None):
                                self._last_txt_grad_mean = self.soft_param.grad.abs().mean().item()
                            if self.vp is not None and self.vp.core is not None and self.vp.core.prefix.grad is not None:
                                self._last_vis_grad_mean = self.vp.core.prefix.grad.abs().mean().item()

                            if self.cfg.max_grad_norm is not None:
                                params_to_clip = []
                                if (self.soft_param is not None) and (not self.visual_only):
                                    params_to_clip.append(self.soft_param)
                                if self.vp is not None:
                                    params_to_clip += list(self.vp.parameters())
                                if params_to_clip:
                                    gnorm_cache = torch.nn.utils.clip_grad_norm_(params_to_clip, self.cfg.max_grad_norm).item()
                            self.opt.step()
                            self.opt.zero_grad(set_to_none=True)
                            self.scheduler.step()

                    # 日志（注意：梯度值取自累积边界时缓存）
                    if (step % self.cfg.eval_every) == 0:
                        lr = self.opt.param_groups[0]["lr"]
                        with torch.no_grad():
                            text_cos = 0.0
                            ortho_mse_text = 0.0
                            if (self.soft_param is not None) and (not self.visual_only):
                                cur_text = F.normalize(self.soft_param, dim=1)
                                init_text = F.normalize(self.soft_init, dim=1)
                                text_cos = (cur_text * init_text).sum(dim=1).mean().item()
                                eye_t = torch.eye(len(self.soft_ids), device=cur_text.device)
                                ortho_mse_text = (cur_text @ cur_text.t() - eye_t).pow(2).mean().item()

                            ortho_mse_visual = 0.0
                            if self.vp is not None and self.vp.core is not None:
                                cur_visual = F.normalize(self.vp.core.prefix, dim=1)
                                eye_v = torch.eye(self.vp_cfg.n_tokens, device=cur_visual.device)
                                ortho_mse_visual = (cur_visual @ cur_visual.t() - eye_v).pow(2).mean().item()

                        print(
                            f"[step={step}] 总损失={total_loss.item():.6f} 原始损失={loss.item():.6f} 正则损失={reg_loss.item():.6f} | "
                            f"文本侧梯度均值={self._last_txt_grad_mean:.6e} 梯度范数={gnorm_cache:.3f} | "
                            f"文本相似度={text_cos:.3f} 文本正交性={ortho_mse_text:.4f} | "
                            f"视觉侧梯度均值={self._last_vis_grad_mean:.6e} 梯度范数={gnorm_cache:.3f} | "
                            f"视觉正交性={ortho_mse_visual:.4f} 学习率={lr:.3e}",
                            flush=True
                        )

                    # 按步保存
                    if (step % self.cfg.save_every_step) == 0:
                        current_lr = self.opt.param_groups[0]["lr"]
                        self._save_step_ckpt(step=step, current_loss=total_loss.item(), current_lr=current_lr)

                    # 评估与最佳
                    if (self.cfg.eval_every > 0) and (dev_loader is not None) and (step % self.cfg.eval_every == 0):
                        val_acc, macro_f1, micro_f1, per_class_f1, val_lat = self.eval_like_infer_generation(dev_loader, self.label_space)
                        if val_acc > best_metric:
                            best_metric = val_acc
                            best_save_dict = {
                                "text_only": self.text_only,
                                "visual_only": self.visual_only,
                                "soft_tokens": (self.soft_tokens if (self.soft_param is not None and not self.visual_only) else None),
                                "soft_vecs": (self.soft_param.detach().cpu() if (self.soft_param is not None and not self.visual_only) else None),
                                "vp_state": (self.vp.state_dict() if self.vp is not None else None),
                                "best_step": step,
                                "best_val_acc": best_metric,
                                "optimizer_state_dict": self.opt.state_dict(),
                                "scheduler_state_dict": self.scheduler.state_dict(),
                                "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
                            }
                            dirname = os.path.dirname(self.cfg.ckpt_best)
                            if dirname:
                                os.makedirs(dirname, exist_ok=True)
                            torch.save(best_save_dict, self.cfg.ckpt_best)
                            print(f"[save] 最佳模型已更新（step={step}，acc={best_metric:.4f}）：{self.cfg.ckpt_best}")
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                            print(f"[early-stop] 未更新最佳模型，计数器：{early_stop_counter}/{self.cfg.early_stop_patience}")
                            if early_stop_counter >= self.cfg.early_stop_patience:
                                print(f"[early-stop] 早停条件触发，训练提前结束")
                                return

                    step += 1
                    if step >= self.cfg.max_steps:
                        break
        finally:
            # 1) 先做集成，得到最终 soft prompt 状态（如果启用了 ensemble）
            ens_state = self._save_ensemble_final_ckpt()

            # 2) 如果有 dev_loader，就对“最终状态”做一次统一评估
            if dev_loader is not None:
                print("[train] 训练结束，对最终软提示在验证集上做一次统一评估...")

                # 如果存在集成结果，就把集成后的参数写回 runtime，再评估
                if ens_state is not None:
                    print("[train] 应用集成后的软提示参数再进行评估（ensemble final）")

                    # 文本 soft prompt
                    if (self.soft_param is not None) and (not self.visual_only) and ("soft_vecs" in ens_state):
                        sp = ens_state["soft_vecs"]
                        self.soft_param.data.copy_(sp.to(self.soft_param.device, dtype=self.soft_param.dtype))
                        # soft_init 也更新一下，避免正则项用的还是老的
                        if self.soft_init is not None:
                            self.soft_init.data.copy_(self.soft_param.data)

                    # 视觉前缀
                    if (self.vp is not None) and ("vp_state" in ens_state):
                        self.vp.load_state_dict(ens_state["vp_state"])

                try:
                    final_acc, final_macro_f1, final_micro_f1, final_per_class_f1, final_lat = self.eval_like_infer_generation(
                        dev_loader,
                        self.label_space,
                    )
                    print(
                        f"[train] 结束时 eval：acc={final_acc:.4f} "
                        f"avg_latency={final_lat:.1f}ms"
                    )
                except Exception as e:
                    print(f"[train] 结束时 eval 失败：{e}")

            # 3) 最后再移除视觉前缀 hook，避免后续误用
            if self.vp is not None:
                self.vp.remove()
                print(f"[visual-hook] 视觉前缀钩子已移除")

        print(f"[训练完成] 最佳模型准确率：{best_metric:.4f}（step={step}）")


    # ---------- Batch 打包 ----------
    def _pack_batch(
        self,
        batch_inputs: Dict,
        target_ids: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        input_ids: torch.Tensor = batch_inputs["input_ids"].to(self.device)
        attn: torch.Tensor = batch_inputs["attention_mask"].to(self.device)
        B, T = input_ids.size()

        others: Dict = {}
        for k, v in batch_inputs.items():
            if k in ["input_ids", "attention_mask"]:
                continue
            if torch.is_tensor(v):
                if k == "pixel_values":
                    target_dtype = torch.bfloat16 if self.compute_dtype == torch.bfloat16 else (
                        torch.float16 if self.compute_dtype == torch.float16 else torch.float32
                    )
                    others[k] = v.to(self.device, dtype=target_dtype)
                else:
                    others[k] = v.to(self.device)
            else:
                others[k] = v

        max_tgt = max(len(t) for t in target_ids) if target_ids else 0
        pad_id = self.pad_id
        if max_tgt == 0:
            labels = torch.full((B, T), -100, device=self.device, dtype=torch.long)
            return input_ids, attn, labels, others

        tgt_ids = torch.full((B, max_tgt), fill_value=pad_id, device=self.device, dtype=torch.long)
        labels = torch.full((B, T + max_tgt), fill_value=-100, device=self.device, dtype=torch.long)

        for i, ids in enumerate(target_ids):
            L = len(ids)
            if L > 0:
                tgt_ids[i, :L] = torch.tensor(ids, device=self.device, dtype=torch.long)
                labels[i, T: T + L] = torch.tensor(ids, device=self.device, dtype=torch.long)

        input_ids2 = torch.cat([input_ids, tgt_ids], dim=1)
        attn2 = torch.cat([attn, (tgt_ids != pad_id).long()], dim=1)
        return input_ids2, attn2, labels, others

    def __del__(self):
        if hasattr(self, "vp") and self.vp is not None:
            self.vp.remove()
