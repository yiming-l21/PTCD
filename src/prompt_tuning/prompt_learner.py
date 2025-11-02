# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, AutoProcessor
#from sp_utils import gpu_mem_snapshot
# --------- ckpt io ---------
def save_prompt_ckpt(path: str, state: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_prompt_ckpt(path: str, map_location="cpu") -> Dict:
    return torch.load(path, map_location=map_location)


# --------- train cfg ---------
@dataclass
class TrainCfg:
    # 软提示通常需要更大的 lr（远大于全参微调）
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    grad_accum: int = 8
    log_every: int = 50
    save_ckpt: str = "prompt_ckpt.pt"
    use_fp16: bool = True  # 若模型是bf16，会自动切到bf16并关闭GradScaler

    eval_every: int = 100
    ckpt_best: str = "prompt_ckpt.best.pt"
    early_stop_patience: int = 5  # 可选：未实现
    monitor: str = "acc"          # "acc" 或 "loss"
    minimize: bool = True         # 可选：未实现

    warmup_steps: int = 200
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    init_prompt: Optional[str] = (
        # 用简短、稳定的英文模板初始化软提示，能显著提升早期可学性
        "You are a helpful assistant that answers by returning JSON like {\"label\": \"<class>\"}."
    )


class SoftPromptLearner:
    """
    仅训练 embedding 中 <soft*> 行（其余权重冻结）。
    使用 nn.utils.parametrize 在前向时“行替换”，
    从而优化器只持有 O(n_soft×H) 的状态，且梯度可达 soft_param。
    """

    class _ReplaceRowsParam(nn.Module):
        def __init__(self, soft_idx: torch.Tensor, soft_param: nn.Parameter,
                    soft_init: torch.Tensor, dropout_p: float = 0.0):
            super().__init__()
            self.register_buffer("soft_idx", soft_idx, persistent=False)
            self.soft_param = soft_param              # (n_soft, H)
            self.register_buffer("soft_init", soft_init, persistent=False)  # (n_soft, H)
            self.dropout_p = float(dropout_p)

        def forward(self, base_weight: torch.Tensor) -> torch.Tensor:
            W2 = base_weight.clone()

            # 训练时才启用；评估/推理时禁用
            if self.training and self.dropout_p > 0.0:
                # 每次前向采样一个 (n_soft,1) 的二值 mask
                with torch.no_grad():
                    mask = (torch.rand((self.soft_param.size(0), 1), device=self.soft_param.device)
                            < self.dropout_p)
                # 在维度上广播：mask==True 用 soft_init，False 用 soft_param
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
        demo_provider,  # 占位，不使用
        label_space: List[str],
        train_cfg: TrainCfg,
        device: torch.device,
        soft_token_names: Optional[List[str]] = None
    ):
        import torch.nn.utils.parametrize as P

        self.model = model
        self.processor = processor
        self.tok = processor.tokenizer
        self.label_space = list(label_space)
        self.cfg = train_cfg
        self.device = device
        self.template_variants = template_variants

        # vocab 对齐
        new_vocab = len(self.tok)
        lm_head_rows = self.model.get_output_embeddings().weight.shape[0]
        if lm_head_rows != new_vocab:
            self.model.resize_token_embeddings(new_vocab)
        self.model.config.vocab_size = new_vocab

        # --- <soft*> tokens ---
        if soft_token_names is None:
            cand = [t for t in (self.tok.additional_special_tokens or []) if t.startswith("<soft")]
            def _key(s: str) -> int:
                s = s.replace("<soft", "").replace(">", "")
                return int(s) if s.isdigit() else 10**9
            self.soft_tokens = sorted(cand, key=_key)
        else:
            self.soft_tokens = list(soft_token_names)

        if not self.soft_tokens:
            raise RuntimeError(
                "未在 tokenizer 中找到任何 <soft*> token。请先注册："
                "tokenizer.add_special_tokens({'additional_special_tokens': ['<soft0>', ...]}); "
                "并调用 model.resize_token_embeddings(len(tokenizer))"
            )
        self.soft_ids = self.tok.convert_tokens_to_ids(self.soft_tokens)
        if any(x < 0 for x in self.soft_ids):
            raise RuntimeError(f"部分软token未在词表中：{self.soft_tokens} → {self.soft_ids}")

        # --- 冻结全模型，仅训练 soft_param ---
        for p in self.model.parameters():
            p.requires_grad_(False)

        emb_layer = self.model.get_input_embeddings()
        self.emb: nn.Parameter = emb_layer.weight  # [V, H]
        V, H = self.emb.shape
        self.emb.requires_grad_(False)
        soft_idx = torch.tensor(self.soft_ids, device=self.emb.device, dtype=torch.long)

        # 训练参数: (n_soft, H)
        self.soft_param = nn.Parameter(self.emb[self.soft_ids].detach().clone())

        print(f"[soft] vocab_size={V} hidden_size={H} n_soft_tokens={len(self.soft_ids)} trainable_params={len(self.soft_ids) * H}")

        # 仅优化 soft_param
        self.opt = torch.optim.AdamW([self.soft_param], lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

        # --------- AMP / GradScaler 适配（bf16/fp16）---------
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
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)  # bf16 不用 scaler
        else:
            self.amp_dtype = torch.float32
            self.use_amp = False
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        # pad id
        self.pad_id = self.tok.pad_token_id
        if self.pad_id is None:
            self.pad_id = getattr(self.tok, "eos_token_id", 0)

        # 学习率调度器：linear warmup + cosine decay
        self.scheduler = self._build_scheduler(self.opt, self.cfg.max_steps, self.cfg.warmup_steps)

        # 软提示初始化（把自然语言模板的 embedding 拷贝/平均到 <soft*> 行的初值）
        if self.cfg.init_prompt:
            self._init_soft_from_gaussian(0, None)

        # 记录初始值用于正则/对比
        with torch.no_grad():
            self.soft_init = self.soft_param.detach().clone()
         # 注册参数化：前向时将 base weight 的软行替换成 soft_param
        P.register_parametrization(
            emb_layer, "weight",
            SoftPromptLearner._ReplaceRowsParam(soft_idx=soft_idx, soft_param=self.soft_param, soft_init=self.soft_init, dropout_p=0.0)
        )
        self._row_replacer = emb_layer.parametrizations.weight[0]
        # 自检打印
        with torch.no_grad():
            vocab = len(self.tok)
            lm_head_rows = self.model.get_output_embeddings().weight.shape[0]
            print(f"[soft] vocab={vocab}  lm_head_rows={lm_head_rows}  config.vocab_size={self.model.config.vocab_size}")
            print(f"[soft] tokens={self.soft_tokens}")
            print(f"[soft] ids={self.soft_ids}")
            print(f"[amp] compute_dtype={self.compute_dtype} amp_dtype={self.amp_dtype} use_amp={self.use_amp} scaler_enabled={self.scaler.is_enabled()}")

    # ---------- scheduler ----------
    def _build_scheduler(self, opt, max_steps: int, warmup_steps: int):
        # 线性 warmup → cosine 衰减到 10% lr 尾值
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
        import math
        emb_layer = self.model.get_input_embeddings()
        # 取“基础表”以估计方差
        if hasattr(emb_layer, "parametrizations") and hasattr(emb_layer.parametrizations, "weight"):
            base_table = emb_layer.parametrizations.weight.original  # [V, H]
        else:
            base_table = emb_layer.weight  # [V, H]

        # 若未显式指定 std，则用 embedding 权重的整体 std，退化兜底 0.02（CLIP 常见初始化量级）
        if std is None:
            try:
                est = float(base_table.std().item())
            except Exception:
                est = 0.02
            std = est if math.isfinite(est) and est > 1e-6 else 0.02

        # 直接在可训练参数上采样
        self.soft_param.normal_(mean=mean, std=std)
        print(f"[init] soft prompts initialized from Gaussian N({mean:.3f}, {std:.3f}^2)")
    # ---------- 用自然语言模板初始化软提示 ----------
    @torch.no_grad()
    def _init_soft_from_text(self, prompt: str):
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            print("[init] template tokenized to empty; skip init.")
            return

        emb_layer = self.model.get_input_embeddings()

        # —— 安全获取“基础表”：如果已经注册过参数化，则取 original；否则直接取 weight —— #
        base_table = None
        if hasattr(emb_layer, "parametrizations") and hasattr(emb_layer.parametrizations, "weight"):
            base_table = emb_layer.parametrizations.weight.original  # [V, H]
        else:
            base_table = emb_layer.weight  # [V, H]

        idx = torch.tensor(ids, device=base_table.device, dtype=torch.long)
        vecs = base_table[idx]  # [L, H]

        if len(self.soft_ids) <= vecs.size(0):
            chosen = vecs[:len(self.soft_ids)]
        else:
            reps = (len(self.soft_ids) + vecs.size(0) - 1) // vecs.size(0)
            chosen = vecs.repeat(reps, 1)[:len(self.soft_ids)]

        noise = torch.randn_like(chosen) * 0.01
        chosen = chosen + noise

        # 初始化到 soft_param（训练参数）
        self.soft_param.data.copy_(chosen)
        print(f"[init] soft prompts initialized from template ({len(ids)} toks)")

    # ---------- batch 打包：把 target token 拼到输入末尾，并构造 labels ----------
    def _pack_batch(
        self,
        batch_inputs: Dict,
        target_ids: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        输入：
          - batch_inputs: processor(...) 的返回（含 input_ids, attention_mask, 以及像素字段）
          - target_ids: List[List[int]]，每条样本的目标 token 序列
        输出：
          - input_ids2: [B, T+max_tgt]
          - attn2:      [B, T+max_tgt]
          - labels:     [B, T+max_tgt]，仅 target 段为真值，其它为 -100
          - others:     视觉/其余字段（需透传给模型）
        """
        input_ids: torch.Tensor = batch_inputs["input_ids"].to(self.device)   # [B,T]
        attn: torch.Tensor = batch_inputs["attention_mask"].to(self.device)   # [B,T]
        B, T = input_ids.size()

        # 透传视觉/其他字段
        others: Dict = {}
        for k, v in batch_inputs.items():
            if k in ["input_ids", "attention_mask"]:
                continue
            if torch.is_tensor(v):
                others[k] = v.to(self.device)
            else:
                others[k] = v

        # 视觉张量 dtype（若存在）
        if "pixel_values" in others and torch.is_tensor(others["pixel_values"]):
            target_dtype = torch.bfloat16 if self.compute_dtype == torch.bfloat16 else (
                torch.float16 if self.compute_dtype == torch.float16 else torch.float32
            )
            others["pixel_values"] = others["pixel_values"].to(target_dtype)

        max_tgt = max(len(t) for t in target_ids) if target_ids else 0
        pad_id = self.pad_id

        if max_tgt == 0:
            labels = torch.full((B, T), -100, device=self.device, dtype=torch.long)
            return input_ids, attn, labels, others

        tgt_ids = torch.full((B, max_tgt), fill_value=pad_id, device=self.device, dtype=torch.long)
        labels  = torch.full((B, T + max_tgt), fill_value=-100, device=self.device, dtype=torch.long)

        for i, ids in enumerate(target_ids):
            L = len(ids)
            if L > 0:
                tgt_ids[i, :L] = torch.tensor(ids, device=self.device, dtype=torch.long)
                labels[i, T: T + L] = torch.tensor(ids, device=self.device, dtype=torch.long)

        input_ids2 = torch.cat([input_ids, tgt_ids], dim=1)                    # [B, T+max_tgt]
        attn2 = torch.cat([attn, (tgt_ids != pad_id).long()], dim=1)           # [B, T+max_tgt]
        assert labels.size(1) == input_ids2.size(1), "labels 与输入长度必须一致"
        return input_ids2, attn2, labels, others

    @torch.no_grad()
    def eval_like_infer_generation(
        self,
        dev_loader,
        label_space,
        max_new_tokens: int = 32,
    ) -> Tuple[float, float]:
        """
        返回 (val_acc, avg_latency_ms)，评估严格复用推理代码路径。
        """
        from utils import parse_label_from_output
        import numpy as np
        import time
        old_p = 0.0
        if hasattr(self, "_row_replacer"):
            old_p = self._row_replacer.dropout_p
            self._row_replacer.dropout_p = 0.0
        try:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            for k in ("top_p", "top_k", "typical_p"):
                if hasattr(self.model.generation_config, k):
                    setattr(self.model.generation_config, k, None)
        except Exception:
            pass
        _GEN_ALLOW = {
            # 文本
            "input_ids", "attention_mask",
            # 图像 / 视频张量
            "pixel_values", "pixel_attention_mask",
            "pixel_values_videos", "pixel_values_videos_mask",
            # 视觉几何信息（Qwen2.5-VL 必需）
            "image_grid_thw", "video_grid_thw", "image_sizes",
            # 其他可能的前置特征
            "input_features", "encoder_outputs", "image_prompt_embeds",
        }

        self.model.eval()
        total = 0
        correct = 0
        latencies = []

        # 统计混淆/每类准确率
        labels = list(label_space)
        idx_of = {c: i for i, c in enumerate(labels)}
        C = len(labels)
        cm = np.zeros((C, C), dtype=int)

        for _, batch in enumerate(dev_loader):
            B = len(batch["gold_label_str"])
            for i in range(B):
                s_full = batch["hf_inputs"]

                # —— 清洗 hf_inputs：只保留 generate 需要的字段 —— #
                s = {k: v for k, v in s_full.items() if k in _GEN_ALLOW}
                # 单条样本切片
                for k, v in list(s.items()):
                    if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == B:
                        s[k] = v[i:i+1].to(self.device)

                gold_label = batch["gold_label_str"][i]

                t1 = time.time()
                out = self.model.generate(
                    **s,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,     # 确定性
                    num_beams=1,
                )
                latencies.append((time.time() - t1) * 1000.0)

                attn = s.get("attention_mask", None)
                if attn is not None:
                    attn_lens = attn[0].sum().item()
                    out = out[0, int(attn_lens):]  # 去掉输入部分
                else:
                    out = out[0]
                text_out = self.tok.decode(out, skip_special_tokens=True)

                pred = parse_label_from_output(text_out, label_space)
                correct += int(pred == gold_label)
                total += 1

                # 混淆矩阵计数
                if gold_label in idx_of and pred in idx_of:
                    cm[idx_of[gold_label], idx_of[pred]] += 1

        acc = correct / max(total, 1)
        avg_latency = float(sum(latencies) / max(len(latencies), 1)) if latencies else 0.0

        # 每类准确率
        per_cls = {}
        for c in range(C):
            n = cm[c].sum()
            per_cls[labels[c]] = (cm[c, c] / n) if n > 0 else 0.0

        # 简要评估报告
        print(f"[eval] acc={acc:.4f} avg_latency_ms={avg_latency:.1f}")
        print(f"[eval] per_class_acc=" + ", ".join(f"{k}:{v:.2f}" for k, v in per_cls.items()))
        print("[eval] confusion_matrix (rows=gold, cols=pred):")
        for r in range(C):
            row = " ".join(f"{cm[r, c]:3d}" for c in range(C))
            print(f"  {labels[r]:>10s} | {row}")
        if hasattr(self, "_row_replacer"):
            self._row_replacer.dropout_p = old_p
        self.model.train()
        return acc, avg_latency

    @torch.no_grad()
    def _avg_logprob(self, input_ids2, attn2, others, texts: List[str]) -> float:
        tok = self.tok
        device = self.device
        B = input_ids2.size(0)
        scores = []
        for i in range(B):
            tgt_ids = tok(texts[i], add_special_tokens=False)["input_ids"]
            T = len(tgt_ids)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(input_ids=input_ids2[i:i+1], attention_mask=attn2[i:i+1], **others)
            logits = out.logits[:, -T:, :]
            log_probs = F.log_softmax(logits, dim=-1)
            tgt = torch.tensor(tgt_ids, device=device).unsqueeze(0)
            token_lp = log_probs.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()
            scores.append(token_lp)
        return float(sum(scores) / max(len(scores), 1))

    @torch.no_grad()
    def _apply_prompt_dropout(self, p=0.2):
        # 不再改 self.soft_param.data，只设置概率即可
        if hasattr(self, "_row_replacer"):
            self._row_replacer.dropout_p = float(max(0.0, min(1.0, p)))


    # ---------- 训练 ----------
    def fit(self, loader: DataLoader, dev_loader: Optional[DataLoader], target_builder):
        """
        训练：只更新 soft_param。保存 best 与最终 ckpt（均只保存软行子矩阵）。
        """
        self.model.train()
        step = 0
        best_metric = -float("inf")

        # 缓存日志指标
        gnorm_cache = 0.0
        gnorm_soft_cache = 0.0

        while step < self.cfg.max_steps:
            for batch in loader:
                #gpu_mem_snapshot(prefix="before forward")
                targets = [target_builder(lbl) for lbl in batch["gold_label_str"]]
                target_ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in targets]
                input_ids2, attn2, labels, others = self._pack_batch(batch["hf_inputs"], target_ids)

                # prompt dropout
                self._apply_prompt_dropout(p=0.2)

                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    #gpu_mem_snapshot(prefix="before forward")
                    out = self.model(input_ids=input_ids2, attention_mask=attn2, labels=labels, **others)
                    #gpu_mem_snapshot(prefix="after model forward")
                    loss = out.loss

                    # 正则项（作用于 soft_param）
                    cur = self.soft_param
                    lambda_anchor = 1e-3
                    anchor_l2 = F.mse_loss(cur, self.soft_init)
                    lambda_ortho = 1e-3
                    S = F.normalize(cur, dim=1)      # (n_soft, H)
                    ortho = (S @ S.t() - torch.eye(len(self.soft_ids), device=S.device)).pow(2).mean()
                    loss = loss + lambda_anchor * anchor_l2 + lambda_ortho * ortho
                #gpu_mem_snapshot(prefix="after forward")
                if self.scaler.is_enabled():
                    self.scaler.scale(loss / max(1, self.cfg.grad_accum)).backward()
                    #gpu_mem_snapshot(prefix="after backward")
                    if ((step + 1) % self.cfg.grad_accum) == 0:
                        self.scaler.unscale_(self.opt)
                        raw_soft_g_cache = (self.soft_param.grad.abs().mean().item()
                                            if self.soft_param.grad is not None else 0.0)
                        if self.cfg.max_grad_norm is not None:
                            gnorm_cache = torch.nn.utils.clip_grad_norm_([self.soft_param], self.cfg.max_grad_norm).item()
                        else:
                            gnorm_cache = (self.soft_param.grad.norm().item()
                                           if self.soft_param.grad is not None else 0.0)
                        gnorm_soft_cache = gnorm_cache

                        self.scaler.step(self.opt)
                        self.scaler.update()
                        self.opt.zero_grad(set_to_none=True)
                        self.scheduler.step()
                else:
                    (loss / max(1, self.cfg.grad_accum)).backward()
                    #gpu_mem_snapshot(prefix="after backward")
                    if ((step + 1) % self.cfg.grad_accum) == 0:
                        raw_soft_g_cache = (self.soft_param.grad.abs().mean().item()
                                            if self.soft_param.grad is not None else 0.0)
                        if self.cfg.max_grad_norm is not None:
                            gnorm_cache = torch.nn.utils.clip_grad_norm_([self.soft_param], self.cfg.max_grad_norm).item()
                        else:
                            gnorm_cache = (self.soft_param.grad.norm().item()
                                           if self.soft_param.grad is not None else 0.0)
                        gnorm_soft_cache = gnorm_cache

                        self.opt.step()
                        self.opt.zero_grad(set_to_none=True)
                        self.scheduler.step()

                if (step % self.cfg.log_every) == 0:
                    lr = self.opt.param_groups[0]["lr"]
                    with torch.no_grad():
                        cur = F.normalize(self.soft_param, dim=1)
                        init = F.normalize(self.soft_init, dim=1)
                        cos_to_init = (cur * init).sum(dim=1).mean().item()
                        ortho_mse = (cur @ cur.t() - torch.eye(len(self.soft_ids), device=cur.device)).pow(2).mean().item()

                    print(
                        f"[soft-prompt] step={step} "
                        f"loss={loss.item():.6f},raw_soft_g={raw_soft_g_cache:.6e} "
                        f"l2={anchor_l2.item():.6f} ortho={ortho.item():.6f} "
                        f"gnorm={gnorm_cache:.3f} gnorm_soft={gnorm_soft_cache:.3f} "
                        f"cos_init={cos_to_init:.3f} ortho_mse={ortho_mse:.4f} lr={lr:.3e}",
                        flush=True
                    )

                if (self.cfg.eval_every > 0) and (dev_loader is not None) and (step >= 0) and (step % self.cfg.eval_every == 0):
                    val_acc, val_lat = self.eval_like_infer_generation(dev_loader, self.label_space)
                    print(f"[soft-prompt] eval@{step}: val_acc={val_acc:.4f} avg_latency_ms={val_lat:.1f}", flush=True)
                    metric = val_acc
                    best_metric = metric
                    soft_vecs = self.soft_param.detach().cpu()
                    save_prompt_ckpt(self.cfg.save_ckpt+f"prompt_ckpt_{step}.pt", {
                        "soft_tokens": self.soft_tokens,
                        "soft_vecs": soft_vecs,
                    })
                    print(f"[soft-prompt] new best -> saved prompt_ckpt_{step}.pt", flush=True)

                step += 1
                if step >= self.cfg.max_steps:
                    break
