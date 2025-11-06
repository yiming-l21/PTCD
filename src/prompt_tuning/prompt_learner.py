# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Union
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, AutoProcessor

# --------- ckpt io ---------
def save_prompt_ckpt(path: str, state: Dict):
    # 确保保存目录存在
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    torch.save(state, path)
    print(f"[save] 模型已保存到：{path}")

def load_prompt_ckpt(path: str, map_location="cpu") -> Dict:
    return torch.load(path, map_location=map_location)

# --------- 视觉软提示初始化工具函数 ---------
def init_visual_soft_tokens(
    model: PreTrainedModel,
    n_tokens: int = 8,
    device: torch.device = None,
    init_std: float = 0.02
) -> Tuple[nn.Parameter, int]:
    """
    初始化视觉软提示（适配Qwen2.5-VL的1280维视觉特征）
    Args:
        model: 预训练模型（Qwen2.5-VL）
        n_tokens: 视觉软提示token数量
        device: 设备
        init_std: 初始化标准差
    Returns:
        visual_sp_param: 视觉软提示参数（n_tokens, 1280）
        visual_embed_dim: 视觉嵌入维度（固定1280）
    """
    if device is None:
        device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cpu")
    
    visual_embed_dim = 1280  # Qwen2.5-VL视觉特征维度固定为1280
    compute_dtype = next(model.parameters()).dtype
    visual_sp_param = nn.Parameter(
        torch.randn(n_tokens, visual_embed_dim, device=device, dtype=compute_dtype) * init_std
    )
    return visual_sp_param, visual_embed_dim

# --------- train cfg ---------
@dataclass
class TrainCfg:
    lr: float = 1e-4
    weight_decay: float = 0.01
    max_steps: int = 1000
    grad_accum: int = 8
    log_every: int = 50
    save_ckpt: str = "prompt_ckpt/final.pt"  # 最终模型保存路径
    use_fp16: bool = True

    eval_every: int = 100
    ckpt_best: str = "prompt_ckpt/best.pt"  # 最佳模型保存路径
    step_ckpt_dir: str = "prompt_ckpt/step_ckpts"  # 按步保存的目录
    save_every_step: int = 100  # 每隔多少步保存一次（可通过配置修改）
    early_stop_patience: int = 5
    monitor: str = "acc"
    minimize: bool = False

    warmup_steps: int = 200
    warmup_ratio: float = 0.10
    max_grad_norm: float = 1.0
    init_prompt: Optional[str] = (
        "You are a helpful assistant that answers by returning JSON like {\"label\": \"<class>\"}."
    )
    visual_sp_dropout: float = 0.1  # 视觉Prompt Dropout

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
                    mask = (torch.rand((self.soft_param.size(0), 1), device=self.soft_param.device)
                            < self.dropout_p)
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
        # 模式开关（从环境变量读取，默认同时启用文本+视觉）
        self.text_only = int(os.getenv("TEXT_PROMPT_ONLY", 0))
        self.visual_only = int(os.getenv("VISUAL_PROMPT_ONLY", 0))
        
        # 校验开关（二选一，不能同时为1）
        if self.text_only and self.visual_only:
            raise ValueError("TEXT_PROMPT_ONLY和VISUAL_PROMPT_ONLY不能同时设为1")
        
        self.cfg = train_cfg
        self.device = device
        self.template_variants = template_variants
        self.use_image = use_image
        self.n_visual_sp = n_visual_sp if not self.text_only else 0  # 仅文本模式时禁用视觉软提示

        # 创建按步保存的目录
        if self.cfg.step_ckpt_dir:
            os.makedirs(self.cfg.step_ckpt_dir, exist_ok=True)
            print(f"[init] 按步保存目录已创建：{self.cfg.step_ckpt_dir}")

        # vocab对齐
        new_vocab = len(self.tok)
        lm_head_rows = self.model.get_output_embeddings().weight.shape[0]
        if lm_head_rows != new_vocab:
            self.model.resize_token_embeddings(new_vocab)
        self.model.config.vocab_size = new_vocab

        # --- 冻结全模型 ---
        for p in self.model.parameters():
            p.requires_grad_(False)
        print(f"[init] 模型主干网络已冻结，仅优化软提示参数")

        # --- 文本软提示初始化（仅文本模式或混合模式）---
        self.soft_tokens: List[str] = []
        self.soft_ids: List[int] = []
        self.soft_param: Optional[nn.Parameter] = None
        self.emb: Optional[nn.Parameter] = None
        self.hidden_size: int = 0

        if not self.visual_only:  # 不是仅视觉模式，启用文本软提示
            if soft_token_names is None:
                # 自动从词表中查找<softx>类特殊token
                cand = [t for t in (self.tok.additional_special_tokens or []) if t.startswith("<soft")]
                def _key(s: str) -> int:
                    s = s.replace("<soft", "").replace(">", "")
                    return int(s) if s.isdigit() else 10**9
                self.soft_tokens = sorted(cand, key=_key)
            else:
                self.soft_tokens = list(soft_token_names)

            # 校验文本软提示token
            if not self.soft_tokens:
                raise RuntimeError("未找到文本软提示Token，请先添加<softx>类特殊Token到词表")
            if len(self.soft_tokens) == 0:
                raise RuntimeError("文本软提示Token数量不能为0")
            self.soft_ids = self.tok.convert_tokens_to_ids(self.soft_tokens)
            if any(x < 0 for x in self.soft_ids):
                raise RuntimeError(f"部分软token未在词表中：{self.soft_tokens} → {self.soft_ids}")

            # 初始化文本软提示参数
            emb_layer = self.model.get_input_embeddings()
            self.emb: nn.Parameter = emb_layer.weight  # [V, H]
            V, self.hidden_size = self.emb.shape
            self.emb.requires_grad_(False)
            soft_idx = torch.tensor(self.soft_ids, device=self.emb.device, dtype=torch.long)
            self.soft_param = nn.Parameter(self.emb[self.soft_ids].detach().clone())
            print(f"[text-sp] 维度: vocab_size={V} hidden_size={self.hidden_size} 可训练参数: {len(self.soft_ids) * self.hidden_size}")
            print(f"[text-sp] 启用的文本软提示Token: {self.soft_tokens}")
        else:
            print(f"[text-sp] 禁用文本软提示（仅视觉模式）")

        # --- 视觉软提示初始化（仅视觉模式或混合模式）---
        self.visual_sp_param: Optional[nn.Parameter] = None
        self.vis_hidden_size: int = 0
        self.merger_hook: Optional[torch.utils.hooks.RemovableHandle] = None  # 视觉钩子

        if not self.text_only and self.use_image and self.n_visual_sp > 0:
            self.visual_sp_param, self.vis_hidden_size = init_visual_soft_tokens(
                model=self.model,
                n_tokens=self.n_visual_sp,
                device=self.device
            )
            # 关键：验证视觉维度是1280（merger输入要求）
            assert self.vis_hidden_size == 1280, f"视觉软提示维度必须为1280（merger输入要求），当前：{self.vis_hidden_size}"
            self.visual_sp_param.requires_grad_(True)
            print(f"[visual-sp] 维度: {self.n_visual_sp} × {self.vis_hidden_size} 可训练参数: {self.n_visual_sp * self.vis_hidden_size}")
            print(f"[visual-sp] 提示：视觉软提示将插入merger前，由模型原生merger自动升维到{self.hidden_size}维（混合模式）" if not self.text_only else "[visual-sp] 仅视觉模式，无需与文本维度对齐")

            # 注册merger前钩子（训练/推理时自动插入视觉软提示）
            self._register_visual_hook()
        else:
            print(f"[visual-sp] 禁用视觉软提示（仅文本模式或未启用图像）")

        # --- 优化器配置（仅优化启用的软提示参数）---
        trainable_params = []
        if self.soft_param is not None:
            trainable_params.append({"params": [self.soft_param], "lr": train_cfg.lr})
        if self.visual_sp_param is not None:
            # 视觉软提示学习率默认是文本的1.5倍
            visual_lr = train_cfg.lr * 1.5
            trainable_params.append({"params": [self.visual_sp_param], "lr": visual_lr})
        
        if not trainable_params:
            raise RuntimeError("没有可训练的软提示参数，请检查模式开关和配置")
        
        self.opt = torch.optim.AdamW(trainable_params, weight_decay=train_cfg.weight_decay)
        print(f"[opt] 优化器初始化完成，可训练参数组数量：{len(trainable_params)}")

        # --- AMP适配 ---
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

        # --- 其他配置 ---
        self.pad_id = self.tok.pad_token_id or getattr(self.tok, "eos_token_id", 0)
        self.scheduler = self._build_scheduler(self.opt, self.cfg.max_steps, self.cfg.warmup_steps)

        # --- 文本软提示初始化（仅文本模式或混合模式）---
        self.soft_init: Optional[torch.Tensor] = None
        if self.soft_param is not None and self.cfg.init_prompt:
            self._init_soft_from_gaussian(0,None)
            with torch.no_grad():
                self.soft_init = self.soft_param.detach().clone()

        # --- 注册文本软提示参数化（仅文本模式或混合模式）---
        self._row_replacer: Optional[SoftPromptLearner._ReplaceRowsParam] = None
        if self.soft_param is not None:
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

        # --- 自检信息汇总 ---
        print(f"\n[init-summary] 训练模式：{'仅文本软提示' if self.text_only else ('仅视觉软提示' if self.visual_only else '文本+视觉软提示')}")
        print(f"[init-summary] 文本软提示：{'启用' if self.soft_param is not None else '禁用'}，Token数：{len(self.soft_tokens) if self.soft_param is not None else 0}")
        print(f"[init-summary] 视觉软提示：{'启用' if self.visual_sp_param is not None else '禁用'}，Token数：{self.n_visual_sp if self.visual_sp_param is not None else 0}")
        print(f"[init-summary] AMP配置：compute_dtype={self.compute_dtype} amp_dtype={self.amp_dtype} use_amp={self.use_amp}")
        print(f"[init-summary] 保存配置：每隔{self.cfg.save_every_step}步保存中间checkpoint")
        print(f"[init-summary] 中间checkpoint目录：{self.cfg.step_ckpt_dir}")
        print(f"[init-summary] 最终模型路径：{self.cfg.save_ckpt}")
        print(f"[init-summary] 最佳模型路径：{self.cfg.ckpt_best}\n")

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
    @torch.no_grad()
    def _init_soft_from_text(self, prompt: str):
        """从文本模板初始化文本软提示"""
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            print("[init] 模板token化为空，跳过文本软提示初始化")
            return

        emb_layer = self.model.get_input_embeddings()
        base_table = emb_layer.parametrizations.weight.original if hasattr(emb_layer, "parametrizations") else emb_layer.weight
        idx = torch.tensor(ids, device=base_table.device, dtype=torch.long)
        vecs = base_table[idx]

        # 适配文本软提示token数量（截断或重复）
        if len(self.soft_ids) <= vecs.size(0):
            chosen = vecs[:len(self.soft_ids)]
        else:
            reps = (len(self.soft_ids) + vecs.size(0) - 1) // vecs.size(0)
            chosen = vecs.repeat(reps, 1)[:len(self.soft_ids)]

        # 添加小噪声避免初始化过于一致
        noise = torch.randn_like(chosen) * 0.01
        self.soft_param.data.copy_(chosen + noise)
        print(f"[init] 文本软提示从模板初始化完成（模板token数：{len(ids)}）")

    def _insert_visual_soft_prompt(self, visual_hidden_states: torch.Tensor) -> torch.Tensor:
        """插入视觉软提示到1280维特征中（merger前）"""
        if self.visual_sp_param is None or self.n_visual_sp <= 0:
            return visual_hidden_states
        
        # 视觉软提示：(n_visual_sp, 1280) → 扩展为 (batch_size, n_visual_sp, 1280)
        batch_size = visual_hidden_states.size(0)
        visual_sp = self.visual_sp_param.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 训练时应用dropout（防止过拟合）
        if self.training and self.cfg.visual_sp_dropout > 0:
            visual_sp = F.dropout(visual_sp, p=self.cfg.visual_sp_dropout)
        
        # 插入到原生视觉特征前面（优先学习视觉Prompt）
        return torch.cat([visual_sp, visual_hidden_states], dim=1)

    def _visual_merger_hook(self, module, input, output):
        """模型钩子：在merger处理前插入视觉软提示"""
        if self.visual_sp_param is None:
            return output
        
        # 解析merger输入：Qwen2_5_VLPatchMerger的输入为(visual_hidden_states, ...)
        visual_hidden_states = input[0]  # (batch_size, num_patches, 1280)
        if visual_hidden_states.dim() != 3 or visual_hidden_states.size(-1) != self.vis_hidden_size:
            return output
        
        # 插入视觉软提示
        visual_hidden_states_with_sp = self._insert_visual_soft_prompt(visual_hidden_states)
        
        # 替换merger输入（保留其他参数）
        new_input = (visual_hidden_states_with_sp,) + input[1:]
        return module(*new_input)

    def _register_visual_hook(self):
        """注册视觉软提示插入钩子（merger前）"""
        if hasattr(self.model.model.visual, "merger"):
            self.merger_hook = self.model.model.visual.merger.register_forward_hook(self._visual_merger_hook)
            print(f"[visual-hook] 视觉软提示钩子已注册到 model.model.visual.merger")
        else:
            raise RuntimeError("未找到model.model.visual.merger模块，无法注册视觉钩子")

    def _remove_visual_hook(self):
        """移除视觉钩子（推理/训练结束时）"""
        if self.merger_hook is not None:
            self.merger_hook.remove()
            self.merger_hook = None
            print(f"[visual-hook] 视觉软提示钩子已移除")

    def _pack_batch(
        self,
        batch_inputs: Dict,
        target_ids: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """打包batch数据（文本+图像输入）"""
        input_ids: torch.Tensor = batch_inputs["input_ids"].to(self.device)
        attn: torch.Tensor = batch_inputs["attention_mask"].to(self.device)
        B, T = input_ids.size()

        others: Dict = {}
        for k, v in batch_inputs.items():
            if k in ["input_ids", "attention_mask"]:
                continue
            if torch.is_tensor(v):
                # 像素值直接传递给模型，由模型内部视觉编码器处理
                if k == "pixel_values":
                    target_dtype = torch.bfloat16 if self.compute_dtype == torch.bfloat16 else (
                        torch.float16 if self.compute_dtype == torch.float16 else torch.float32
                    )
                    others[k] = v.to(self.device, dtype=target_dtype)
                else:
                    others[k] = v.to(self.device)
            else:
                others[k] = v

        # 处理目标序列
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

    @torch.no_grad()
    def eval_like_infer_generation(
        self,
        dev_loader,
        label_space,
        max_new_tokens: int = 32,
    ) -> Tuple[float, float]:
        """评估函数（模拟推理时的生成过程）"""
        from utils import parse_label_from_output
        import numpy as np
        import time
        
        # 禁用文本Prompt Dropout
        old_p = self._row_replacer.dropout_p if (self._row_replacer is not None) else 0.0
        if self._row_replacer is not None:
            self._row_replacer.dropout_p = 0.0

        # 配置生成参数
        try:
            self.model.generation_config.do_sample = False
            self.model.generation_config.temperature = None
            for k in ("top_p", "top_k", "typical_p"):
                if hasattr(self.model.generation_config, k):
                    setattr(self.model.generation_config, k, None)
        except Exception:
            pass

        # 允许传递给generate的输入字段
        _GEN_ALLOW = {
            "input_ids", "attention_mask",
            "pixel_values", "pixel_attention_mask",
            "pixel_values_videos", "pixel_values_videos_mask",
            "image_grid_thw", "video_grid_thw", "image_sizes",
            "input_features", "encoder_outputs", "image_embeddings",
        }

        self.model.eval()
        total = 0
        correct = 0
        latencies = []

        labels = list(label_space)
        idx_of = {c: i for i, c in enumerate(labels)}
        C = len(labels)
        cm = np.zeros((C, C), dtype=int)

        for _, batch in enumerate(dev_loader):
            B = len(batch["gold_label_str"])
            for i in range(B):
                s_full = batch["hf_inputs"]
                s = {k: v for k, v in s_full.items() if k in _GEN_ALLOW}
                for k, v in list(s.items()):
                    if torch.is_tensor(v) and v.dim() >= 1 and v.size(0) == B:
                        s[k] = v[i:i+1].to(self.device)

                gold_label = batch["gold_label_str"][i]

                # 生成预测
                t1 = time.time()
                out = self.model.generate(
                    **s,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                )
                latencies.append((time.time() - t1) * 1000.0)  # 毫秒级延迟

                # 解析生成结果
                attn = s.get("attention_mask", None)
                if attn is not None:
                    attn_lens = attn[0].sum().item()
                    out = out[0, int(attn_lens):]
                else:
                    out = out[0]
                text_out = self.tok.decode(out, skip_special_tokens=True)

                # 解析标签
                pred = parse_label_from_output(text_out, label_space)
                correct += int(pred == gold_label)
                total += 1

                # 更新混淆矩阵
                if gold_label in idx_of and pred in idx_of:
                    cm[idx_of[gold_label], idx_of[pred]] += 1

        # 计算指标
        acc = correct / max(total, 1)
        avg_latency = float(sum(latencies) / max(len(latencies), 1)) if latencies else 0.0

        # 每类准确率
        per_cls = {}
        for c in range(C):
            n = cm[c].sum()
            per_cls[labels[c]] = (cm[c, c] / n) if n > 0 else 0.0

        # 打印评估结果
        print(f"\n[eval] 准确率(acc)={acc:.4f} 平均延迟(ms)={avg_latency:.1f}")
        print(f"[eval] 每类准确率: " + ", ".join(f"{k}:{v:.4f}" for k, v in per_cls.items()))
        print("[eval] 混淆矩阵（行=真实标签，列=预测标签）:")
        for r in range(C):
            row = " ".join(f"{cm[r, c]:3d}" for c in range(C))
            print(f"  {labels[r]:>10s} | {row}")
        print()

        # 恢复文本Prompt Dropout
        if self._row_replacer is not None:
            self._row_replacer.dropout_p = old_p
        self.model.train()
        return acc, avg_latency

    @torch.no_grad()
    def _apply_prompt_dropout(self, p=0.1):
        """应用文本Prompt Dropout"""
        if self._row_replacer is not None:
            self._row_replacer.dropout_p = float(max(0.0, min(1.0, p)))

    def _save_step_ckpt(self, step: int, current_loss: float, current_lr: float):
        """保存带step后缀的checkpoint（包含模式开关标识）"""
        if not self.cfg.step_ckpt_dir:
            return
        
        # 构建带step后缀的文件名（6位数字对齐）
        step_ckpt_path = os.path.join(
            self.cfg.step_ckpt_dir,
            f"prompt_ckpt_step_{step:06d}.pt"
        )
        print(f"[save] 保存step={step}的中间checkpoint：{step_ckpt_path}")
        
        # 保存的内容（包含模式开关，用于推理时适配）
        step_save_dict = {
            # 模式开关标识（关键）
            "text_only": self.text_only,
            "visual_only": self.visual_only,
            # 文本软提示参数
            "soft_tokens": self.soft_tokens if self.soft_param is not None else None,
            "soft_vecs": self.soft_param.detach().cpu() if self.soft_param is not None else None,
            # 视觉软提示参数
            "visual_sp_n_tokens": self.n_visual_sp if self.visual_sp_param is not None else 0,
            "visual_sp_vecs": self.visual_sp_param.detach().cpu() if self.visual_sp_param is not None else None,
            # 训练状态
            "step": step,
            "current_loss": current_loss,
            "current_lr": current_lr,
            "optimizer_state_dict": self.opt.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }
        
        save_prompt_ckpt(step_ckpt_path, step_save_dict)

    def fit(self, loader: DataLoader, dev_loader: Optional[DataLoader], target_builder):
        """训练主函数"""
        self.model.train()
        step = 0
        best_metric = -float("inf")
        gnorm_cache = 0.0
        raw_soft_g_cache = 0.0
        early_stop_counter = 0

        print(f"\n[train] 开始训练，总步数：{self.cfg.max_steps}，梯度累积步数：{self.cfg.grad_accum}")
        print(f"[train] 评估间隔：{self.cfg.eval_every}步，早停耐心值：{self.cfg.early_stop_patience}")
        print(f"[train] 训练模式：{'仅文本软提示' if self.text_only else ('仅视觉软提示' if self.visual_only else '文本+视觉软提示')}\n")

        try:
            while step < self.cfg.max_steps:
                for batch in loader:
                    # 构建目标序列
                    targets = [target_builder(lbl) for lbl in batch["gold_label_str"]]
                    target_ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in targets]
                    input_ids2, attn2, labels, others = self._pack_batch(batch["hf_inputs"], target_ids)

                    # 应用文本Prompt Dropout（仅文本模式或混合模式）
                    self._apply_prompt_dropout(p=0.2)

                    # 前向传播（AMP自动混合精度）
                    with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                        out = self.model(input_ids=input_ids2, attention_mask=attn2, labels=labels,** others)
                        loss = out.loss

                        # 正则项（防止过拟合）
                        reg_loss = 0.0
                        if self.soft_param is not None:
                            # 文本软提示：锚点损失（接近初始化值）
                            lambda_anchor = 1e-3
                            anchor_l2 = F.mse_loss(self.soft_param, self.soft_init)
                            reg_loss += lambda_anchor * anchor_l2

                            # 文本软提示：正交性约束（避免token冗余）
                            lambda_ortho = 1e-3
                            S_text = F.normalize(self.soft_param, dim=1)
                            ortho_text = (S_text @ S_text.t() - torch.eye(len(self.soft_ids), device=S_text.device)).pow(2).mean()
                            reg_loss += lambda_ortho * ortho_text

                        # 视觉软提示：正交性约束（仅视觉模式或混合模式）
                        if self.visual_sp_param is not None:
                            lambda_ortho_visual = 1e-3
                            S_visual = F.normalize(self.visual_sp_param, dim=1)
                            ortho_visual = (S_visual @ S_visual.t() - torch.eye(self.n_visual_sp, device=S_visual.device)).pow(2).mean()
                            reg_loss += lambda_ortho_visual * ortho_visual

                        # 总损失
                        total_loss = loss + reg_loss

                    # 反向传播（梯度累积）
                    if self.scaler.is_enabled():
                        self.scaler.scale(total_loss / max(1, self.cfg.grad_accum)).backward()
                        if ((step + 1) % self.cfg.grad_accum) == 0:
                            # 梯度裁剪
                            self.scaler.unscale_(self.opt)
                            raw_soft_g_cache = (self.soft_param.grad.abs().mean().item() if (self.soft_param is not None and self.soft_param.grad is not None) else 0.0)
                            if self.cfg.max_grad_norm is not None:
                                params_to_clip = [p for p in [self.soft_param, self.visual_sp_param] if p is not None]
                                gnorm_cache = torch.nn.utils.clip_grad_norm_(params_to_clip, self.cfg.max_grad_norm).item()
                            # 优化器步骤
                            self.scaler.step(self.opt)
                            self.scaler.update()
                            self.opt.zero_grad(set_to_none=True)
                            self.scheduler.step()
                    else:
                        (total_loss / max(1, self.cfg.grad_accum)).backward()
                        if ((step + 1) % self.cfg.grad_accum) == 0:
                            # 梯度裁剪
                            raw_soft_g_cache = (self.soft_param.grad.abs().mean().item() if (self.soft_param is not None and self.soft_param.grad is not None) else 0.0)
                            if self.cfg.max_grad_norm is not None:
                                params_to_clip = [p for p in [self.soft_param, self.visual_sp_param] if p is not None]
                                gnorm_cache = torch.nn.utils.clip_grad_norm_(params_to_clip, self.cfg.max_grad_norm).item()
                            # 优化器步骤
                            self.opt.step()
                            self.opt.zero_grad(set_to_none=True)
                            self.scheduler.step()

                    # 日志打印
                    if (step % self.cfg.log_every) == 0:
                        lr = self.opt.param_groups[0]["lr"]
                        with torch.no_grad():
                            # 文本软提示相关指标
                            text_cos = 0.0
                            ortho_mse_text = 0.0
                            if self.soft_param is not None:
                                cur_text = F.normalize(self.soft_param, dim=1)
                                init_text = F.normalize(self.soft_init, dim=1)
                                text_cos = (cur_text * init_text).sum(dim=1).mean().item()
                                ortho_mse_text = (cur_text @ cur_text.t() - torch.eye(len(self.soft_ids), device=cur_text.device)).pow(2).mean().item()

                            # 视觉软提示相关指标
                            ortho_mse_visual = 0.0
                            if self.visual_sp_param is not None:
                                cur_visual = F.normalize(self.visual_sp_param, dim=1)
                                ortho_mse_visual = (cur_visual @ cur_visual.t() - torch.eye(self.n_visual_sp, device=cur_visual.device)).pow(2).mean().item()

                        # 打印日志
                        print(
                            f"[step={step}] 总损失={total_loss.item():.6f} 原始损失={loss.item():.6f} 正则损失={reg_loss.item():.6f} | "
                            f"梯度均值={raw_soft_g_cache:.6e} 梯度范数={gnorm_cache:.3f} | "
                            f"文本相似度={text_cos:.3f} 文本正交性={ortho_mse_text:.4f} | "
                            f"视觉正交性={ortho_mse_visual:.4f} 学习率={lr:.3e}",
                            flush=True
                        )

                    # 按步保存中间checkpoint
                    if (step % self.cfg.save_every_step) == 0:
                        current_lr = self.opt.param_groups[0]["lr"]
                        self._save_step_ckpt(step=step, current_loss=total_loss.item(), current_lr=current_lr)

                    # 评估与保存最佳模型
                    if (self.cfg.eval_every > 0) and (dev_loader is not None) and (step % self.cfg.eval_every == 0):
                        val_acc, val_lat = self.eval_like_infer_generation(dev_loader, self.label_space)
                        
                        # 早停逻辑
                        if val_acc > best_metric:
                            best_metric = val_acc
                            # 保存最佳模型
                            best_save_dict = {
                                "text_only": self.text_only,
                                "visual_only": self.visual_only,
                                "soft_tokens": self.soft_tokens if self.soft_param is not None else None,
                                "soft_vecs": self.soft_param.detach().cpu() if self.soft_param is not None else None,
                                "visual_sp_n_tokens": self.n_visual_sp if self.visual_sp_param is not None else 0,
                                "visual_sp_vecs": self.visual_sp_param.detach().cpu() if self.visual_sp_param is not None else None,
                                "best_step": step,
                                "best_val_acc": best_metric,
                                "optimizer_state_dict": self.opt.state_dict(),
                                "scheduler_state_dict": self.scheduler.state_dict(),
                                "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
                            }
                            save_prompt_ckpt(self.cfg.ckpt_best, best_save_dict)
                            print(f"[save] 最佳模型已更新（step={step}，acc={best_metric:.4f}）：{self.cfg.ckpt_best}")
                            early_stop_counter = 0
                        else:
                            early_stop_counter += 1
                            print(f"[early-stop] 未更新最佳模型，计数器：{early_stop_counter}/{self.cfg.early_stop_patience}")
                            if early_stop_counter >= self.cfg.early_stop_patience:
                                print(f"[early-stop] 早停条件触发，训练提前结束")
                                return

                    # 步数递增
                    step += 1
                    if step >= self.cfg.max_steps:
                        break
        finally:
            # 训练结束移除视觉钩子（避免内存泄漏）
            self._remove_visual_hook()

        # 训练结束：保存最终模型
        final_save_dict = {
            "text_only": self.text_only,
            "visual_only": self.visual_only,
            "soft_tokens": self.soft_tokens if self.soft_param is not None else None,
            "soft_vecs": self.soft_param.detach().cpu() if self.soft_param is not None else None,
            "visual_sp_n_tokens": self.n_visual_sp if self.visual_sp_param is not None else 0,
            "visual_sp_vecs": self.visual_sp_param.detach().cpu() if self.visual_sp_param is not None else None,
            "final_step": step,
            "final_val_acc": best_metric,
            "optimizer_state_dict": self.opt.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }
        save_prompt_ckpt(self.cfg.save_ckpt, final_save_dict)
        print(f"\n[训练完成] 最终模型已保存到：{self.cfg.save_ckpt}")
        print(f"[训练完成] 最佳模型准确率：{best_metric:.4f}（step={step}）")

    def __del__(self):
        """销毁时移除钩子，避免内存泄漏"""
        self._remove_visual_hook()