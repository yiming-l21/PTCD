# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase, AutoProcessor
from retrieve_demo import DemoProvider

# --------- ckpt io ---------
def save_prompt_ckpt(path: str, state: Dict):
    torch.save(state, path)


def load_prompt_ckpt(path: str, map_location="cpu") -> Dict:
    return torch.load(path, map_location=map_location)


# --------- train cfg ---------
@dataclass
class TrainCfg:
    # 软提示通常需要更大的 lr（远大于全参微调）
    lr: float = 5e-2
    weight_decay: float = 0.0
    max_steps: int = 3000
    grad_accum: int = 4
    log_every: int = 20
    save_ckpt: str = "prompt_ckpt.pt"
    use_fp16: bool = True  # 若模型是bf16，会自动切到bf16并关闭GradScaler

    eval_every: int = 200
    ckpt_best: str = "prompt_ckpt.best.pt"
    early_stop_patience: int = 5  # 可选：未实现
    monitor: str = "acc"          # "acc" 或 "loss"
    minimize: bool = True         # 可选：未实现

    # 新增：warmup/decay、梯度裁剪、sanity check、初始化模板
    warmup_steps: int = 200
    max_grad_norm: float = 1.0
    do_sanity_overfit: bool = False
    sanity_steps: int = 200
    init_prompt: Optional[str] = (
        # 用简短、稳定的英文模板初始化软提示，能显著提升早期可学性
        "You are a helpful assistant that answers by returning JSON like {\"label\": \"<class>\"}."
    )


class SoftPromptLearner:
    """
    方案1：离散软token。仅训练 embedding 中 <soft*> 行，其他权重全冻结。
    监督：让模型生成 {"label": "<gold>"}。
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processor: AutoProcessor,
        template_variants: List[str],
        demo_provider: DemoProvider,
        label_space: List[str],
        train_cfg: TrainCfg,
        device: torch.device,
        soft_token_names: Optional[List[str]] = None
    ):
        self.model = model
        self.processor = processor
        self.tok = processor.tokenizer
        self.label_space = list(label_space)
        self.cfg = train_cfg
        self.device = device
        self.demo_provider = demo_provider
        self.template_variants = template_variants
        new_vocab = len(self.tok)
        lm_head_rows = self.model.get_output_embeddings().weight.shape[0]
        if lm_head_rows != new_vocab:
            self.model.resize_token_embeddings(new_vocab)
        self.model.config.vocab_size = new_vocab

        # --- 发现并校验 <soft*> tokens ---
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
                "未在 tokenizer 中找到任何 <soft*> token。请在外部先注册："
                " tokenizer.add_special_tokens({'additional_special_tokens': ['<soft0>',...]}), "
                "并调用 model.resize_token_embeddings(len(tokenizer))"
            )

        self.soft_ids = self.tok.convert_tokens_to_ids(self.soft_tokens)
        if any(x < 0 for x in self.soft_ids):
            raise RuntimeError(f"部分软token未在词表中：{self.soft_tokens} → {self.soft_ids}")

        # --- 冻结全模型，仅训练 embedding 的软行 ---
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.emb: nn.Parameter = self.model.get_input_embeddings().weight  # [vocab, hidden]
        self.emb.requires_grad_(True)
        V, H = self.emb.shape
        n_soft = len(self.soft_ids)
        effective_trainable = int(n_soft * H)
        print(f"[soft] vocab_size={V} hidden_size={H} n_soft_tokens={n_soft} trainable_params={effective_trainable}")

        # 只更新软token所在的行：通过hook屏蔽其它行的梯度
        soft_idx = torch.tensor(self.soft_ids, device=self.emb.device, dtype=torch.long)
        mask_rows = torch.zeros(self.emb.size(0), device=self.emb.device, dtype=self.emb.dtype)
        mask_rows[soft_idx] = 1.0

        def grad_mask_hook(grad: torch.Tensor) -> torch.Tensor:
            # grad: [vocab, hidden]
            return grad * mask_rows.view(-1, 1)

        self._hook_handle = self.emb.register_hook(grad_mask_hook)

        # 只优化 embedding（其余参数冻结且无梯度）
        self.opt = torch.optim.AdamW([self.emb], lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

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
            # FutureWarning: 推荐用 torch.amp.GradScaler('cuda', ...)
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        elif want_amp and self.compute_dtype == torch.bfloat16:
            self.amp_dtype = torch.bfloat16
            self.use_amp = True
            # bf16 不用 scaler
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)
        else:
            self.amp_dtype = torch.float32
            self.use_amp = False
            self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.pad_id = self.tok.pad_token_id

        # 学习率调度器：linear warmup + cosine decay
        self.scheduler = self._build_scheduler(self.opt, self.cfg.max_steps, self.cfg.warmup_steps)

        # 软提示初始化（把自然语言模板的 embedding 拷贝/平均到 <soft*> 行）
        if self.cfg.init_prompt:
            self._init_soft_from_text(self.cfg.init_prompt)

        # 启动自检
        with torch.no_grad():
            vocab = len(self.tok)
            lm_head_rows = self.model.get_output_embeddings().weight.shape[0]
            print(f"[soft] vocab={vocab}  lm_head_rows={lm_head_rows}  config.vocab_size={self.model.config.vocab_size}")
            print(f"[soft] tokens={self.soft_tokens}")
            print(f"[soft] ids={self.soft_ids}")
            print(f"[amp] compute_dtype={self.compute_dtype} amp_dtype={self.amp_dtype} use_amp={self.use_amp} scaler_enabled={self.scaler.is_enabled()}")

    # ---------- scheduler ----------
    def _build_scheduler(self, opt, max_steps: int, warmup_steps: int):
        # 先线性 warmup，再 cosine 衰减到 10% lr 尾值
        self.base_lr = self.cfg.lr
        self.min_lr = self.cfg.lr * 0.1

        def lr_lambda(step: int):
            if step < warmup_steps and warmup_steps > 0:
                return max(step / max(1, warmup_steps), 1e-8)
            t = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            t = min(max(t, 0.0), 1.0)
            return (self.min_lr / self.base_lr) + 0.5 * (1 - self.min_lr / self.base_lr) * (1 + math.cos(math.pi * t))

        return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # ---------- 用自然语言模板初始化软提示 ----------
    @torch.no_grad()
    def _init_soft_from_text(self, prompt: str):
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        if len(ids) == 0:
            print("[init] template tokenized to empty; skip init.")
            return
        emb_table = self.model.get_input_embeddings().weight  # [V,H]
        vecs = emb_table[torch.tensor(ids, device=emb_table.device, dtype=torch.long)]  # [L,H]
        if len(self.soft_ids) <= vecs.size(0):
            chosen = vecs[:len(self.soft_ids)]
        else:
            reps = (len(self.soft_ids) + vecs.size(0) - 1) // vecs.size(0)
            chosen = vecs.repeat(reps, 1)[:len(self.soft_ids)]
        noise = (torch.randn_like(chosen) * 0.01)
        chosen = chosen + noise
        self.emb[self.soft_ids] = chosen
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
        others = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                  for k, v in batch_inputs.items() if k not in ["input_ids", "attention_mask"]}

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
        dev_loader,              # 一组样本对象（和你推理那套 reader.read() 一致）
        label_space,
        max_new_tokens: int = 32,
    ) -> Tuple[float, float]:
        """
        返回 (val_acc, avg_latency_ms)，评估严格复用推理代码路径。
        """
        from utils import parse_label_from_output
        import numpy as np

        self.model.eval()
        total = 0
        correct = 0
        for idx, batch in enumerate(dev_loader):
            for i in range(len(batch["gold_label_str"])):
                s = batch["hf_inputs"]
                gold_label = batch["gold_label_str"][i]
                with torch.inference_mode():
                    out = self.model.generate(
                        **s,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                    attn = s["attention_mask"]
                    if attn is not None:
                        attn_lens = attn[i].sum().item()
                        out = out[i, attn_lens:]  # 去掉输入部分
                    else:
                        out = out[i]
                    text_out = self.tok.decode(out, skip_special_tokens=True)
                pred = parse_label_from_output(text_out, label_space)
                correct += int(pred == gold_label)
                total += 1

        acc = correct / max(total, 1)
        self.model.train()
        return acc


    # ---------- 小 batch 过拟合 ----------
    def _sanity_overfit(self, loader: DataLoader, target_builder):
        self.model.train()
        it = iter(loader)
        batch = next(it)
        for step in range(self.cfg.sanity_steps):
            targets = [target_builder(lbl) for lbl in batch["gold_label_str"]]
            target_ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in targets]
            input_ids2, attn2, labels, others = self._pack_batch(batch["hf_inputs"], target_ids)
            with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                out = self.model(input_ids=input_ids2, attention_mask=attn2, labels=labels, **others)
                loss = out.loss

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                if self.cfg.max_grad_norm is not None:
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_([self.emb], self.cfg.max_grad_norm)
                self.scaler.step(self.opt)
                self.scaler.update()
            else:
                loss.backward()
                if self.cfg.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_([self.emb], self.cfg.max_grad_norm)
                self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.scheduler.step()

            if step % 20 == 0:
                with torch.no_grad():
                    gold_txts = [target_builder(lbl) for lbl in batch["gold_label_str"]]
                    probs = self._avg_logprob(input_ids2, attn2, others, gold_txts)
                    g_all = (self.emb.grad.norm().item() if self.emb.grad is not None else 0.0)
                    g_soft = (self.emb.grad[self.soft_ids].norm().item() if self.emb.grad is not None else 0.0)
                    print(f"[sanity] step={step} loss={loss.item():.4f} gnorm={g_all:.3f} gnorm_soft={g_soft:.3f} avg_gold_lp={probs:.4f}")

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

    def fit(self, loader: DataLoader, dev_loader: Optional[DataLoader], target_builder):
        """
        训练：只更新 embedding 中的软行。保存 best 与最终 ckpt（均只保存软行子矩阵）。
        """
        # sanity check：先确保能在单 batch 上迅速过拟合
        if self.cfg.do_sanity_overfit:
            print("[sanity] begin single-batch overfit…")
            self._sanity_overfit(loader, target_builder)
            print("[sanity] done.")

        self.model.train()
        step = 0
        best_metric = -float("inf")

        # 为了正确打印，把梯度范数缓存变量准备好
        gnorm_cache = 0.0
        gnorm_soft_cache = 0.0

        while step < self.cfg.max_steps:
            for batch in loader:
                targets = [target_builder(lbl) for lbl in batch["gold_label_str"]]
                target_ids = [self.tok(t, add_special_tokens=False)["input_ids"] for t in targets]
                input_ids2, attn2, labels, others = self._pack_batch(batch["hf_inputs"], target_ids)
                with torch.autocast("cuda", dtype=self.amp_dtype, enabled=self.use_amp):
                    out = self.model(input_ids=input_ids2, attention_mask=attn2, labels=labels, **others)
                    loss = out.loss

                if self.scaler.is_enabled():
                    self.scaler.scale(loss / max(1, self.cfg.grad_accum)).backward()
                    if ((step + 1) % self.cfg.grad_accum) == 0:
                        # 在 step/zero_grad 之前计算并缓存梯度范数
                        self.scaler.unscale_(self.opt)
                        if self.cfg.max_grad_norm is not None:
                            gnorm_cache = torch.nn.utils.clip_grad_norm_([self.emb], self.cfg.max_grad_norm).item()
                        else:
                            gnorm_cache = (self.emb.grad.norm().item() if self.emb.grad is not None else 0.0)
                        gnorm_soft_cache = (self.emb.grad[self.soft_ids].norm().item()
                                            if self.emb.grad is not None else 0.0)

                        self.scaler.step(self.opt)
                        self.scaler.update()
                        self.opt.zero_grad(set_to_none=True)
                        self.scheduler.step()
                else:
                    (loss / max(1, self.cfg.grad_accum)).backward()
                    if ((step + 1) % self.cfg.grad_accum) == 0:
                        # 在 step/zero_grad 之前计算并缓存梯度范数
                        if self.cfg.max_grad_norm is not None:
                            gnorm_cache = torch.nn.utils.clip_grad_norm_([self.emb], self.cfg.max_grad_norm).item()
                        else:
                            gnorm_cache = (self.emb.grad.norm().item() if self.emb.grad is not None else 0.0)
                        gnorm_soft_cache = (self.emb.grad[self.soft_ids].norm().item()
                                            if self.emb.grad is not None else 0.0)

                        self.opt.step()
                        self.opt.zero_grad(set_to_none=True)
                        self.scheduler.step()

                if (step % self.cfg.log_every) == 0:
                    lr = self.opt.param_groups[0]["lr"]
                    print(f"[soft-prompt] step={step} loss={loss.item():.4f} "
                          f"gnorm={gnorm_cache:.3f} gnorm_soft={gnorm_soft_cache:.3f} lr={lr:.3e}",
                          flush=True)

                if (self.cfg.eval_every > 0) and (dev_loader is not None) and (step > 0) and (step % self.cfg.eval_every == 0):
                    val_acc = self.eval_like_infer_generation(dev_loader, self.label_space)
                    print(f"[soft-prompt] eval@{step}: val_acc={val_acc:.4f}", flush=True)
                    metric = val_acc
                    if metric > best_metric:
                        best_metric = metric
                        soft_vecs = self.emb[self.soft_ids].detach().cpu()
                        save_prompt_ckpt(self.cfg.ckpt_best, {
                            "soft_tokens": self.soft_tokens,
                            "soft_vecs": soft_vecs,
                        })
                        print(f"[soft-prompt] new best -> saved {self.cfg.ckpt_best}", flush=True)

                step += 1
                if step >= self.cfg.max_steps:
                    break

        # 保存最终 ckpt
        soft_vecs = self.emb[self.soft_ids].detach().cpu()
        save_prompt_ckpt(self.cfg.save_ckpt, {
            "soft_tokens": self.soft_tokens,
            "soft_vecs": soft_vecs,
        })
        print(f"[soft-prompt] saved: {self.cfg.save_ckpt}", flush=True)
