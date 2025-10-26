# VLM-MSA: A VLM-based Multimodal Prompt Learning Framework for Few-shot Sentiment Analysis

**VLM‑MSA** is a research framework for **few‑shot multimodal sentiment analysis** (coarse‑grained & fine‑grained). It takes **image + text** as input and predicts sentiment. In few‑shot settings, full finetuning is often sub‑optimal; we instead train **lightweight prompt parameters** and retrieve **in‑context demonstrations** to boost a strong **VLM backend (Qwen2.5‑VL)**.

---

## Key Features

* ✅ **RAG (Retrieval‑Augmented Generation)**: retrieve demonstrations from the **training split** with offline indices; supports **global**, **balanced**, and **per‑class** modes.
* ✅ **Soft Visual‑Text Prompts**: learn discrete soft tokens injected into the tokenizer/vocabulary; only update the corresponding embedding rows.
* 🚧 **Prompt‑as‑Expert (MoE)**: Mixture‑of‑Experts style fusion of diverse prompts (hard/soft, vis‑text). *(Planned)*
* 🚧 **Contrastive Decoding**: negative candidates / anti‑prompts to sharpen decisions. *(Planned)*


---

## Table of Contents

* [Installation](#installation)
* [Datasets & Labels](#datasets--labels)
* [Data Preparation (RAG indices)](#data-preparation-rag-indices)
* [Training: Soft Prompt Tuning](#training-soft-prompt-tuning)
* [Evaluation / Inference](#evaluation--inference)
* [RAG Module Details](#rag-module-details)
* [Templates (Coarse & Fine)](#templates-coarse--fine)
* [Results](#results)
* [FAQ](#faq)
* [License & Citation](#license--citation)
* [Acknowledgements](#acknowledgements)

---

## Installation
**Pretrained Backends**

* **Qwen2.5‑VL‑7B‑Instruct** (VLM): `Qwen/Qwen2.5-VL-7B-Instruct` or a local path (e.g. `/home/lym/models/qwen2.5_vl`).
* **Sentence Embedding** model for RAG: `sentence-transformers/sbert-roberta-large` or local path (e.g. `/home/lym/MultiPoint/models/sbert-roberta-large`).

You can configure paths via environment variables or CLI flags (see below).

---

## Datasets & Labels

Supported datasets:

* **Coarse**: `mvsa-s`, `mvsa-m`, `tumemo`, `tumblr`
* **Fine (Aspect)**: `t2015`, `t2017`, `masad`

Label spaces (examples):

* `mvsa-*`: `negative / neutral / positive`
* `tumemo`: `angry, bored, calm, fear, happy, love, sad`
* `masad`: `negative / positive`
* `t2015 / t2017`: `negative / neutral / positive` (with **aspect**)

See [`dataset_info.py`](./src/dataset_info.py or similar) for full registry.

---

## Data Preparation (RAG indices)

Run **once** under `datasets/` to preprocess TSVs, build text embeddings, and precompute offline top‑k mappings.

```bash
cd datasets
bash run_data.sh
```

`run_data.sh` does the following:

1. Convert TSV to JSON (`data_tsv2json.py`) for splits: `train_few1`, `dev_few1`, `test`.
2. Generate text embeddings (`generate.sh`) using your sentence encoder (e.g., SBERT).
3. Precompute similarity from **test→train/dev** for offline retrieval (`topk.sh` for `val` / `test`).

**Environment / Paths**

* SBERT path example: `/home/lym/MultiPoint/models/sbert-roberta-large`
* Qwen2.5‑VL path: `/home/lym/models/qwen2.5_vl`

---

## Training: Soft Prompt Tuning

Train discrete **soft tokens** (only the corresponding embedding rows are updated). Full model weights are **frozen**.

**One‑liner**

```bash
bash train_softprompt.sh
```

**Script (reference)**

```bash
CUDA_VISIBLE_DEVICES=0 \
SP_N_TOKENS=8 \
python src/prompt_tuning/train_soft_prompt.py \
  --model "$QWEN_VL" \
  --data_dir "datasets/mvsa-s" \
  --img_dir  "datasets/mvsa-s/imgs" \
  --tsv      "test.tsv" \
  --train_tsv train_few1.tsv \
  --dev_tsv   dev_few1.tsv \
  --dtype bf16 \
  --min_pixels 224 --max_pixels 1024 \
  --batch_size 4 \
  --sp_mode generic \
  --sp_lr 5e-3 --sp_steps 1000 \
  --sp_ckpt ./prompt_ckpt.pt
```

Training highlights:

* **Only** the `<soft*>` token rows in the embedding are trained (others frozen).
* AMP (fp16/bf16) supported; cosine LR with warmup; optional prompt‑dropout & anchor loss.
* Dev evaluation mimics inference (`generate`) to avoid train/test mismatch.

Artifacts:

* `prompt_ckpt.pt` and `prompt_ckpt.best.pt` contain only `{soft_tokens, soft_vecs}`.

---

## Evaluation / Inference

Batch evaluation with (optional) multi‑GPU **DDP** and offline **RAG** demos.

```bash
bash run.sh
```

Key envs in `run.sh`:

* `USE_DEMO=0|1` — enable/disable RAG demos
* `DEMO_TOPK` — number of retrieved demos per sample
* `DEMO_MODE=global|balanced|perclass`
* `TRAIN_JSONL` — path to `train_few1.json` (used when `USE_DEMO=1`)
* `SOFT_PROMPT_CKPT` — path to soft‑prompt checkpoint

Outputs:

* Prediction logs and optional raw generations
* If RAG is on, diagnostics and retrieval stats per sample

---

## RAG Module Details

* **Indexing**: offline precomputation of **test→train/dev** nearest neighbors with SBERT embeddings.
* **Modes**:

  * **global**: top‑K nearest overall
  * **balanced**: round‑robin from per‑class pools (class balance)
  * **perclass**: top‑K within each class
* **Demo formatting**: each retrieved item becomes a `(user, assistant)` pair (image optional), assistant returns `{"label": "<class>"}` as supervision/ICL signal.
* **Fine‑grained (aspect)**: replace `$T$ / $t$` placeholder with the concrete aspect; optionally include explicit `Aspect:` line in demos.

---

## Templates (Coarse & Fine)

Two canonical variants are provided for both **coarse** and **fine** settings. Examples:

* **Coarse**

  * `t1_coarse`: `[CLS] [MASK] ... [SEP]` (+ image in default multimodal)
  * `t2_coarse`: `Text: "..." Sentiment of text: [MASK]`
* **Fine (with Aspect)**

  * `t1_fine`: text‑only / default multimodal
  * `t2_fine`: `Text: "..." Aspect: "..." Sentiment of aspect: [MASK]`

See implementations in `labels_and_templates.py` / `dataset_info.py` and `prompts/`.

---

## Results

**Main comparison** (Accuracy / Macro‑F1 / Weighted‑F1). Numbers from your runs; see Feishu for full context.

| dataset                 |    MVSA‑S |           |           |    MVSA‑M |           |           |     MASAD |           |           |
| ----------------------- | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: | --------: |
| metric                  |       ACC |    MAC‑F1 |    Wtd‑F1 |       ACC |    MAC‑F1 |    Wtd‑F1 |       ACC |    MAC‑F1 |    Wtd‑F1 |
| **UP‑MPF (T1)**         |     58.21 |     51.08 |     58.49 |     55.97 |     44.63 |     57.79 |     75.82 |     74.63 |     76.19 |
| **UP‑MPF (T2)**         |     57.77 |     52.10 |     59.74 |     58.10 |     45.29 |     58.50 |     74.85 |     73.73 |     75.24 |
| **MultiPoint w/o Demo** | **62.25** | **56.07** | **64.37** | **62.81** | **51.52** | **63.05** | **80.39** | **78.27** | **80.20** |
| **MultiPoint w/ Demo**  |     60.50 |     55.03 |     63.30 |     59.65 |     50.89 |     61.09 |     79.03 |     76.89 |     79.30 |
| **Zero‑shot MMVLM**     |     56.13 |     50.69 |     59.95 | **63.68** | **52.05** | **64.04** |     77.00 |     71.84 |     75.26 |

| dataset                 | Twitter‑15 |           |           | Twitter‑17 |           |           |    TumEmo |           |           |
| ----------------------- | ---------: | --------: | --------: | ---------: | --------: | --------: | --------: | --------: | --------: |
| metric                  |        ACC |    MAC‑F1 |    Wtd‑F1 |        ACC |    MAC‑F1 |    Wtd‑F1 |       ACC |    MAC‑F1 |    Wtd‑F1 |
| **UP‑MPF (T1)**         |      52.73 |     49.64 |     53.33 |      50.95 |     48.99 |     50.47 |     48.08 |     48.31 |     48.06 |
| **UP‑MPF (T2)**         |      56.03 |     53.06 |     56.41 |      52.61 |     52.44 |     51.96 |     48.27 |     48.61 |     48.34 |
| **MultiPoint w/o Demo** |  **61.72** | **60.04** | **62.09** |  **55.59** | **56.08** | **55.09** |     50.70 |     50.51 |     50.56 |
| **MultiPoint w/ Demo**  |      60.75 |     58.80 |     61.15 |      53.32 |     54.08 |     53.10 | **51.14** | **50.38** | **50.70** |
| **Zero‑shot MMVLM**     |      51.88 |     52.45 |     49.32 |  **57.78** | **56.50** | **54.17** |     47.56 |     47.70 |     47.67 |

**Prompt variant study (highlights)**

* Variants include `Strict`, `Image First`, `Text First`, `Conflict‑aware`, `Sarcasm/Irony‑aware`, and ensembles (`EN3`, `EN5`).
* Best variants differ by dataset; e.g., `Text First` often improves on MVSA‑S/M; ensembles provide modest gains without demos.

**Few‑shot k & retrieval mode study**

* Compared **Zero‑shot**, **1/3/5‑shot** baselines (MMVLM) and our **balanced/perclass** retrieval.
* On MVSA‑S, **2‑shot per‑class** achieves **64.71 ACC / 56.98 MAC‑F1 / 66.48 Wtd‑F1**; trends are similar across datasets.

> Full tables & figures: [https://i0hfv1fsook.feishu.cn/docx/La0rd2WHRoeyXTxTcBjcdobHnJc](https://i0hfv1fsook.feishu.cn/docx/La0rd2WHRoeyXTxTcBjcdobHnJc)

---

## Project Structure

```
VLM-MSA/
├─ datasets/
│  ├─ mvsa-s/ mvsa-m/ masad/ t2015/ t2017/ tumemo/
│  ├─ run_data.sh  data_tsv2json.py  generate.sh  precompute_topk.py  topk.sh  generate_emb.py
├─ logs/
│  ├─ logs.py  masad/  mvsa-m/  mvsa-s/  t2015/  t2017/  tumemo/
├─ src/
│  ├─ run.py                 # main eval/infer (DDP ready)
│  ├─ dataset.py             # TSV/JSON readers
│  ├─ dataset_info.py        # labels & templates registry
│  ├─ prompts.py             # prompt builders & templates
│  ├─ retrieve_demo.py       # RAG provider & offline indices
│  ├─ utils.py               # helpers
│  ├─ ensemble.py            # voting / ensembling utils
│  └─ prompt_tuning/
│     ├─ train_soft_prompt.py
│     ├─ prompt_learner.py   # SoftPromptLearner
│     └─ sp_utils.py         # soft token init
├─ train_softprompt.sh  run.sh  run_prompts.sh  run_ensemble.sh
├─ prompt_ckpt.pt  prompt_ckpt.best.pt  README.md  requirements.txt
└─ datasets/ imgs/ etc.
```

---

## FAQ

**Q1. How are soft prompts trained without touching the full model?**
We mask gradients to **only** update the embedding rows of `<soft*>` tokens; all other parameters are frozen.

**Q2. How is RAG different from on‑the‑fly retrieval?**
We precompute **offline** indices to make evaluation stable and fast; modes `global/balanced/perclass` trade off relevance and class balance.

---

## License & Citation

* **License**: Apache-2.0
* **Citation**: (TBD)

---

## Acknowledgements

* **Qwen2.5‑VL‑7B‑Instruct** as the VLM backbone.
* **SBERT (RoBERTa‑large)** for sentence embeddings.
* Datasets: **MVSA‑S/M**, **MASAD**, **Twitter 2015/2017 (ABSA)**, **TUMEMO**.
