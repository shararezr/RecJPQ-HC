# RecJPQ-HC

**Memory-efficient sequential recommendation with Joint Product Quantization (JPQ) and pluggable centroid-assignment strategies.**

RecJPQ-HC implements and compares *Joint Product Quantization* for the item-embedding tables of two popular sequential recommenders — **SASRec / gSASRec** and **BERT4Rec** — and studies how the choice of **centroid-assignment strategy** (SVD, BPR, random, Quotient–Remainder, SVD+OPQ) affects accuracy, embedding-table size, code utilization and the effective rank of the learned representation.

The repository is organised as two self-contained experiment suites that share the same JPQ layer and the same centroid-assignment code:

| Directory | Backbone model | Training signal |
|-----------|----------------|-----------------|
| [`sasrec_jpq/`](sasrec_jpq) | SASRec / gSASRec (causal Transformer decoder) | Next-item prediction (cross-entropy or gBCE) |
| [`bert4rec_jpq/`](bert4rec_jpq) | BERT4Rec (bidirectional HuggingFace BERT encoder) | Masked-item prediction (cross-entropy) |

---

## Table of contents

- [Why JPQ?](#why-jpq)
- [How it works](#how-it-works)
  - [The JPQ item-embedding layer](#the-jpq-item-embedding-layer)
  - [Centroid-assignment strategies](#centroid-assignment-strategies)
- [Repository layout](#repository-layout)
- [Installation](#installation)
- [Datasets](#datasets)
- [Quick start](#quick-start)
  - [SASRec / gSASRec + JPQ](#sasrec--gsasrec--jpq)
  - [BERT4Rec + JPQ](#bert4rec--jpq)
- [Configuration reference](#configuration-reference)
- [Diagnostics & measurements](#diagnostics--measurements)
- [Reproducibility](#reproducibility)
- [Acknowledgements](#acknowledgements)

---

## Why JPQ?

In sequential recommendation the **item-embedding table is usually the largest part of the model**. A standard table needs `num_items × embedding_dim` trainable parameters, which becomes prohibitive for catalogues with millions of items and hurts *cold / long-tail* items that rarely receive gradient updates.

**Product Quantization (PQ)** replaces the full table with:

* a small, **trainable codebook** of centroids, and
* a compact, **non-trainable code** per item (a few bytes).

Each item embedding is *reconstructed on the fly* by concatenating one centroid from each of `M` sub-spaces. This shrinks the per-item cost from `embedding_dim` floats to `M` bytes, while centroids are shared across all items so the long tail benefits from parameters learned on popular items.

**Joint PQ (JPQ)** trains the centroids *jointly with the recommender* (rather than freezing a pre-trained quantizer), so the quantized embeddings are optimised directly for the recommendation objective.

This repo additionally treats **how the discrete codes are initially assigned** as a first-class experimental knob — see [centroid-assignment strategies](#centroid-assignment-strategies).

---

## How it works

### The JPQ item-embedding layer

The shared layer lives in `rec_jpq_layer.py` (identical interface in both suites) and exposes a drop-in replacement for `nn.Embedding`.

Given `embedding_dim = D` and `pq_m = M`:

* The embedding space is split into `M` sub-spaces, each of size `D / M` (so `D` **must be divisible by `M`**).
* `item_codes`: a non-trainable `[num_items + 1, M]` buffer of `uint8` codes (the extra row is the padding item).
* `centroids`: a trainable `[M, K, D/M]` parameter, where `K = 256` codes per sub-space (`K = ceil(sqrt(num_items))` for the `qr` strategy).
* **Forward** (`item_id → embedding`): look up each item's `M` codes, gather the matching centroid from every sub-space, and concatenate them into a `D`-dimensional vector. Padding positions are masked to zero.
* **Scoring** is done directly in code space for efficiency:
  * `score_sequence_items` — scores a set of candidate items per position (used by the sampled-softmax / gBCE loss).
  * `score_all_items` — scores the full catalogue from a sequence representation (used for full cross-entropy and for inference / top-k retrieval).

`StandardItemEmbedding` provides the same interface backed by a plain `nn.Embedding`, so you can run an un-quantized baseline by setting `use_jpq=False` (SASRec suite).

```
item id ──► [c₁, c₂, …, c_M]            (M uint8 codes, fixed before training)
                │   │        │
                ▼   ▼        ▼
          centroid lookups in M codebooks (trainable)
                │   │        │
                └───┴───…────┘ concat ──► D-dim item embedding
```

### Centroid-assignment strategies

Before training, every item is assigned its `M` discrete codes by an *assignment strategy* (selected via `codes_strategy`). Strategies live in `centroid_assignment_strategies/` and all subclass `CentroidAssignmentStragety`:

| `codes_strategy` | Module | Idea |
|------------------|--------|------|
| `svd`     | `svd_strategy.py`     | TruncatedSVD of the user×item matrix → per-component **quantile binning** (`KBinsDiscretizer`) into 256 codes. |
| `bpr`     | `bpr_strategy.py`     | LightFM **MF-BPR** item factors → quantile binning into 256 codes. |
| `random`  | `random_strategy.py`  | Random codes — a baseline / ablation. |
| `qr`      | `qr_strategy.py`      | **Quotient–Remainder** trick (requires `pq_m = 2`): code = `(id // d, id % d)` with `d = ceil(sqrt(num_items))`. |
| `svd_opq` | `svd_opqfixed.py`     | SVD item embeddings → optional **OPQ rotation** → **FAISS `ProductQuantizer`**. Returns *both* codes **and** initialised centroids, so the codebook starts from the PQ solution rather than random. |

> A strategy may return either just the codes, or a `(codes, centroids_init)` tuple. When centroids are returned (e.g. `svd_opq`), the JPQ layer initialises its trainable codebook from them.

---

## Repository layout

```
RecJPQ-HC/
├── sasrec_jpq/                     # SASRec / gSASRec + JPQ
│   ├── gsasrec.py                  # GSASRec model (Transformer decoder + JPQ layer)
│   ├── transformer_decoder.py      # SASRec-style multi-head attention block
│   ├── rec_jpq_layer.py            # JPQ ItemCodeLayer + StandardItemEmbedding
│   ├── config.py                   # GSASRecExperimentConfig (hyper-parameters)
│   ├── train.py                    # Training entry point
│   ├── evaluate.py                 # Test-set evaluation from a checkpoint
│   ├── eval_utils.py               # ir_measures-based evaluation loop
│   ├── measurements.py             # Quantization error, effective rank, cold-start eval
│   ├── dataset_utils.py            # Datasets / dataloaders / negative sampling
│   ├── utils.py                    # Config loading, model building, device
│   ├── centroid_assignment_strategies/   # svd / bpr / random / qr / svd_opq …
│   ├── config/                     # Per-dataset experiment configs
│   └── datasets/                   # ml1m / beauty / sports / toys + preprocessing
│
└── bert4rec_jpq/                   # BERT4Rec + JPQ
    ├── bert4rec_jpq_model.py       # BERT4RecJPQ (HuggingFace BertModel + JPQ layer)
    ├── bert4rec.py                 # Recommender wrapper (aprec-style API)
    ├── collator.py                 # MaskingCollator (masked-item training/eval)
    ├── config.py                   # BERT4RecExperimentConfig
    ├── train.py / evaluate.py      # Entry points
    ├── eval_utils.py / measurements.py / dataset_utils.py / utils.py
    ├── aprec/                       # Minimal API (Action, Recommender, ItemId, …)
    ├── centroid_assignment_strategies/
    ├── config/
    └── datasets/
```

---

## Installation

Python 3.9+ and PyTorch are required. The two suites share the same core dependencies.

```bash
# 1. Create an environment
conda create -n recjpq python=3.10 -y
conda activate recjpq

# 2. Core dependencies
pip install torch numpy scipy scikit-learn tqdm torchinfo ir_measures

# 3. Product-quantization backend (used by svd_opq and as a baseline)
#    conda is the most reliable way to install FAISS:
conda install -c pytorch faiss-cpu      # or faiss-gpu

# 4. Strategy- and model-specific extras
pip install lightfm          # BPR centroid-assignment strategy
pip install transformers     # BERT4Rec backbone (HuggingFace)
pip install mmh3 requests tensorboard   # BERT4Rec wrapper + dataset download + logging
```

> **Note on FAISS / LightFM:** these are optional unless you use the strategies that need them (`svd_opq` → FAISS, `bpr` → LightFM). `pip install faiss-cpu` also works on many platforms but conda is recommended.

There is no `setup.py`; scripts are run directly from inside `sasrec_jpq/` or `bert4rec_jpq/` (paths such as `datasets/...` and `config/...` are resolved relative to the working directory).

---

## Datasets

Four standard sequential-recommendation benchmarks are supported:

| Name | `dataset_name` | Source |
|------|----------------|--------|
| MovieLens-1M | `ml1m` | downloaded automatically (SASRec-mapped version) |
| Amazon Beauty | `beauty` | `Beauty.txt` (included) |
| Amazon Sports & Outdoors | `sports` | `Sports_and_Outdoors.txt` (included) |
| Amazon Toys & Games | `toys` | `Toys_and_Games.txt` (included) |

### Preprocessing

Each dataset folder contains a `preprocess_<name>.py` script. It performs a **leave-one-out** temporal split (last interaction → test, second-to-last → validation for 512 randomly chosen users), remaps item IDs to a contiguous range, and writes:

```
datasets/<name>/
├── dataset_stats.json          # {num_users, num_items, num_interactions}
├── train/input.txt             # one space-separated item sequence per user
├── val/input.txt, val/output.txt
└── test/input.txt, test/output.txt
```

Run it once per dataset before training, from inside the suite directory:

```bash
cd sasrec_jpq                         # or bert4rec_jpq
python datasets/ml1m/preprocess_ml1m.py        # also downloads ml-1m
python datasets/beauty/preprocess_beauty.py
python datasets/sports/preprocess_sports.py
python datasets/toys/preprocess_toys.py
```

The padding item is assigned id `num_items` (the JPQ layer reserves an extra embedding row for it), and `dataset_stats.json` is read at runtime to recover `num_items`.

---

## Quick start

### SASRec / gSASRec + JPQ

```bash
cd sasrec_jpq

# 1. Preprocess (once)
python datasets/ml1m/preprocess_ml1m.py

# 2. Train (config selects dataset + hyper-parameters)
python train.py --config config/config_ml1m.py

# 3. Evaluate the best checkpoint on the test set
python evaluate.py --config config/config_ml1m.py \
                   --checkpoint models/gsasrec-ml1m-...best....pt
```

* Training validates every epoch with `nDCG@10` and keeps the best checkpoint under `models/` (early stopping after `early_stopping_patience` non-improving epochs).
* Loss is selectable per config: `loss='ce'` (full-catalogue cross-entropy) or `loss='gbce'` (generalised BCE with `negs_per_pos` sampled negatives and temperature `gbce_t`).
* Set `use_jpq=False` to train a standard (un-quantized) SASRec baseline.

### BERT4Rec + JPQ

```bash
cd bert4rec_jpq

# 1. Preprocess (once)
python datasets/ml1m/preprocess_ml1m.py

# 2. Train
python train.py --config config/config_ml1m.py

# 3. Evaluate
python evaluate.py --config config/config_ml1m.py \
                   --checkpoint models/bert4rec-ml1m-...best....pt
```

BERT4Rec uses a HuggingFace `BertModel` encoder fed with JPQ embeddings (`inputs_embeds`), a learnable `[MASK]` embedding, left-padding with remapped position ids, and a `MaskingCollator` that masks `masking_prob` of each training sequence (and the final item for validation/test). Training optimises cross-entropy over the masked positions and tracks TensorBoard logs + an optional profiler trace.

---

## Configuration reference

Experiments are plain Python files under `config/` that instantiate an experiment-config object. Pass one with `--config`. A template is provided in `config/config_template.py`.

### `GSASRecExperimentConfig` (SASRec suite)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_name` | — | `ml1m` / `beauty` / `sports` / `toys`. |
| `sequence_length` | `50` | Max items per user sequence. |
| `embedding_dim` | `128` | Item/sequence embedding size `D` (must be divisible by `pq_m`). |
| `num_heads`, `num_blocks` | `1`, `2` | Transformer attention heads / decoder blocks. |
| `dropout_rate` | `0.5` | Dropout. |
| `pq_m` | `16` | Number of PQ sub-spaces `M`. Larger → smaller codebook footprint, coarser embeddings. |
| `codes_strategy` | `svd` | Centroid-assignment strategy (`svd` / `bpr` / `random` / `qr` / `svd_opq`). |
| `use_jpq` | `True` | `False` falls back to a standard embedding table. |
| `loss` | `ce` | `ce` (cross-entropy) or `gbce` (generalised BCE). |
| `negs_per_pos` | `1` | Negatives per positive for the gBCE loss. |
| `gbce_t` | `0` | gBCE temperature (0 ≈ standard BCE; →1 sharpens toward full softmax). |
| `learning_rate` | `1e-3` | AdamW learning rate. |
| `train_batch_size` / `eval_batch_size` | `128` / `512` | Batch sizes. |
| `max_epochs` / `max_batches_per_epoch` | `10000` / `100` | Training budget. |
| `early_stopping_patience` | `200` | Stop after N non-improving validations. |
| `metrics` / `val_metric` | `nDCG@10, R@10, R@1` / `nDCG@10` | `ir_measures` metrics. |
| `filter_rated` | `True` | Exclude already-seen items from recommendations. |
| `reuse_item_embeddings` | `True` | Share input/output embeddings. |
| `seed` | `42` | Random seed. |

### `BERT4RecExperimentConfig` (BERT4Rec suite)

Mirrors the above with BERT-specific names: `embedding_size`, `num_attention_heads`, `num_hidden_layers`, `hidden_act`, `attention_probs_dropout_prob`, `hidden_dropout_prob`, `masking_prob`, `max_steps_per_epoch`, `lr`, plus the shared `pq_m`, `codes_strategy`, `metrics`, etc.

Pre-made configs are provided for every dataset (`config_<dataset>.py`).

---

## Diagnostics & measurements

`measurements.py` (both suites) records the things that matter for a quantization study, before and after training:

* **Code utilization** — how many of the 256 codes per sub-space are actually used (a proxy for representational diversity / collapse).
* **Quantization error** (MSE / RMSE) when reference embeddings are available.
* **Effective rank** — `exp(entropy(normalised singular values))` of the reconstructed item-embedding matrix, i.e. how many dimensions are effectively in use.
* **Peak GPU memory** (`torch.cuda.max_memory_allocated`).

The SASRec suite also includes **cold-start evaluation** helpers (`compute_item_popularity`, `get_cold_start_items`, `evaluate_coldstart`) that report metrics separately on the least-popular (long-tail) items vs. the rest.

Results are written to `results/` as JSON (e.g. `eval_<dataset>_emb<D>_pq<M>.json`, `tmp_measurements.json`, `tmp_gpu_mem.json`).

---

## Reproducibility

All entry points seed `random`, `numpy` and `torch` (seed `42` by default) and enable deterministic cuDNN. FAISS is pinned to a single thread in the PQ-based strategies to keep code assignment deterministic. Note that `train.py` in the SASRec suite seeds explicitly to `42` regardless of `config.seed`.

---

## Acknowledgements

This work builds on the SASRec, gSASRec and BERT4Rec sequential-recommendation models and on the RecJPQ line of work on product-quantized item embeddings. The MovieLens-1M data uses the SASRec-mapped version, and evaluation is performed with [`ir_measures`](https://ir-measur.es/). FAISS is used for product quantization and OPQ, and LightFM for the BPR assignment strategy.
