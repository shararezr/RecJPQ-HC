# RecJPQ-HC

**Higher Capacity Transfer into JPQ Item Codes for Scalable
Sequential Recommendation.**

Although transformer models have improved the effectiveness of sequential recommendation, their computational and memory costs make it challenging to scale to item catalogues containing millions of items. To address this limitation, RecJPQ, a recommendation method based on joint product quantisation, represents each item using compact discrete codes rather than large embedding tables, substantially reducing training memory requirements. These codes are typically initialised from a memory-efficient bootstrap representation, such as truncated SVD, before being quantised. However, because quantisation is inherently lossy, the quality of the resulting code assignments depends on how effectively information from the bootstrap representation space is transferred to the quantised item space, a problem we term capacity transfer. To address this issue, we initialise JPQ item codes using a higher-dimensional bootstrap representation that is subsequently compressed via product quantisation prior to training; we refer to this method as RecJPQ- HC (Higher Capacity). Experimental results show that RecJPQ-HC consistently improves downstream recommendation performance over the original RecJPQ code assignment strategy. Across multiple
datasets and sequential recommendation models, it achieves statistically significant average relative improvements of approximately
30% in NDCG@10 while preserving memory efficiency

The repository is organised as two self-contained experiment suites that share the same JPQ layer and the same centroid-assignment code:

| Directory | Backbone model | Training signal |
|-----------|----------------|-----------------|
| [`sasrec_jpq/`](sasrec_jpq) | SASRec / gSASRec (causal Transformer decoder) | Next-item prediction (cross-entropy or gBCE) |
| [`bert4rec_jpq/`](bert4rec_jpq) | BERT4Rec (bidirectional HuggingFace BERT encoder) | Masked-item prediction (cross-entropy) |

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


---

## Acknowledgements

All entry points seed `random`, `numpy` and `torch` (seed `42` by default) and enable deterministic cuDNN. FAISS is pinned to a single thread in the PQ-based strategies to keep code assignment deterministic. Note that `train.py` in the SASRec suite seeds explicitly to `42` regardless of `config.seed`.

