import os
import json
import torch
import random
import numpy as np
from argparse import ArgumentParser

from utils import load_config, build_model
from dataset_utils import load_actions, load_test_actions
from eval_utils import validate


# ==========================================================
# SEED (FIXED)
# ==========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




# ==========================================================
# Arguments
# ==========================================================
parser = ArgumentParser()
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--checkpoint', type=str, required=True)
parser.add_argument('--split', type=str, choices=['val', 'test'], default='test')
args = parser.parse_args()


# ==========================================================
# Load config + model
# ==========================================================
config = load_config(args.config)
set_seed(config.seed)
model = build_model(config)


# ==========================================================
# Load train + val data
# ==========================================================
load_actions(config.dataset_name, model)


# ==========================================================
# IMPORTANT: load test separately
# ==========================================================
test_user_ids, test_gts = load_test_actions(config.dataset_name, model)
model.test_users = test_user_ids
model.test_gts = test_gts


# ==========================================================
# Build full model state
# ==========================================================
model.rebuild_model()


# ==========================================================
# Load checkpoint
# ==========================================================
checkpoint = torch.load(args.checkpoint, map_location=model.device)
model.model.load_state_dict(checkpoint)
model.model.eval()


# ==========================================================
# Evaluation
# ==========================================================
if args.split == "val":
    evaluation_result = validate(
        model,
        model.val_users,
        model.val_ndcg_at,
        model.val_batch_size
    )
    eval_label = "val"


elif args.split == "test":
    evaluation_result = validate(
        model,
        model.test_users,
        model.val_ndcg_at,   # ideally replace with config.test_k
        model.val_batch_size
    )
    eval_label = "test"


# ==========================================================
# Print results
# ==========================================================
print(f"\n=== Evaluation result ({config.dataset_name} | {eval_label}) ===")
print(evaluation_result)


# ==========================================================
# Serialization helper
# ==========================================================
def serialize(x):
    if isinstance(x, dict):
        return {k: serialize(v) for k, v in x.items()}
    if isinstance(x, list):
        return [serialize(v) for v in x]
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    return x


# ==========================================================
# Save results
# ==========================================================
os.makedirs("results", exist_ok=True)

output_path = os.path.join(
    "results",
    f"eval_{eval_label}_{config.dataset_name}_emb{config.embedding_size}_pq{config.pq_m}.json"
)

with open(output_path, "w") as f:
    json.dump(serialize(evaluation_result), f, indent=4)


print(f"\nSaved results to: {output_path}")



