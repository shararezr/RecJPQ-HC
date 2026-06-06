import os
import json
import torch
from argparse import ArgumentParser

from dataset_utils import get_num_items, get_test_dataloader
from eval_utils import evaluate
from utils import build_model, get_device, load_config
from dataset_utils import get_train_dataloader, get_num_items, get_val_dataloader

import numpy as np
import random
import faiss

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# ----------------------------
# Argument parsing
# ----------------------------
parser = ArgumentParser()
parser.add_argument('--config', type=str, required=True, help="Path to the config file")
parser.add_argument('--checkpoint', type=str, required=True, help="Path to the trained model checkpoint")
args = parser.parse_args()

# ----------------------------
# Load config and model
# ----------------------------
config = load_config(args.config)
num_items = get_num_items(config.dataset_name)
device = get_device()

model = build_model(config)
model = model.to(device)

checkpoint = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

# ----------------------------
# Prepare test data
# ----------------------------
test_dataloader = get_test_dataloader(
    config.dataset_name,
    batch_size=config.eval_batch_size,
    max_length=config.sequence_length
)

# ----------------------------
# Evaluate
# ----------------------------
evaluation_result = evaluate(
    model,
    test_dataloader,
    config.metrics,
    config.recommendation_limit,
    config.filter_rated,
    device=device
)

print(f"Evaluation result for dataset {config.dataset_name}:")
print(evaluation_result)

# ----------------------------
# Helper: make result JSON-serializable
# ----------------------------
def serialize_eval_result(result):
    if isinstance(result, dict):
        return {str(k): serialize_eval_result(v) for k, v in result.items()}
    elif isinstance(result, list):
        return [serialize_eval_result(x) for x in result]
    elif isinstance(result, torch.Tensor):
        return result.detach().cpu().tolist()
    elif isinstance(result, (int, float, str, bool)) or result is None:
        return result
    else:
        return str(result)  # fallback

evaluation_result_serializable = serialize_eval_result(evaluation_result)

# ----------------------------
# Save results to file
# ----------------------------
save_dir = "results"
os.makedirs(save_dir, exist_ok=True)

# Use dynamic naming based on dataset, embedding dim, PQ length
emb = config.embedding_dim
pq_m = config.pq_m

output_path = os.path.join(
    save_dir,
    f"eval_{config.dataset_name}_emb{emb}_pq{pq_m}.json"
)

with open(output_path, "w") as f:
    json.dump(evaluation_result_serializable, f, indent=4)

print(f"Saved evaluation results to {output_path}")
