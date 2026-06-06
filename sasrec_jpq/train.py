from argparse import ArgumentParser
import os
import time
import torch.cuda

import torch
from utils import load_config, build_model, get_device
from dataset_utils import get_train_dataloader, get_num_items, get_val_dataloader
from tqdm import tqdm
from eval_utils import evaluate
from torchinfo import summary
import json
import torch.nn.functional as F

import numpy as np
import random
import faiss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------
# Setup
# --------------------------------------------------
models_dir = "models"
os.makedirs(models_dir, exist_ok=True)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/config_ml1m.py')
args = parser.parse_args()
config = load_config(args.config)
#set_seed(config.seed)

import numpy as np
import random
import faiss

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)



#padding index was num_items
num_items = get_num_items(config.dataset_name)
padding_idx = num_items   # matches dataset_utils get_padding_value

device = get_device()
model = build_model(config).to(device)

# reset GPU memory stats before training
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)

# --------------------------------------------------
# Data loaders
# --------------------------------------------------
train_dataloader = get_train_dataloader(
    config.dataset_name,
    batch_size=config.train_batch_size,
    max_length=config.sequence_length,
    train_neg_per_positive=config.negs_per_pos
)

val_dataloader = get_val_dataloader(
    config.dataset_name,
    batch_size=config.eval_batch_size,
    max_length=config.sequence_length
)

# --------------------------------------------------
# Prepare JPQ code assignment input
# --------------------------------------------------
all_train_users_tensor = []
for positives, _ in train_dataloader:
    all_train_users_tensor.append(positives)

all_train_users_tensor = torch.cat(all_train_users_tensor, dim=0)
#item_id <num_items

formatted_users = []
for seq in all_train_users_tensor:
    user_seq = []
    for item_id in seq:
        item_id = int(item_id)
        if 0 <= item_id < num_items:
            user_seq.append(item_id)
    formatted_users.append(user_seq)

# Debug
all_item_ids = [i for seq in formatted_users for i in seq]
print(f"[JPQ] Min item: {min(all_item_ids)}, Max item: {max(all_item_ids)}, Num items: {num_items}")

# --------------------------------------------------
# Assign JPQ item codes
# --------------------------------------------------
model.fit_item_codes(formatted_users)

# --------------------------------------------------
# Quantization error BEFORE training
# --------------------------------------------------
from measurements import compute_quantization_error, compute_effective_rank
qe_before = compute_quantization_error(model)
if qe_before:
    print(f"[QE before training] code_utilization={qe_before['mean_code_utilization']:.1f}")

# --------------------------------------------------
# Optimizer
# --------------------------------------------------
optimiser = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=getattr(config, "learning_rate", 1e-3)
)


batches_per_epoch = min(config.max_batches_per_epoch, len(train_dataloader))

best_metric = float("-inf")
best_model_name = None
step = 0
steps_not_improved = 0

# Optional: inspect model summary
summary(model, (config.train_batch_size, config.sequence_length), batch_dim=None)

# --------------------------------------------------
# Training loop
# --------------------------------------------------
epoch_start = time.time()

for epoch in range(config.max_epochs):
    model.train()
    loss_sum = 0.0

    batch_iter = iter(train_dataloader)
    pbar = tqdm(range(batches_per_epoch), desc=f"Epoch {epoch}")

    for batch_idx in pbar:
        step += 1
        positives, negatives = [x.to(device) for x in next(batch_iter)]

        model_input = positives[:, :-1].long()
        labels = positives[:, 1:].long()
        negatives = negatives[:, 1:, :].long()

        # --------------------------------------------------
        # Forward
        # --------------------------------------------------
        seq_emb, _ = model(model_input)
        mask = (model_input != padding_idx).float()

        output_layer = model.get_output_embeddings()

        if config.loss == 'ce':
            # --------------------------------------------------
            # Cross-entropy over all items
            # --------------------------------------------------
            valid_pos = (model_input != padding_idx)
            valid_hidden = seq_emb[valid_pos]
            valid_labels = labels[valid_pos]
            scores = output_layer.score_all_items(valid_hidden)
            loss_task = F.cross_entropy(scores, valid_labels)

        elif config.loss == 'gbce':
            # --------------------------------------------------
            # GBCE with sampled negatives
            # --------------------------------------------------
            # Clamp padding labels to 0 for safe indexing (masked out in loss anyway)
            safe_labels = labels.clone()
            safe_labels[mask == 0] = 0
            pos_neg_concat = torch.cat([safe_labels.unsqueeze(-1), negatives], dim=-1)
            logits = output_layer.score_sequence_items(seq_emb, pos_neg_concat)

            gt = torch.zeros_like(logits)
            gt[:, :, 0] = 1.0

            eps = 1e-10
            alpha = config.negs_per_pos / (num_items - 1)
            t = config.gbce_t
            beta = alpha * ((1 - 1/alpha) * t + 1/alpha)

            positive_logits = logits[:, :, 0:1].to(torch.float64)
            negative_logits = logits[:, :, 1:].to(torch.float64)

            positive_probs = torch.clamp(torch.sigmoid(positive_logits), eps, 1 - eps)
            positive_probs_adjusted = torch.clamp(
                positive_probs.pow(-beta), 1 + eps, torch.finfo(torch.float64).max
            )

            to_log = torch.clamp(1.0 / (positive_probs_adjusted - 1), eps, torch.finfo(torch.float64).max)
            positive_logits_transformed = to_log.log()

            logits_adjusted = torch.cat([positive_logits_transformed, negative_logits], dim=-1)

            loss_per_element = F.binary_cross_entropy_with_logits(
                logits_adjusted, gt, reduction='none'
            ).mean(-1) * mask
            loss_task = loss_per_element.sum() / mask.sum()

        else:
            raise ValueError(f"Unknown loss type: {config.loss}")

        # --------------------------------------------------
        # Centroid regularization
        # --------------------------------------------------

        # Combine losses
        loss_total = loss_task

        # --------------------------------------------------
        # Backprop
        # --------------------------------------------------
        optimiser.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimiser.step()

        loss_sum += loss_total.item()
        pbar.set_description(f"Epoch {epoch} | Loss {loss_sum / (batch_idx + 1):.4f}")

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------
    eval_result = evaluate(
        model,
        val_dataloader,
        config.metrics,
        config.recommendation_limit,
        config.filter_rated,
        device=device
    )

    print(f"Epoch {epoch} evaluation: {eval_result} loss: {loss_sum / (batch_idx + 1)}")
    print(f"Epoch {epoch} time: {time.time() - epoch_start:.2f}s")

    # --------------------------------------------------
    # Early stopping
    # --------------------------------------------------
    val_metric = eval_result[config.val_metric]
    if val_metric > best_metric:
        best_metric = val_metric
        model_name = (
            f"models/gsasrec-{config.dataset_name}-step:{step}-"
            f"emb:{config.embedding_dim}-dropout:{config.dropout_rate}-"
            f"metric:{best_metric:.6f}.pt"
        )
        print(f"Saving best model -> {model_name}")
        if best_model_name and os.path.exists(best_model_name):
            os.remove(best_model_name)
        torch.save(model.state_dict(), model_name)
        best_model_name = model_name
        steps_not_improved = 0
    else:
        steps_not_improved += 1
        print(f"Validation metric did not improve for {steps_not_improved} steps")
        if steps_not_improved >= config.early_stopping_patience:
            print(f"Early stopping. Best model: {best_model_name}")
            break

# --------------------------------------------------
# Post-training measurements
# --------------------------------------------------
total_epochs = epoch + 1
print(f"[METRICS] Total epochs: {total_epochs}")

qe_after = compute_quantization_error(model)
if qe_after:
    print(f"[QE after training] code_utilization={qe_after['mean_code_utilization']:.1f}")

eff_rank = compute_effective_rank(model)
if eff_rank:
    print(f"[Effective rank] {eff_rank['effective_rank']:.2f} / {eff_rank['embedding_dim']}")

# Save measurements to file
os.makedirs("results", exist_ok=True)
measurements = {
    "total_epochs": total_epochs,
    "qe_before_training": qe_before,
    "qe_after_training": qe_after,
    "effective_rank": eff_rank,
}
with open("results/tmp_measurements.json", "w") as f:
    json.dump(measurements, f, indent=4)

# --------------------------------------------------
# GPU memory reporting
# --------------------------------------------------
if torch.cuda.is_available():
    peak_mem_bytes = torch.cuda.max_memory_allocated(device)
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    print(f"[GPU] Peak memory allocated: {peak_mem_gb:.3f} GB")

    tmp_gpu_file = "results/tmp_gpu_mem.json"
    with open(tmp_gpu_file, "w") as f:
        json.dump({"peak_gpu_mem_gb": peak_mem_gb}, f)
