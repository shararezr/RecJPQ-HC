
from argparse import ArgumentParser
import os
import time
import json
import copy
import random
import numpy as np

import torch
import torch.nn.functional as F
import torch.profiler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils import load_config, build_model
from dataset_utils import load_actions
from eval_utils import validate
from utils import load_config, build_model
from dataset_utils import load_actions
from eval_utils import validate
from measurements import compute_quantization_error, compute_effective_rank


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
# Reproducibility
# --------------------------------------------------


# --------------------------------------------------
# Setup
# --------------------------------------------------
models_dir = "models"
results_dir = "results"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/config_ml1m.py')
args = parser.parse_args()
config = load_config(args.config)
set_seed(config.seed)
experiment_start_time = time.time()

# --------------------------------------------------
# Build model & load data
# --------------------------------------------------
model = build_model(config)
load_actions(config.dataset_name, model)
model.rebuild_model()

# --------------------------------------------------
# Assign JPQ item codes
# --------------------------------------------------
all_users = list(model.user_actions.keys())
formatted_users = []

for uid in all_users:
    seq = [item_id for _, item_id in model.user_actions[uid]]
    # Exclude val GT (last item for val users) from JPQ code assignment
    if uid in model.val_users_internal:
        seq = seq[:-1]
    formatted_users.append(seq)


all_items = [i for seq in formatted_users for i in seq]
print(f"[JPQ] Min item: {min(all_items)}, Max item: {max(all_items)}")

model.model.fit_item_codes(formatted_users)


# --------------------------------------------------
# Quantization error BEFORE training
# --------------------------------------------------
qe_before = compute_quantization_error(model.model)
if qe_before:
    print(f"[QE before training] code_utilization={qe_before['mean_code_utilization']:.1f}")

num_items = model.items.size()
loss_type = config.loss
negs_per_pos = config.negs_per_pos
# gbce_t = config.gbce_t


# --------------------------------------------------
# Training setup
# --------------------------------------------------
tensorboard_dir = model.get_tensorboard_dir()
tb_writer = SummaryWriter(tensorboard_dir)

train_loader = model.get_train_loader()
batches_per_epoch = min(
    model.max_steps_per_epoch,
    len(all_users) // model.train_batch_size
)

optimiser = torch.optim.Adam(model.model.parameters(), lr=model.lr)

best_ndcg = float('-inf')
best_epoch = -1
best_model_weights = None
best_model_name = None
epochs_since_best = 0

# --------------------------------------------------
# Profiler (optional)
# --------------------------------------------------
profiler = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=1, active=10, repeat=1),
    on_trace_ready=torch.profiler.tensorboard_trace_handler(
        tensorboard_dir + '/profiler'
    ),
    profile_memory=False,
    with_modules=False,
)

# --------------------------------------------------
# Training loop
# --------------------------------------------------
for epoch in range(model.max_epochs):
    print(f"\nEpoch {epoch}")
    model.model.train()

    pbar = tqdm(
        total=batches_per_epoch,
        ascii=True,
        bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}',
        ncols=70
    )

    epoch_loss_sum = 0.0
    profiler.start()

    for step, batch in enumerate(train_loader):
        if step >= batches_per_epoch:
            break

        optimiser.zero_grad()

        seq = batch['seq'].to(model.device)
        labels = batch['labels'].to(model.device)
        attention_mask = batch['attn'].to(model.device)

        # Forward
        hidden = model.model(seq, attention_mask=attention_mask)

        # Masked positions
        masked_pos = (labels != -100)
        masked_hidden = hidden[masked_pos]
        masked_labels = labels[masked_pos]

        # Scores
        scores = model.model.score_all_items(masked_hidden)

        # Loss
        loss = F.cross_entropy(scores, masked_labels)

        loss.backward()
        optimiser.step()
        profiler.step()

        epoch_loss_sum += loss.item()
        epoch_loss_mean = epoch_loss_sum / (step + 1)

        model.trained_samples += len(seq)

        pbar.update()
        pbar.set_description(f"Loss: {epoch_loss_mean:.4f}")

    profiler.stop()
    pbar.close()

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------
    epoch_result = validate(
        model,
        model.val_users,
        model.val_ndcg_at,
        model.val_batch_size
    )

    print(f"Train loss: {epoch_loss_mean:.4f}")
    print(f"Val loss: {epoch_result['loss']:.4f}")
    print(f"Val NDCG@{model.val_ndcg_at}: {epoch_result['ndcg']:.4f}")

    tb_writer.add_scalar("loss/train", epoch_loss_mean, model.trained_samples)
    tb_writer.add_scalar("loss/val", epoch_result['loss'], model.trained_samples)
    tb_writer.add_scalar("ndcg@10/val", epoch_result['ndcg'], model.trained_samples)

    # --------------------------------------------------
    # Save best model
    # --------------------------------------------------
    if epoch_result['ndcg'] > best_ndcg:
        best_ndcg = epoch_result['ndcg']
        best_epoch = epoch
        best_model_weights = copy.deepcopy(model.model.state_dict())
        epochs_since_best = 0

        best_model_name = (
            f"models/bert4rec-{config.dataset_name}-epoch:{epoch}-"
            f"emb:{config.embedding_size}-pq:{config.pq_m}-"
            f"metric:{best_ndcg:.6f}.pt"
        )

        torch.save(best_model_weights, best_model_name)
        print(f"Saved BEST model → {best_model_name}")

    else:
        epochs_since_best += 1

    print(f"Best NDCG: {best_ndcg:.4f} at epoch {best_epoch}")
    print(f"Epochs since best: {epochs_since_best}")

    # Early stopping
    if epochs_since_best >= model.early_stop_epochs:
        print(f"Early stopping at epoch {epoch}")
        break

    tb_writer.flush()

# --------------------------------------------------
# Restore best model
# --------------------------------------------------
print(f"\nRestoring best model from epoch {best_epoch}")
model.model.load_state_dict(best_model_weights)

# --------------------------------------------------
# Save final model + metadata
# --------------------------------------------------
total_time = time.time() - experiment_start_time
'''
final_model_name = (
    f"{models_dir}/bert4rec-{config.dataset_name}-"
    f"pq{getattr(config, 'pq_m', 'NA')}-"
    f"emb{getattr(config, 'embedding_size', 'NA')}-final.pt"
)
'''
torch.save(model.model.state_dict(), best_model_name)

# Model size
model_size_mb = os.path.getsize(best_model_name) / (1024 ** 2)

# Save summary JSON
results = {
    "dataset": config.dataset_name,
    "best_ndcg": best_ndcg,
    "best_epoch": best_epoch,
    "total_epochs_trained": epoch + 1,
    "training_time_sec": total_time,
    "model_size_mb": model_size_mb,
    "best_model_path": best_model_name,
    "config": {
        "pq_m": getattr(config, "pq_m", None),
        "embedding_size": getattr(config, "embedding_size", None),
        "num_attention_heads": getattr(config, "num_attention_heads", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
    }
}

results_path = (
    f"{results_dir}/bert4rec_{config.dataset_name}_"
    f"pq{getattr(config, 'pq_m', 'NA')}_summary.json"
)

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved FINAL model → {best_model_name}")
print(f"Saved summary → {results_path}")
print(f"Training time: {total_time:.2f}s")
print(f"Model size: {model_size_mb:.2f} MB")



# --------------------------------------------------
# Restore best model
# --------------------------------------------------
if best_model_name:
    print(f"Restoring best model from epoch {best_epoch}")
    model.model.load_state_dict(torch.load(best_model_name, map_location=model.device))

# --------------------------------------------------
# Post-training measurements
# --------------------------------------------------
total_epochs = epoch + 1
print(f"[METRICS] Total epochs: {total_epochs}")

qe_after = compute_quantization_error(model.model)
if qe_after:
    print(f"[QE after training] code_utilization={qe_after['mean_code_utilization']:.1f}")

eff_rank = compute_effective_rank(model.model)
if eff_rank:
    print(f"[Effective rank] {eff_rank['effective_rank']:.2f}")

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
    peak_mem_bytes = torch.cuda.max_memory_allocated(model.device)
    peak_mem_gb = peak_mem_bytes / (1024 ** 3)
    print(f"[GPU] Peak memory allocated: {peak_mem_gb:.3f} GB")

    tmp_gpu_file = "results/tmp_gpu_mem.json"
    with open(tmp_gpu_file, "w") as f:
        json.dump({"peak_gpu_mem_gb": peak_mem_gb}, f)

