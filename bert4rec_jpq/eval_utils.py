import torch
from torch.utils.data import DataLoader

import torch, numpy as np, random, faiss

import random
import numpy as np
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from torch.utils.data import DataLoader

def validate(model, val_users, val_ndcg_at, val_batch_size):
    model.model.eval()
    positions = torch.arange(1, val_ndcg_at + 1, dtype=torch.float)
    ndcg_discounts = torch.unsqueeze(1 / torch.log2(positions + 1), 0)

    ndcgs = []
    recalls = [] 
    losses = []
    #I add this here!
    val_users = list(val_users)
    for batch in DataLoader(val_users, batch_size=val_batch_size, shuffle=False):
        recommendations = model.recommend_impl(batch, val_ndcg_at, 'val')
        label_items = torch.unsqueeze(recommendations['labels'], 1)  # [B, 1]
        hit = (recommendations['items'].cpu() == label_items).float()  # [B, K]

        # NDCG
        rec_ndcg = torch.sum(hit * ndcg_discounts, 1)
        ndcgs.append(rec_ndcg)

        # Recall@K: fraction of users whose label is in top-K
        rec_recall = (hit.sum(1) > 0).float()  # 1 if label in top-K else 0
        recalls.append(rec_recall)

        # Loss
        losses.append(-recommendations['gt_logprobs'])

    val_result = {
        "ndcg": torch.cat(ndcgs).mean().item(),
        "rec@10": torch.cat(recalls).mean().item(), 
        "loss": torch.cat(losses).mean().item()
    }

    return val_result
