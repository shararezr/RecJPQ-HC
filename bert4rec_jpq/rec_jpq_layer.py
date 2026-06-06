import torch
import torch.nn as nn
import math
import numpy as np
# --- Import assignment strategies ---
from centroid_assignment_strategies.bpr_strategy import BPRAssignmentStrategy
from centroid_assignment_strategies.qr_strategy import QuotientRemainder
from centroid_assignment_strategies.random_strategy import RandomAssignmentStrategy
from centroid_assignment_strategies.svd_strategy import SVDAssignmentStrategy
from centroid_assignment_strategies.svd_opq import SVDAssignmentStrategy_opq


def get_codes_strategy(
    codes_strategy,
    item_code_bytes,
    num_items,
    sub_embedding_size,
    strategy_weights,
    strategy_partitions
):
    if codes_strategy == "svd":
        return SVDAssignmentStrategy(item_code_bytes, num_items, sub_embedding_size)
    elif codes_strategy == "bpr":
        return BPRAssignmentStrategy(item_code_bytes, num_items, sub_embedding_size)
    elif codes_strategy == "random":
        return RandomAssignmentStrategy(item_code_bytes, num_items, sub_embedding_size)
    elif codes_strategy == "qr":
        return QuotientRemainder(item_code_bytes, num_items, sub_embedding_size)
    elif codes_strategy == "svd_llm":
        return SVDAssignmentStrategy_llm(item_code_bytes, num_items, sub_embedding_size)
    elif codes_strategy == "svd_combo":
        return SVDAssignmentStrategy_combo(
            item_code_bytes, num_items, sub_embedding_size, strategy_weights
        )
    elif codes_strategy == "svd_concat":
        return SVDAssignmentStrategy_concat(
            item_code_bytes, num_items, sub_embedding_size, strategy_partitions
        )
    elif codes_strategy == "svd_opq":
        return SVDAssignmentStrategy_opq(item_code_bytes, num_items, sub_embedding_size)

    else:
        raise ValueError(f"Unknown code assignment strategy: {codes_strategy}")


class ItemCodeLayer(nn.Module):
    """
    JPQ / OPQ Item Embedding Layer with safe padding handling.
    """

    def __init__(
        self,
        embedding_size,
        pq_m,
        num_items,
        sequence_length,
        padding_idx=None,
        strategy_weights=None,
        strategy_partitions=None,
        codes_strategy="svd_opq",
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.pq_m = pq_m
        self.num_items = num_items
        self.sequence_length = sequence_length
        self.padding_idx = padding_idx

        self.sub_embedding_size = embedding_size // pq_m
        self.item_code_bytes = pq_m

        if codes_strategy != "qr":
            self.vals_per_dim = 256
        else:
            self.vals_per_dim = math.ceil(math.sqrt(num_items))

        # --- Non-trainable item codes (with extra row for padding) ---
        self.register_buffer(
            "item_codes",
            torch.zeros(num_items + 2, self.item_code_bytes, dtype=torch.long)
        )

        # --- Trainable centroids ---
        self.centroids = nn.Parameter(
            torch.empty(self.item_code_bytes, self.vals_per_dim, self.sub_embedding_size)
        )
        nn.init.uniform_(self.centroids, -0.05, 0.05)

        # --- Assignment strategy ---
        self.item_codes_strategy = get_codes_strategy(
            codes_strategy,
            self.item_code_bytes,
            num_items,
            self.sub_embedding_size,
            strategy_weights,
            strategy_partitions
        )

    # ------------------------------------------------------------------
    # Assign JPQ codes before training
    # ------------------------------------------------------------------
    def assign_codes(self, train_users=None):
        output = self.item_codes_strategy.assign(train_users)

        # Handle tuple (codes, centroids_init) or just codes
        if isinstance(output, tuple):
            codes, centroids_init = output
        else:
            codes, centroids_init = output, None

        # Assign codes
        codes_tensor = torch.tensor(codes, dtype=torch.long, device=self.centroids.device)
        self.item_codes[:self.num_items].copy_(codes_tensor[:self.num_items])


        # Initialize centroids if returned
        if centroids_init is not None:
            centroids_tensor = torch.tensor(centroids_init, dtype=torch.float32, device=self.centroids.device)
            if centroids_tensor.shape != self.centroids.shape:
                raise ValueError(f"Centroids shape mismatch: {centroids_tensor.shape} vs {self.centroids.shape}")
            self.centroids.data.copy_(centroids_tensor)

    # ------------------------------------------------------------------
    # Forward: item IDs -> embeddings
    # ------------------------------------------------------------------
    def forward(self, input_ids):
        """
        input_ids: [B, L] (long)
        returns:   [B, L, embedding_size]
        """
        B, L = input_ids.shape
        if not input_ids.dtype in [torch.long, torch.int64]:
            input_ids = input_ids.long()

        if self.padding_idx is not None:
            valid = (input_ids >= 0) & (input_ids < self.num_items)
            mask = valid.unsqueeze(-1).float()
            safe_ids = input_ids.clamp(0, self.num_items-1)
        else:
            mask = torch.ones(B, L, 1, device=input_ids.device)
            safe_ids = input_ids

        input_codes = self.item_codes[safe_ids]  # [B, L, M]

        code_idx = torch.arange(self.item_code_bytes, device=input_ids.device).view(1, 1, -1).expand(B, L, -1)
        flat_idx = torch.stack([code_idx, input_codes], dim=-1).view(-1, 2)

        sub_embs = self.centroids[flat_idx[:, 0], flat_idx[:, 1]]  # [B*L*M, D_sub]
        emb = sub_embs.view(B, L, self.embedding_size)

        # Zero out padding embeddings
        emb = emb * mask
        return emb

    # ------------------------------------------------------------------
    # Sequence scoring (training)
    # ------------------------------------------------------------------
    def score_sequence_items(self, seq_emb, target_ids):
        """ seq_emb: [B, L, D] target_ids:[B, L, N] returns: [B, L, N] """
        B, L, _ = seq_emb.shape
        N = target_ids.shape[2]
        seq_sub = seq_emb.view(B, L, self.item_code_bytes, self.sub_embedding_size)
        centroid_scores = torch.einsum("bsmd,mhd->bsmh", seq_sub, self.centroids) # [B, L, M, vals]
        target_ids = target_ids.long()
        target_codes = self.item_codes[target_ids] # [B, L, N, M]
        target_codes = target_codes.permute(0, 1, 3, 2) # [B, L, M, N]
        batch_idx = torch.arange(B, device=seq_emb.device).view(B, 1, 1, 1).expand(B, L, self.item_code_bytes, N)
        seq_idx = torch.arange(L, device=seq_emb.device).view(1, L, 1, 1).expand(B, L, self.item_code_bytes, N)
        m_idx = torch.arange(self.item_code_bytes, device=seq_emb.device).view(1, 1, -1, 1).expand(B, L, self.item_code_bytes, N)
        sub_scores = centroid_scores[batch_idx, seq_idx, m_idx, target_codes]
        return sub_scores.sum(dim=2) # [B, L, N]

    # ------------------------------------------------------------------
    # Full item scoring (inference)
    # ------------------------------------------------------------------
    def score_all_items(self, seq_emb):
        """
        seq_emb: [B, D]
        returns: [B, num_items]
        """

        B = seq_emb.shape[0]
        seq_sub = seq_emb.view(B, self.item_code_bytes, self.sub_embedding_size)
        centroid_scores = torch.einsum("bmd,mhd->bmh", seq_sub, self.centroids)  # [B, M, vals]

        target_codes = self.item_codes[:self.num_items]  # [num_items, M]
        scores = torch.zeros(B, self.num_items, device=seq_emb.device)
        for m in range(self.item_code_bytes):
            scores += centroid_scores[:, m, target_codes[:, m]]
        return scores


