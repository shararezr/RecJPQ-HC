import torch
import torch.nn.functional as F
from transformer_decoder import TransformerBlock
from rec_jpq_layer import ItemCodeLayer, StandardItemEmbedding


class GSASRec(torch.nn.Module):

    def __init__(
        self,
        num_items,
        sequence_length=50,
        embedding_dim=256,
        num_heads=1,
        num_blocks=1,
        dropout_rate=0.5,
        reuse_item_embeddings=True,
        use_jpq=True,
        pq_m=32,
        strategy_weights=None,
        strategy_partitions=None,
        codes_strategy='svd_opq'
    ):
        super().__init__()

        self.num_items = num_items
        self.use_jpq = use_jpq

        # Align padding_idx with the training data (matches dataset_utils)
        self.padding_idx = num_items 

        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.embeddings_dropout = torch.nn.Dropout(dropout_rate)
        self.reuse_item_embeddings = reuse_item_embeddings

        # --- Item embedding layer ---
        if use_jpq:
            self.item_embedding = ItemCodeLayer(
                embedding_size=embedding_dim,
                pq_m=pq_m,
                num_items=num_items,
                sequence_length=sequence_length,
                strategy_weights=strategy_weights,
                strategy_partitions=strategy_partitions,
                codes_strategy=codes_strategy,
                padding_idx=self.padding_idx
            )
        else:
            self.item_embedding = StandardItemEmbedding(
                embedding_size=embedding_dim,
                num_items=num_items,
                padding_idx=self.padding_idx
            )

        # --- Positional embeddings ---
        self.position_embedding = torch.nn.Embedding(sequence_length, embedding_dim)

        # --- Transformer blocks ---
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, embedding_dim, dropout_rate)
            for _ in range(num_blocks)
        ])

        self.seq_norm = torch.nn.LayerNorm(embedding_dim)

        # Optional separate output embedding
        if not reuse_item_embeddings:
            self.output_embedding = torch.nn.Embedding(
                num_items + 1,
                embedding_dim,
                padding_idx=self.padding_idx
            )

    # ------------------------------------------------------------------
    # Assign JPQ item codes
    # ------------------------------------------------------------------
    def fit_item_codes(self, train_users):
        self.item_embedding.assign_codes(train_users)
        if self.use_jpq:
            print("JPQ item codes assigned.")

    # ------------------------------------------------------------------
    # Return embedding layer used for scoring
    # ------------------------------------------------------------------
    def get_output_embeddings(self):
        return self.item_embedding if self.reuse_item_embeddings else self.output_embedding

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, input_ids):
        """
        input_ids: [B, L] (long)
        returns: [B, L, embedding_dim], attentions list
        """

        if input_ids.dtype not in [torch.long, torch.int64]:
            input_ids = input_ids.long()

        mask = (input_ids != self.padding_idx).float().unsqueeze(-1)

        seq = self.item_embedding(input_ids)

        positions = torch.arange(seq.shape[1], device=input_ids.device).unsqueeze(0)
        seq = seq + self.position_embedding(positions)
        seq *= mask
        seq = self.embeddings_dropout(seq)

        attentions = []
        for block in self.transformer_blocks:
            seq, attn = block(seq, mask)
            attentions.append(attn)

        seq_emb = self.seq_norm(seq)
        return seq_emb, attentions

    # ------------------------------------------------------------------
    # Full-item scoring (inference)
    # ------------------------------------------------------------------
    def score_all_items(self, input_ids):
        seq_emb, _ = self.forward(input_ids)
        last_emb = seq_emb[:, -1, :]
        output_layer = self.get_output_embeddings()
        return output_layer.score_all_items(last_emb)  # [B, num_items]

    def get_predictions(self, input_ids, limit, rated=None):
        """
        Compute top-k predictions for last timestep.
        input_ids: [B, L]
        rated: list of sets/lists per batch item
        """

        with torch.no_grad():
            seq_emb, _ = self.forward(input_ids)
            last_emb = seq_emb[:, -1, :]  # [B, D]

            output_layer = self.get_output_embeddings()
            scores = output_layer.score_all_items(last_emb)  # [B, num_items]

            # Safe masking of rated items
            if rated is not None:
                for i, rated_items in enumerate(rated):
                    if len(rated_items) == 0:
                        continue

                    # Convert to tensor with correct dtype
                    indices = torch.tensor(
                        list(rated_items),
                        dtype=torch.long,
                        device=scores.device
                    )

                    # Remove out-of-bounds indices
                    indices = indices[
                        (indices >= 0) & (indices < scores.shape[1])
                    ]

                    if indices.numel() > 0:
                        scores[i, indices] = float('-inf')

            # Return top-k items and scores
            topk_scores, topk_items = torch.topk(scores, k=limit, dim=-1)
            return topk_items, topk_scores
