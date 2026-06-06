import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from rec_jpq_layer import ItemCodeLayer


class BERT4RecJPQ(nn.Module):
   def __init__(
       self,
       num_items,
       embedding_size=64,
       num_attention_heads=2,
       num_hidden_layers=3,
       hidden_act="gelu",
       attention_probs_dropout_prob=0.2,
       hidden_dropout_prob=0.2,
       initializer_range=0.02,
       type_vocab_size=2,
       sequence_length=200,
       pq_m=16,
       codes_strategy='svd_opq',
       strategy_weights=None,
       strategy_partitions=None,
   ):
       super().__init__()


       self.num_items = num_items
       self.embedding_size = embedding_size
       self.sequence_length = sequence_length
       self.pad_item_id = num_items
       self.mask_item_id = num_items + 1


       # --- JPQ item embedding layer ---
       self.item_embedding = ItemCodeLayer(
           embedding_size=embedding_size,
           pq_m=pq_m,
           num_items=num_items,
           sequence_length=sequence_length,
           padding_idx=self.pad_item_id,
           strategy_weights=strategy_weights,
           strategy_partitions=strategy_partitions,
           codes_strategy=codes_strategy,
       )


       # --- Learnable mask token embedding ---
       self.mask_embedding = nn.Parameter(torch.empty(embedding_size))
       nn.init.normal_(self.mask_embedding, std=initializer_range)


       # --- BERT encoder (no MLM head, just transformer) ---
       self.bert_config = BertConfig(
           vocab_size=2,  # unused — we pass inputs_embeds directly
           hidden_size=embedding_size,
           intermediate_size=embedding_size * 4,
           max_position_embeddings=sequence_length,
           attention_probs_dropout_prob=attention_probs_dropout_prob,
           hidden_act=hidden_act,
           hidden_dropout_prob=hidden_dropout_prob,
           initializer_range=initializer_range,
           num_attention_heads=num_attention_heads,
           num_hidden_layers=num_hidden_layers,
           type_vocab_size=type_vocab_size,
       )
       self.bert = BertModel(self.bert_config)


   # ------------------------------------------------------------------
   # Assign JPQ item codes
   # ------------------------------------------------------------------
   def fit_item_codes(self, train_users):
       if hasattr(self.item_embedding.item_codes_strategy, "assign"):
           self.item_embedding.assign_codes(train_users)
           print("JPQ item codes assigned.")


   # ------------------------------------------------------------------
   # Forward: input_ids -> hidden states
   # ------------------------------------------------------------------
   def forward(self, input_ids, attention_mask=None):
       """
       input_ids: [B, L] — may contain pad_item_id and mask_item_id
       attention_mask: [B, L]
       returns: [B, L, D] hidden states
       """
       B, L = input_ids.shape


       # Get JPQ embeddings (padding is handled by ItemCodeLayer)
       embeds = self.item_embedding(input_ids)  # [B, L, D]


       # Replace mask positions with learnable mask embedding
       mask_positions = (input_ids == self.mask_item_id)
       if mask_positions.any():
           embeds[mask_positions] = self.mask_embedding


       # Compute position_ids for left-padded sequences.
       # Default HF positions are 0..L-1, which gives pad tokens real positions.
       # This remaps so pad positions get 0 and real tokens start from 0 sequentially.
       if attention_mask is not None:
           position_ids = attention_mask.long().cumsum(-1) - 1
           position_ids.masked_fill_(attention_mask == 0, 0)
       else:
           position_ids = None


       # Run through BERT encoder
       outputs = self.bert(
           inputs_embeds=embeds,
           attention_mask=attention_mask,
           position_ids=position_ids,
       )
       return outputs.last_hidden_state  # [B, L, D]


   # ------------------------------------------------------------------
   # Score all items from hidden states
   # ------------------------------------------------------------------
   def score_all_items(self, hidden_states):
       """
       hidden_states: [B, D] (e.g. last position or masked position)
       returns: [B, num_items]
       """
       return self.item_embedding.score_all_items(hidden_states)


   # ------------------------------------------------------------------
   # Get predictions (inference)
   # ------------------------------------------------------------------
   def get_predictions(self, input_ids, attention_mask, limit):
       """
       input_ids: [B, L] — last position should be mask_item_id
       returns: top-k items and scores
       """
       with torch.no_grad():
           hidden = self.forward(input_ids, attention_mask)
           last_hidden = hidden[:, -1, :]  # [B, D]
           scores = self.score_all_items(last_hidden)  # [B, num_items]


           # Mask out padding and special tokens
           scores[:, self.num_items:] = float('-inf')


           topk_scores, topk_items = torch.topk(scores, k=limit, dim=-1)
           return topk_items, topk_scores




