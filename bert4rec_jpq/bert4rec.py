from collections import defaultdict
import tempfile
from aprec.api.action import Action
from aprec.recommenders.recommender import Recommender
from aprec.utils.item_id import ItemId
from torch.utils.data import DataLoader
import torch
import mmh3

from collator import MaskingCollator
from config import BERT4RecExperimentConfig
from bert4rec_jpq_model import BERT4RecJPQ




class BERT4RecPytorchRecommender(Recommender):
   def __init__(self, config: BERT4RecExperimentConfig):
           super().__init__()
           self.config = config
           self.users = ItemId()
           self.items = ItemId()
           self.masking_prob = config.masking_prob
           self.sequence_length = config.sequence_length
           self.user_actions = defaultdict(list)
           self.trained_samples = 0
           self.train_batch_size = config.train_batch_size
           self.flags = {}
           self.max_steps_per_epoch = config.max_steps_per_epoch
           self.lr = config.lr
           self.val_batch_size = config.eval_batch_size
           self.val_ndcg_at = config.recommendation_limit
           self.max_epochs = config.max_epochs
           self.early_stop_epochs = config.early_stopping_patience
           if torch.cuda.is_available():
               self.device = torch.device("cuda")
           else:
               self.device = torch.device("cpu")
           self.positions = torch.arange(1, self.val_ndcg_at + 1, dtype=torch.float)
           self.ndcg_discounts = torch.unsqueeze(1 / torch.log2(self.positions + 1), 0)


   def get_tensorboard_dir(self):
        if not hasattr(self, "tensorboard_dir") or self.tensorboard_dir is None:
            self.tensorboard_dir = tempfile.mkdtemp()
        return self.tensorboard_dir


   def add_action(self, action: Action):
       user_id_internal = self.users.get_id(action.user_id)
       action_id_internal = self.items.get_id(action.item_id)
       self.user_actions[user_id_internal].append((action.timestamp, action_id_internal))


   def set_val_users(self, val_users):
        return super().set_val_users(val_users)


   def sort_actions(self):
       for user_id in self.user_actions:
           self.user_actions[user_id].sort(key=lambda x: (x[0], mmh3.hash(f"{x[1]}_{user_id}")) )


   # ------------------------------------------------------------------
   # Initialize model and collators (call before training)
   # ------------------------------------------------------------------
   def rebuild_model(self):
       self.sort_actions()
       self.pad_item_id = self.items.size()
       self.mask_item_id = self.items.size() + 1


       # --- BERT4Rec with JPQ ---
       self.model = BERT4RecJPQ(
           num_items=self.items.size(),
           embedding_size=self.config.embedding_size,
           num_attention_heads=self.config.num_attention_heads,
           num_hidden_layers=self.config.num_hidden_layers,
           hidden_act=self.config.hidden_act,
           attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
           hidden_dropout_prob=self.config.hidden_dropout_prob,
           initializer_range=self.config.initializer_range,
           type_vocab_size=self.config.type_vocab_size,
           sequence_length=self.sequence_length,
           pq_m=self.config.pq_m,
           codes_strategy=self.config.codes_strategy,
           strategy_weights=self.config.strategy_weights,
           strategy_partitions=self.config.strategy_partitions,
       ).to(self.device)


       val_users_internal = set()
       for user in self.val_users:
           val_users_internal.add(self.users.get_id(user))


       self.val_users_internal = val_users_internal
       self.val_collator = MaskingCollator(self.user_actions, self.sequence_length, val_users_internal, self.masking_prob, self.pad_item_id, self.mask_item_id, mode='val')
       self.test_collator = MaskingCollator(self.user_actions, self.sequence_length, val_users_internal, self.masking_prob, self.pad_item_id, self.mask_item_id, mode='test')


   def get_train_loader(self):
       train_collator = MaskingCollator(self.user_actions, self.sequence_length, self.val_users_internal, self.masking_prob, self.pad_item_id, self.mask_item_id, mode='train')
       all_users = list(self.user_actions.keys())
       return DataLoader(all_users, batch_size=self.train_batch_size, collate_fn=train_collator, shuffle=True)


   # ------------------------------------------------------------------
   # Inference
   # ------------------------------------------------------------------
   def recommend_impl(self, user_ids, limit, mode):
       with torch.no_grad():
           if mode == 'val':
               collator = self.val_collator
           elif mode == 'test':
               collator = self.test_collator
           else:
               raise ValueError(f"Unknown mode {mode}")
           internal_user_ids = [self.users.get_id(user_id) for user_id in user_ids]
           batch = collator(internal_user_ids)
           seq = batch['seq'].to(self.device)
           attn = batch['attn'].to(self.device)


           # Get hidden states and score from last position
           hidden = self.model(seq, attention_mask=attn)
           scores = self.model.score_all_items(hidden[:, -1, :])


           if self.flags.get('filter_seen', False):
               for idx, user in enumerate(internal_user_ids):
                   seen_items = [x[1] for x in self.user_actions[user]]
                   if mode == 'val':
                       seen_items = seen_items[:-1]
                   scores[idx, seen_items] = float('-inf')
           scores[:, self.items.size():] = float('-inf')


           log_probs = torch.nn.functional.log_softmax(scores, dim=1)
           top_k = torch.topk(log_probs, limit, dim=1)
           result = {
               'items': top_k.indices,
               'scores': top_k.values,
           }
           if mode == 'val':
               labels = batch['labels'][:, -1]
               gt_logprobs = log_probs[range(len(internal_user_ids)), labels]
               result['labels'] = labels
               result['gt_logprobs'] = gt_logprobs
           return result


   def recommend(self, user_id, limit, features=None):
       requests = [(user_id, features)]
       result = self.recommend_batch(requests, limit)
       return result[0]


   def recommend_batch(self, recommendation_requests, limit):
       self.model.eval()
       user_ids = [x[0] for x in recommendation_requests]
       result = []
       for users_batch in DataLoader(user_ids, batch_size=self.val_batch_size, shuffle=False):
           recommendations = self.recommend_impl(users_batch, limit, 'test')
           rec_items = recommendations['items'].cpu().numpy()
           rec_scores = recommendations['scores'].cpu().numpy()
           for idx in range(len(users_batch)):
               items = [self.items.reverse_id(x.item()) for x in rec_items[idx]]
               scores = rec_scores[idx]
               user_result = list(zip(items, scores))
               result.append(user_result)
       return result



