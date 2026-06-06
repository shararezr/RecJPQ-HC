import json
from aprec.api.action import Action




def get_num_items(dataset_name):
   with open(f"datasets/{dataset_name}/dataset_stats.json", 'r') as f:
       stats = json.load(f)
   return stats['num_items']




def load_actions(dataset_name, model):
   """
   Read preprocessed dataset files and populate model via add_action().
   Sets val_users on the model.


   Expects:
       datasets/{name}/train/input.txt  — one line per user, space-separated item IDs
       datasets/{name}/val/input.txt    — val user sequences
       datasets/{name}/val/output.txt   — val ground truth items
   """
   dataset_dir = f"datasets/{dataset_name}"


   # --- Load train sequences ---
   with open(f"{dataset_dir}/train/input.txt", 'r') as f:
       train_sequences = [list(map(int, line.strip().split())) for line in f.readlines()]


   # --- Load val sequences + ground truth ---
   with open(f"{dataset_dir}/val/input.txt", 'r') as f:
       val_sequences = [list(map(int, line.strip().split())) for line in f.readlines()]


   with open(f"{dataset_dir}/val/output.txt", 'r') as f:
       val_outputs = [int(line.strip()) for line in f.readlines()]


   # --- Pre-register all items (including test-only items) so ItemId mapping is stable.
   # Without this, items that appear only as test_gt would be assigned new internal IDs
   # during test evaluation, breaking pad_item_id and mask_item_id.
   try:
       with open(f"{dataset_dir}/test/input.txt", 'r') as f:
           for line in f:
               for tok in line.strip().split():
                   model.items.get_id(int(tok))
       with open(f"{dataset_dir}/test/output.txt", 'r') as f:
           for line in f:
               model.items.get_id(int(line.strip()))
   except FileNotFoundError:
       pass


   # --- Feed train actions into model ---
   for user_idx, seq in enumerate(train_sequences):
       user_id = f"train_{user_idx}"
       for t, item_id in enumerate(seq):
           action = Action(user_id=user_id, item_id=item_id, timestamp=t)
           model.add_action(action)


   # --- Feed val actions (sequence + ground truth as last item) ---
   val_user_ids = []
   for user_idx, (seq, target) in enumerate(zip(val_sequences, val_outputs)):
       user_id = f"val_{user_idx}"
       val_user_ids.append(user_id)
       for t, item_id in enumerate(seq):
           action = Action(user_id=user_id, item_id=item_id, timestamp=t)
           model.add_action(action)
       # Ground truth as the last action — collator will hold this out for val
       action = Action(user_id=user_id, item_id=target, timestamp=len(seq))
       model.add_action(action)


   model.set_val_users(val_user_ids)




def load_test_actions(dataset_name, model):
   """
   Load test_input sequences as test_{X} users.


   Following the all_full.txt pattern: test_gt is NOT appended to user_actions.
   It's held externally and returned alongside test_user_ids so the caller can
   compare predictions to ground truth after recommend_impl(mode='test').


   The model's existing test_collator (mode='test') will append mask_id at the
   end of each test_input sequence, so predicting at the mask yields seq[-1] = test_gt.


   Must be called AFTER load_actions() + model.rebuild_model().


   Returns: (test_user_ids, test_gts) — parallel lists.
   """
   dataset_dir = f"datasets/{dataset_name}"


   with open(f"{dataset_dir}/test/input.txt", 'r') as f:
       test_sequences = [list(map(int, line.strip().split())) for line in f.readlines()]
   with open(f"{dataset_dir}/test/output.txt", 'r') as f:
       test_gts = [int(line.strip()) for line in f.readlines()]


   test_user_ids = []
   for user_idx, seq in enumerate(test_sequences):
       user_id = f"test_{user_idx}"
       test_user_ids.append(user_id)
       for t, item_id in enumerate(seq):
           action = Action(user_id=user_id, item_id=item_id, timestamp=t)
           model.add_action(action)


   model.sort_actions()
   return test_user_ids, test_gts








