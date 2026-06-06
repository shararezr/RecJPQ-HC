'''
from collections import defaultdict
from pathlib import Path
import numpy as np
import json

DATASET_DIR = Path(__file__).parent
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
FILE_NAME = DATASET_DIR / "Beauty.txt"   


def train_val_test_split():
    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)

    user_items = {}
    items = set()
    num_interactions = 0

    # -------------------------
    #   READ "uid item1 item2 ..."
    # -------------------------
    with open(FILE_NAME) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            user = int(parts[0])
            seq = list(map(int, parts[1:]))  # list of item ids

            user_items[user] = seq
            items.update(seq)
            num_interactions += len(seq)

    num_users = len(user_items)

    # -------------------------
    #  gSASRec-style validation users
    # -------------------------
    rng = np.random.RandomState(42)
    val_users = set(rng.choice(num_users, 512, replace=False))

    dataset_stats = {
        "num_users": num_users,
        "num_items": len(items),
        "num_interactions": num_interactions
    }

    print("Dataset stats: ", json.dumps(dataset_stats, indent=4))
    with open(DATASET_DIR / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)

    # --------------------------------------------
    # Create train/val/test sequences
    # --------------------------------------------
    train_sequences = []
    val_input = []
    val_gt = []
    test_input = []
    test_gt = []

    for uid, seq in user_items.items():
        if uid in val_users:
            # validation user
            train_sequences.append(seq[:-3])

            val_input.append(seq[:-2])
            val_gt.append(seq[-2])

            test_input.append(seq[:-1])
            test_gt.append(seq[-1])
        else:
            # non-validation user
            train_sequences.append(seq[:-2])

            test_input.append(seq[:-1])
            test_gt.append(seq[-1])

    # --------------------------------------------
    # Write output files
    # --------------------------------------------
    with open(TRAIN_DIR / "input.txt", "w") as f:
        for seq in train_sequences:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(VAL_DIR / "input.txt", "w") as f:
        for seq in val_input:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(VAL_DIR / "output.txt", "w") as f:
        for a in val_gt:
            f.write(str(a) + "\n")

    with open(TEST_DIR / "input.txt", "w") as f:
        for seq in test_input:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(TEST_DIR / "output.txt", "w") as f:
        for a in test_gt:
            f.write(str(a) + "\n")


if __name__ == "__main__":
    train_val_test_split()


'''

from collections import defaultdict
from pathlib import Path
import numpy as np
import json


DATASET_DIR = Path(__file__).parent
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
FILE_NAME = DATASET_DIR / "Beauty.txt"


def train_val_test_split():
    TRAIN_DIR.mkdir(exist_ok=True)
    VAL_DIR.mkdir(exist_ok=True)
    TEST_DIR.mkdir(exist_ok=True)

    user_items = {}
    items = set()
    num_interactions = 0

    # -------------------------
    # Load data
    # -------------------------
    with open(FILE_NAME) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            user = int(parts[0])
            seq = list(map(int, parts[1:]))

            user_items[user] = seq
            items.update(seq)
            num_interactions += len(seq)

    # -------------------------
    # Filter users with < 5 interactions
    # -------------------------
    
    min_len = 5
    user_items = {
        u: seq for u, seq in user_items.items()
        if len(seq) >= min_len
    }
    

    # -------------------------
    # Recompute stats AFTER filtering
    # -------------------------
    items = set()
    num_interactions = 0
    for seq in user_items.values():
        items.update(seq)
        num_interactions += len(seq)

    num_users = len(user_items)



    # -------------------------
    # Remap item IDs to 0..N-1   IMPORTANT
    # -------------------------
    item2id = {item: idx for idx, item in enumerate(sorted(items))}
    user_items = {
        u: [item2id[i] for i in seq]
        for u, seq in user_items.items()
    }
    
    # -------------------------
    # Validation users (FIXED) 
    # -------------------------
    rng = np.random.RandomState(42)
    val_users = set(rng.choice(list(user_items.keys()), 512, replace=False))
    

    dataset_stats = {
        "num_users": num_users,
        "num_items": len(items),
        "num_interactions": num_interactions
    }

    print("Dataset stats:", json.dumps(dataset_stats, indent=4))

    with open(DATASET_DIR / "dataset_stats.json", "w") as f:
        json.dump(dataset_stats, f, indent=4)

    # --------------------------------------------
    # Split containers
    # --------------------------------------------
    train_sequences = []

    val_input = []
    val_gt = []

    test_input = []
    test_gt = []

    # --------------------------------------------
    # Core split logic (CONSISTENT RULESET)
    # --------------------------------------------
    for uid, seq in user_items.items():

        # must have at least 2 items
        if len(seq) < 2:
            continue

        if uid in val_users:

            # TRAIN: remove ONLY last item (GT)
            train_sequences.append(seq[:-1])

            # VAL: predict last item
            val_input.append(seq[:-1])
            val_gt.append(seq[-1])

            # TEST: same structure
            test_input.append(seq[:-1])
            test_gt.append(seq[-1])

        else:

            # TRAIN: remove last item (GT)
            train_sequences.append(seq[:-1])

            # TEST only
            test_input.append(seq[:-1])
            test_gt.append(seq[-1])

    # --------------------------------------------
    # Write files
    # --------------------------------------------
    with open(TRAIN_DIR / "input.txt", "w") as f:
        for seq in train_sequences:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(VAL_DIR / "input.txt", "w") as f:
        for seq in val_input:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(VAL_DIR / "output.txt", "w") as f:
        for a in val_gt:
            f.write(str(a) + "\n")

    with open(TEST_DIR / "input.txt", "w") as f:
        for seq in test_input:
            f.write(" ".join(map(str, seq)) + "\n")

    with open(TEST_DIR / "output.txt", "w") as f:
        for a in test_gt:
            f.write(str(a) + "\n")


if __name__ == "__main__":
    train_val_test_split()



