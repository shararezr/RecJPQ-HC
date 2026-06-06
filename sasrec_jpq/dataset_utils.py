import json
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):
    def __init__(self, input_file, padding_value, output_file=None, max_length=200):
        with open(input_file, 'r') as f:
            # 🔧 SHIFT TO ZERO-BASED (int(x) - 1) for x in line.strip().split()]
            self.inputs = [
                [(int(x)) for x in line.strip().split()]
                for line in f.readlines()
            ]

        if output_file:
            with open(output_file, 'r') as f:
                # 🔧 SHIFT OUTPUTS TOO   self.outputs = [int(line.strip()) - 1 for line in f.readlines()]
                self.outputs = [int(line.strip())  for line in f.readlines()]
        else:
            self.outputs = None

        self.max_length = max_length
        self.padding_value = padding_value

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        inp = self.inputs[idx]

        # rated items (exclude padding)
        rated = set(x for x in inp if x != self.padding_value)

        if len(inp) > self.max_length:
            inp = inp[-self.max_length:]
        elif len(inp) < self.max_length:
            inp = [self.padding_value] * (self.max_length - len(inp)) + inp

        inp_tensor = torch.tensor(inp, dtype=torch.long)

        if self.outputs is not None:
            out_tensor = torch.tensor(self.outputs[idx], dtype=torch.long)
            return inp_tensor, rated, out_tensor

        return inp_tensor,


def collate_with_random_negatives(batch, pad_value, num_negatives):
    seqs = torch.stack([x[0] for x in batch], dim=0)  # [B, L]
    B, L = seqs.shape

    # 🔧 NEGATIVES NOW 0 ... num_items-1
    negatives = torch.randint(
        low=0,
        high=pad_value,
        size=(B, L, num_negatives),
        dtype=torch.long
    )

    seqs_exp = seqs.unsqueeze(-1)  # [B, L, 1]
    mask = (negatives == seqs_exp) & (seqs_exp != pad_value)

    while mask.any():
        negatives[mask] = torch.randint(
            low=0,
            high=pad_value,
            size=(mask.sum(),),
            dtype=torch.long
        )
        mask = (negatives == seqs_exp) & (seqs_exp != pad_value)

    return seqs, negatives


def collate_val_test(input_batch):
    input = torch.stack([input_batch[i][0] for i in range(len(input_batch))], dim=0)
    rated = [input_batch[i][1] for i in range(len(input_batch))]
    output = torch.stack([input_batch[i][2] for i in range(len(input_batch))], dim=0)
    return [input, rated, output]


def get_num_items(dataset):
    with open(f"datasets/{dataset}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    return stats['num_items']


def get_padding_value(dataset_dir):
    with open(f"{dataset_dir}/dataset_stats.json", 'r') as f:
        stats = json.load(f)
    # 🔧 PADDING IS NOW num_items
    padding_value = stats['num_items']
    return padding_value


def get_train_dataloader(dataset_name, batch_size=32, max_length=200, train_neg_per_positive=256):
    dataset_dir = f"datasets/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)

    train_dataset = SequenceDataset(
        f"{dataset_dir}/train/input.txt",
        max_length=max_length + 1,  # +1 for sequence shifting
        padding_value=padding_value
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda x: collate_with_random_negatives(x, padding_value, train_neg_per_positive)
    )
    return train_loader


def get_val_or_test_dataloader(dataset_name, part='val', batch_size=32, max_length=200):
    dataset_dir = f"datasets/{dataset_name}"
    padding_value = get_padding_value(dataset_dir)

    dataset = SequenceDataset(
        f"{dataset_dir}/{part}/input.txt",
        padding_value,
        f"{dataset_dir}/{part}/output.txt",
        max_length=max_length
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_val_test
    )
    return dataloader


def get_val_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_val_or_test_dataloader(dataset_name, 'val', batch_size, max_length)


def get_test_dataloader(dataset_name, batch_size=32, max_length=200):
    return get_val_or_test_dataloader(dataset_name, 'test', batch_size, max_length)


