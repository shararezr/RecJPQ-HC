import torch


class MaskingCollator(object):
    def __init__(self, user_actions, sequence_length, val_users,
                    masking_prob, pad_id, mask_id,  ignore_val = -100, mode='train') -> None:
        self.val_users = val_users
        self.sequence_length = sequence_length
        self.user_actions = user_actions
        self.masking_prob = masking_prob
        self.pad_id = pad_id
        self.ignore_val = ignore_val
        self.mask_id = mask_id
        self.mode = mode
        self.is_validation = mode == 'val'
        self.is_train = mode == 'train'
        self.is_test = mode == 'test'


    def __call__(self, batch):
        seqs = []
        labels_all = []
        attns = []
        for user_id in batch:
            seq = [x[1] for x in  self.user_actions[user_id]] # sequence of the item id (x[1]) in the actions of the user
            # if self.is_train and (user_id in self.val_users):
            #     seq = seq[:-1] # if the current user is in the validation users and the collator is the train one, the last item in the sequence is removed, I might need to remove this part
            #  I need to comment this one 
            # 
            # if self.is_test:
                # 
                # seq.append(self.mask_id) # if the collator is the test one, the mask id is added to the sequence

            if len(seq) > self.sequence_length:
                seq = seq[-self.sequence_length:] # if the sequence is longer than the max sequence length, the firt exceeding items are removed from the sequence

            seq = torch.tensor(seq, dtype=torch.long, requires_grad=False)
            attn = torch.ones_like(seq, requires_grad=False) # initialization of the attention mask
            if self.is_validation or self.is_test: # if the collator is the validation or test one
                num_masked_items = 1
                masked_positions = torch.tensor([len(seq) - 1], requires_grad=False) # the only masked position is the last one (len(seq) - 1) -> is a tensor of shape 1
            else:
                num_masked_items = max(1, int(len(seq) * self.masking_prob)) # the number of masked items for the training sequences depends by the masking probability
                masked_positions = torch.randperm(len(seq))[:num_masked_items] # the position of the masks is randomly chosen
            masked_mask = torch.zeros(len(seq), dtype=torch.long, requires_grad=False)
            masked_mask[masked_positions] = 1 # the masked mask is set to 1 in the masked positions and to 0 in all the other ones
            labels = seq.clone() * masked_mask + self.ignore_val * (1 - masked_mask) # the label is set to the right one if the input token is a mask and to a ignore value otherwise
            seq[masked_positions] = self.mask_id # the mask id is put in the masked positions of the sequence
            if(len(seq) < self.sequence_length): # if the sequence is shorted than the max sequence length
                pad = torch.tensor([self.pad_id] * (self.sequence_length - len(seq)), requires_grad=False)
                ignore_pad = torch.tensor([self.ignore_val] * (self.sequence_length - len(seq)), requires_grad=False)
                zero_pad = torch.zeros_like(pad, requires_grad=False)
                seq = torch.cat([pad, seq], dim=0) # pad ids are added on the left of the sequence
                labels = torch.cat([ignore_pad, labels], dim=0) # ignore values are added on the left of the labels
                attn = torch.cat([zero_pad, attn], dim=0) # zero values are added on the left of the attention mask
            seqs.append(seq)
            attns.append(attn)
            labels_all.append(labels)
        batch = {"seq": torch.stack(seqs), "attn": torch.stack(attns), "labels": torch.stack(labels_all)} # stacked along batch dimension
        return batch

