import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader


class R2DataLoader(DataLoader):
    def __init__(self, args, dataset, tokenizer, split, shuffle):
        self.args      = args
        self.shuffle   = shuffle
        self.tokenizer = tokenizer
        self.split = split
        if(self.split == "train"):
            self.batch_size  = args.batch_size_train
            self.num_workers = args.num_workers_train
        else:
            self.batch_size  = args.batch_size_valid
            self.num_workers = args.num_workers_valid

        self.dataset = dataset
        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        images_id, labels, images, reports_ids, reports_masks, seq_lengths = zip(*data)

        # volume
        images = torch.stack(images, 0)

        # labels
        labels = torch.stack(labels, 0)

        # reports
        max_seq_length = max([max(x) for x in seq_lengths])
        targets        = np.full((len(reports_ids), len(reports_ids[0]), max_seq_length), fill_value=50256, dtype=int)
        targets_masks  = np.zeros((len(reports_ids), len(reports_ids[0]), max_seq_length), dtype=int)

        # iterate over sample in the batch
        for idx_batch in range(len(reports_ids)):
            # iterate over 18 sentences
            for idx_sequence, (report_ids, report_masks) in enumerate(zip(reports_ids[idx_batch], reports_masks[idx_batch])):
                # fill with token ids
                targets[idx_batch, idx_sequence, :len(report_ids)] = report_ids
                # fill with attention mask
                targets_masks[idx_batch, idx_sequence, :len(report_masks)] = report_masks

        return images_id, labels, images, torch.LongTensor(targets), torch.FloatTensor(targets_masks)

