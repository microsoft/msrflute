# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import numpy as np
from core.dataloader import BaseDataLoader
from experiments.fednewsrec.dataloaders.dataset import Dataset

class DataLoader(BaseDataLoader):
    def __init__(self, mode, num_workers=0, **kwargs):
        args = kwargs['args']
        self.batch_size = args['batch_size']
        self.mode = mode

        dataset = Dataset(
            data=kwargs['data'],
            test_only=(not mode=='train'),
            user_idx=kwargs.get('user_idx', None),
        )

        super().__init__(
            dataset,
            batch_size=self.batch_size,
            shuffle=(mode=='train'),
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        if self.mode == "train": # For training
            click, sample, label = list(zip(*batch))
            click = torch.tensor(click)
            sample = torch.tensor(sample)
            label = torch.tensor(label)
            return {'x': (click, sample), 'y': label}

        else: # For testing -- data format is different
            nv_hist = torch.stack(batch[0][0]).squeeze(1) 
            nv_imp = torch.stack(batch[0][1]).squeeze(1)
            label = batch[0][2]
            return {'x': (nv_hist, nv_imp), 'y': label}



        
        