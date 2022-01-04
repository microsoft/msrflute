# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from experiments.ecg_cnn.dataloaders.text_dataset import TextDataset

import torch
from torch.utils.data import DataLoader

class TextDataLoader(DataLoader):
    def __init__(self, mode, num_workers=0, **kwargs):
        args = kwargs['args']
        self.batch_size = args['batch_size']

        dataset = TextDataset(
            data=kwargs['data'],
            test_only=(not mode=='train'),
            user_idx=kwargs.get('user_idx', None),
            file_type='hdf5',
        )

        super().__init__(
            dataset,
            batch_size=self.batch_size,
            shuffle=(mode=='train'),
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

    def create_loader(self):
        return self

    def collate_fn(self, batch):
        x, y = list(zip(*batch))
        return {'x': torch.tensor(x), 'y': torch.tensor(y)}