# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch

from core.dataloader import BaseDataLoader
from experiments.classif_cnn.dataloaders.dataset import Dataset

class DataLoader(BaseDataLoader):
    def __init__(self, mode, num_workers=0, **kwargs):
        args = kwargs['args']
        self.batch_size = args['batch_size']

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
        x, y = list(zip(*batch))
        return {'x': torch.tensor(x), 'y': torch.tensor(y)}