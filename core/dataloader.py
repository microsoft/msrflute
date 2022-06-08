# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.utils.data import DataLoader as PyTorchDataLoader
from abc import ABC

class BaseDataLoader(ABC, PyTorchDataLoader):
    '''This is a wrapper class for PyTorch dataloaders.'''

    def create_loader(self):
        '''Returns the dataloader'''
        return self

        
    
