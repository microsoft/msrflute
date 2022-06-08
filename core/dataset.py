# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.utils.data import Dataset as PyTorchDataset
from abc import ABC, abstractmethod

class BaseDataset(ABC, PyTorchDataset):
    '''This is a wrapper class for PyTorch datasets.'''

    @abstractmethod
    def __init__(self,**kwargs):
        super(BaseDataset, self).__init__()
        
    @abstractmethod
    def __getitem__(self, idx, **kwargs):
        '''Fetches a data sample for a given key'''
        pass
    
    @abstractmethod
    def __len__(self):
        '''Returns the size of the dataset'''
        pass
    
    @abstractmethod
    def load_data(self,**kwargs):
        '''Wrapper method to read/instantiate the dataset'''
        pass
