# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
from core.dataset import BaseDataset
from experiments.semisupervision.dataloaders.cifar_dataset import CIFAR100

class Dataset(BaseDataset):
    def __init__(self, data, test_only=False, user_idx=0, **kwargs):
        self.test_only = test_only
        self.user_idx = user_idx
        args = kwargs.get('args',None)
        
        # Get all data
        self.user_list, self.user_data, self.user_data_label, self.num_samples = self.load_data(data, self.test_only, args)

        if user_idx != -1:
            if self.test_only:  # combine all data into single array
                self.user = 'test_only'
                self.features = np.vstack([user_data for user_data in self.user_data.values()])
                self.labels = np.hstack([user_label for user_label in self.user_data_label.values()])
            else:  # get a single user's data
                if user_idx is None:
                    raise ValueError('in train mode, user_idx must be specified')

                self.user = self.user_list[user_idx]
                self.features = self.user_data[self.user]
                self.labels = self.user_data_label[self.user]

    def __getitem__(self, idx):
        return np.array(self.features[idx]).astype(np.float32), self.labels[idx]

    def __len__(self):
        return len(self.features)

    def load_data(self, data, test_only, sup_config):
        '''Wrapper method to read/instantiate the dataset'''

        if data == None:
            dataset = CIFAR100(self.user_idx, test_only, sup_config)
            data = dataset.data
        
        users = data['users']
        features = data['user_data']
        labels = data['user_data_label']
        num_samples = data['num_samples']
            
        return users, features, labels, num_samples