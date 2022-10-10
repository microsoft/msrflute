# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from core.dataset import BaseDataset
from experiments.fednewsrec.dataloaders.preprocess_mind import MIND

class Dataset(BaseDataset):
    def __init__(self, data, test_only=False, user_idx=0, **kwargs):
        self.test_only = test_only
        self.user_idx = user_idx

        # Get all data
        self.user_list, self.user_data, self.user_data_label, self.num_samples = self.load_data(data, self.test_only)

        if user_idx != -1:
            if self.test_only:  # combine all data into single array
                self.user = 'test_only'
                self.labels = [user_label for user_label in self.user_data_label.values()]
                self.features_x = [user_data['x'] for user_data in self.user_data.values()]
                self.features_y = [user_data['y'] for user_data in self.user_data.values()]
            else:  # get a single user's data
                if user_idx is None:
                    raise ValueError('in train mode, user_idx must be specified')

                self.user = self.user_list[user_idx]
                self.features_x = self.user_data[self.user]['x']
                self.features_y = self.user_data[self.user]['y']
                self.labels = self.user_data_label[self.user]

    def __getitem__(self, idx):
        return self.features_x[idx], self.features_y[idx], self.labels[idx]

    def __len__(self):
        return len(self.features_x)

    def load_data(self, data, test_only):
        '''Wrapper method to read/instantiate the dataset'''

        if data == None:
            dataset = MIND(root_data_path="/mnt/data/MIND_large", embedding_path="/mnt/data/MIND_large")
            data = dataset.testset if test_only else dataset.trainset
        
        users = data['users']
        features = data['user_data']
        labels = data['user_data_label']
        num_samples = data['num_samples']
            
        return users, features, labels, num_samples