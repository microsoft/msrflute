# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np

from core.dataset import BaseDataset
from experiments.cv.data import prepare_dataset

class Dataset(BaseDataset):
    def __init__(self, data, test_only=False, user_idx=0, **kwargs):
        self.test_only = test_only
        self.user_idx = user_idx

        # Get all data
        self.user_list, self.user_data, self.user_data_label, self.num_samples = self.load_data(data, self.test_only)

        if self.test_only:  # combine all data into single array
            self.user = 'test_only'
            self.features = np.vstack([user_data['x'] for user_data in self.user_data.values()])
            self.labels = np.hstack(list(self.user_data_label.values()))

        else:  # get a single user's data
            if user_idx is None:
                raise ValueError('in train mode, user_idx must be specified')

            self.user = self.user_list[user_idx]
            self.features = np.vstack([user_data['x'] for user_data in self.user_data.values()])
            self.labels = np.hstack(list(self.user_data_label.values()))

    def __getitem__(self, idx):
        return self.features[idx].astype(np.float32).T, self.labels[idx]

    def __len__(self):
        return len(self.features)

    def load_data(self, data, test_only):
        '''Download or load data from disk/memory.
        
        The `data` argument can be either the path to the JSON
        or HDF5 file that contains the expected dictionary, or the
        actual dictionary. In case data cannot be loaded, will be 
        downloaded through `prepare_dataset` method.'''

        if data == None:
            training_dataset, test_dataset = prepare_dataset(rnd=2020,
                                                                img_size=40, 
                                                                image='cifar', 
                                                                total_num_clients=100, 
                                                                image_path="./",
                                                                save_to_disk= False,
                                                                alpha= 1.0,
                                                                wantTransform= False)
            data = test_dataset if test_only else training_dataset
        
        users = data['users']
        features = data['user_data']
        labels = data['user_data_label']
        num_samples = data['num_samples']

        return users, features, labels, num_samples