# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import h5py
import json
import numpy as np
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data, test_only=False, user_idx=None, file_type=None):
        self.test_only = test_only
        self.user_idx = user_idx
        self.file_type = file_type

        # Get all data
        self.user_list, self.user_data, self.user_data_label, self.num_samples = self.load_data(data, self.file_type)

        if self.test_only:  # combine all data into single array
            self.user = 'test_only'
            self.features = np.vstack([user_data['x'] for user_data in self.user_data.values()])
            self.labels = np.hstack(list(self.user_data_label.values()))
        else:  # get a single user's data
            if user_idx is None:
                raise ValueError('in train mode, user_idx must be specified')

            self.user = self.user_list[user_idx]
            self.features = self.user_data[self.user]['x']
            self.labels = self.user_data_label[self.user]

    def __getitem__(self, idx):
        return self.features[idx].astype(np.float32).T, self.labels[idx]

    def __len__(self):
        return len(self.features)

    @staticmethod
    def load_data(data, file_type):
        '''Load data from disk or memory.
        
        The :code:`data` argument can be either the path to the JSON
        or HDF5 file that contains the expected dictionary, or the
        actual dictionary.'''

        if isinstance(data, str):
            if file_type == 'json':
                with open(data, 'r') as fid:
                    data = json.load(fid)
            elif file_type == 'hdf5':
                data = h5py.File(data, 'r')

        users = data['users']
        features = data['user_data']
        labels = data['user_data_label']
        num_samples = data['num_samples']

        return users, features, labels, num_samples