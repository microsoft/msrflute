# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import h5py
import numpy as np

from core.dataset import BaseDataset

class Dataset(BaseDataset):
    def __init__(self, data, test_only=False, user_idx=0, **kwargs):
        self.test_only = test_only
        self.user_idx = user_idx

        # Get all data
        self.user_list, self.user_data, self.user_data_label, self.num_samples = self.load_data(data)

        if self.test_only:  # combine all data into single array
            self.user = 'test_only'
            self.features = np.vstack([user_data['x'] for user_data in self.user_data.values()])
            self.labels = np.hstack([user_label['x'] for user_label in self.user_data_label.values()])
        else:  # get a single user's data
            if user_idx is None:
                raise ValueError('in train mode, user_idx must be specified')

            self.user = self.user_list[user_idx]
            self.features = self.user_data[self.user]['x']
            self.labels = self.user_data_label[self.user]['x']

    def __getitem__(self, idx):
        items = self.features[idx].astype(np.float32).T.reshape(1,187)
        return items, self.labels[idx]

    def __len__(self):
        return len(self.features)

    def load_data(self,data):
        '''Load data from disk or memory'''

        if isinstance(data, str):
            try:
                data = h5py.File(data, 'r')
            except:
                raise ValueError('Only HDF5 format is allowed for this experiment')

            users = []
            num_samples = data['num_samples']
            features, labels = dict(), dict()
            
            # Decoding bytes from hdf5
            decode_if_str = lambda x: x.decode() if isinstance(x, bytes) else x
            for user in data['users']:
                user = decode_if_str(user)
                users.append(user)
                features[user] = {'x': data['user_data'][user]['x'][()]}
                labels[user] = {'x': data['user_data_label'][user][()]}

        else:
        
            users = data['users']
            features = data['user_data']
            labels = data['user_data_label']
            num_samples = data['num_samples']
            
        return users, features, labels, num_samples