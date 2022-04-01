# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import h5py
import time
import tqdm
import csv
import pandas as pd
from sklearn.utils import resample

def _dump_dict_to_hdf5(data_dict: dict, hdf5_file: h5py.File):
    '''Dump dict with expected structure to HDF5 file'''

    hdf5_file.create_dataset('users', data=data_dict['users'])
    hdf5_file.create_dataset('num_samples', data=data_dict['num_samples'])

    # Store actual data in groups
    user_data_group = hdf5_file.create_group('user_data')
    for user, user_data in tqdm.tqdm(data_dict['user_data'].items()):
        user_subgroup = user_data_group.create_group(user)
        user_subgroup.create_dataset('x', data=user_data) 

    user_data_label_group = hdf5_file.create_group('user_data_label')
    for user, user_data_label in tqdm.tqdm(data_dict['user_data_label'].items()):
        user_data_label_group.create_dataset(user, data=user_data_label) 

def _process_and_save_to_disk(dataset, n_users, output):
    '''Process the dataset to expected format and save to disk'''

    # Split training data equally among all users
    total_samples = len(dataset)
    samples_per_user = total_samples // n_users
    assert total_samples % n_users == 0

    # Function for getting a given user's data indices
    user_idxs = lambda user_id: slice(user_id * samples_per_user, (user_id + 1) * samples_per_user)

    # Convert training data to expected format
    print('Converting data to expected format...')
    start_time = time.time()

    data_dict = {  # the data is expected to have this format
        'users' : [f'{user_id:04d}' for user_id in range(n_users)],
        'num_samples' : n_users * [samples_per_user],
        'user_data' : {f'{user_id:04d}': dataset.data[user_idxs(user_id)] for user_id in range(n_users)},
        'user_data_label': {f'{user_id:04d}': dataset.targets[user_idxs(user_id)] for user_id in range(n_users)},
    }
    print(f'Finished converting data in {time.time() - start_time:.2f}s.')

    # Save training data to disk
    print('Saving data to disk...')
    start_time = time.time()

    with h5py.File(output + '.hdf5', 'w') as hdf5_file:
        _dump_dict_to_hdf5(data_dict=data_dict, hdf5_file=hdf5_file)
    print(f'Finished saving data in {time.time() - start_time:.2f}s.')

class HeartDataSet: 
    def __init__(self, heartdata, cutoff):
        self.data = [row[:187] for row in heartdata][:cutoff]
        self.targets = [int(float(row[187])) for row in heartdata][:(round(len(heartdata), -3))][:cutoff]

    def __len__(self):
        return len(self.data)  


# From https://www.kaggle.com/gregoiredc/arrhythmia-on-ecg-classification-using-cnn/notebook
# Can be used to creating resampled training set for less class imbalance
def resampleSet(train_df): 
    train_df[187]=train_df[187].astype(float).astype(int)
    df_1=train_df[train_df[187]==1]
    df_2=train_df[train_df[187]==2]
    df_3=train_df[train_df[187]==3]
    df_4=train_df[train_df[187]==4]
    df_0=(train_df[train_df[187]==0]).sample(n=40001,random_state=42)

    df_1_upsample=resample(df_1,replace=True,n_samples=10000,random_state=123)
    df_2_upsample=resample(df_2,replace=True,n_samples=20000,random_state=124)
    df_3_upsample=resample(df_3,replace=True,n_samples=5000,random_state=125)
    df_4_upsample=resample(df_4,replace=True,n_samples=20000,random_state=126)

    train_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])
    return train_df

# Uncomment lines below for resampled dataset
with open('../data/mitbih/mitbih_test.csv') as f: 
    testset = list(csv.reader(f , delimiter=','))
TestDataset = HeartDataSet(testset, 21000)
_process_and_save_to_disk(TestDataset,1000,'../data/test_data')

with open('../data/mitbih/mitbih_train.csv') as f: 
    trainset = csv.reader(f , delimiter=',')
    trainsetlist = list(trainset) 
TrainDataset = HeartDataSet(trainsetlist, 87000)
_process_and_save_to_disk(TrainDataset,1000,'../data/train_data')
