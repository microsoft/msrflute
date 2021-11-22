# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import h5py
from tqdm import tqdm
import time

json_file = r'C:\Users\train.tsv'

def local_time():
    return str(time.strftime("%H:%M:%S",time.localtime()))

print(local_time() + " Starting script " )   
with open(json_file, 'r') as f:
    json_file = json.load(f)
print(local_time() + " JSON file read " )   

hdf_file = h5py.File(r"C:\Users\train.hdf5", "w")
dset_0 = hdf_file.create_dataset("users",data=json_file['users'])
dset_1 = hdf_file.create_dataset("num_samples",data=json_file['num_samples'])
print(local_time() + " users and num_samples stored " )   

user_data = hdf_file.create_group("user_data")
for user in tqdm(json_file['user_data']):
    user_group = user_data.create_group(user)
    dset_2 = user_group.create_dataset('x',data=json_file['user_data'][user]['x'])

print(local_time() + " end of script " )   