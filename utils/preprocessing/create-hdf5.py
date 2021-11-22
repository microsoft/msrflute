# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import h5py
import time
from tqdm import tqdm
import pandas as pd


path = r'C:\Users\train.tsv'

def local_time():
    return str(time.strftime("%H:%M:%S",time.localtime()))


print(local_time() + " Starting script " )    
columns = ['author','num1','content','str1','str2','num2','subreddit']
df = pd.read_csv(path, sep='\t', names=columns, header=None)
print(local_time() + " File has been read "  )

df_authors = pd.DataFrame(df['author'])
df_content = pd.DataFrame(df['content'])
df_file = pd.concat([df_authors,df_content], axis=1)
print(local_time() + " Data needed has been concatenated ")


users_group = df_file.groupby('author')
group0 = df_file.groupby(['author','content'])
group1 = pd.Series(users_group.size())
users = (group1.index).to_numpy() 
print(local_time() + " users been formatted ")
num_samples = group1.values 
print(local_time() + " num_samples has been formatted ")
user_data_dict= {}

user_data_dict= {i: {'x':list()} for i in tqdm(users)}

for i in tqdm(range(len(df_file))):
    if df_file['content'][i] not in user_data_dict[df_file['author'][i]]['x']:
        user_data_dict[df_file['author'][i]]['x'].append(df_file['content'][i])
        

print(local_time() + " user_data has been formatted ")
f = h5py.File(r"C:\Users\train.hdf5", "w")
dset_0 = f.create_dataset("num_samples",data=num_samples)
dset_1= f.create_dataset("users", data =users)
print(local_time() + " starting to store dictionary ")

user_data = f.create_group("user_data")
for user in tqdm(user_data_dict):
    user_group = user_data.create_group(user)
    user_data_dict[user]['x'] = [str(e).encode('utf8') for e in  user_data_dict[user]['x']]
    x_dset = user_group.create_dataset('x',data=user_data_dict[user]['x'])

print(local_time() + " end of script ")