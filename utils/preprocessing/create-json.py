# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
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
        

f = open(r'C:\Users\train.json', "w")
new_data = {'users': users.tolist(), 'num_samples': num_samples.tolist(), 'user_data': user_data_dict}
json.dump(new_data,f)
print(local_time() + " end of script ")