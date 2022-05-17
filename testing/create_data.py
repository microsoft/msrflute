# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import csv
import json
import random
import argparse
import platform
from collections import OrderedDict
from itertools import islice

import tqdm
import h5py
import torchvision
import torchvision.transforms as transforms
from google_drive_downloader import GoogleDriveDownloader as gdd

def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    return parser

def reduce_users(file):

    with open(file, 'r') as f:
            json_file = json.load(f)

    num_samples = json_file['num_samples'][0:25]
    user_data = dict(OrderedDict(islice(json_file['user_data'].items(), 0, 25)))
    users_list = list(user_data.keys())

    return users_list, num_samples, user_data

def _process_and_save_to_disk(dataset, n_users, exp, output):
    '''Process a Torchvision dataset to expected format and save to disk'''

    # Split training data equally among all users
    total_samples = len(dataset)
    samples_per_user = total_samples // n_users
    assert total_samples % n_users == 0

    # Function for getting a given user's data indices
    user_idxs = lambda user_id: slice(user_id * samples_per_user, (user_id + 1) * samples_per_user)

    data_dict = {  # the data is expected to have this format
        'users' : [f'{user_id:04d}' for user_id in range(n_users)],
        'num_samples' : n_users * [samples_per_user],
        'user_data' : {f'{user_id:04d}': dataset.data[user_idxs(user_id)].tolist() if exp =="classif_cnn" else dataset.data[user_idxs(user_id)] for user_id in range(n_users)},
        'user_data_label': {f'{user_id:04d}': dataset.targets[user_idxs(user_id)] for user_id in range(n_users)},
    }

    with h5py.File(output + '.hdf5', 'w') as hdf5_file:
        _dump_dict_to_hdf5(data_dict=data_dict, hdf5_file=hdf5_file)


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

class HeartDataSet: 
    def __init__(self, heartdata, cutoff):
        self.data = [row[:187] for row in heartdata][:cutoff]
        self.targets = [int(float(row[187])) for row in heartdata][:(round(len(heartdata), -3))][:cutoff]

    def __len__(self):
        return len(self.data)  

def main():

    parser = get_arg_parser()
    args = parser.parse_args()
    args = vars(args)
    exp = args["task"]

    # Create data folder
    os.system("mkdir data")

    if exp == "nlg_gru" or exp == "mlm_bert":
        
        # Download preprocessed reddit dataset by LEAF: A Benchmark for Federated Settings
        gdd.download_file_from_google_drive(file_id='1ISzp69JmaIJqBpQCX-JJ8-kVyUns8M7o', dest_path='./data/nlg_gru.zip', unzip=True)

        files = ["train_data", "val_data", "test_data"]
        for file in files:
            orig_file = os.path.join("data","new_small_data",str(file+".json"))
            users_list, num_samples, user_data = reduce_users(orig_file)
            
            # Preprocess data
            if exp == "nlg_gru":
                os.makedirs("data/nlg_gru", exist_ok= True) if platform.system() == "Windows" else os.system("mkdir data/nlg_gru") 
                for users in user_data:
                    listToStr = ''
                    for i, sentences in enumerate(user_data[users]['x']):
                        for j, pieces in enumerate(sentences):
                            listToStr = ' '.join([elem for elem in pieces])
                            user_data[users]['x'][i][j] = listToStr
                        full_sentence = ' '.join([elem for elem in sentences])
                        full_sentence = full_sentence.replace('<PAD>', '').replace('<EOS>', '').replace('<BOS>', '').strip()
                        user_data[users]['x'][i] = full_sentence
                        user_data[users].pop('y',None)

            elif exp == "mlm_bert":
                os.makedirs("data/mlm_bert", exist_ok= True) if platform.system() == "Windows" else os.system("mkdir data/mlm_bert")
                user_data_aux = dict()
                for users in user_data:
                    listToStr = ''
                    for i, sentences in enumerate(user_data[users]['x']):
                        for j, pieces in enumerate(sentences):
                            listToStr = ' '.join([elem for elem in pieces])
                            listToStr = listToStr.replace('<PAD>', '').replace('<EOS>', '').replace('<BOS>', '').strip()
                            user_data[users]['x'][i][j] = listToStr
                        user_data[users].pop('y',None)
                    user_data_aux[users] = user_data[users]['x']
                user_data = user_data_aux

            # Create new dictionary
            new_dict = {'users':users_list ,'num_samples':num_samples, 'user_data':user_data}

            # Save preprocessed files
            ext = ".json" if exp=="nlg_gru" else ".txt"
            new_file = os.path.join("data",exp,str(file+ ext))
            f = open(new_file,'w')
            json.dump(new_dict,f)
            f.close()

            # Build vocabulary
            os.system(str("python build_vocab.py --data-dir ./data/"+ exp +" --target-dir ./models"))
            
    elif exp == "classif_cnn":
        os.makedirs("data/classif_cnn", exist_ok= True) if platform.system() == "Windows" else os.system("mkdir data/classif_cnn")
        
        # Get training and testing sets from torchvision
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)

        # Saving datasets
        _process_and_save_to_disk(trainset, n_users=50, exp=exp, output='./data/classif_cnn/train_data')
        _process_and_save_to_disk(testset, n_users=50, exp=exp, output='./data/classif_cnn/test_data')
    
    elif exp == "ecg_cnn":
        os.makedirs("data/ecg_cnn", exist_ok= True) if platform.system() == "Windows" else os.system("mkdir data/ecg_cnn")
        
        # Create dummy datasets
        for set in ['train_data.csv', 'test_data.csv']:
            data= [random.random() for i in range(188)]
            with open(os.path.join('data',exp,set), 'w', newline='') as f:
                write = csv.writer(f)
                for row in range(87554):
                    write.writerow(data)

        # Preprocess datasets
        for set in ['train_data', 'test_data']: 
            with open(os.path.join('data',exp,str(set+".csv"))) as f: 
                testset = list(csv.reader(f , delimiter=','))
            TestDataset = HeartDataSet(testset, 21000)
            _process_and_save_to_disk(TestDataset,1000,exp,os.path.join('data',exp,set))

if __name__ == '__main__':
    main()
