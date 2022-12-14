# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os 
import time
import json

import torch
import numpy as np
import pathlib

from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader
from numpy.random import RandomState

TRAINSET = "trainset.json"
TRAINSET_UNLAB = "trainset_unlab.json"
TRAINSET_UNLAB_RAND = "trainset_unlab_rand.json"
TESTSET = "testset.json"
ROOT = './data'


class CIFAR100:
    def __init__(self, user_idx=None, test_only=None, args=None, read_data=True) :
        if read_data: # Reads the data previously saved on files
            if user_idx == -1:
                if test_only:
                    print("Reading testing file")
                    file = os.path.join(ROOT,TESTSET)
                else:
                    print("Reading training labeled file")
                    file = os.path.join(ROOT,TRAINSET)
            elif user_idx == -2:
                print("Reading unlabeled training file")
                file = os.path.join(ROOT, TRAINSET_UNLAB)
            elif user_idx == -3:
                print("Reading unlabeled random training file")
                file = os.path.join(ROOT, TRAINSET_UNLAB_RAND)

            with open(file, 'r') as f:
                json_file = json.load(f)

            self.data = json_file
        else: # Create, preprocess and save the datasets
            from RandAugment import RandAugment
            trans = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            transform_unlabeltrain = transforms.Compose([ 
                RandAugment(1, 10),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

            # Download and preprocess datasets
            trainset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_train)
            unlabel_trainset = datasets.CIFAR100('./data', train=True, download=True, transform=transform_unlabeltrain)
            self.pretestset = datasets.CIFAR100('./data', train=False, download=True, transform=trans)

            train_loader = DataLoader(trainset, batch_size=len(trainset))
            ultrain_loader = DataLoader(unlabel_trainset, batch_size=len(unlabel_trainset))

            X_train = next(iter(train_loader))[0].numpy()
            Y_train = next(iter(train_loader))[1].numpy()
            X_unlabel_train = next(iter(ultrain_loader))[0].numpy()
            Y_unlabel_train = next(iter(ultrain_loader))[1].numpy()

            self.pretrainset, trainset_unlab_rand, trainset_unlab, \
            self.embed_dim = partition_imagedataset(X_train, Y_train, X_unlabel_train, Y_unlabel_train,args)
            self.trainset = _process(self.pretrainset, train=True)
            self.trainset_unlab = _process(trainset_unlab, train=True)
            self.trainset_unlab_rand = _process(trainset_unlab_rand, train=True)
            self.testset = _process(self.pretestset, train=False)

            save_json(self.trainset, TRAINSET)
            save_json(self.trainset_unlab, TRAINSET_UNLAB)
            save_json(self.trainset_unlab_rand, TRAINSET_UNLAB_RAND)
            save_json(self.testset, TESTSET)

def save_json(dict, filename):
    f = open(os.path.join('./data',filename), "w")
    json.dump(dict,f)
    f.close()

def _process(dataset, train=True):
    '''Process a Torchvision/preprocessed dataset to expected FLUTE format'''

    print('Converting data to expected format...')
    start_time = time.time()

    data_dict = {'users':[], 'num_samples': [], 'user_data':{}, 'user_data_label':{}}
    
    for i in range(len(dataset)):

        if train:
            x, y = dataset[i]['x'], dataset[i]['y']
        else:
            x, y = dataset[i]

        data_dict['users'].append(f'{i:04d}')
        data_dict['num_samples'].append(len(y) if train else 1)
        data_dict['user_data'][f'{i:04d}'] = [xi.tolist() for xi in x] if train else [x.tolist()]
        data_dict['user_data_label'][f'{i:04d}'] = [yi.tolist() for yi in y] if train else y

    print(f'Finished converting data in {time.time() - start_time:.2f}s.')

    return data_dict

def partition_imagedataset(X_train, Y_train, X_unlabel_train, Y_unlabel_train, args):

    if args['isclust'] == 1:
        partition = __getClusteredData__(Y_train, args['ensize'])

    elif args['isclust'] == 2:
        partition = __getClusteredMixedData__(Y_train, args['ensize'])
    else:
        partition = __getDirichletData__(Y_train, args)

    dataset_train = []
    dataset_val = []
    dataset_val_norand = []
    dataset_test = []

    train_ratio = args['train_ratio']
    val_ratio = args['val_ratio']
    test_ratio = args['test_ratio']
    x_for_embed = np.shape(X_train[0])
    for (i, ind) in enumerate(partition):

        x = X_train[ind]
        y = Y_train[ind]

        x_ul = X_unlabel_train[ind]
        y_ul = Y_unlabel_train[ind]

        n_i = len(ind)

        train_size = int(train_ratio * n_i)
        val_size = int(val_ratio * n_i) 
        test_size = int(test_ratio * n_i)

        x_train = torch.Tensor(x[val_size:val_size + train_size])
        y_train = torch.LongTensor(y[val_size:val_size + train_size])

        dataset_train_torch = {'x': x_train, 'y':y_train}

        if val_size == 0:
            x_val = x_train
            y_cal = y_train
            dataset_val_torch = dataset_train_torch
            dataset_val_torch_norand = dataset_train_torch
        else:
            x_val = torch.Tensor(x[:val_size])
            y_val = torch.LongTensor(y[:val_size])
            x_ul_val = torch.Tensor(x_ul[:val_size])
            y_ul_val = torch.LongTensor(y_ul[:val_size])
            dataset_val_torch = {'x': x_ul_val, 'y': y_ul_val}
            dataset_val_torch_norand = {'x':x_val, 'y':y_val}

        dataset_train.append(dataset_train_torch)
        dataset_val.append(dataset_val_torch)
        dataset_val_norand.append(dataset_val_torch_norand)

    return dataset_train, dataset_val, dataset_val_norand, x_for_embed

def __getDirichletData__(y, args):

    n = args['ensize']
    n_nets = args['ensize']
    K = args['num_classes']
    num_c = args['num_classes']
    labelList_true = y

    min_size = 0
    N = len(labelList_true)
    rnd = 0
    rann = RandomState(rnd)
    net_dataidx_map = {}
    p_client = np.zeros((n, num_c))

    for i in range(n):
        p_client[i] = rann.dirichlet(np.repeat(args['alpha'], num_c))

    idx_batch = [[] for _ in range(n_nets)]

    for k in range(K):
        idx_k = np.where(labelList_true == k)[0]
        rann.shuffle(idx_k)
        proportions = p_client[:, k]
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]

    for j in range(n_nets):
        if args['shuffle'] == 1:
            rann.shuffle(idx_batch[j])

        net_dataidx_map[j] = idx_batch[j]

    net_cls_counts_label = {}
    net_cls_counts_unlabel = {}

    for net_i in range(len(idx_batch)):
        n_i = len(idx_batch[net_i])
        train_size = int(args['train_ratio'] * n_i)
        val_size = int(args['val_ratio'] * n_i)
        unq, unq_cnt = np.unique(labelList_true[idx_batch[net_i][val_size:val_size + train_size]], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts_label[net_i] = tmp

        unq1, unq_cnt1 = np.unique(labelList_true[idx_batch[net_i][:val_size]], return_counts=True)
        tmp1 = {unq1[i]: unq_cnt1[i] for i in range(len(unq1))}
        net_cls_counts_unlabel[net_i] = tmp1

    local_sizes = []
    for i in range(n_nets):
        local_sizes.append(len(net_dataidx_map[i]))
    local_sizes = np.array(local_sizes)
    weights = local_sizes / np.sum(local_sizes)

    return idx_batch

if __name__ == "__main__":

    # Download and preprocess data
    args= {'name': 'FedVATnew', 'isaml':0, 'uda':1 , 'dataset': 'cifar100',
            'num_classes': 100, 'isclust': 0, 'alpha': 0.1, 'train_ratio': 0.2, 'val_ratio':0.8,
            'shuffle':1, 'vat_ptb':0.0 , 'vat_consis':0.05, 'unsup_lamb':1, 'l2_lambda':10,
            'bo': 50, 'thre': 0.3, 'comp': 'var', 'eta': 0.003, 'bs':64, 'unl_bs':128, 'train_ep':30,
            'unsuptrain_ep':10, 'rounds':2000, 'ensize':100, 'size': 10, 'model': 'RES50', 'seed': 0,
            'test_ratio': 0.0}

    data = CIFAR100(read_data=False, args=args)