# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import h5py
import json
import os

import torchvision
from torchvision import transforms
import numpy as np
from numpy.random import RandomState

from utils import print_rank

class DataPartitioner(object):
    """ Partitions a dataset into different chunks. """

    def __init__(self, data, sizes=None, rnd=0, alpha=0, num_c=10,
                 dataset=None, lab_distr=None, ratio=1, img_size=32, wantTrans=True):
        self.data = data
        self.dataset  = dataset
        self.total_num= len(sizes) if sizes is not None else len(lab_distr)
        self.img_size= img_size
        self.wantTrans= wantTrans

        if lab_distr is not None:
            self.partitions, self.dat_stat = self.__use_fixed_lab_distr__(data, lab_distr,
                                                                           ratio, rnd, num_c)
        else:
            self.partitions, self.ratio, self.dat_stat, self.endat_size = self.__getDirichletData__(data, sizes,
                                                                                                    alpha, num_c, rnd)


    def get_lab_distr(self):
        return self.dat_stat


    def return_partition(self, partition, flag='data', is_train_set=True):

        if flag != 'data':
            return [self.data[idx][1] for idx in self.partitions[partition]]
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]

        if self.wantTrans:
            dc = {'resize': 0.5 if is_train_set else None,
                  'pad': None,
                  'crop': None,
                  'flip': False,
                  'rotate': (-180+2*int(partition*180/self.total_num), -180+2*int((partition+1)*180/self.total_num)) if is_train_set else \
                            (-180+2*int(partition*180/self.total_num)+2, -180+2*int(partition*180/self.total_num)+2),
                  'normalize': [mean, std]}
        else:
            dc = {'resize': None,
                  'pad': None,
                  'crop': None,
                  'flip': False,
                  'rotate': None,
                  'normalize': [mean, std]}

        transform = get_transform(transform=dc,img_size=self.img_size)

        return {'x': [transform(self.data[idx][0]).tolist() for idx in self.partitions[partition]]}


    def __use_fixed_lab_distr__(self, data, lab_distr, ratio, rnd, num_c):
        n_nets = []
        idx_batch = []
        labelList = np.array(data.targets)
        rann = RandomState(rnd)

        # Find where all labels are
        label_dict={lab: np.where(labelList == lab)[0] for lab in range(num_c)}

        # Process the prefixed label distributions one by one
        for lab_indices in list(lab_distr.keys())[:-1]:
            net_dataidx_map = {}

            for lab, num in lab_distr[lab_indices].items():
                len_k = len(label_dict[lab])
                idx_k = label_dict[lab][:min(int(num*ratio), len_k)]
                label_dict[lab] = label_dict[lab][min(int(num*ratio), len_k):]
                if len(idx_k)>0:
                    net_dataidx_map[lab] = list(idx_k)
            n_nets.append(net_dataidx_map)

        net_dataidx_map = {}
        for lab, idx_k in label_dict.items():
            if len(idx_k)>0:
                net_dataidx_map[lab] = idx_k
        n_nets.append(net_dataidx_map)

        for i, lab_indices in enumerate(n_nets):
            idx_batch.append([item for sublist in lab_indices.values() for item in sublist])

        net_cls_counts = {}
        for net_i, dataidx in enumerate(idx_batch):
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        print_rank('Data statistics: %s' % str(net_cls_counts), loglevel=logging.DEBUG)

        if 0:
            count=0
            tot_count={i:0 for i in range(10)}
            for _, client in net_cls_counts.items():
                for lab, num in client.items():
                    tot_count[lab]+=num
                    count+=num
            print('Debugging:', tot_count, count)


        return idx_batch, net_cls_counts


    # Getting this function from FedML -- 02-17-22
    def __getDirichletData__(self, data, psizes, alpha, num_c, rnd):
        n_nets = len(psizes)
        K = num_c
        labelList = np.array(data.targets)
        min_size = 0
        N = len(labelList)
        rann = RandomState(rnd)

        net_dataidx_map = {}
        while min_size < K:
            idx_batch = [[] for _ in range(n_nets)]
            # for each class in the dataset
            for k in range(K):
                idx_k = np.where(labelList == k)[0]
                rann.shuffle(idx_k)
                proportions = rann.dirichlet(np.repeat(alpha, n_nets))
                ## Balance
                proportions = np.array([p * (len(idx_j) < N / n_nets) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_nets):
            rann.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

        net_cls_counts = {}
        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(labelList[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp

        local_sizes = []
        for i in range(n_nets):
            local_sizes.append(len(net_dataidx_map[i]))
        local_sizes = np.array(local_sizes)
        weights = local_sizes / np.sum(local_sizes)

        print_rank('Data statistics: %s' % str(net_cls_counts), loglevel=logging.DEBUG)
        print_rank('Data ratio: %s' % str(weights), loglevel=logging.DEBUG)


        if 0:
            count=0
            tot_count={i:0 for i in range(10)}
            for _, client in net_cls_counts.items():
                for lab, num in client.items():
                    tot_count[lab]+=num
                    count+=num
            print('Debugging:', tot_count, count)

        return idx_batch, weights, net_cls_counts, np.sum(local_sizes)


def partition_dataset(rnd, img_size, image, total_num_clients, image_path, alpha, wantTransform):

    partition_sizes = [1.0/total_num_clients for _ in range(total_num_clients)]

    if image == 'cifar':
        trainset = torchvision.datasets.CIFAR10(
                                            root=os.path.join(image_path, image),
                                            train=True,
                                            download=True,
                                            transform=None)
        train_partition = DataPartitioner(trainset, partition_sizes, rnd,
                                            alpha=alpha,
                                            num_c=10,
                                            img_size=img_size,
                                            wantTrans=wantTransform)

        testset = torchvision.datasets.CIFAR10(
                                            root=os.path.join(image_path, image),
                                            train=False,
                                            download=True,
                                            transform=None)

        if 0:
            lab_distr= train_partition.get_lab_distr()
            test_partition = DataPartitioner(testset, lab_distr=lab_distr, rnd=rnd, ratio=0.2,
                                                num_c=10,
                                                img_size=img_size,
                                                wantTrans=wantTransform)
        else:
            test_partition = DataPartitioner(testset, partition_sizes, rnd,
                                              alpha=alpha,
                                              num_c=10,
                                              img_size=img_size,
                                              wantTrans=wantTransform)

    elif image == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
                                            root=os.path.join(image_path, image),
                                            train=True,
                                            download=True,
                                            transform=transform_train) # NOTE: Is this working?
        train_partition = DataPartitioner(trainset, partition_sizes, rnd,
                                            alpha=alpha,
                                            num_c=100)

        testset = torchvision.datasets.CIFAR100(
                                            root=os.path.join(image_path, image),
                                            train=False,
                                            download=True,
                                            transform=transform_test)
        test_partition = DataPartitioner(testset, partition_sizes, rnd,
                                            alpha=alpha,
                                            num_c=100)

    return train_partition, test_partition


# Setup all necessary image datasets for training
def prepare_dataset(rnd=2020, img_size=40, image='cifar', total_num_clients=100, image_path="./", alpha= 1.0, wantTransform=False, save_to_disk=False):
    
    train_partition, test_partition = partition_dataset(rnd=rnd, 
                                                        img_size=img_size, 
                                                        image=image, 
                                                        total_num_clients=total_num_clients, 
                                                        image_path=image_path,
                                                        alpha=alpha,
                                                        wantTransform= wantTransform)

    datasets = ["train_dataset.hdf5", "test_dataset.hdf5"]
    print_rank('Processing {}... '.format(datasets), loglevel=logging.DEBUG)
    output = [_process_and_save_to_disk(train_partition if set == "train_dataset.hdf5" else test_partition, 
                                            save_to_disk, 
                                            file_format= set.split('.')[-1], 
                                            output=set, 
                                            is_train_set=True if set == "train_dataset.hdf5" else False) for set in datasets]

    return output[0], output[1]


def _dump_dict_to_hdf5(data_dict: dict, hdf5_file: h5py.File):
    '''Dump dict with expected structure to HDF5 file'''

    hdf5_file.create_dataset('users', data=data_dict['users'])
    hdf5_file.create_dataset('num_samples', data=data_dict['num_samples'])

    # Store actual data in groups
    user_data_group = hdf5_file.create_group('user_data')
    for user, user_data in data_dict['user_data']['x'].items():
        user_subgroup = user_data_group.create_group(user)
        user_subgroup.create_dataset('x', data=user_data)

    user_data_label_group = hdf5_file.create_group('user_data_label')
    for user, user_data_label in data_dict['user_data_label'].items():
        user_data_label_group.create_dataset(user, data=user_data_label)


def _process_and_save_to_disk(dataset, save_to_disk, file_format, output, is_train_set=True):
    '''Process a Torchvision dataset to expected format and save to disk'''

    n_users = len(dataset.partitions)

    # Convert training data to expected format
    print_rank('Converting data to expected format...', loglevel=logging.DEBUG)

    data_dict = {
        'users': [f'{user_id:04d}' for user_id in range(n_users)],
        'num_samples': [len(dataset.partitions[user_id]) for user_id in range(n_users)],
        'user_data': {f'{user_id:04d}': dataset.return_partition(user_id, 'data', is_train_set) for user_id in range(n_users)},
        'user_data_label': {f'{user_id:04d}': dataset.return_partition(user_id, 'labels', is_train_set) for user_id in range(n_users)},
    }

    # Save training data to disk
    print_rank('Saving data to disk...', loglevel=logging.DEBUG)
    if save_to_disk:
        if file_format == 'json':
            outfile =output + '.json'
            with open(outfile, 'w') as json_file:
                json.dump(data_dict, json_file)
        elif file_format == 'hdf5':
            outfile =output + '.hdf5'
            with h5py.File(outfile, 'w') as hdf5_file:
                _dump_dict_to_hdf5(data_dict=data_dict, hdf5_file=hdf5_file)
        else:
            raise ValueError('unknown format.')
        print_rank('Finished saving data...{}'.format(outfile), loglevel=logging.DEBUG)
    else:
        outfile=data_dict

    return outfile


def get_transform(transform, img_size=32):
    """Unpack transformations and apply to train or test splits"""

    transform_list = [transforms.ToTensor()]
    # resize
    if transform['resize'] is not None:
        transform_list.append(transforms.RandomResizedCrop(img_size, scale=(transform['resize'], 2*transform['resize'])))
        transform_list.append(torchvision.transforms.Pad(4))
    else:
        transform_list.append(transforms.RandomCrop(img_size, padding=4))
        #transform_list.append(transforms.Resize(img_size))
    # padding
    if transform['pad'] is not None:
        transform_list.append(transforms.Pad(transform['pad']))

    # crop
    if transform['crop'] is not None:
        transform_list.append(transforms.RandomResizedCrop(transform['crop']))

    if transform['rotate'] is not None:
        transform_list.append(transforms.RandomRotation(transform['rotate']))

    # flips
    if transform['flip']:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomVerticalFlip())

    # normalization
    if transform['normalize'] is not None:
        transform_list.append(transforms.Normalize(mean=transform['normalize'][0], std=transform['normalize'][1]))

    return transforms.Compose(transform_list)
