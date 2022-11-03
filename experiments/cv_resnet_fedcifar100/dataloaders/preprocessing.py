import os
import wget
import zipfile
import tarfile
import h5py

data_cache_dir = "./data"
DEFAULT_TRAIN_FILE = "fed_cifar100_train.h5"
DEFAULT_TEST_FILE = "fed_cifar100_test.h5"

''' 
    The FedCIFAR100 dataset is taken from FedML repository. For more information regarding this dataset, 
    please refer to https://github.com/FedML-AI/FedML/tree/master/python/fedml/data/fed_cifar100.

    In order to download the data run the following commands:
        - wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/fed_cifar100.tar.bz2
        - tar -xvf fed_cifar100.tar.bz2
'''

class FEDCIFAR100:
    def __init__(self) :

        download_files(data_cache_dir)

        # Preprocess datasets
        train_h5 = h5py.File(os.path.join(data_cache_dir,'fed_cifar100', DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(os.path.join(data_cache_dir, 'fed_cifar100',DEFAULT_TEST_FILE), "r")
        test_dict = {'users': [], 'num_samples': [], 'user_data': dict(), 'user_data_label': dict()}
        train_dict = {'users': [], 'num_samples': [], 'user_data': dict(), 'user_data_label': dict()}

        for user in test_h5['examples'].keys():
            test_dict['users'].append(user)
            test_dict['num_samples'].append(len(test_h5['examples'][user]['image'][()]))
            test_dict['user_data'][user] = test_h5['examples'][user]['image'][()]
            test_dict['user_data_label'][user] = test_h5['examples'][user]['label'][()]
            
        for user in train_h5['examples'].keys():
            train_dict['users'].append(user)
            train_dict['num_samples'].append(len(train_h5['examples'][user]['image'][()]))
            train_dict['user_data'][user] = train_h5['examples'][user]['image'][()]
            train_dict['user_data_label'][user] = train_h5['examples'][user]['label'][()]

        print(" Dictionaries ready .. ")
        self.trainset, self.testset = train_dict, test_dict

def download_files(data_cache_dir):

    URL = "https://fedml.s3-us-west-1.amazonaws.com/fed_cifar100.tar.bz2"

    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    file_path = os.path.join(data_cache_dir,"fed_cifar100.tar.bz2") 

    # Download and decompress the file (if we haven't already)
    if not os.path.exists(file_path):
        wget.download(URL, out=file_path)

        file = tarfile.open(file_path)
        file.extractall(os.path.join(data_cache_dir,'fed_cifar100'))
        file.close()
