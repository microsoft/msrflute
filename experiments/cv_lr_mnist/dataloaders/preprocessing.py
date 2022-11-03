import os
import wget
import zipfile
import numpy as np
import json

FEDML_DATA_MNIST_URL = "https://fedcv.s3.us-west-1.amazonaws.com/MNIST.zip"
data_cache_dir = "./data"

''' 
    The MNIST dataset is taken from FedML repository. For more information regarding this dataset, 
    please refer to https://github.com/FedML-AI/FedML/tree/master/python/fedml/data/MNIST.

    In order to download the data run the following commands:
        - wget --no-check-certificate --no-proxy https://fedcv.s3.us-west-1.amazonaws.com/MNIST.zip
        - unzip MNIST.zip
'''

class MNIST:
    def __init__(self) :
        
        download_mnist(data_cache_dir)
        self.trainset, self.testset = read_data(
            train_data_dir = os.path.join(data_cache_dir,'MNIST','train'),
            test_data_dir= os.path.join(data_cache_dir,'MNIST','test'),
        )
        print("Dictionaries ready ..")

def download_mnist(data_cache_dir):
    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    file_path = os.path.join(data_cache_dir,"MNIST.zip") 

    # Download the file (if we haven't already)
    if not os.path.exists(file_path):
        wget.download(FEDML_DATA_MNIST_URL, out=file_path)

    with zipfile.ZipFile(file_path, "r") as zip_ref:
        zip_ref.extractall(data_cache_dir)

def read_data(train_data_dir, test_data_dir):

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith(".json")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        with open(file_path, "r") as inf:
            train_data = json.load(inf)

    train_data['user_data_label'] = dict()
    for user in train_data['user_data']:
        train_data['user_data_label'][user] = train_data['user_data'][user]['y']
        train_data['user_data'][user] = train_data['user_data'][user]['x']
    
    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith(".json")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        with open(file_path, "r") as inf:
            test_data = json.load(inf)  
        
    test_data['user_data_label'] = dict()
    for user in test_data['user_data']:
        test_data['user_data_label'][user] = test_data['user_data'][user]['y']
        test_data['user_data'][user] = test_data['user_data'][user]['x']
        
    return train_data, test_data