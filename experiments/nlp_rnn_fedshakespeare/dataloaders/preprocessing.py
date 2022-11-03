import logging
import os
import wget
import tarfile
import h5py
import collections
import numpy as np

data_cache_dir = "./data"
DEFAULT_TRAIN_FILE = "shakespeare_train.h5"
DEFAULT_TEST_FILE = "shakespeare_test.h5"

word_dict = None
word_list = None
_pad = "<pad>"
_bos = "<bos>"
_eos = "<eos>"

''' 
    The FedeShakespeare dataset is taken from FedML repository. For more information regarding this dataset, 
    please refer to https://github.com/FedML-AI/FedML/tree/master/python/fedml/data/fed_shakespeare.

    In order to download the data run the following commands:
        - wget --no-check-certificate --no-proxy https://fedml.s3-us-west-1.amazonaws.com/shakespeare.tar.bz2
        - tar -xvf shakespeare.tar.bz2
    
    This code follows the steps of preprocessing in tff shakespeare dataset: 
    https://github.com/google-research/federated/blob/master/utils/datasets/shakespeare_dataset.py

'''

SEQUENCE_LENGTH = 80  # from McMahan et al AISTATS 2017
# Vocabulary re-used from the Federated Learning for Text Generation tutorial.
# https://www.tensorflow.org/federated/tutorials/federated_learning_for_text_generation

CHAR_VOCAB = list("dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#'/37;?bfjnrvzBFJNRVZ\"&*.26:\naeimquyAEIMQUY]!%)-159\r")

def preprocess(sentences, max_seq_len=SEQUENCE_LENGTH):

    sequences = []

    def to_ids(sentence, num_oov_buckets=1):
        """
        map list of sentence to list of [idx..] and pad to max_seq_len + 1
        Args:
            num_oov_buckets : The number of out of vocabulary buckets.
            max_seq_len: Integer determining shape of padded batches.
        """
        tokens = [char_to_id(c) for c in sentence]
        tokens = [char_to_id(_bos)] + tokens + [char_to_id(_eos)]
        if len(tokens) % (max_seq_len + 1) != 0:
            pad_length = (-len(tokens)) % (max_seq_len + 1)
            tokens += [char_to_id(_pad)] * pad_length
        return (
            tokens[i : i + max_seq_len + 1]
            for i in range(0, len(tokens), max_seq_len + 1)
        )

    for sen in sentences:
        sequences.extend(to_ids(sen))
    return sequences

def char_to_id(char):
    word_dict = get_word_dict()
    if char in word_dict:
        return word_dict[char]
    else:
        return len(word_dict)

def get_word_dict():
    global word_dict
    if word_dict == None:
        words = [_pad] + CHAR_VOCAB + [_bos] + [_eos]
        word_dict = collections.OrderedDict()
        for i, w in enumerate(words):
            word_dict[w] = i
    return word_dict

def split(dataset):
    ds = np.asarray(dataset)
    x = ds[:, :-1]
    y = ds[:, 1:]
    return x, y

def download_files(data_cache_dir):

    URL = "https://fedml.s3-us-west-1.amazonaws.com/shakespeare.tar.bz2"

    if not os.path.exists(data_cache_dir):
        os.makedirs(data_cache_dir)

    file_path = os.path.join(data_cache_dir,"shakespeare.tar.bz2") 

    # Download and decompress the file (if we haven't already)
    if not os.path.exists(file_path):
        wget.download(URL, out=file_path)

        file = tarfile.open(file_path)
        file.extractall(os.path.join(data_cache_dir,'fed_shakespeare'))
        file.close()

class FEDSHAKESPEARE:
    def __init__(self) :

        download_files(data_cache_dir)
        train_h5 = h5py.File(os.path.join(data_cache_dir,'fed_shakespeare', DEFAULT_TRAIN_FILE), "r")
        test_h5 = h5py.File(os.path.join(data_cache_dir, 'fed_shakespeare',DEFAULT_TEST_FILE), "r")
        test_dict = {'users': [], 'num_samples': [], 'user_data': dict(), 'user_data_label': dict()}
        train_dict = {'users': [], 'num_samples': [], 'user_data': dict(), 'user_data_label': dict()}

        for user in train_h5['examples'].keys():
            train_dict['users'].append(user)
            raw_train = train_h5['examples'][user]['snippets'][()]
            raw_train = [x.decode("utf8") for x in raw_train]
            user_data = preprocess(raw_train)
            train_dict['num_samples'].append(len(user_data))

            # split data
            train_x, train_y = split(user_data)
            train_dict['user_data'][user] = train_x
            train_dict['user_data_label'][user] = train_y

        for user in test_h5['examples'].keys():
            test_dict['users'].append(user)
            raw_test = test_h5['examples'][user]['snippets'][()]
            raw_test = [x.decode("utf8") for x in raw_test]
            user_data = preprocess(raw_test)
            test_dict['num_samples'].append(len(user_data))

            # split data
            test_x, test_y = split(user_data)
            test_dict['user_data'][user] = test_x
            test_dict['user_data_label'][user] = test_y
            
        print(" Dictionaries ready .. ")
        self.trainset, self.testset = train_dict, test_dict

