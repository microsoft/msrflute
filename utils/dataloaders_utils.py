# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import logging
from importlib.machinery import SourceFileLoader
from utils import print_rank

def get_exp_dataloader(task):
    """ Detect the dataloader declared in the experiment folder

    Args:
        task (str): task parsed from the console
    """
    
    try:
        dir = os.path.join('experiments',task,'dataloaders','text_dataloader.py')
        loader = SourceFileLoader("TextDataLoader",dir).load_module()
        loader = loader.TextDataLoader
    except:
        print_rank("Dataloader not found, please make sure is located inside the experiment folder")

    return loader


def detect_loader_type(my_data, loader_type):
    """ Detect the loader type declared in the configuration file

    Inside this function should go the implementation of 
    specific detection for any kind of loader. 

    Args:
        my_data (str): path of file or chunk file set 
        loader_type (str): loader description in yaml file
    """

    if not loader_type == "auto_detect":
        return loader_type
    
    # Here should go the implementation for the rest of loaders
    else:
        raise ValueError("Unknown format: {}".format(loader_type))


def make_train_dataloader(data_config, data_path, clientx, task=None, vec_size=300, data_strct=None):
    """ Create a dataloader for training on either server or client side """

    mode = 'train'
    tokenizer_type= data_config.get('tokenizer_type', 'not_applicable')

    # Training list for a server
    if clientx is None:  
        if not "train_data_server" in data_config or data_config["train_data_server"] is None:
            print_rank("No server training set is defined")
            return None
        my_data = os.path.join(data_path, data_config["train_data_server"])
        mode='val'
        
    # Training list on a client side
    else:  
        if tokenizer_type != 'not_applicable':
            assert clientx >=0 and clientx < len(data_config["train_data"]), "Invalid client index {}".format(clientx)
            my_data = data_config["train_data"][clientx]
        else:
            my_data = data_config["list_of_train_data"]

    # Find the loader_type
    loader_type = detect_loader_type(my_data, data_config["loader_type"])

    if loader_type == 'text':
        TextDataLoader = get_exp_dataloader(task)
        train_dataloader = TextDataLoader(
                                    data = data_strct if data_strct is not None else my_data,
                                    user_idx = clientx,
                                    mode = mode,
                                    args=data_config
                                    )
    else:
        raise NotImplementedError("Not supported {}: detected_type={} loader_type={} audio_format={}".format(my_data, loader_type, data_config["loader_type"], data_config["audio_format"]))
    return train_dataloader



def make_val_dataloader(data_config, data_path, task=None, data_strct=None):
    """ Return a data loader for a validation set """

    if not "val_data" in data_config or data_config["val_data"] is None:
        print_rank("Validation data list is not set", loglevel=logging.DEBUG)
        return None

    loader_type = detect_loader_type(data_config["val_data"], data_config["loader_type"])

    if loader_type == 'text':
        TextDataLoader = get_exp_dataloader(task)

        val_dataloader = TextDataLoader(
                                data = data_strct if data_strct is not None else os.path.join(data_path, data_config["val_data"]),
                                user_idx = 0,
                                mode = 'val',
                                args=data_config
                                )
    else:
        raise NotImplementedError("Not supported loader_type={} audio_format={}".format(loader_type, data_config["audio_format"]))
    return val_dataloader


def make_test_dataloader(data_config, data_path, task=None, data_strct=None):
    """ Return a data loader for an evaluation set. """

    if not "test_data" in data_config or data_config["test_data"] is None:
        print_rank("Test data list is not set")
        return None

    loader_type = detect_loader_type(data_config["test_data"], data_config["loader_type"])

    if loader_type == 'text':
        TextDataLoader = get_exp_dataloader(task)

        test_dataloader = TextDataLoader(
                                data = data_strct if data_strct is not None else os.path.join(data_path, data_config["test_data"]),
                                user_idx = 0,
                                mode = 'test',
                                args=data_config
                                )

    else:
        raise NotImplementedError("Not supported loader_type={} audio_format={}".format(loader_type, data_config["audio_format"]))
    return test_dataloader


