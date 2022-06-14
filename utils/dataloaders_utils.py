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
        dir = os.path.join('experiments',task,'dataloaders','dataloader.py')
        loader = SourceFileLoader("DataLoader",dir).load_module()
        loader = loader.DataLoader
    except:
        print_rank("Dataloader not found, please make sure is located inside the experiment folder")

    return loader

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

    DataLoader = get_exp_dataloader(task)
    train_dataloader = DataLoader(data = data_strct if data_strct is not None else my_data,
                                    user_idx = clientx,
                                    mode = mode,
                                    args=data_config
                                    )

    return train_dataloader



def make_val_dataloader(data_config, data_path, task=None, data_strct=None, train_mode=False):
    """ Return a data loader for a validation set """

    DataLoader = get_exp_dataloader(task)
    val_file = os.path.join(data_path, data_config["val_data"]) if data_config["val_data"] != None and data_path != None else None
    val_dataloader = DataLoader(data = data_strct if data_strct is not None else val_file,
                                user_idx = 0,
                                mode = 'val',
                                args=data_config
                                )

    return val_dataloader


def make_test_dataloader(data_config, data_path, task=None, data_strct=None):
    """ Return a data loader for an evaluation set. """

    DataLoader = get_exp_dataloader(task)
    test_file = os.path.join(data_path, data_config["test_data"]) if data_config["test_data"] != None and data_path != None else None
    test_dataloader = DataLoader(data = data_strct if data_strct is not None else test_file,
                                user_idx = 0,
                                mode = 'test',
                                args=data_config
                                )

    return test_dataloader


