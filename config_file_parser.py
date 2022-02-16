# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
A collection of functions for checking the format of configuration values
"""
import os

def check_server_config(config, default_server_conf):

    assert "server_config" in config, "server config setting is missing"

    # Checking parameters for server-side training
    if "train" in config["server_config"]["data_config"]:
        if "train_data_server" in config["server_config"]["data_config"]["train"]:
            assert "server_replay_config" in config["server_config"], "Training dataset is defined on the server but training parameters are not set"
            assert "optimizer_config" in config["server_config"]["server_replay_config"], "Missing \"optimizer_config\" in server_replay server training config"
            assert "server_iterations" in config["server_config"]["server_replay_config"],  "Missing \"server_iterations\" in server_replay server training config"

    # Setting the default values if missing
    for key in default_server_conf.keys():
        if not key in config["server_config"]:
            config["server_config"][key] = default_server_conf[key]

    server_type = config["server_config"]["type"]
    if not (server_type == "model_averaging" or \
        server_type == "optimization" or \
        server_type == "model_optimization" or \
        server_type == "cluster_finetuning" or \
        server_type == "cluster_parallel") :
        raise ValueError("Invalid server type {} in federated learning config".format(server_type))

    assert "best_model_criterion" in config["server_config"], "Missing \"best_model_criterion\" in server config"

    if server_type == "model_optimization" or server_type == "cluster_finetuning" or server_type == "cluster_parallel":
        assert "initial_lr_client" in config["server_config"], "Missing \"initial_lr_client\" in server config"
        assert "lr_decay_factor" in config["server_config"], "Missing \"lr_decay_factor\" in server config"
        assert "aggregate_median" in config["server_config"], "Missing \"aggregate_median\" in server config"

    if "nbest_task_scheduler" in config["server_config"]:
        assert "num_tasks" in config["server_config"]["nbest_task_scheduler"], "Define \"num_tasks\" in [\"nbest_task_scheduler\"]"
        assert "iteration_per_task" in config["server_config"]["nbest_task_scheduler"], "Define \"iteration_per_task\" in [\"nbest_task_scheduler\"]"
        assert len(config["server_config"]["nbest_task_scheduler"]["num_tasks"]) == len(config["server_config"]["nbest_task_scheduler"]["iteration_per_task"]), \
            "Length mismatched: {}!={}".format(len(config["server_config"]["nbest_task_scheduler"]["num_tasks"]), len(config["server_config"]["nbest_task_scheduler"]["iteration_per_task"]))

    data_path = config['data_path']
    if 'vocab_dict' in config["server_config"]["data_config"]["val"]:
        config["server_config"]["data_config"]["val"]["vocab_dict"]=os.path.join(data_path, config["server_config"]["data_config"]["val"]["vocab_dict"])
    if 'vocab_dict' in config["server_config"]["data_config"]["test"]:
        config["server_config"]["data_config"]["test"]["vocab_dict"]=os.path.join(data_path, config["server_config"]["data_config"]["test"]["vocab_dict"])
    if 'vocab_dict' in config["server_config"]["data_config"]["test"]:
        config["server_config"]["data_config"]["train"]["vocab_dict"]=os.path.join(data_path, config["server_config"]["data_config"]["train"]["vocab_dict"])


    # BERT specific parameters
    if 'model_config' in config and 'BERT' in config['model_config']:
        if 'model_name_or_path' in config['model_config']['BERT']['model']:
            config['server_config']['data_config']['val']['model_name_or_path'] =config['model_config']['BERT']['model']['model_name_or_path']
            config['server_config']['data_config']['test']['model_name_or_path']=config['model_config']['BERT']['model']['model_name_or_path']
        else:
            config['server_config']['data_config']['val']['model_name_or_path'] =config['model_config']['BERT']['model']['model_name']
            config['server_config']['data_config']['test']['model_name_or_path']=config['model_config']['BERT']['model']['model_name']

        if 'process_line_by_line' in config['model_config']['BERT']['model']:
            config['server_config']['data_config']['val']['process_line_by_line'] =config['model_config']['BERT']['model']['process_line_by_line']
            config['server_config']['data_config']['test']['process_line_by_line']=config['model_config']['BERT']['model']['process_line_by_line']

    if "initial_val" in config['server_config']:
        config['server_config']['initial_val'] = config['server_config']['initial_val']
    else:
        config['server_config']['initial_val'] = False

    if "initial_rec" in config['server_config']:
        config['server_config']['initial_rec'] = config['server_config']['initial_rec']
    else:
        config['server_config']['initial_rec'] = False
        
    return config


def check_client_config(config, default_client_conf):

    assert "client_config" in config, "client config setting is missing"

    # Setting the default values if missing
    for key in default_client_conf.keys():
        if not key in config["client_config"]:
            config["client_config"][key] = default_client_conf[key]

    client_type = config["client_config"]["type"]
    if not (client_type == "gradient_computation" or client_type == "optimization"):
        raise ValueError("Invalid client option {} in federated learning config".format(client_type))

    if not "ss_config" in config["client_config"]:
        config["client_config"]["ss_config"] = None

    if "list_of_train_data" in config["client_config"]["data_config"]["train"] and "train_data" in config["client_config"]["data_config"]["train"]:
        raise ValueError("\"list_of_train_data\" and \"train_data\" cannot be defined at the same time")

    assert "list_of_train_data" in config["client_config"]["data_config"]["train"] or "train_data" in config["client_config"]["data_config"]["train"], "Define either \"list_of_train_data\" and \"train_data\""

    # Adjust path to vocab_dict
    data_path = config['data_path']
    if 'vocab_dict' in config["client_config"]["data_config"]["train"]:
        config["client_config"]["data_config"]["train"]["vocab_dict"]=os.path.join(data_path, config["client_config"]["data_config"]["train"]["vocab_dict"])

    # BERT specific parameters
    if 'model_config' in config and 'train' in config['client_config']['data_config'] and 'BERT' in config['model_config']:
        if 'model_name_or_path' in config['model_config']['BERT']['model']:
            config['client_config']['data_config']['train']['model_name_or_path']=config['model_config']['BERT']['model']['model_name_or_path']
        else:
            config['client_config']['data_config']['train']['model_name_or_path']=config['model_config']['BERT']['model']['model_name']
        if 'process_line_by_line' in config['model_config']['BERT']['model']:
            config['client_config']['data_config']['train']['process_line_by_line'] =config['model_config']['BERT']['model']['process_line_by_line']

    return config
