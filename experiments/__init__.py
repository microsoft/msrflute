# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from utils import print_rank, print_cuda_stats, to_device
from importlib.machinery import SourceFileLoader

def make_model(model_config, dataloader_type=None, input_dim=-1, output_dim=-1):
    print('Preparing model .. Initializing')
    
    try:
        dir = "./"+ str(model_config["model_folder"])
        model_class = model_config["model_type"]
        loader = SourceFileLoader(model_class,dir).load_module()
        model_type = getattr(loader,model_class )
    except:
        raise ValueError("{} model not found, make sure to indicate the model path in the .yaml file".format(model_config["type"]))

    model = model_type(model_config)
    print(model)

    if not "weight_init" in model_config or model_config["weight_init"] == "default":
        print_rank("initialize model with default settings")
        pass
    elif model_config["weight_init"] == "xavier_normal":
        print_rank("initialize model with xavier_normal")
        for p in model.parameters():
            if p.dim() > 1: # weight
                torch.nn.init.xavier_normal_(p.data)
            elif p.dim() == 1: # bias
                p.data.zero_()
        for m in model.modules():
            if isinstance(m, (torch.nn.Embedding, torch.nn.LayerNorm, torch.nn.BatchNorm2d)):
                m.reset_parameters()
    else:
        return ValueError("{} not supported".format(model_config["weight_init"]))

    print_rank("trying to move the model to GPU")
    model = to_device(model)
    print_rank("model: {}".format(model))
    print_cuda_stats()

    return model
