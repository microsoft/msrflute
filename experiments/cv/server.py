# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
In this file, we define the local server that lives inside the client.
'''

from core.server import OptimizationServer

class PersonalizationServer(OptimizationServer):
    def __init__(self, num_clients, model, optimizer, ss_scheduler, data_path, model_path, train_dataloader, train_dataset,
                 val_dataloader, test_dataloader, config, config_server):
        """
        Personalization Server. 
        
        Customized routines for server can be included here.
        """
        super().__init__(num_clients, model, optimizer, ss_scheduler, data_path, model_path, train_dataloader, train_dataset,
                 val_dataloader, test_dataloader, config, config_server)