# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
In this file we define the functions for running
test and validation tasks inside the Server.
'''

import logging
import torch
import numpy as np

# Internal imports
from core.globals import TRAINING_FRAMEWORK_TYPE
if TRAINING_FRAMEWORK_TYPE == 'mpi':
    import core.federated as federated
else:
    raise NotImplementedError('{} is not supported'.format(TRAINING_FRAMEWORK_TYPE))

from core.client import Client
from utils import (
    print_rank
)

# AzureML-related libs
from azureml.core import Run
run = Run.get_context()

class Evaluation():

    def __init__(self, config, model_path, process_testvalidate, val_dataloader, test_dataloader):

        self.config = config
        self.model_path = model_path
        self.process_testvalidate = process_testvalidate
        self.test_dataloader = val_dataloader
        self.val_dataloader = test_dataloader

        super().__init__()
    
    def run(self, eval_list, req, metric_logger=None):
        '''Run test/validation taks depending on the modes
        received in the eval_list.
        
        Args:
            eval_list (arr): Contains the tasks to run.
            req (dict): information for test/val tasks
            metric_logger (callback, optional): callback used for logging.
                Defaults to None, in which case AML logger is used.
        '''      
        
        self.worker_trainer = req['worker_trainer']
        save_model = False 
        if metric_logger is None:
            metric_logger = run.log

        for mode in eval_list:

            # Skipping validation round when RL is enabled
            if 'wantRL' in self.config['server_config'] and self.config['server_config']['wantRL'] and mode == "val":
                continue
            
            # Compute avg_loss and avg_acc
            self.metrics = self.run_distributed_inference(mode)
            req = self.initialize_req(req) if len(req) == 1 else req

            # Log metrics
            for key, value in self.metrics.items():
                metric_logger(str(mode + " " + key).capitalize(), value['value'])
                print_rank('LOG: {}_{}={}: best_{}_{}={}'.format(mode, key, value['value'], mode, key, req[str("best_"+ mode + "_" + key)]))

            for key,value in self.metrics.items():
                attr = str("best_"+ mode + "_" + key)
                if value['higher_is_better']:
                    if self.metrics[key]['value'] > req[attr]: 
                        req[attr] = self.metrics[key]['value']
                        save_model = True
                else:
                    if self.metrics[key]['value'] < req[attr]:
                        req[attr] = self.metrics[key]['value']
                        save_model = True
                
                if save_model and mode == 'val':
                    self.worker_trainer.save(
                        model_path=self.model_path,
                        token=str('best_'+ mode +'_'+key),
                        config=self.config['server_config']
                    )
                    save_model = False
        
        return req
    
    def initialize_req(self, req):
        '''Update the keys, to have the same as metrics dictionary. This 
        function is only used during itr=0 for initializing the req 
        dictionary. 

        Args:
            req (dict): Best results for all the metrics (e.g. best_val_acc).
        '''
        for mode in ['test','val']:
            for key in self.metrics.keys():
                attr = "best_"+ mode + "_" + key 
                req[attr] = -1.0 if self.metrics[key]['higher_is_better'] else float('inf')

        return req

    def run_distributed_inference(self, mode):
        '''Call `run_distributed_evaluation` specifically for test or validation.
        
        This is just a helper function that fetches the dataloader depending on
        the mode and calls `run_distributed_evaluation` using that dataloader.

        Args:
            mode (str): `test` or `val`.
        '''
        if mode == 'val':
            dataloader = self.val_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        else:
            raise NotImplementedError('Unsupported mode: {}'.format(mode))
        return self.run_distributed_evaluation(dataloader, mode)

    def run_distributed_evaluation(self, dataloader, mode):
        '''Perform evaluation using available workers.

        See also `process_test_validate` on federated.py.

        Args:
            dataloader (torch.utils.data.DataLoader): used to fetch data.
            mode (str): `test` or `val`.
        '''
        val_clients = list(self.make_eval_clients(dataloader))
        print_rank(f'mode: {mode} evaluation_clients {len(val_clients)}', loglevel=logging.DEBUG)

        total = 0
        self.logits = {'predictions': [], 'probabilities': [], 'labels': []}
        server_data = (0.0, [p.data.to(torch.device('cpu')) for p in self.worker_trainer.model.parameters()])

        for result in self.process_testvalidate(val_clients, server_data, mode):
            output, metrics, count = result
            val_metrics =  {key: {'value':0, 'higher_is_better': False} for key in metrics.keys()} if total == 0 else val_metrics
 
            for key in val_metrics:
                val_metrics[key]['value'] += metrics[key]['value']* count
                val_metrics[key]['higher_is_better'] = metrics[key]['higher_is_better']
            total+= count
            
            if output is not None:
                self.logits['predictions'].append(output['predictions'])
                self.logits['probabilities'].append(output['probabilities'])
                self.logits['labels'].append(output['labels'])

        if  self.logits['probabilities'] and self.logits['predictions'] and self.logits['labels']:
            self.logits['predictions'] = np.concatenate(self.logits['predictions'])
            self.logits['probabilities'] = np.concatenate(self.logits['probabilities'])
            self.logits['labels'] = np.concatenate(self.logits['labels'])

        
        for key in val_metrics:
                val_metrics[key]['value'] = val_metrics[key]['value']/total 
            
        self.losses = [val_metrics['loss']['value'], val_metrics['acc']['value']] # For compatibility with Server
        return val_metrics

    def make_eval_clients(self, dataloader):
        '''Generator that yields clients for evaluation, continuously.

        Args:
            dataloader (torch.utils.data.DataLoader): used to get client's data
        '''

        total = sum(dataloader.dataset.num_samples)
        clients = federated.size() - 1
        delta = total / clients + 1
        threshold = delta
        current_users_idxs = list()
        current_total = 0

        # Accumulate users until a threshold is reached to form client
        for i in range(len(dataloader.dataset.user_list)):
            current_users_idxs.append(i)
            count = dataloader.dataset.num_samples[i]
            current_total += count
            if current_total > threshold:
                print_rank(f'sending {len(current_users_idxs)} users', loglevel=logging.DEBUG)
                yield Client(current_users_idxs, self.config, False, dataloader)
                current_users_idxs = list() 
                current_total = 0

        if len(current_users_idxs) != 0:
            print_rank(f'sending {len(current_users_idxs)} users -- residual', loglevel=logging.DEBUG)
            yield Client(current_users_idxs, self.config, False, dataloader)
