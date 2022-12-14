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
import core.federated as federated
from core.client import Client
from utils import print_rank

# AzureML-related libs
from azureml.core import Run
run = Run.get_context()

class Evaluation():

    def __init__(self, config, model_path, process_testvalidate, idx_val_clients, idx_test_clients):

        self.config = config
        self.model_path = model_path
        self.process_testvalidate = process_testvalidate
        self.server_type = config['server_config']['type']
        self.idx_val_clients = idx_val_clients
        self.idx_test_clients = idx_test_clients
        self.send_dicts = config['server_config'].get('send_dicts', False)
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
        if self.send_dicts:
            global_model_values = [self.worker_trainer.model.state_dict()[param_key].to(torch.device('cpu')) for param_key in self.worker_trainer.model.state_dict()]
        else:
            global_model_values = [p.data.to(torch.device('cpu')) for p in self.worker_trainer.model.parameters()]

        if 'tmp_unsup' in req:
            unsup_values = req['tmp_unsup'].values()
            sup_values = req['tmp_sup'].values()
            semisupervision_inference = True
        else:
            semisupervision_inference = False

        save_model = False 
        
        if metric_logger is None:
            metric_logger = run.log

        for mode in eval_list:

            # Skipping validation round when RL is enabled
            if 'wantRL' in self.config['server_config'] and self.config['server_config']['wantRL'] and mode == "val":
                continue
            
            # Compute avg_loss and avg_acc
            self.metrics = self.run_distributed_inference(mode, global_model_values)
            req = self.initialize_req(req) if len(req) == 1 else req

            # Only if for semisupervision
            if semisupervision_inference:
                unsup_metrics = self.run_distributed_inference(mode, unsup_values)
                sup_metrics = self.run_distributed_inference(mode, sup_values)

                for key, value in unsup_metrics.items():
                    metric_logger(str("Unsup" +mode + " " + key).capitalize(), value['value'])
                    print_rank('LOG UNSUP: {}_{}={}'.format(mode, key, value['value']))
                
                for key, value in sup_metrics.items():
                    metric_logger(str("Sup" + mode + " " + key).capitalize(), value['value'])
                    print_rank('LOG SUP: {}_{}={}'.format(mode, key, value['value']))

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

    def run_distributed_inference(self, mode, model):
        '''Call `run_distributed_evaluation` specifically for test or validation.
        
        This is just a helper function that fetches the clients depending on
        the mode and calls `run_distributed_evaluation` using that list.

        Args:
            mode (str): `test` or `val`.
        '''
        if mode == 'val':
            clients = self.idx_val_clients
        elif mode == 'test':
            clients = self.idx_test_clients
        else:
            raise NotImplementedError('Unsupported mode: {}'.format(mode))

        return self.run_distributed_evaluation(mode, clients, model)

    def run_distributed_evaluation(self, mode, clients, model):
        '''Perform evaluation using available workers.

        See also `process_test_validate` on federated.py.

        Args:
            mode (str): `test` or `val`.
            clients (list): clients for test/val round.
        '''

        total = 0
        self.logits = {'predictions': [], 'probabilities': [], 'labels': []}
        server_data = (0.0, model, 0)
        for result in self.process_testvalidate(clients, server_data, mode):
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

def make_eval_clients(dataset, config):
    '''Generator that yields clients for evaluation, continuously.

    Args:
        dataset (torch.utils.data.Dataset): used to get client's data
        config (dict): used for the client's constructor
    '''

    total = sum(dataset.num_samples)
    clients = federated.size() - 1
    delta = total / clients + 1
    threshold = delta
    current_users_idxs = list()
    current_total = 0

    if config["server_config"]["type"] == "personalization":  
        for i in range(len(dataset.user_list)):
            yield Client([i], config, False)
    else:
        for i in range(len(dataset.user_list)):
            current_users_idxs.append(i)
            count = dataset.num_samples[i]
            current_total += count
            if current_total > threshold:
                print_rank(f'sending {len(current_users_idxs)} users', loglevel=logging.DEBUG)
                yield Client(current_users_idxs, config, False)
                current_users_idxs = list()
                current_total = 0

        if len(current_users_idxs) != 0:
            print_rank(f'sending {len(current_users_idxs)} users -- residual', loglevel=logging.DEBUG)
            yield Client(current_users_idxs, config, False)
