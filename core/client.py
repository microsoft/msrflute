# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
The Client object is short-lived, instantiated inside of worker 0 and moved to
workers 1 to N for processing a given client's data. It's main method is the
`process_round` function, used to update the model given a client's data.
'''

import copy
import json
import logging
import os
import time
from easydict import EasyDict as edict

import h5py
import numpy as np
import torch

# Internal imports
from core.globals import TRAINING_FRAMEWORK_TYPE
if TRAINING_FRAMEWORK_TYPE == 'mpi':
    import core.federated as federated
else:
    raise NotImplementedError('{} is not supported'.format(TRAINING_FRAMEWORK_TYPE))

from .strategies import select_strategy
from .trainer import (
    Trainer,
    run_validation_generic,
    set_component_wise_lr,
)
from utils import (
    ScheduledSamplingScheduler,
    make_optimizer,
    print_rank,
    scrub_empty_clients,
)
from utils.dataloaders_utils import (
    make_train_dataloader,
    make_val_dataloader,
    make_test_dataloader,
)

import extensions.privacy
from extensions.privacy import metrics as privacy_metrics
from experiments import make_model


# A per-process cache of the training data, so clients don't have to repeatedly re-load
# TODO: deprecate this in favor of passing dataloader around
_data_dict = None
_file_ext = None


class Client:
    # It's unclear why, but sphinx refuses to generate method docs
    # if there is no docstring for this class.
    """Client class for specifying individual client training tasks"""

    def __init__(self, client_id, config, send_gradients, dataloader):
        '''
        Client side processing: computing gradients, update the model and send them back to the server

        Args:
            client_id (int): identifier for grabbing that client's data.
            config (dict): dictionary with parameters loaded from config file.
            send_gradients (bool): if True, model gradients are sent back;
                otherwise, model weights are sent back.
            dataloader (torch.utils.data.DataLoader): dataloader that generates
                training data for the client.
        '''
        super().__init__()

        self.client_id = client_id
        self.client_data = self.get_data(client_id,dataloader)
        self.config = copy.deepcopy(config)
        self.send_gradients = send_gradients

    def get_client_data(self):
        '''"Getter" method that returns all object's attributes at once.'''
        return self.client_id, self.client_data, self.config, self.send_gradients

    @staticmethod
    def get_num_users(filename):
        '''Count users given a JSON or HDF5 file.

        This function will fill the global data dict. Ideally we want data
        handling not to happen here and only at the dataloader, that will be the
        behavior in future releases.

        Args:
            filename (str): path to file containing data.
        '''

        global _data_dict
        global _file_ext
        _file_ext = filename.split('.')[-1]
        
        try:
            if _file_ext == 'json' or _file_ext == 'txt':
                if _data_dict is None:
                    print_rank('Reading training data dictionary from JSON')
                    with open(filename,'r') as fid:
                        _data_dict = json.load(fid)  # pre-cache the training data
                    _data_dict = scrub_empty_clients(_data_dict)  # empty clients MUST be scrubbed here to match num_clients in the entry script
                    print_rank('Read training data dictionary', loglevel=logging.DEBUG)

            elif _file_ext == 'hdf5':
                print_rank('Reading training data dictionary from HDF5')
                _data_dict = h5py.File(filename, 'r')
                print_rank('Read training data dictionary', loglevel=logging.DEBUG)
        
        except:
            raise ValueError('Error reading training file. Please make sure the format is allowed')

        num_users = len(_data_dict['users'])
        return num_users

    @staticmethod
    def get_data(client_id, dataloader):
        '''Load data from the dataloader given the client's id.

        This function will load the global data dict. Ideally we want data
        handling not to happen here and only at the dataloader, that will be the
        behavior in future releases.

        Args:
            client_id (int or list): identifier(s) for grabbing client's data.
            dataloader (torch.utils.data.DataLoader): dataloader that
                provides the trianing 
        '''

        # Auxiliary function for decoding only when necessary
        decode_if_str = lambda x: x.decode() if isinstance(x, bytes) else x

        # During training, client_id will be always an integer
        if isinstance(client_id, int):
            user_name = decode_if_str(_data_dict['users'][client_id])
            num_samples = _data_dict['num_samples'][client_id]
            
            if _file_ext == 'hdf5':
                arr_data = [decode_if_str(e) for e in _data_dict['user_data'][user_name]['x'][()]]
                user_data = {'x': arr_data}
            elif _file_ext == 'json' or _file_ext == 'txt':
                user_data = _data_dict['user_data'][user_name]

            if 'user_data_label' in _data_dict:  # supervised problem
                labels = _data_dict['user_data_label'][user_name]
                if _file_ext == 'hdf5':  # transforms HDF5 Dataset into Numpy array
                    labels = labels[()]

                return edict({'users': [user_name],
                        'user_data': {user_name: user_data},
                        'num_samples': [num_samples],
                        'user_data_label': {user_name: labels}})
            else:
                print_rank('no labels present, unsupervised problem', loglevel=logging.DEBUG)
                return edict({'users': [user_name],
                        'user_data': {user_name: user_data},
                        'num_samples': [num_samples]})

        # During validation and test, client_id might be a list of integers
        elif isinstance(client_id, list):
            if 'user_data_label' in _data_dict:
                users_dict = {'users': [], 'num_samples': [], 'user_data': {}, 'user_data_label': {}}
            else:
                users_dict = {'users': [], 'num_samples': [], 'user_data': {}}
        
            for client in client_id:
                user_name = decode_if_str(dataloader.dataset.user_list[client])
                users_dict['users'].append(user_name)
                users_dict['num_samples'].append(dataloader.dataset.num_samples[client])

                if _file_ext == 'hdf5':
                    arr_data = dataloader.dataset.user_data[user_name]['x']
                    arr_decoded = [decode_if_str(e) for e in arr_data]
                    users_dict['user_data'][user_name] = {'x': arr_decoded}
                elif _file_ext == 'json':
                    users_dict['user_data'][user_name] = {'x': dataloader.dataset.user_data[user_name]['x']}
                elif _file_ext == 'txt':  # using a different line for .txt since our files have a different structure
                    users_dict['user_data'][user_name] = dataloader.dataset.user_data[user_name]

                if 'user_data_label' in _data_dict:
                    labels = dataloader.dataset.user_data_label[user_name]
                    if _file_ext == 'hdf5':
                        labels = labels[()]
                    users_dict['user_data_label'][user_name] = labels
                    
            return users_dict

    @staticmethod
    def run_testvalidate(client_data, server_data, mode, model):
        '''Called by worker to run test/validation sample on a client.

        This functions assumes set_model_for_round has already been called to
        push the model to the client (see federated.py).

        Args:
            client_data (tuple): client data and config. It is a tuple with 4
                components; importantly, the second component is a dict
                containing the data, and the third component is a dict with the
                config parsed from the YAML file.
            server_data (tuple): server data (model parameters mostly). It is
                a tuple with 3 components; importantly, the third component
                consists of the current model parameters.
            mode (str): whether to `test` or `validate`.
            model (torch.nn.Module): actual model without parameters.
        '''

        # Process inputs and initialize variables
        _, data_strct, config, _ = client_data
        _, _, model_parameters = server_data
        config = copy.deepcopy(config)

        begin = time.time()  

        # Use the server's data config since we're distributing test/validate from the server
        data_config = config['server_config']['data_config'][mode]
        want_logits = data_config.get('wantLogits', False)

        # Create dataloader 
        dataloader = None
        print_rank('making dataloader with task {}'.format(config['server_config']['task']), loglevel=logging.DEBUG)
        if mode == 'test':
            dataloader = make_test_dataloader(data_config, data_path=None, task=config['server_config']['task'], data_strct=data_strct)
        elif mode == 'val':
            dataloader = make_val_dataloader(data_config, data_path=None, task=config['server_config']['task'], data_strct=data_strct)

        # Set model parameters
        n_layers, n_params = len([f for f in model.parameters()]), len(model_parameters)
        print_rank(f'Copying model parameters... {n_layers}/{n_params}', loglevel=logging.DEBUG)
        model.cuda() if torch.cuda.is_available() else model
        for p, data in zip(model.parameters(), model_parameters):
            p.data = data.detach().clone().cuda() if torch.cuda.is_available() else data.detach().clone()
        print_rank(f'Model setup complete. {time.time() - begin}s elapsed.', loglevel=logging.DEBUG)

     
        num_instances = sum(data_strct['num_samples'])
        print_rank(f'Validating {num_instances}', loglevel=logging.DEBUG)

        # Compute output and metrics on the test or validation data
        output, metrics = run_validation_generic(model, dataloader)
        output = None if not want_logits else output

        return output, metrics, num_instances

    @staticmethod
    def process_round(client_data, server_data, model, data_path, eps=1e-7):
        '''Compute gradients given client's data and update model.

        Args:
            client_data (tuple): client data and config. It is a tuple
                consisting of 4 components: an int indicating the client's id, a
                dict containing that client's data, a dict with the config
                parsed from the YAML file, and a bool indicating whether or not
                gradients should be sent.
            server_data (tuple): server data (model parameters mostly). It is
                a tuple consisting of 3 components; importantly, the first is
                a float giving the client's learning rate, and the third a list
                of torch.Tensor's with current model parameters. The second one
                is not used, right now.
            model (torch.nn.Module): actual model without parameters.
            data_path (str): where to get data from.
            eps (float): lower bound for aggregation weights.
        '''

        # Ensure the client is assigned to the correct GPU
        if torch.cuda.is_available() and torch.cuda.device_count() == federated.size():
            torch.cuda.set_device(federated.local_rank())

        # Process inputs and initialize variables
        client_id, data_strct, config, send_gradients = client_data
        initial_lr, _, model_parameters = server_data
        config = copy.deepcopy(config)

        model_config = config['model_config']
        client_config = config['client_config']
        data_config = client_config['data_config']['train']
        task = client_config.get('task', {})
        trainer_config = client_config.get('trainer_config', {})
        privacy_metrics_config = config.get('privacy_metrics_config', None)

        StrategyClass = select_strategy(config['strategy'])
        strategy = StrategyClass('client', config)
        print_rank(f'Client successfully instantiated strategy {strategy}', loglevel=logging.DEBUG)

        begin = time.time()  
        client_stats = {}      
        
        # Update the location of the training file
        data_config['list_of_train_data'] = os.path.join(data_path, data_config['list_of_train_data'])

        user = data_strct['users'][0]
        if 'user_data_label' in data_strct.keys():  # supervised case
            input_strct = edict({
                'users': [user],
                'user_data': {user: data_strct['user_data'][user]},
                'num_samples': [data_strct['num_samples'][0]],
                'user_data_label': {user: data_strct['user_data_label'][user]}
            })
        else:
            input_strct = edict({
                'users': [user],
                'user_data': {user: data_strct['user_data'][user]},
                'num_samples': [data_strct['num_samples'][0]]
            })

        print_rank('Loading : {}-th client with name: {}, {} samples, {}s elapsed'.format(
            client_id, user, data_strct['num_samples'][0], time.time() - begin), loglevel=logging.INFO)

        # Get dataloaders
        train_dataloader = make_train_dataloader(data_config, data_path, task=task, clientx=0, data_strct=input_strct)
        val_dataloader   = make_val_dataloader(data_config, data_path)

        # Instantiate the model object
        if model is None:
            model = make_model(
                model_config,
                dataloader_type=train_dataloader.__class__.__name__,
                input_dim=data_config['input_dim'],
                vocab_size=train_dataloader.vocab_size,
            )

        # Set model parameters
        n_layers, n_params = len([f for f in model.parameters()]), len(model_parameters)
        print_rank(f'Copying model parameters... {n_layers}/{n_params}', loglevel=logging.DEBUG)
        model.cuda() if torch.cuda.is_available() else model
        for p, data in zip(model.parameters(), model_parameters):
            p.data = data.detach().clone().cuda() if torch.cuda.is_available() else data.detach().clone()
        print_rank(f'Model setup complete. {time.time() - begin}s elapsed.', loglevel=logging.DEBUG)

        # Fix parameters of layers
        if 'updatable_names' in trainer_config:
            set_component_wise_lr(model, client_config['optimizer_config'], trainer_config['updatable_names'])

        # Create the optimizer on the workers
        # NOTE: the server dictates the learning rate for the clients
        client_config['optimizer_config']['lr'] = initial_lr
        optimizer = make_optimizer(client_config['optimizer_config'], model)

        # Make the scheduled sampling scheduler
        ss_scheduler = None
        if 'ss_config' in client_config and client_config['ss_config'] is not None:
            ss_scheduler = ScheduledSamplingScheduler(model=model, **client_config['ss_config'])

        # Make the trainer
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            ss_scheduler=ss_scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            server_replay_config =client_config,
            max_grad_norm=client_config['data_config']['train'].get('max_grad_norm', None),
            anneal_config=client_config['annealing_config'] if 'annealing_config' in client_config else None,
            num_skips_threshold=client_config['num_skips_threshold'] if 'num_skips_threshold' in client_config else -1,
            ignore_subtask=client_config['ignore_subtask']
        )

        if trainer.optimizer is not None:
            initial_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())

        annealing_config = client_config['annealing_config'] if 'annealing_config' in client_config else None

        assert 'desired_max_samples' in client_config['data_config']['train'], 'Missing \'desired_max_samples\' entry in data config parameter'
        desired_max_samples = client_config['data_config']['train']['desired_max_samples']

        if trainer.optimizer is not None:  # reset the optimizer state
            if initial_lr > 0:
                trainer.optimizer.param_groups[0].update({'lr': initial_lr})
            initial_optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
            trainer.reset_optimizer(initial_optimizer_state, annealing_config)

        # Mark the end of setup
        end = time.time()
        client_stats['setup'] = end - begin
        print_rank(f'Client setup cost {client_stats["setup"]}s', loglevel=logging.DEBUG)               
        begin_training = end
        
        # Training begins here
        trainer.model.train()
        trainer.model.zero_grad()

        # Save the client batches if we want to evaluate the privacy metrics
        apply_privacy_metrics = (False if privacy_metrics_config is None else privacy_metrics_config['apply_metrics'])

        # This is where training actually happens
        train_loss, num_samples = trainer.train_desired_samples(desired_max_samples=desired_max_samples, apply_privacy_metrics=apply_privacy_metrics)
        print_rank('client={}: training loss={}'.format(client_id, train_loss), loglevel=logging.DEBUG)

        # Estimate gradient magnitude mean/var
        trainer.sufficient_stats['mean'] = trainer.sufficient_stats['sum'] / trainer.sufficient_stats['n']
        trainer.sufficient_stats['mag'] = np.sqrt(trainer.sufficient_stats['sq_sum'] / trainer.sufficient_stats['n'])
        trainer.sufficient_stats['var'] = trainer.sufficient_stats['sq_sum'] / trainer.sufficient_stats['n'] - \
            trainer.sufficient_stats['mag'] ** 2
        trainer.sufficient_stats['norm'] = np.sqrt(trainer.sufficient_stats['sq_sum'])

        trainer.train_loss = train_loss
        trainer.num_samples = num_samples

        # Compute pseudo-gradient
        for p, data in zip(trainer.model.parameters(), model_parameters):
            data = data.cuda() if torch.cuda.is_available() else data
            p.grad = data - p.data

        payload = strategy.generate_client_payload(trainer) if send_gradients else None

        # Mark that training (including post-processing) is finished
        end = time.time()
        client_stats['training'] = end - begin_training
        client_stats['full cost'] = end - begin
        print_rank(f'Client training cost {end - begin_training}s', loglevel=logging.DEBUG)      
        print_rank(f'Client full cost {end - begin}s', loglevel=logging.DEBUG)

        # Create dictionary that is sent back to server
        client_output = {
            'cs': client_stats, 
            'tl': train_loss, 
            'mg': trainer.sufficient_stats['mag'],
            'vg': trainer.sufficient_stats['var'],
            'ng': trainer.sufficient_stats['mean'],
            'rg': trainer.sufficient_stats['norm'],
            'ns': num_samples,
            'pl': payload,
        }
       
        # Apply privacy metrics
        if privacy_metrics_config and privacy_metrics_config['apply_metrics']:
            print_rank('Applying privacy metrics', loglevel=logging.DEBUG)

            privacy_stats = {'Dropped clients': 0}
            batches = trainer.cached_batches
            trainer.cached_batches = []
            gradients = extensions.privacy.unroll_network(model.named_parameters(), select_grad=True)[0]

            if privacy_metrics_config['apply_indices_extraction']:
                allowed_word_rank = privacy_metrics_config.get('allowed_word_rank', 9000)
                embed_dim, vocab_size = model_config['embed_dim'], model_config['vocab_size']
                overlap, indices = privacy_metrics.extract_indices_from_embeddings(gradients, batches, embed_dim, vocab_size)

                max_overlap =  privacy_metrics_config.get('max_allowed_overlap', None)
                if max_overlap is not None and overlap > max_overlap:
                    print_rank('Removing this client because we extracted {}% words and the maximum allowed is {}%'.format(overlap * 100, max_overlap * 100))
                    client_output['wt'] = 0.0
                    privacy_stats['Dropped clients'] = 1

                privacy_stats['Extracted indices percentage'] = overlap
                privacy_stats['Words percentage above ' + str(allowed_word_rank) + ' word rank'] = (indices > allowed_word_rank).mean() if len(indices)>0 else 0
          
            if privacy_metrics_config['apply_leakage_metric']:
                print_rank('Applying leakage metric', loglevel=logging.DEBUG)

                orig_params = {n: p for (n, _), p in zip(trainer.model.named_parameters(), model_parameters)}
                max_ratio = np.exp(privacy_metrics_config['max_leakage'])
                optim_config = privacy_metrics_config['attacker_optimizer_config']
                is_leakage_weighted = privacy_metrics_config['is_leakage_weighted']

                leakage = privacy_metrics.practical_epsilon_leakage(orig_params,
                    trainer.model, batches, is_leakage_weighted, max_ratio, optim_config)                
                print_rank('privacy leakage: {}'.format(leakage), loglevel=logging.DEBUG)

                max_leakage =  privacy_metrics_config.get('max_allowed_leakage', None)
                if max_leakage is not None and leakage > max_leakage:
                    print_rank('Removing this client because the information leakage/practical epsilon is {} and the maximum allowed is {}'.format(leakage, max_leakage))
                    client_output['wt'] = 0.0
                    privacy_stats['Dropped clients'] = 1

                privacy_stats['Practical epsilon (Max leakage)'] = leakage
            
            client_output['ps'] = privacy_stats

        client_output['ts'] = time.time()
        return client_output
