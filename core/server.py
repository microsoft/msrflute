# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
In this file, we define the classes that live inside 'worker 0', the worker
responsible for orchestration and aggregation. The main class is the
OptimizationServer, which sends clients to the other workers to process and
combines the resulting models.
'''

import copy
import json
import logging
import os
import random
import shutil
import time
from collections import defaultdict

import numpy as np
import torch

# Internal imports
from core.globals import TRAINING_FRAMEWORK_TYPE
if TRAINING_FRAMEWORK_TYPE == 'mpi':
    import core.federated as federated
else:
    raise NotImplementedError('{} is not supported'.format(TRAINING_FRAMEWORK_TYPE))

from core.client import Client
from .trainer import (
    ModelUpdater,
    Trainer,
    set_component_wise_lr,
)
from utils import (
    compute_grad_cosines,
    get_lr,
    print_rank,
    update_json_log,
)
from utils.utils import _to_cuda

from extensions import (
    RL,
    privacy,
)

# For profiling
import cProfile
import pstats

# AzureML-related libs
from azureml.core import Run
run = Run.get_context()


class OptimizationServer(federated.Server):
    def __init__(self, num_clients, model, optimizer, ss_scheduler, data_path, model_path, train_dataloader,
                 val_dataloader, test_dataloader, config, config_server):
        '''Implement Server's orchestration and aggregation.

        This is the main Server class, that actually implements orchestration
        and aggregation, inheriting from `federated.Server`, which deals with
        communication only.

        The `train` method is central in FLUTE, as it defines good part of what
        happens during training.

        Args:
            num_clients (int): total available clients.
            model (torch.nn.Module): neural network model.
            optimizer (torch.optim.Optimizer): optimizer.
            ss_scheduler: scheduled sampling scheduler.
            data_path (str): points to where data is.
            model_path (str): points to where pretrained model is.
            train_dataloader (torch.utils.data.DataLoader): dataloader for training
            val_dataloader (torch.utils.data.DataLoader): dataloader for validation
            test_dataloader (torch.utils.data.DataLoader): dataloader for test, can be None
            config (dict): JSON style configuration parameters
            config_server: deprecated, kept for API compatibility only.
        '''

        super().__init__()

        # Initialize all attributes from arguments
        self.client_idx_list = list(range(num_clients))
        self.config = config
        server_config = config['server_config']
        decoder_config = config.get('decoder_config', None)

        self.max_iteration = server_config['max_iteration']
        self.do_clustering = server_config.get('clustering', False)

        self.num_clients_per_iteration = [int(x) for x in server_config['num_clients_per_iteration'].split(',')] \
            if isinstance(server_config['num_clients_per_iteration'], str) \
            else [server_config['num_clients_per_iteration']]

        self.val_freq = server_config['val_freq']
        self.rec_freq = server_config['rec_freq']
        self.model_backup_freq = server_config.get('model_backup_freq', 100)
        self.worker_trainer_config = server_config.get('trainer_config', {})

        self.aggregate_median = server_config['aggregate_median']
        self.initial_lr_client = server_config.get('initial_lr_client', -1.0)
        self.lr_decay_factor = server_config.get('lr_decay_factor', 1.0)

        self.model_type = config['model_config']['model_type']
        self.quant_thresh = config['client_config'].get('quant_thresh', None)
        self.quant_bits = config['client_config'].get('quant_bits', 10)

        self.list_of_train_data = config['client_config']['data_config']['train']['list_of_train_data']
        self.data_path = data_path

        # Get max grad norm from data config
        if 'train' in server_config['data_config']:
            max_grad_norm = server_config['data_config']['train'].get('max_grad_norm', None)
        else:
            max_grad_norm = None

        # Creating an instance to update the model with stats aggregated from workers
        self.worker_trainer = ModelUpdater(
            model=model,
            optimizer=optimizer,
            ss_scheduler=ss_scheduler,
            train_dataloader=train_dataloader if train_dataloader is not None else val_dataloader,
            val_dataloader=val_dataloader,
            max_grad_norm=max_grad_norm,
            anneal_config=server_config['annealing_config'],
            model_type=self.model_type,
            decoder_config=decoder_config
        )

        self.val_dataloader = val_dataloader

        # Creating an instance for the server-side trainer (runs mini-batch SGD)
        self.server_replay_iterations = None
        self.server_trainer = None
        if train_dataloader is not None:
            assert 'server_replay_config' in server_config, 'server_replay_config is not set'
            assert 'optimizer_config' in server_config[
                'server_replay_config'], 'server-side replay training optimizer is not set'
            self.server_optimizer_config = server_config['server_replay_config']['optimizer_config']
            self.server_trainer_config = server_config['server_replay_config'].get('trainer_config', {})
            self.server_replay_iterations = server_config['server_replay_config']['server_iterations']
            self.server_trainer = Trainer(
                model=model,
                optimizer=None,
                ss_scheduler=ss_scheduler,
                train_dataloader=train_dataloader,
                server_replay_config=server_config['server_replay_config'],
                val_dataloader=None,
                max_grad_norm=server_config['server_replay_config']\
                                            .get('max_grad_norm',server_config['data_config']['train']\
                                                .get('max_grad_norm',None)),
                anneal_config=server_config['server_replay_config'].get('annealing_config', None)
            )

        self.skip_model_update = False  # will not update the model if True

        self.train_loss = 0.0
        self.model_path = model_path
        self.best_model_criterion = server_config['best_model_criterion']
        self.fall_back_to_best_model = server_config['fall_back_to_best_model']
        self.last_model_path = os.path.join(self.model_path, 'latest_model.tar')
        self.best_model_path = os.path.join(self.model_path,
            'best_val_{}_model.tar'.format(self.best_model_criterion))
        self.log_path = os.path.join(self.model_path, 'status_log.json')
        self.cur_iter_no = 0  # keep the iteration number for Tensor board plotting
        self.best_val_loss= float('inf')
        self.best_val_acc = -1.0
        self.best_test_loss= float('inf')
        self.best_test_acc= -1.0
        self.lr_weight = 1.0

        self.weight_sum_stale = 0.0
        self.client_parameters_stack_stale = []
        self.stale_prob = server_config.get('stale_prob', 0.0)

        self.losses = []
        self.no_label_updates = 0  # no. label updates

        # Update the parameters above if the log file
        if server_config.get('resume_from_checkpoint', False):
            self.load_saved_status()

        # Decoding config
        self.test_dataloader = test_dataloader
        self.decoder_config = decoder_config
        self.spm_model = server_config['data_config']['test'].get('spm_model', None)

        self.do_profiling = server_config.get('do_profiling', False)

        self.wantRL = server_config.get('wantRL', False)
        self.aggregate_fast = server_config.get('fast_aggregation', False)
        if self.aggregate_fast:
            print_rank('It is NOT possible to enable RL with fast_aggregation, RL is set to False', loglevel=logging.INFO)
            self.wantRL = False
            print_rank('It is NOT possible in Current Implementation to have stale gradients with fast_aggregation, stale_prob is set to 0.0', loglevel=logging.INFO)
            self.stale_prob = 0.0

        if self.wantRL:
            self.RL = RL(config=server_config)

        # Parallel processing
        self.clients_in_parallel = config['client_config'].get('clients_in_parallel', None)

    def load_saved_status(self):
        '''Load checkpoint from disk'''

        # Check if model is on disk, if so loads it onto trainer
        if os.path.exists(self.last_model_path):
            print_rank('Resuming from checkpoint model {}'.format(self.last_model_path))
            self.worker_trainer.load(self.last_model_path, update_lr_scheduler=True, update_ss_scheduler=True)
            if self.server_trainer is not None:
                self.server_trainer.model = self.worker_trainer.model  # make sure that the models are in sync

        # Check if log is on disk, if so loads it onto current stats
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r') as logfp:  # loading the iteration no., best loss and CER
                elems = json.load(logfp)
                self.cur_iter_no = elems.get('i', 0)
                self.best_val_loss = elems.get('best_val_loss', float('inf'))
                self.best_val_acc = elems.get('best_val_acc', float('inf'))
                self.best_test_loss = elems.get('best_test_loss', float('inf'))
                self.best_test_acc = elems.get('best_test_acc', float('inf'))
                self.lr_weight = elems.get('weight', 1.0)
                self.no_label_updates = elems.get('num_label_updates', 0)
                print_rank(f'Resuming from status_log: cur_iter: {self.cur_iter_no}')

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

    def run_distributed_evaluation(self, dataloader, mode):
        '''Perform evaluation using available workers.
        
        See also `process_test_validate` on federated.py.

        Args:
            dataloader (torch.utils.data.DataLoader): used to fetch data.
            mode (str): `test` or `val`.
        '''
        val_clients = list(self.make_eval_clients(dataloader))
        print_rank(f'mode: {mode} evaluation_clients {len(val_clients)}', loglevel=logging.DEBUG)

        usl_json = None  # NOTE: deprecated
        val_loss = val_acc = total = 0
        self.logits = {'predictions': [], 'probabilities': [], 'labels': []}
        server_data = (0.0, usl_json, [p.data.to(torch.device('cpu')) for p in self.worker_trainer.model.parameters()])

        for result in self.process_testvalidate(val_clients, server_data, mode):
            output, (loss, cer, count) = result
            val_loss += loss * count
            val_acc += cer * count
            total += count

            if output is not None:
                self.logits['predictions'].append(output['predictions'])
                self.logits['probabilities'].append(output['probabilities'])
                self.logits['labels'].append(output['labels'])

        if  self.logits['probabilities'] and self.logits['predictions'] and self.logits['labels']:
            self.logits['predictions'] = np.concatenate(self.logits['predictions'])
            self.logits['probabilities'] = np.concatenate(self.logits['probabilities'])
            self.logits['labels'] = np.concatenate(self.logits['labels'])

        return val_loss / total, val_acc / total

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

    def run(self):
        '''Trigger training.
        
        This is a simple wrapper to the `train` method.
        '''
        print_rank('server started')
        self.train()
        print_rank('server terminated')

    def train(self):
        '''Main method for training.'''

        self.run_stats = {
            'secsPerClientRound': [],
            'secsPerClient': [],
            'secsPerClientTraining': [],
            'secsPerClientSetup': [],
            'secsPerClientFull': [],
            'secsPerRoundHousekeeping': [],
            'secsPerRoundTotal': [],
            'mpiCosts': []
        }

        run.log('Max iterations', self.max_iteration)
        try:
            self.worker_trainer.model.cuda() if torch.cuda.is_available() else None

            # Do an initial validation round to understand the pretrained model's validation accuracy
            # Skip if we resumed from a checkpoint (cur_iter_no > 0)
            if self.cur_iter_no == 0:
                
                if self.config['server_config']['initial_rec']:
                    eval_list.append('test')
                if self.config['server_config']['initial_val']:
                    eval_list.append('val')

                print_rank("Running {} at itr={}".format(eval_list, self.cur_iter_no))
                self.update_metrics(self.evaluation.run(eval_list, self.update_val_test_req(), metric_logger=run.log))
                eval_list=[] # some cleanup

            # Dump all the information in aggregate_metric
            print_rank('Saving Model Before Starting Training', loglevel=logging.INFO)
            for token in ['best_val_loss', 'best_val_acc', 'best_test_acc', 'latest']:
                self.worker_trainer.save(
                    model_path=self.model_path,
                    token=token,
                    config=self.config['server_config']
                )

            # Training loop 
            self.worker_trainer.model.train()
            for i in range(self.cur_iter_no, self.max_iteration):
                begin = time.time()
                metrics_payload = {}

                def log_metric(k, v):
                    metrics_payload[k] = v

                print_rank('==== iteration {}'.format(i))
                log_metric('Current iteration', i)
                usl_json = None  # deprecated

                # Initial value for the learning rate of the worker
                initial_lr = self.initial_lr_client * self.lr_weight
                print_rank('Client learning rate {}'.format(initial_lr))

                # Run training on clients
                self.worker_trainer.model.zero_grad()
                self.train_loss = []
                server_data = (
                    initial_lr,
                    usl_json,
                    [p.data.to(torch.device('cpu')) for p in self.worker_trainer.model.parameters()]
                )

                # Random number of clients per iteration
                if len(self.num_clients_per_iteration) > 1:
                    num_clients_curr_iter = random.randint(
                        self.num_clients_per_iteration[0],
                        self.num_clients_per_iteration[1]
                    )
                else:
                    num_clients_curr_iter = self.num_clients_per_iteration[0]
                log_metric('Clients for round', num_clients_curr_iter)

                # Perform annealing in quantization threshold
                if self.quant_thresh is not None:
                    self.config['client_config']['quant_thresh'] *= self.config['client_config'].get('quant_anneal', 1.0)
                    self.quant_thresh = self.config['client_config']['quant_thresh']
                    log_metric('Quantization Thresh.', self.config['client_config']['quant_thresh'])

                #  Create the pool of clients -- sample from this pool to assign to workers
                sampled_idx_clients = random.sample(self.client_idx_list,
                    num_clients_curr_iter) if num_clients_curr_iter > 0 else self.client_idx_list
                sampled_clients = [
                    Client(
                        client_id,
                        self.config,
                        self.config['client_config']['type'] == 'optimization',
                        None
                    ) for client_id in sampled_idx_clients
                ]

                # Initialize stats
                clients_begin = time.time()

                client_losses = []
                client_weights = []
                client_mag_grads = []
                client_mean_grads = []
                client_var_grads = []
                client_norm_grads = []

                self.client_parameters_stack = []
                self.run_stats['secsPerClient'].append([])
                self.run_stats['secsPerClientFull'].append([])
                self.run_stats['secsPerClientTraining'].append([])
                self.run_stats['secsPerClientSetup'].append([])
                self.run_stats['mpiCosts'].append([])

                # Check if we want privacy metrics
                apply_privacy_metrics = self.config.get('privacy_metrics_config', None) and \
                    self.config['privacy_metrics_config']['apply_metrics']
                adaptive_leakage = apply_privacy_metrics and \
                    self.config['privacy_metrics_config'].get('adaptive_leakage_threshold', None)
                if apply_privacy_metrics:
                    privacy_metrics_stats = defaultdict(list)                    

                # Initialize profiler
                profiler = None
                if self.do_profiling:
                    profiler = cProfile.Profile()
                    profiler.enable()

                # Reset gradient for the model before assigning the new gradients
                self.worker_trainer.model.zero_grad()

                for client_output in self.process_clients(sampled_clients, server_data, self.clients_in_parallel):
                    # Process client output
                    client_timestamp = client_output['ts'] 
                    client_stats = client_output['cs']
                    client_loss = client_output['tl']
                    client_weight = client_output['wt']
                    client_mag_grad = client_output['mg'] 
                    client_var_grad = client_output['vg']
                    client_mean_grad = client_output['ng']
                    client_norm_grad = client_output['rg']
                    num_samples = client_output['ns']

                    # Client_output may contain 'gr' or 'pm' for grads or params.
                    # For the time being we just support gradients.
                    client_parameters = client_output['gr']
                                        
                    if apply_privacy_metrics:
                        privacy_stats = client_output['ps']
                        for metric, value in privacy_stats.items():
                            privacy_metrics_stats[metric].append(value)
                    
                    self.run_stats['mpiCosts'][-1].append(time.time() - client_timestamp)

                    # Ignore clients with agg. weight == 0.0
                    if client_weight == 0.0:
                        print_rank('Dropping client Due to issues with weighting', loglevel=logging.DEBUG)
                        num_clients_curr_iter -= 1
                        continue

                    # Get actual pseudo-gradients for aggregation
                    if self.aggregate_fast:
                        self.aggregate_gradients_inplace(client_parameters)
                    else:
                        self.client_parameters_stack.append(client_parameters)

                    # Aggregate stats
                    self.train_loss.append(client_loss)
                    client_losses.append(client_loss)
                    client_weights.append(client_weight)
                    client_mean_grads.append(client_mean_grad.item())
                    client_var_grads.append(client_var_grad.item())
                    client_norm_grads.append(client_norm_grad.item())

                    # Mark the end of client processing
                    client_end = time.time()

                    self.run_stats['secsPerClientFull'][-1].append(client_stats['full cost'])
                    self.run_stats['secsPerClientTraining'][-1].append(client_stats['training'])
                    self.run_stats['secsPerClientSetup'][-1].append(client_stats['setup'])
                    self.run_stats['secsPerClient'][-1].append(client_end - clients_begin)

                # Tear down profiler
                if self.do_profiling:
                    profiler.disable()
                    stats = pstats.Stats(profiler)
                    stats.sort_stats('cumulative').print_stats()

                # Prepare output
                client_weights = np.array(client_weights)
                client_mag_grads = np.array(client_mag_grads)
                client_mean_grads = np.array(client_mean_grads)
                client_var_grads = np.array(client_var_grads)
                client_norm_grads = np.array(client_norm_grads)
                
                dump_norm_stats = self.config.get('dump_norm_stats', False)
                if dump_norm_stats:
                    with open(os.path.join(self.model_path, 'norm_stats.txt'), 'a', encoding='utf-8') as outF:
                        outF.write('{}\n'.format(json.dumps(list(client_norm_grads))))

                # Print the privacy metrics
                if apply_privacy_metrics:
                    for metric, values in privacy_metrics_stats.items():
                        if metric == 'Dropped clients':
                            log_metric(metric, sum(values))
                        else:
                            log_metric(metric, max(values))

                if type(adaptive_leakage) is float:
                    values = privacy_metrics_stats['Practical epsilon (Max leakage)']
                    new_threshold = list(sorted(values))[int(adaptive_leakage*len(values))]
                    print_rank('Updating leakage threshold to {}'.format(new_threshold))
                    self.config['privacy_metrics_config']['max_allowed_leakage'] = new_threshold

                # Mark that all clients have been processed
                end = time.time()
                self.run_stats['secsPerClientRound'].append(end - begin)
                begin = end

                if self.wantRL:
                    rl_model = self.run_RL_inference(client_weights, client_mag_grads, client_mean_grads, client_var_grads)

                # Aggregation step
                if dump_norm_stats:
                    cps_copy = [[g.clone().detach() for g in x] for x in self.client_parameters_stack]
                weight_sum = self.aggregate_gradients(num_clients_curr_iter, client_weights, metric_logger=log_metric)
                print_rank('Sum of weights: {}'.format(weight_sum), loglevel=logging.DEBUG)

                torch.cuda.empty_cache()

                # Normalize with weight_sum
                for p in self.worker_trainer.model.parameters():
                    p.grad /= weight_sum

                if dump_norm_stats:
                    cosines = compute_grad_cosines(cps_copy, [p.grad.clone().detach() for p in self.worker_trainer.model.parameters()])
                    with open(os.path.join(self.model_path, 'cosines.txt'), 'a', encoding='utf-8') as outfile:
                        outfile.write('{}\n'.format(json.dumps(cosines)))

                # DP-specific steps
                privacy.apply_global_dp(self.config, self.worker_trainer.model, num_clients_curr_iter=num_clients_curr_iter, select_grad=True, metric_logger=log_metric)
                eps = privacy.update_privacy_accountant(self.config, len(self.client_idx_list), curr_iter=i, num_clients_curr_iter=num_clients_curr_iter)
                if eps:
                    print_rank(f'DP result: {eps}')

                # Log the training loss to tensorboard/AML
                log_metric('Training loss', sum(self.train_loss))

                if self.skip_model_update is True:
                    print_rank('Skipping model update')
                    continue

                # Run optimization with gradient/model aggregated from clients
                print_rank('Updating model')
                self.worker_trainer.update_model()
                print_rank('Updating learning rate scheduler')
                self.losses = self.worker_trainer.run_lr_scheduler(force_run_val=False)

                if self.wantRL:
                    self.run_RL_training(i, rl_model, client_weights, client_mag_grads, client_mean_grads, client_var_grads, log_metric)

                # Run a couple of iterations of training data on the server
                if self.server_trainer is not None:
                    print_rank('Running replay iterations on server')

                    if 'updatable_names' in self.server_trainer_config:
                        set_component_wise_lr(
                            self.worker_trainer.model,
                            self.server_optimizer_config,
                            self.server_trainer_config['updatable_names']
                        )
                    self.server_trainer.prepare_iteration(self.worker_trainer.model)
                    self.server_trainer.train_desired_samples(self.server_replay_iterations)
                    self.worker_trainer.model.load_state_dict(self.server_trainer.model.state_dict())
                    torch.cuda.empty_cache()

                # Update a sampling scheduler
                print_rank('Run ss scheduler')
                self.worker_trainer.run_ss_scheduler()

                # Run inference and score on val/test depending on the iter. number
                self.run_val_test(i + 1, metric_logger=log_metric)

                # Backup the current best models
                self.backup_models(i)

                # Fall back to the best model if the option is enabled
                self.fall_back_to_prev_best_status()

                # Logging the latest best values
                update_json_log(
                    self.log_path,
                    {
                        'i': i + 1,
                        'best_val_loss': float(self.best_val_loss),
                        'best_val_acc': float(self.best_val_acc),
                        'best_test_loss': float(self.best_test_loss),
                        'best_test_acc': float(self.best_test_acc),
                        'weight': float(self.lr_weight),
                        'num_label_updates': int(self.no_label_updates)
                    },
                )

                end = time.time()

                # Aggregate stats
                self.run_stats['secsPerRoundHousekeeping'].append(end - begin)
                self.run_stats['secsPerRoundTotal'].append(self.run_stats['secsPerClientRound'][-1] + \
                    self.run_stats['secsPerRoundHousekeeping'][-1])

                log_metric('secsPerRoundTotal', self.run_stats['secsPerRoundTotal'][-1])
                if self.do_profiling:
                    log_metric('secsPerClientRound', self.run_stats['secsPerClientRound'][-1])
                    log_metric('secsPerRoundHousekeeping', self.run_stats['secsPerRoundHousekeeping'][-1])

                    metrics_for_stats = [
                        'secsPerClient',
                        'secsPerClientTraining',
                        'secsPerClientFull',
                        'secsPerClientSetup',
                        'mpiCosts',
                    ]

                    for metric in metrics_for_stats:
                        log_metric(f'{metric}Mean', np.mean(self.run_stats[metric][-1]))
                        log_metric(f'{metric}Median', np.median(self.run_stats[metric][-1]))
                        log_metric(f'{metric}Max', max(self.run_stats[metric][-1]))

                    for k in self.run_stats:
                        if k in metrics_for_stats:
                            print_rank('{}: {}'.format(k, max(self.run_stats[k][-1])), loglevel=logging.DEBUG)
                        else:
                            print_rank('{}: {}'.format(k, self.run_stats[k][-1]), loglevel=logging.DEBUG)

                # Log all the metrics
                for k in metrics_payload:
                    run.log(k, metrics_payload[k])

        finally:  # perform cleanup even if error was raised above
            self.terminate_workers(terminate=(not self.do_clustering))

    def backup_models(self, i):
        '''Save the current best models.
        
        Save CER model, the best loss model and the best WER model. This occurs
        at a specified period.

        Args:
            i: no. of iterations.
        '''

        # Always save the latest model
        self.worker_trainer.save(
            model_path=self.model_path,
            token='latest',
            config=self.config['server_config'],
        )

        if (i % self.model_backup_freq) == 0:  # save the current best models
            self.worker_trainer.save(
                model_path=self.model_path,
                token='epoch{}'.format(i),
                config=self.config['server_config']
            )

            for bodyname in ['best_val_acc', 'best_val_loss', 'best_test_acc']:
                src_model_path = os.path.join(self.model_path, '{}_model.tar'.format(bodyname))
                if os.path.exists(src_model_path):
                    dst_model_path = os.path.join(self.model_path, 'epoch{}_{}_model.tar'.format(i, bodyname))
                    shutil.copyfile(src_model_path, dst_model_path)
                    print_rank('Saved {}'.format(dst_model_path))

    def fall_back_to_prev_best_status(self):
        '''Go back to the past best status and switch to the recent best model.'''

        if self.fall_back_to_best_model:
            print_rank('falling back to model {}'.format(self.best_model_path))

            # Save current learning rate
            tmp_lr = get_lr(self.worker_trainer.optimizer)

            # Load previous best model
            self.worker_trainer.load(self.best_model_path, update_lr_scheduler=False, update_ss_scheduler=False)

            # Update previous learning rate on optimizer
            for g in self.worker_trainer.optimizer.param_groups:
                g['lr'] = tmp_lr

            if self.server_trainer is not None:
                self.server_trainer.model = self.worker_trainer.model  # make sure that the models are in sync

    def run_RL_inference(self, client_weights, client_mag_grads, client_mean_grads, client_var_grads):
        '''Uses RL to estimate weights, using DGA.

        Args:
            client_weights (numpy.ndarray): original weights for aggregation.
            client_mag_grads (numpy.ndarray): gradient stats for RL (magnitudes).
            client_mean_grads (numpy.ndarray): gradient stats for RL (means).
            client_var_grads (numpy.ndarray): gradient stats for RL (vars).

        Returns:
            list of torch.Tensor: parameters of model used to perform RL.
        '''

        weight_sum = 0
        original_model = copy.copy([p for p in self.worker_trainer.model.parameters()])

        # Reinforcement learning for estimating weights
        print_rank('RL estimation of the aggregation weights', loglevel=logging.INFO)
        rl_weights = self.RL.forward(
            np.concatenate((client_weights, client_mag_grads, client_mean_grads, client_var_grads), axis=0)).cpu().detach().np()
        if rl_weights.ndim > 1:
            rl_weights = rl_weights[-1, :]
        rl_weights = np.exp(rl_weights)

        print_rank('RL Weights BEFORE filtering: {}'.format(rl_weights), loglevel=logging.DEBUG)
        index = np.argwhere(np.isnan(rl_weights))
        rl_weights[index] = 0
        index = np.argwhere(np.isinf(rl_weights))
        rl_weights[index] = 0
        print_rank('RL Weights AFTER filtering: {}'.format(rl_weights), loglevel=logging.DEBUG)

        for client_parameters, orig_weight, rl_weight in zip(self.client_parameters_stack, client_weights, rl_weights):
            # Model parameters are already multiplied with weight on client, we only have to sum them up
            for p, client_grad in zip(self.worker_trainer.model.parameters(), client_parameters):
                if p.grad is None:
                    p.grad = _to_cuda(client_grad) * rl_weight / orig_weight
                else:
                    p.grad += _to_cuda(client_grad) * rl_weight / orig_weight
            weight_sum += rl_weight

        # Normalize with weight_sum
        for p in self.worker_trainer.model.parameters():
            p.grad /= weight_sum
        
        # Run optimization with gradient/model aggregated from clients
        self.worker_trainer.update_model()

        # Get the validation result back
        (rl_val_loss, rl_val_acc) = self.worker_trainer.run_lr_scheduler(force_run_val=True)

        # Save model and revert to previous one
        rl_model = copy.copy([p.data for p in self.worker_trainer.model.parameters()])
        for p, p_ in zip(self.worker_trainer.model.parameters(), original_model):
            p.data = p_.data.detach().clone()

        # Set the current set of weights
        self.RL.set_weights(rl_weights)
        self.RL.set_losses((rl_val_loss, rl_val_acc))

        # Return the resulting RL-based model
        return rl_model

    def run_RL_training(self, iter, rl_model, client_weights, client_mag_grads, client_mean_grads, client_var_grads, metric_logger):
        '''Trains RL for estimating weights, following DGA recipe.
        
        Args:
            iter (int): current iteration.
            rl_model (list of torch.Tensor): parameters of model used to perform RL.
            client_weights (numpy.ndarray): original weights for aggregation.
            client_mag_grads (numpy.ndarray): gradient stats for RL (magnitudes).
            client_mean_grads (numpy.ndarray): gradient stats for RL (means).
            client_var_grads (numpy.ndarray): gradient stats for RL (vars).
            metric_logger (callback, optional): callback used for logging.
                Defaults to None, in which case AML logger is used.
        '''

        # Get the validation result back
        if None in self.losses:
            self.losses = self.run_distributed_inference(mode='val')

        # Expected structure of batch
        print_rank('Performing RL training on the aggregation weights')
        if abs(self.losses[1] - self.RL.rl_losses[1]) < 0.001:
            reward = 0.1
            print_rank(
                'Iter:{}  val_ACC={}  rl_val_ACC={}  reward={}'.format(iter, self.losses[1], self.RL.rl_losses[1], reward))
            if 'marginal_update_RL' in self.config['server_config'] and \
                    self.config['server_config']['marginal_update_RL']:
                self.losses = self.RL.rl_losses
                for p, p_ in zip(self.worker_trainer.model.parameters(), rl_model):
                    p.data= p_.data.detach().clone()

        elif (self.losses[1] - self.RL.rl_losses[1]) > 0:
            reward = 1.0
            print_rank(
                'Iter:{}  val_ACC={}  rl_val_ACC={}  reward={}'.format(iter, self.losses[1], self.RL.rl_losses[1], reward))
            self.losses = self.RL.rl_losses
            for p, p_ in zip(self.worker_trainer.model.parameters(), rl_model):
                p.data = p_.data.detach().clone()

        else:
            reward = -1.0
            print_rank(
                'Iter:{}  val_ACC={}  rl_val_ACC={}  reward={}'.format(iter, self.losses[1], self.RL.rl_losses[1], reward))

        # Taking the policy from a game-based RL
        batch = (
            (np.concatenate((client_weights, client_mag_grads, client_mean_grads, client_var_grads), axis=0)),
            (self.RL.rl_weights),
            [reward]
        )

        print_rank('RL Model Update -- Training')
        self.RL.train(batch)

        print_rank('RL State Saving')
        self.RL.save(iter)

        print_rank('RL logging')
        metric_logger('RL Running Loss', self.RL.runningLoss)
        metric_logger('RL Rewards', reward)

    def run_val_test(self, i, metric_logger=None):
        '''Run validation or test, depending on current iteration i.
        
        Args:
            i (int): current iteration.
            metric_logger (callback, optional): callback used for logging.
                Defaults to None, in which case AML logger is used.
        '''

        if metric_logger is None:
            metric_logger = run.log

        # Run validation and update the LR scheduler
        if (i % self.val_freq) == 0:  # print loss info to Tensorboard on Philly
            if 'wantRL' not in self.config['server_config'] or not self.config['server_config']['wantRL']:
                print_rank('Running validation at itr={}'.format(i))
                self.losses = self.run_distributed_inference(mode='val')

            # Log changes
            metric_logger('LR for agg. opt.', get_lr(self.worker_trainer.optimizer))
            metric_logger('Val Loss', self.losses[0])
            metric_logger('Val Acc', self.losses[1])

            print_rank('LOG: val_loss={}: best_val_loss={}'.format(self.losses[0], self.best_val_loss))
            print_rank('LOG: val_acc={}: best_val_acc={}'.format(self.losses[1], self.best_val_acc))

            if self.losses[0] < self.best_val_loss:  # save the model when loss is improved
                self.worker_trainer.save(
                    model_path=self.model_path,
                    token='best_val_loss',
                    config=self.config['server_config']
                )
                self.best_val_loss = self.losses[0]
            else:
                # Create a schedule for the initial_lr (for the worker)
                self.lr_weight *= self.lr_decay_factor
                print_rank('LOG: Client weight of learning rate {}..'.format(self.lr_weight))

            if self.losses[1] > self.best_val_acc:  # save the model when CER is improved
                self.worker_trainer.save(
                    model_path=self.model_path,
                    token='best_val_acc',
                    config=self.config['server_config']
                )
                self.best_val_acc = self.losses[1]

        # Run full testing
        if (i % self.rec_freq) == 0 and self.test_dataloader is not None:
            print_rank('Running Testing at itr={}'.format(i))

            aggregated_metrics = self.run_distributed_inference(mode='test')

            metric_logger('Test Loss', aggregated_metrics[0])
            metric_logger('Test Acc', aggregated_metrics[1])
            print_rank('LOG: test_loss={}: best_test_loss={}'.format(aggregated_metrics[0], self.best_test_loss))
            print_rank('LOG: test_acc={}: best_test_acc={}'.format(aggregated_metrics[1], self.best_test_acc))

            if aggregated_metrics[0] < self.best_test_loss:
                self.best_test_loss=aggregated_metrics[0]

            if aggregated_metrics[1] > self.best_test_acc:
                self.best_test_acc = aggregated_metrics[1]
                self.worker_trainer.save(
                    model_path=self.model_path,
                    token='best_test_acc',
                    config=self.config['server_config'],
                )

    def aggregate_gradients_inplace(self, client_parameters):
        '''Aggregate list of tensors into model gradients.
        
        Args:
            client_parameters (list): list of tensors to aggregate.
        '''
        for p, client_grad in zip(self.worker_trainer.model.parameters(), client_parameters):
            if p.grad is None:
                p.grad = _to_cuda(client_grad)
            else:
                p.grad += _to_cuda(client_grad)

    def aggregate_gradients(self, num_clients_curr_iter, client_weights, metric_logger=None):
        '''Go through stored gradients, aggregate and put them inside model.

        Args:
            num_clients_curr_iter (int): how many clients were processed.
            client_weights: weight for each client.
            metric_logger (callback, optional): callback used for logging.
                Defaults to None, in which case AML logger is used.

        Returns:
            float: sum of weights for all clients.
        '''

        weight_sum = 0
        if metric_logger is None:
            metric_logger = run.log

        if not self.aggregate_fast:
            metric_logger('Stale Gradients Ratio', len(self.client_parameters_stack_stale) / num_clients_curr_iter)
            if len(self.client_parameters_stack_stale) > 0:
                weight_sum = self.weight_sum_stale
                for client_parameters in self.client_parameters_stack_stale:
                    # Model parameters are already multiplied with weight on client, we only have to sum them up
                    self.aggregate_gradients_inplace(client_parameters)
                self.client_parameters_stack_stale = []
                self.weight_sum_stale = 0

            for client_weight, client_parameters in zip(client_weights, self.client_parameters_stack):
                if np.random.random() > self.stale_prob:
                    # Model parameters are already multiplied with weight on client, we only have to sum them up
                    self.aggregate_gradients_inplace(client_parameters)
                else:
                    self.weight_sum_stale += client_weight
                    self.client_parameters_stack_stale.append(client_parameters)

            # Some cleaning
            self.client_parameters_stack = []

        weight_sum += sum(client_weights) - self.weight_sum_stale
        return weight_sum


def select_server(server_type, config):
    '''Select a server type using different possible strings.

    Right now this just returns `OptimizationServer`, but this
    function could be useful when there are multiple choices of
    server.
    
    Args:
        server_type (str): indicates server choice.
        config (dict): config parsed from YAML, passed so that
            parameters can be used to select a given server.
    '''
    return OptimizationServer