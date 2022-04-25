# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import logging
import math
import os

import numpy as np
import torch

from extensions import privacy, RL, quant_model
from utils import compute_grad_cosines, print_rank, to_device
from core.strategies import BaseStrategy
from core.strategies.utils import (
    aggregate_gradients_inplace,
    filter_weight,
)

from azureml.core import Run
run = Run.get_context()

MIN_WEIGHT = 1e-7


class DGA(BaseStrategy):
    '''Dynamic Gradient Aggregation'''

    def __init__(self, mode, config, model_path=None):
        ''' Dynamic Gradient Aggregation (DGA) strategy.

        For more info see arXiv:2106.07578.

        Args:
            mode (str): which part the instantiated object should play,
                typically either :code:`client` or :code:`server`.
            config (dict): initial config dict.
            model_path (str): where to find model, needed for debugging only.
        '''

        super().__init__(mode=mode, config=config, model_path=model_path)

        if mode not in ['client', 'server']:
            raise ValueError('mode in strategy must be either `client` or `server`')

        self.config = config
        self.model_path = model_path
        self.mode = mode

        # Parse config
        self.model_config = config['model_config']
        self.client_config = config['client_config']
        self.server_config = config['server_config']

        self.dp_config = config.get('dp_config', None)

        if mode == 'client':
            self.stats_on_smooth_grad = self.client_config.get('stats_on_smooth_grad', False)
            self.quant_threshold = self.client_config.get('quant_thresh', None)
            self.quant_bits = self.client_config.get('quant_bits', 10)
        elif mode == 'server':
            self.dump_norm_stats = self.config.get('dump_norm_stats', False)
            self.aggregate_fast = self.server_config.get('fast_aggregation', False)
            self.want_rl = self.server_config.get('wantRL', False)
            self.stale_prob = self.server_config.get('stale_prob', 0.0)

            self.skip_model_update = False

            # Do some checks and create objects based on configs
            if self.aggregate_fast:
                print_rank('It is NOT possible to enable RL with fast_aggregation, RL is set to False', loglevel=logging.INFO)
                self.want_rl = False

                print_rank('It is NOT possible in Current Implementation to have stale gradients with fast_aggregation, stale_prob is set to 0.0', loglevel=logging.INFO)
                self.stale_prob = 0.0

            if self.want_rl:
                self.rl = RL(config=self.server_config)

            # Initialize accumulators
            self.client_parameters_stack = []
            self.client_parameters_stack_stale = []
            self.client_weights = []

            self.weight_sum_stale = 0.0

    def generate_client_payload(self, trainer):
        '''Generate client payload

        Args:
            trainer (core.Trainer object): trainer on client.

        Returns:
            dict containing payloads in some specified format.
        '''

        if self.mode != 'client':
            raise RuntimeError('this method can only be invoked by the client')

        # Get weights for aggregation, potentially using DGA
        weight = 1.0
        add_weight_noise = False

        # Reset gradient stats and recalculate them on the smooth/pseudo gradient
        if self.stats_on_smooth_grad:
            trainer.reset_gradient_power()
            trainer.estimate_sufficient_stats()

        # If we are using softmax based on training loss, it needs DP noise
        if self.config['server_config']['aggregate_median'] == 'softmax':
            # This matters when DP is required
            add_weight_noise = True

            if 'weight_train_loss' not in self.config['server_config'] or self.config['server_config']['weight_train_loss'] == 'train_loss':
                training_weight = trainer.train_loss / trainer.num_samples
            elif self.config['server_config']['weight_train_loss'] == 'mag_var_loss':
                training_weight = trainer.sufficient_stats['var']
            elif self.config['server_config']['weight_train_loss'] == 'mag_mean_loss':
                training_weight = trainer.sufficient_stats['mean']
            else:
                training_weight = trainer.sufficient_stats['mag']

            try:
                weight = math.exp(-self.config['server_config']['softmax_beta'] * training_weight)
            except:
                print_rank('There is an issue with the weight -- Reverting to {}'.format(MIN_WEIGHT), loglevel=logging.DEBUG)
                weight = MIN_WEIGHT
            weight = filter_weight(weight)

        # Add local DP noise here.
        # When weight == 0, something went wrong. So we'll skip adding noise and return a zero gradient.
        if weight > 0.0 and self.dp_config is not None and self.dp_config.get('enable_local_dp', False):
            weight = privacy.apply_local_dp(trainer, weight, self.dp_config, add_weight_noise)

        # In all other cases we can compute the weight after adding noise
        if not add_weight_noise:
            assert self.config['server_config']['aggregate_median'] == 'mean'
            assert weight == 1.0

        # Weight the gradient and remove gradients of the layers we want to freeze
        for n, p in trainer.model.named_parameters():
            p.grad = weight * p.grad
            if self.model_config.get('freeze_layer', None) and n == self.model_config['freeze_layer']:
                print_rank('Setting gradient to zero for layer: {}'.format(n), loglevel=logging.INFO)
                p.grad.mul_(0)

        # Gradient quantization step -- if quant_threshold is None, the code returns without doing anything
        quant_model(trainer.model, quant_threshold=self.quant_threshold, quant_bits=self.quant_bits, global_stats=False)

        payload = {}
        payload['weight'] = weight
        payload['gradients'] = [p.grad.to(torch.device('cpu')) for p in trainer.model.parameters()]

        return payload

    def process_individual_payload(self, worker_trainer, payload):
        '''Process client payload

        Args:
            worker_trainer (core.Trainer object): trainer on server
                (aka model updater).
            payload (dict): whatever is generated by
                :code:`generate_client_payload`.

        Returns:
            True if processed succesfully, False otherwise.
        '''

        if self.mode != 'server':
            raise RuntimeError('this method can only be invoked by the server')

        if payload['weight'] == 0.0:
            return False

        self.client_weights.append(payload['weight'])
        if self.aggregate_fast:
            aggregate_gradients_inplace(worker_trainer.model, payload['gradients'])
        else:
            self.client_parameters_stack.append(payload['gradients'])
        return True

    def combine_payloads(self, worker_trainer, curr_iter, num_clients_curr_iter, client_stats, logger=None):
        '''Combine payloads to update model

        Args:
            worker_trainer (core.Trainer object): trainer on server
                (aka model updater).
            curr_iter (int): current iteration.
            num_clients_curr_iter (int): number of clients on current iteration.
            client_stats (dict): stats being collected.
            logger (callback): function called to log quantities.

        Returns:
            losses, computed for use with LR scheduler.
        '''

        if self.mode != 'server':
            raise RuntimeError('this method can only be invoked by the server')

        if self.want_rl:
            rl_model = self._run_rl_inference(self.client_weights, *client_stats)

        # Aggregation step
        if self.dump_norm_stats:
            cps_copy = [[g.clone().detach() for g in x] for x in self.client_parameters_stack]
        weight_sum = self._aggregate_gradients(worker_trainer, num_clients_curr_iter, self.client_weights, metric_logger=logger)
        print_rank('Sum of weights: {}'.format(weight_sum), loglevel=logging.DEBUG)

        torch.cuda.empty_cache()

        # Normalize with weight_sum
        for p in worker_trainer.model.parameters():
            p.grad /= weight_sum

        if self.dump_norm_stats:
            cosines = compute_grad_cosines(cps_copy, [p.grad.clone().detach() for p in worker_trainer.model.parameters()])
            with open(os.path.join(self.model_path, 'cosines.txt'), 'a', encoding='utf-8') as outfile:
                outfile.write('{}\n'.format(json.dumps(cosines)))

        # DP-specific steps
        privacy.apply_global_dp(self.config, worker_trainer.model, num_clients_curr_iter=num_clients_curr_iter, select_grad=True, metric_logger=logger)
        eps = privacy.update_privacy_accountant(self.config, num_clients_curr_iter, curr_iter=curr_iter, num_clients_curr_iter=num_clients_curr_iter)
        if eps:
            print_rank(f'DP result: {eps}')

        if self.skip_model_update is True:
            print_rank('Skipping model update')
            return

        # Run optimization with gradient/model aggregated from clients
        print_rank('Updating model')
        worker_trainer.update_model()
        print_rank('Updating learning rate scheduler')
        losses = worker_trainer.run_lr_scheduler(force_run_val=False)

        if self.want_rl:
            self._run_rl_training(curr_iter, rl_model, self.client_weights, *client_stats, logger)

        return losses

    def _aggregate_gradients(self, worker_trainer, num_clients_curr_iter, client_weights, metric_logger=None):
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
                    aggregate_gradients_inplace(worker_trainer.model, client_parameters)
                self.client_parameters_stack_stale = []
                self.weight_sum_stale = 0

            for client_weight, client_parameters in zip(client_weights, self.client_parameters_stack):
                if np.random.random() > self.stale_prob:
                    # Model parameters are already multiplied with weight on client, we only have to sum them up
                    aggregate_gradients_inplace(worker_trainer.model, client_parameters)
                else:
                    self.weight_sum_stale += client_weight
                    self.client_parameters_stack_stale.append(client_parameters)

        weight_sum += sum(client_weights) - self.weight_sum_stale

        # Some cleaning
        self.client_parameters_stack = []
        self.client_weights = []

        return weight_sum

    def _run_rl_inference(self, client_weights, client_mag_grads, client_mean_grads, client_var_grads):
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
        rl_weights = self.rl.forward(
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
                    p.grad = to_device(client_grad) * rl_weight / orig_weight
                else:
                    p.grad += to_device(client_grad) * rl_weight / orig_weight
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
        self.rl.set_weights(rl_weights)
        self.rl.set_losses((rl_val_loss, rl_val_acc))

        # Return the resulting RL-based model
        return rl_model

    def _run_rl_training(self, iter, rl_model, client_weights, client_mag_grads, client_mean_grads, client_var_grads, metric_logger):
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
        if abs(self.losses[1] - self.rl.rl_losses[1]) < 0.001:
            reward = 0.1
            print_rank(
                'Iter:{}  val_ACC={}  rl_val_ACC={}  reward={}'.format(iter, self.losses[1], self.rl.rl_losses[1], reward))
            if 'marginal_update_RL' in self.config['server_config'] and \
                    self.config['server_config']['marginal_update_RL']:
                self.losses = self.rl.rl_losses
                for p, p_ in zip(self.worker_trainer.model.parameters(), rl_model):
                    p.data= p_.data.detach().clone()

        elif (self.losses[1] - self.rl.rl_losses[1]) > 0:
            reward = 1.0
            print_rank(
                'Iter:{}  val_ACC={}  rl_val_ACC={}  reward={}'.format(iter, self.losses[1], self.rl.rl_losses[1], reward))
            self.losses = self.rl.rl_losses
            for p, p_ in zip(self.worker_trainer.model.parameters(), rl_model):
                p.data = p_.data.detach().clone()

        else:
            reward = -1.0
            print_rank(
                'Iter:{}  val_ACC={}  rl_val_ACC={}  reward={}'.format(iter, self.losses[1], self.rl.rl_losses[1], reward))

        # Taking the policy from a game-based RL
        batch = (
            (np.concatenate((client_weights, client_mag_grads, client_mean_grads, client_var_grads), axis=0)),
            (self.rl.rl_weights),
            [reward]
        )

        print_rank('RL Model Update -- Training')
        self.rl.train(batch)

        print_rank('RL State Saving')
        self.rl.save(iter)

        print_rank('RL logging')
        metric_logger('RL Running Loss', self.rl.runningLoss)
        metric_logger('RL Rewards', reward)