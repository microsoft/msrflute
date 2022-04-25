# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from utils import ( make_lr_scheduler,
                    print_rank,
                    torch_save,
                    try_except_save,
                    make_optimizer,
                    to_device)

class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = x.contiguous()
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True,dropout=0.0,multi=1):
        super(BatchRNN, self).__init__()
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.batch_norm_activate = batch_norm
        self.bidirectional  = bidirectional
        self.multi          = multi
        self.dropout        = dropout

        if self.batch_norm_activate:
            self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size))
        self.rnn = rnn_type(input_size   = input_size,
                            hidden_size  = hidden_size,
                            bidirectional= bidirectional,
                            bias         = True,
                            batch_first  = True,
                            dropout      = self.dropout)
        self.num_directions = 2 if bidirectional else 1


    def forward(self, x):
        if x.dim()==2:
            x=x.unsqueeze(1)

        if self.batch_norm_activate:
            x = x.contiguous()
            x = self.batch_norm(x)
        x, _ = self.rnn(x)

        if self.bidirectional and self.multi<2:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        return x


class NeuralNetwork(nn.Module):
    def __init__(self, params, wantLSTM=False, batch_norm=False):
        super(NeuralNetwork, self).__init__()

        """
        The following parameters need revisiting
        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        optimizer = optim.Adam(model.parameters(), lr=1e-6)
        criterion = nn.MSELoss()

        """
        self.wantLSTM  = wantLSTM
        self.batch_norm= batch_norm
        params = [int(x) for x in params.split(',')]
        layers = []

        self.softmax = nn.Softmax(dim = 1)
        if self.wantLSTM:
            # Recurrent Component of the architecture
            rnns = []
            for i in range(1, len(params) - 2):
                multi = 1 if i==1 else 1
                rnn = BatchRNN(input_size    = params[i-1]*multi,
                                hidden_size  = params[i],
                                rnn_type     = nn.LSTM,
                                bidirectional= True,
                                batch_norm   = batch_norm,
                                multi        = 1,
                                dropout      = 0.0)
                rnns.append(('%d' %(i-1), rnn))
            self.rnn = nn.Sequential(OrderedDict(rnns))

            layers.append(nn.Linear(params[-3], params[-2], bias=True))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(params[-2], params[-1], bias=True))
            mlp = nn.Sequential(*layers)
            self.mlp = nn.Sequential(SequenceWise(mlp),)

        else:
            if self.batch_norm:
                self.batch_norm = nn.BatchNorm1d(params[0])

            for i in range(1, len(params)-1):
                layers.append(nn.Linear(params[i-1], params[i], bias=True))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(params[-2], params[-1], bias=True))
            self.mlp = nn.Sequential(*layers) 


    def forward(self, x):
        if self.wantLSTM:
            x = self.rnn(x)

        if self.batch_norm:
            x = self.batch_norm(x)
        out = self.mlp(x)
        out = out.squeeze()

        return out




class RL:
    def __init__(self, config=None):

        # Finalized config-file
        self.config= config

        self.out_size = config["num_clients_per_iteration"]
        self.wantLSTM = config['RL']['wantLSTM'] if 'wantLSTM' in config['RL'] else False
        self.replay_memory= []
        self.state_memory = []
        self.epsilon= config['RL']['initial_epsilon']
        self.step =0 
        self.runningLoss =0

        model_descriptor = config['RL']['model_descriptor_RL'] if 'model_descriptor_RL'  in config['RL'] else 'Default'
        self.model_name = os.path.join(config['RL']['RL_path'], 'rl_{}.{}.model'.format(self.out_size, model_descriptor))
        self.stats_name = os.path.join(config['RL']['RL_path'], 'rl_{}.{}.stats'.format(self.out_size, model_descriptor))

        # Initialize RL model
        self.make_model()
        self.load_saved_status()

        # Set the RL weights
        self.rl_weights=None
        self.rl_losses=None

        self.criterion = nn.MSELoss()

    def set_losses(self, losses):
        self.rl_losses=losses

    def set_weights(self, weights):
        self.rl_weights = weights

    def forward(self, state=None):
        # epsilon greedy exploration

        if self.wantLSTM:
            N = len(state)
            state.resize(1, N)
            if len(self.state_memory)==0:
                self.state_memory = np.zeros((self.config['RL']['minibatch_size'], N))
            self.state_memory = np.concatenate((self.state_memory[1:], state), axis=0)
            state = self.state_memory

        if random.random() <= self.epsilon:
            print_rank("Performed random action!")
            action= to_device(torch.rand(self.out_size))
        else:
            state = to_device(torch.from_numpy(state))
            print_rank(f'RL_state: {state.shape}')
            action= self.model(state.float())
        return action



    def train(self, batch=None):
        # save transition to replay memory
        self.replay_memory.append(batch)

        # if replay memory is full, remove the oldest transition
        if len(self.replay_memory) > self.config['RL']['max_replay_memory_size']:
            self.replay_memory.pop(0)

        # epsilon annealing
        self.epsilon *= self.config['RL']['epsilon_gamma'] if self.epsilon*self.config['RL']['epsilon_gamma']>self.config['RL']['final_epsilon'] else 1.0

        # sample random minibatch
        if self.wantLSTM:
            if len(self.replay_memory)>= self.config['RL']['minibatch_size']:
                minibatch = self.replay_memory[-self.config['RL']['minibatch_size']:]
            else:
                minibatch = self.replay_memory 
        else:
            minibatch = random.sample(self.replay_memory, min(len(self.replay_memory), self.config['RL']['minibatch_size']))

        # unpack minibatch
        state_batch  = torch.tensor(tuple(d[0] for d in minibatch)).float()
        action_batch = torch.tensor(tuple(d[1] for d in minibatch)).float()
        reward_batch = torch.tensor(tuple(d[2] for d in minibatch)).float()

        state_batch = to_device(state_batch)
        action_batch = to_device(action_batch)
        reward_batch = to_device(reward_batch)


        # set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
        y_batch = reward_batch

        # extract Q-value
        print_rank(f'RL state_batch: {state_batch.shape}', loglevel=logging.DEBUG)
        state_output = self.model(state_batch)
        print_rank(f'RL train shapes: {state_batch.shape} {action_batch.shape} {state_output.shape}', loglevel=logging.DEBUG)
        q_value = torch.sum(state_output * action_batch, dim=1)

        # reset gradient
        self.optimizer.zero_grad()

        # returns a new Tensor, detached from the current graph, the result will never require gradient
        y_batch = y_batch.detach()

        # calculate loss
        loss = self.criterion(q_value, y_batch)

        # do backward pass
        loss.backward()
        self.optimizer.step()

        # Tracking a running average of loss
        if self.runningLoss==0:
            self.runningLoss = loss.item()
        else:
            self.runningLoss = 0.95 * self.runningLoss + 0.05 * loss.item()
        print_rank('Running Loss for RL training process: {}'.format(self.runningLoss))

        # Decay learning rate
        self.lr_scheduler.step()


    def make_model(self):
        # make model
        self.model = NeuralNetwork(self.config['RL']['network_params'], \
                        self.config['RL']['wantLSTM'] if 'wantLSTM' in self.config['RL'] else False, \
                        self.config['RL']['batchNorm'] if 'batchNorm' in self.config['RL'] else False)
        print(self.model)
        model = to_device(model)

        # make optimizer
        self.optimizer = make_optimizer(self.config['RL']["optimizer_config"], self.model)

        # make lr_scheduler
        self.lr_scheduler = make_lr_scheduler(
                                            self.config['RL']['annealing_config'],
                                            self.optimizer,
                                            num_batches=1)


    def load_saved_status(self):
        if os.path.exists(self.model_name):
            print_rank("Resuming from checkpoint model {}".format(self.model_name))
            self.load()

        if os.path.exists(self.stats_name):
            with open(self.stats_name, 'r') as logfp: # loading the iteration no., val_loss and lr_weight
                elems = json.load(logfp)
                self.cur_iter_no= elems["i"]
                self.val_loss   = elems["val_loss"]
                self.val_cer    = elems["val_cer"]
                self.runningLoss= elems["weight"]



    def load(self):
        print_rank("Loading checkpoint: {}".format(self.model_name))
        checkpoint = torch.load(self.model_name)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        anl_st_dict = checkpoint.get('lr_scheduler_state_dict')
        if anl_st_dict and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(anl_st_dict)


    def save(self, i):
        """
        Save a model as well as training information
        """

        save_state = {
                'model_state_dict' : self.model.state_dict(),
                'optimizer_state_dict' : self.optimizer.state_dict() if self.optimizer is not None else None,
                'lr_scheduler_state_dict' : self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None
            }

        outputdir = os.path.dirname(self.model_name)
        if os.path.exists(outputdir) is False:
            os.makedirs(outputdir, exist_ok=True)

        print_rank("Saving model to: {}".format(self.model_name))
        try_except_save(torch_save, state_or_model=save_state,
                                        save_path=self.model_name)

        # logging the latest best values
        print_rank(f'Saving stats to {self.stats_name}')
        with open(self.stats_name, 'w') as logfp:
            json.dump({"i":i+1,
                        "val_loss":float(self.rl_losses[0]),
                        "val_cer":float(self.rl_losses[1]),
                        "weight":float(self.runningLoss)},
                        logfp)



