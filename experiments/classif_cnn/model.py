# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score

from core.model import BaseModel

class Net(nn.Module):
    '''The standard PyTorch model we want to federate'''

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(BaseModel):
    '''This is a PyTorch model with some extra methods'''

    def __init__(self, model_config):
        super().__init__()
        self.net = Net()

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        '''Performs forward step and computes the loss'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)
        return F.cross_entropy(output, labels.long())

    def inference(self, input):
        '''Performs forward step and computes metrics'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)

        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output, dim=1) == labels).float()).item()
        f1 = f1_score(labels.cpu(), torch.argmax(output, dim=1).cpu(), average='micro')

        # NOTE: Only the keys 'output','acc' and 'batch_size' does not require 
        # extra fields as 'value' and 'higher is better'. FLUTE requires this 
        # format only for customized metrics.

        return {'output':output, 'acc': accuracy, 'batch_size': n_samples, \
                'f1_score': {'value':f1,'higher_is_better': True}} 


        