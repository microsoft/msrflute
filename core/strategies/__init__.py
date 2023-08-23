# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from .base import BaseStrategy
from .fedavg import FedAvg
from .dga import DGA
from .fedlabels import FedLabels

def select_strategy(strategy):
    ''' Selects the aggregation strategy class
    
    NOTE: FedProx uses FedAvg weights during aggregation, 
    which are proportional to the number of samples in 
    each client.
    '''
    if strategy.lower() == 'dga':
        return DGA
    elif strategy.lower() in ['fedavg', 'fedprox']:
        return FedAvg
    elif strategy.lower() == 'fedlabels':
        return FedLabels
    else:
        raise ValueError(f'cannot use strategy f{strategy}')