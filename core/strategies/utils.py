# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

import numpy as np

from utils import print_rank, to_device


def filter_weight(weight):
    '''Handles aggregation weights if something messed them up'''
    print_rank('Client Weight BEFORE filtering: {}'.format(weight), loglevel=logging.DEBUG)
    if np.isnan(weight) or not np.isfinite(weight):
        weight = 0.0
    elif weight > 100:
        weight = 100
    print_rank('Client Weights AFTER filtering: {}'.format(weight), loglevel=logging.DEBUG)
    return weight

def aggregate_gradients_inplace(model, gradients):
    '''Aggregate list of tensors into model gradients.

    Args:
        model (torch.nn.Module): model to which gradients will be summed.
        gradients (list): list of gradients to sum to model.
    '''

    for p, client_grad in zip(model.parameters(), gradients):
        if p.grad is None:
            p.grad = to_device(client_grad)
        else:
            p.grad += to_device(client_grad)