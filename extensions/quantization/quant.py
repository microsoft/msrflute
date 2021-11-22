# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import torch
from utils import print_rank
from typing import Optional, Tuple

def quant_model(
        model: torch.nn.Module,
        quant_bits: int = 8,
        quant_threshold: Optional[int] = None,
        global_stats: bool = False
    ):
    '''Quantize the gradients using the desired number of bits.

    Nothing is returned as gradients inside :code:`model` are modified
    in-place.

    Args:
        model: model which gradients we want to quantize.
        quant_bits: how many bits will we use to quantize the gradients.
        quant_threshold: fraction of components to be set to zero; defaults to
            None, in which case no quantization happens.
        global_stats: use a single histogram for all layers when binning,
            defaults to False.
    '''

    # If no `quant_threshold`, does nothing
    if quant_threshold is None:
        return
    print_rank('Performing Gradient Quantization with Prob. Threshold: {}'.format(
        quant_threshold), loglevel=logging.INFO)

    # If `global_stats` is true, min/max and thresh are computed across all layers
    if global_stats:
        flattened_grad = torch.cat([p.grad.data.flatten() for p in model.parameters()])
        min_grad, max_grad, thresh = find_min_max_gradient(flattened_grad,
            quant_threshold)

    # Loop through all layers
    for p in model.parameters():
        if not global_stats:
            min_grad, max_grad, thresh = find_min_max_gradient(p.grad.data,
                quant_threshold)

        # Perform binning and sparsification of components
        binned_grad = quant_bins(p.grad.data, 2 ** quant_bits, min_grad, max_grad)
        p.grad = torch.where(torch.abs(p.grad.data) > thresh, binned_grad,
            torch.tensor(0.).to(p.grad))


def find_min_max_gradient(
        gradient: torch.Tensor,
        quant_threshold: Optional[float] = None
    ) -> Tuple[float, float, float]:
    '''Get min and max gradients, as well as threshold gradient.

    Args:
        gradient: tensor over which statistics will be computed.
        quant_threshold: which quantile to look for to compute threshold, must
            be between 0 and 1.
    '''

    # Computes min/max and quantile corresponding to `quant_threshold`
    min_grad, max_grad = gradient.min(), gradient.max()
    thresh = torch.quantile(torch.abs(gradient), quant_threshold)

    print_rank('Min. and Max. Gradients: {}, {}'.format(min_grad, max_grad),
        loglevel=logging.INFO)
    print_rank('Grad. Threshold: {}'.format(thresh), loglevel=logging.INFO)

    return min_grad, max_grad, thresh


def quant_bins(
        gradients: torch.Tensor,
        n_bins: int,
        min_grad: float,
        max_grad: float
    ) -> torch.Tensor:
    '''Perform quantization using binning.

    Creates histogram with `n_bins` bins between `min_grad` and `max_grad`.
    Returns a tensor similar to gradients but with components corresponding to
    bin labels.

    Args:
        gradients: tensor we want to quantize.
        n_bins: how many bins to use for binning.
        min_grad: min. value for bins.
        max_grad: max. value for bins.
    '''

    # We remove half bin width, as bucketize always takes the ceil instead of rounding
    bin_labels = torch.linspace(min_grad, max_grad, n_bins).to(gradients)
    bin_width = bin_labels[1] - bin_labels[0]
    grad_bins = torch.bucketize(gradients - .5 * bin_width, bin_labels, right=False)

    return bin_labels[grad_bins]
