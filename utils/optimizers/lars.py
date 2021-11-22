# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""distoptim.hit package"""
import logging
import torch

LOG = logging.getLogger(__name__)

class LarsSGDV1(torch.optim.SGD):
    """ LARS SGD V1, based on https://arxiv.org/abs/1708.03888
        2018.
        Refer to torch.optim.SGD for paramters.
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        LOG.info("Init LarsSGDV1")
        super(LarsSGDV1, self).__init__(
            params, lr, momentum, dampening, weight_decay, nesterov)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            # dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                p_n = p.data.norm()
                d_p_n = d_p.norm()

                if weight_decay != 0:
                    d_p_n.add_(weight_decay, p_n)
                    d_p.add_(weight_decay, p.data)

                alpha = 0.001 * p_n / d_p_n  # This is the LARS eta from the paper
                lr = alpha * group['lr']
                lr = min(lr, 5.0) 

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                            torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(lr, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-1, d_p)

        return loss


class LarsSGD(torch.optim.SGD):
    """ LARS SGD, based on https://arxiv.org/abs/1904.00962 Algorithm 1
        2019, a newer version.
        Refer to torch.optim.SGD for paramters.
    """

    def __init__(self, params, lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        LOG.info("Init LarsSGD")
        super(LarsSGD, self).__init__(
            params, lr, momentum, dampening, weight_decay, nesterov)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            # dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add(p.data, alpha=weight_decay)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = \
                            torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - momentum, d_p)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                lr = group['lr'] * p.data.norm() / (d_p.norm() + 1e-8)
                lr.clamp_(0, 10)
                p.data.add_(d_p, alpha=-lr)

        return loss
