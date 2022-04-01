# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch as T
import logging
import math
import json
from utils import print_rank
from azureml.core import Run
from scipy.special import betainc, betaln

run = Run.get_context()

def compute_LDP_noise_std(eps, max_sensitivity, delta):
    return np.sqrt(2 * np.log(1.25 / delta)) * max_sensitivity / eps

    
def _beta2betainc_ratio(a, x):
    return 1 / betainc(a, a, x)


def _log_m1(d, alpha, gamma):
    return alpha * np.log(1 - gamma**2) - (d - 2) * np.log(2) - np.log(d - 1)


def _log_m2(p, tau, alpha):
    return np.log(p / (_beta2betainc_ratio(alpha, tau) - 1) - (1 - p)) + np.log(_beta2betainc_ratio(alpha, tau)) - betaln(alpha, alpha)


def _efficient_m(d, gamma, p):
    alpha = (d - 1) / 2
    tau = (1 + gamma) / 2 
    return np.exp(_log_m1(d, alpha, gamma) + _log_m2(p, tau, alpha))


def privacy_parameters(eps0, eps, d):
    exp_eps0 = np.exp(eps0)
    exp_eps = np.exp(eps)
    if exp_eps0 == np.inf:
        p0 = 1
    else:
        p0 = exp_eps0 / (1 + exp_eps0)
    if exp_eps == np.inf:
        gamma = np.sqrt(np.pi / (2 * (d - 1)))
    else:
        gamma = ((exp_eps - 1) / (exp_eps + 1)) * np.sqrt(np.pi / (2 * (d - 1)))
    return p0, gamma


def private_unit2(grad, gamma, prob):
    np.testing.assert_almost_equal(grad.norm().cpu().item(), 1, decimal=5)
    assert prob >= 0.5
    assert (0 <= gamma <= 1)
    p = T.rand(())
    while True:
        # create a uniform distriubtion over d-sphere
        V = T.normal(0, 1, grad.shape, device=grad.device)
        V = V / V.norm()
        dot_prod = T.dot(V, grad)
        if (dot_prod >= gamma and p < prob) or (dot_prod < gamma and p >= prob):
            break
    d = grad.shape[0]
    m = _efficient_m(d, gamma, prob)
    return V / m


def add_gaussian_noise(grad, eps, max_grad, delta):
    sigma = compute_LDP_noise_std(eps, max_grad, delta)
    #sigma = np.sqrt(2 * np.log(1.25 / delta)) * max_grad / eps
    noisy_grad = sigma * T.randn(grad.shape, device=grad.device) + grad
    return noisy_grad, sigma


def add_private_unit2_noise(eps, grad):
    eps0 = 0.01 * eps
    eps1 = 0.99 * eps
    samp_prob, gamma = privacy_parameters(eps0, eps1, grad.shape[0])
    return private_unit2(grad, gamma, samp_prob)


def scalar_DP(r, eps, k, r_max):
    r = np.minimum(r, r_max)
    val = k * r / r_max
    f_val = math.floor(val)
    c_val = math.ceil(val)
    J = f_val if T.rand(()) < (c_val - val) else c_val
    exp_eps = np.exp(eps)
    rand_prob = exp_eps / (exp_eps + k)
    if T.rand(()) >= rand_prob:
        while True:
            J_ = T.randint(0, k + 1, ()).item()
            if J != J_:
                J = J_
                break
    a = ((exp_eps + k) / (exp_eps - 1)) * (r_max / k)
    b = (k * (k + 1)) / (2 * (exp_eps + k))
    return a * (J - b)


def laplace_noise(max_sens, eps, vocab_size):
    return np.random.laplace(0.0, max_sens/eps, vocab_size)


def unroll_network(named_params, select_grad=False):
    # Unroll the network as 1D vector and save original values indices
    params_ids, flat_params  = {}, []
    cur_idx = 0
    for n, p in named_params:
        dat = p.grad if select_grad else p.data
        flat_params.append(dat.view(-1))
        next_idx = cur_idx + flat_params[-1].shape[0]
        params_ids[n] = (cur_idx, next_idx)
        cur_idx = next_idx
    return T.cat(flat_params), params_ids


def update_network(named_params, params_ids, flat_params, apply_to_grad=False):
    # Roll back the network parameters to layers
    for n, p in named_params:
        s_id, e_id = params_ids[n]
        if apply_to_grad:
            p.grad.copy_(flat_params[s_id : e_id].view(*p.grad.shape))
        else:
            p.data.copy_(flat_params[s_id : e_id].view(*p.data.shape))


def apply_global_dp(config, model, num_clients_curr_iter, select_grad=True, metric_logger=None):
    # Add global DP noise here
    dp_config = config.get('dp_config', None)
    if dp_config is not None and dp_config.get('enable_global_dp', False):
        # enable_local_dp must be enabled - client-side gradient clipping must be enabled.
        assert (dp_config['enable_local_dp'])
        # Unroll the network grads as 1D vectors
        flat_grad, params_ids = unroll_network(model.named_parameters(), select_grad=select_grad)

        sigma = dp_config['global_sigma']
        max_grad = dp_config['max_grad']
        noise_scale = sigma * max_grad / num_clients_curr_iter
        noise = T.normal(0, 1, flat_grad.shape, device=flat_grad.device) * noise_scale
        flat_noisy_grad = flat_grad + noise
        print_rank('Error from noise {} is {}. grad norm: {} noisy_grad norm: {}'.format(noise_scale, (
                    flat_grad - flat_noisy_grad).norm(), flat_grad.norm(), flat_noisy_grad.norm()))

        # Return back to the network gradients
        update_network(model.named_parameters(), params_ids, flat_noisy_grad,
                               apply_to_grad=select_grad)

        if metric_logger is None:
            metric_logger = Run.get_context().log
        metric_logger('Gradient Norm', flat_grad.norm().cpu().item())


def apply_local_dp(trainer, weight, dp_config, add_weight_noise):
    '''Apply client-side DP, possibly given a data-dependent aggregation weight

    Args:
        trainer (core.Trainer object): trainer on client.
        dp_config (dict): DP config on original config file.
        add_weight_noise (bool): whether noise should be added to aggregation weight.
    '''

    # Unroll the network grads as 1D vectors
    flat_grad, params_ids = unroll_network(trainer.model.named_parameters(), select_grad=True)
    grad_norm = flat_grad.norm().cpu().item()

    if dp_config['eps'] < 0:
        # clip, but don't add noise
        if grad_norm > dp_config['max_grad']:
            flat_grad = flat_grad * (dp_config['max_grad'] / grad_norm)
            update_network(trainer.model.named_parameters(), params_ids, flat_grad, apply_to_grad=True)

    else:
        # Get Gaussian LDP noise
        dp_eps = dp_config['eps']
        delta = dp_config.get('delta', 1e-7) # TODO pre-compute in config
        weight_ = weight

        # Scaling the weight down so we don't impact the noise too much
        weight = dp_config.get('weight_scaler', 1) * weight
        weight = min(dp_config['max_weight'], weight)
        flat_noisy_grad = dp_config['max_grad'] * (flat_grad / flat_grad.norm())
        max_sensitivity = np.sqrt(dp_config['max_grad']**2 + (dp_config['max_weight']**2 if add_weight_noise else 0.0))
        flat_noisy_grad = T.cat([flat_noisy_grad, T.tensor([weight], device=flat_noisy_grad.device)], dim=0)
        flat_noisy_grad, _ = add_gaussian_noise(flat_noisy_grad, dp_eps, max_sensitivity, delta)
        weight = min(max(flat_noisy_grad[-1].item(), dp_config['min_weight']), dp_config['max_weight'])

        # Scaling the weight back up after noise addition (This is a DP-protect transformation)
        weight = weight / dp_config.get('weight_scaler', 1)
        if not add_weight_noise:
            weight = weight_
        flat_noisy_grad = flat_noisy_grad[:-1]

        print_rank('Cosine error from noise {}'.format(T.nn.functional.cosine_similarity(flat_grad, flat_noisy_grad, dim=0)), loglevel=logging.DEBUG)
        print_rank('Error from noise is {}'.format((flat_grad-flat_noisy_grad).norm()), loglevel=logging.DEBUG)
        print_rank('weight is {} and noisy weight is {}'.format(weight_, weight), loglevel=logging.DEBUG)

        # Return back to the network
        update_network(trainer.model.named_parameters(), params_ids, flat_noisy_grad, apply_to_grad=True)

    return weight


def update_privacy_accountant(config, num_clients, curr_iter, num_clients_curr_iter):
    # Privacy accounting starts here
    # We will dump all the needed parameters to the log so as not to slow down training.
    dp_config = config.get('dp_config', None)
    if dp_config is not None and dp_config.get('enable_global_dp', False) or dp_config.get('enable_local_dp',
                                                                                           False):
        from math import sqrt, exp, log
        import extensions.privacy.analysis as privacy_analysis

        K = 1  # from DP perspective each user is contributing one gradient
        B = num_clients_curr_iter  # batch size
        n = num_clients
        T = curr_iter + 1
        _delta = dp_config.get('delta', min(1e-7, 1. / (n * log(n))))  # TODO should be precomputed in config
        if dp_config.get('global_sigma', None) is None:
            max_sensitivity = np.sqrt(dp_config['max_grad'] ** 2 + dp_config['max_weight'] ** 2)
            noise_scale = compute_LDP_noise_std(dp_config['eps'], max_sensitivity, _delta)
            global_sigma = noise_scale * np.sqrt(B) / max_sensitivity
        else: 
            global_sigma = dp_config['global_sigma']
            noise_scale = global_sigma * dp_config['max_grad'] / B

        try:
            mu = K * B / n * sqrt(T * exp((1. / global_sigma) ** 2 - 1))
        except OverflowError:
            print_rank(f"Error computing mu {global_sigma} {K} {B} {n} {T}")
            mu = -1

        orders = ([1.25, 1.5, 1.75, 2., 2.25, 2.5, 3., 3.5, 4., 4.5] + list(range(5, 64)) + [128, 256, 512])
        q = B / n
        _sigma = global_sigma  # was: noise_scale but we should apply the noise multiplier.
        rdp = privacy_analysis.compute_rdp(q, _sigma, T, orders)

        rdp_epsilon, opt_order = privacy_analysis.get_privacy_spent(orders, rdp, _delta)

        props = {
            'dp_global_K': K,  # gradients per user
            'dp_global_B': B,  # users per batch
            'dp_global_n': n,  # total users
            'dp_global_T': T,  # how many iterations
            'dp_sigma': _sigma,  # noise_multiplier. Should be combined global+local sigma.
            'dp_global_mu': mu,
            # 'dp_epsilon_fdp': fdp_epsilon,
            'dp_epsilon_rdp': rdp_epsilon,
            # 'dp_epsilon_exact': exact_eps,
            'dp_opt_order': opt_order,
            'dp_delta': _delta,
            'dp_noise_scale': noise_scale  # Note: not needed for accounting.
        }

        print_rank(f'DP accounting: {json.dumps(props)}')
        for k in props:
            run.log(k, props[k])

        return rdp_epsilon
    else:
        return None
