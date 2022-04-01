# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import numpy as np
import torch as T
from copy import deepcopy
from utils import make_optimizer, print_rank

def extract_indices_from_embeddings(gradients, batch, embed_size, vocab_size):
    # Extract the Input gradient embeddings
    batch = T.cat([b.view(-1) for b in batch]).cpu().detach().numpy()
    embed_grad = gradients[:embed_size * vocab_size].reshape(vocab_size, embed_size)
    valid_batch = batch[batch > 0]
    tot_valid_tokens, tot_tokens = len(valid_batch), len(batch)
    # The embedding gradients of the indices seen in the batch have higher l2 norm,
    # because dl/dembed_i = dl/dembed_input_i * (if word_i is in batch) + dl/dembed_output_i
    extracted_indices = T.argsort(embed_grad.norm(dim=-1), descending=True)[:tot_tokens].cpu().detach().numpy()
    # Get the overlap ratio
    extracted_ratio = np.isin(valid_batch, extracted_indices).mean()
    # Find True positive extracted indices
    return extracted_ratio, np.intersect1d(extracted_indices, valid_batch)


def compute_perplexity(encoded_batch, model):
    outputs = model.inference(encoded_batch)    
    (batch_size, seq_len, vocab_size) = outputs['output'].shape    
    perplex = T.nn.functional.log_softmax(outputs['output'], dim=-1)
    return perplex.reshape(-1, vocab_size)[np.arange(batch_size * seq_len),
                    encoded_batch.reshape(-1)].reshape(batch_size, seq_len)


def practical_epsilon_leakage(original_params, model, encoded_batches, is_weighted_leakage=True,
                              max_ratio=1e9, optimizer_config=None):
    # Copy the gradients and save the model.
    current_params = deepcopy(model.state_dict())
    current_gradients = dict((n,p.grad.clone().detach()) for n,p in model.named_parameters())
    model.load_state_dict(original_params)
    pre_perplex, post_perplex = [], []
    # This is just to initialise the gradients
    model.loss(encoded_batches[0][:1]).backward()
    model.zero_grad()
    tolerance = 1 / max_ratio
    max_leakage = 0
    with T.no_grad():
        # Original model before training on client
        for encoded_batch in encoded_batches:
            pre_perplex.append(compute_perplexity(encoded_batch, model))
        # The attacker doesn't not he optimal gradient magnitude but using Adamax with high lr, is proved to be effective    
        for n, p in model.named_parameters():
            p.grad = current_gradients[n] #.grad
            print_rank('grad l2: {}'.format(p.grad), loglevel=logging.DEBUG)
        if optimizer_config is None:
            optimizer_config = {'lr': 0.03, 'amsgrad': False, 'type': 'adamax'}
        #T.optim.Adamax(model.parameters(), lr=optim_lr).step()
        make_optimizer(optimizer_config, model).step()
        #model.zero_grad()
        # The model after training on the client data
        for encoded_batch in encoded_batches:
            post_perplex.append(compute_perplexity(encoded_batch, model))
      
        for pre, post in zip(pre_perplex, post_perplex):
            # Compute the ratio of preplexity and weight it be the probability of correctly predicting the word
            leakage = ((pre + tolerance) / (post + tolerance)).clamp_(0, max_ratio)
            print_rank('perplexities leakage: {} '.format(leakage), loglevel=logging.DEBUG)
            if is_weighted_leakage:
                weight_leakage = T.max(pre.exp(), post.exp()) * leakage
            else:
                weight_leakage = leakage
            max_leakage = max(max_leakage, weight_leakage.max().item())
    print_rank('raw max leakage: {}'.format(max_leakage), loglevel=logging.DEBUG)
    model.load_state_dict(current_params)
    for n,p in model.named_parameters():
        p.grad = current_gradients[n]
    # WE return the log to match epsilon
    return max(np.log(max_leakage), 0)