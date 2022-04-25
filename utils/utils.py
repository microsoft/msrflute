# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import sys
import numpy as np
import logging
import yaml
import time
import math
import json
import copy
import io
import pstats
import functools
import torch
from collections import OrderedDict
from utils.optimizers.lars import LarsSGD
from utils.optimizers.lamb import LAMB
from utils.optimizers.adamW import AdamW
from core.globals import TRAINING_FRAMEWORK_TYPE
from easydict import EasyDict as edict
from torch.optim.lr_scheduler import (
    StepLR, 
    MultiStepLR, 
    ReduceLROnPlateau )


if TRAINING_FRAMEWORK_TYPE == 'mpi':
    from mpi4py import MPI
else:
    raise NotImplementedError('Training framework is not yet supported')

def make_optimizer(optimizer_config, model):
    """Initialization for optimizer."""

    tmp_config = copy.deepcopy(optimizer_config)
    if optimizer_config["type"] == "sgd":
        tmp_config.pop("type", None)
        return torch.optim.SGD(model.parameters(), **tmp_config)

    elif optimizer_config["type"] == "adam":
        tmp_config.pop("type", None)
        return torch.optim.Adam(model.parameters(), **tmp_config)

    elif optimizer_config["type"] == "adamax":
        tmp_config.pop("type", None)
        tmp_config.pop("amsgrad", None)
        return torch.optim.Adamax(model.parameters(), **tmp_config)

    elif optimizer_config["type"] == "lars":
        tmp_config.pop("type", None)
        from torchlars import LARS
        base_optimizer = torch.optim.SGD(model.parameters(), **tmp_config)
        return LARS(optimizer=base_optimizer, eps=1e-8, trust_coef=0.001)
    
    elif optimizer_config["type"] == "LarsSGD":
        tmp_config.pop("type", None)
        return LarsSGD(model.parameters(),**tmp_config)

    elif optimizer_config["type"] == "lamb":
        tmp_config.pop("type", None)
        return LAMB(model.parameters(), **tmp_config)

    elif optimizer_config["type"] == "adamW":
        tmp_config.pop("type", None)
        tmp_config.pop("amsgrad", None)
        return AdamW(model.parameters(), **tmp_config)
        
    else:
        raise ValueError("{} optimizer not supported".format(optimizer_config["type"]))


def get_lr(optimizer):
    """Obtain LR."""
    for param_group in optimizer.param_groups:
        return param_group['lr']

def get_lr_all(optimizer):
    """Double checking for get_lr."""
    for param_group in optimizer.param_groups:
        yield param_group['lr']


def softmax(X, theta = 1.0, axis = None):
    """Compute the softmax of each element along an axis of X.

    Args:
        X (ndarray): x, probably should be floats.
        theta (float): used as a multiplier prior to exponentiation. Default = 1.0
        axis : axis to compute values along. Default is the first non-singleton axis.

    Returns:
        An array the same size as X. The result will sum to 1 along the specified axis.
    """
    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


class AverageMeter(object):
    """ Will calculate running micro and macro averages for various
    (error/efficiency) rates.
    """
    def __init__(self, metric_name):
        self.numerators, self.denominators = list(), list()
        self.metric_name = metric_name

    def add(self, top, bottom):
        self.numerators.append(top)
        self.denominators.append(bottom)

    def get_macro_average(self):
        scores = [float(self.numerators[i]) / self.denominators[i] \
                            for i in range(len(self.denominators))]
        return self.get_average(scores)

    def get_micro_average(self):
        return float(sum(self.numerators)) / sum(self.denominators)

    # accepts a list and returns average
    def get_average(self, l):
        return sum(l) / float(len(l))

    def reset(self):
        self.numerators, self.denominators = list(), list()

    def display_results(self, loglevel=logging.INFO):
        print_rank("{} Macro average: {}".format(self.metric_name,
                                                self.get_macro_average()), loglevel)
        print_rank("{} Micro average: {}".format(self.metric_name,
                                                self.get_micro_average()), loglevel)


def make_lr_scheduler(annealing_config, optimizer, num_batches=1):
    """Set learning rate scheduler."""

    annealing_config = copy.deepcopy(annealing_config)
    annealing_type = annealing_config.pop("type")

    # per epoch or per iter
    step_interval='epoch'
    if "step_interval" in annealing_config:
        step_interval = annealing_config.pop("step_interval")

    if annealing_type == "step_lr":
        # convert epoch steps to iter steps
        # expochs can also be floats like 1.5
        if step_interval == "epoch":
            annealing_config["step_size"] = int(num_batches * \
                                    annealing_config["step_size"])
        lr_scheduler =  StepLR(optimizer=optimizer,
                                **annealing_config)
    elif annealing_type == "multi_step_lr":
        # convert epoch steps to iter steps
        if step_interval == "epoch":
            annealing_config["milestones"] = [int(i * num_batches) for i in annealing_config["milestones"]]
        lr_scheduler =  MultiStepLR(optimizer=optimizer,
                                **annealing_config)
    elif annealing_type == "rampup-keep-expdecay-keep":
        # emulate SpecAugment scheduling
        lr_scheduler =  RampupKeepExpdecayKeepLRScheduler(optimizer=optimizer,
                                        **annealing_config)
    elif annealing_type == 'val_loss':
        lr_scheduler =  ReduceLROnPlateau(optimizer,
                                        **annealing_config)
    else:
        raise ValueError("{} LR scheduler not supported".format(
                                                annealing_type))
    return lr_scheduler


class RampupKeepExpdecayKeepLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Implements the LR schedule described in the specaugment paper."""

    def __init__(self, optimizer, peak_lr=0.001, floor_lr=0.00001, sr=1000, si=40000, sf=160000, last_epoch=-1):
        assert(peak_lr>=floor_lr)
        self.peak_lr = peak_lr
        self.floor_lr = floor_lr
        assert(sr<=si)
        assert(si<=sf)
        self.sr = sr
        self.si = si
        self.sf = sf
        self.gamma = math.log(self.floor_lr/self.peak_lr)/(float(self.sf-self.si))
        print('self.gamma')
        print(self.gamma)
        self.step_count = 0
        super(RampupKeepExpdecayKeepLRScheduler, self).__init__(optimizer, last_epoch=last_epoch)

    def step(self, epoch=None):
        for p, lr in zip(self.optimizer.param_groups, self.get_lr()):
            p['lr'] = lr
        self.step_count += 1

    def get_lr(self):
        lr = self.floor_lr
        if self.step_count < self.sr:
            # linear ramp up
            lr = self.peak_lr * float(self.step_count) / float(self.sr)
        elif self.step_count < self.si:
            # keep peak_lr
            lr = self.peak_lr
        elif self.step_count < self.sf:
            # exponential decay from peak_lr to floor_lr
            lr = self.peak_lr * math.exp(self.gamma * (float(self.step_count-self.si)))

        return [lr for base_lr in self.base_lrs]



class ScheduledSamplingScheduler():
    """ Implementing the schedule sampling rate schedule.

    0 - ramp_start          = initial_rate
    ramp_start - ramp_end   = {linearly increase to final_rate}
    ramp_end - infinity     = final_rate
    """

    def __init__(self, model, ramp_start, ramp_stop,
                            initial_rate, final_rate):
        self.model = model
        self.ramp_start = ramp_start
        self.ramp_stop = ramp_stop
        self.initial_rate = initial_rate
        self.final_rate = final_rate
        self.iter = 0

    def step(self):
        if self.iter < self.ramp_start:
            self.model.scheduled_sampling_rate = self.initial_rate
        elif self.iter >= self.ramp_start and self.iter <= self.ramp_stop:
            self.model.scheduled_sampling_rate = self.initial_rate + (self.final_rate - self.initial_rate) * ( (self.iter - self.ramp_start) / (self.ramp_stop - self.ramp_start))
        else:
            self.model.scheduled_sampling_rate = self.final_rate

        self.model.scheduled_sampling = (self.model.scheduled_sampling_rate != 0)
        self.iter += 1

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'model'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)


class NBestTaskScheduler():
    """ Implementing the scheduler for multi-task training.

    num_tasks[0]: 0                     <= i < iteration_per_task[0]
    num_tasks[1]: iteration_per_task[0] <= i < iteration_per_task[1]
    """
    def __init__(self, num_tasks, iteration_per_task):
        assert len(num_tasks) == len(iteration_per_task), "Mismatched length {}!={}".format(len(num_tasks), len(iteration_per_task))
        self.iter = 0
        self.stagex = 0
        self.num_tasks = num_tasks
        self.iteration_per_task = iteration_per_task

    def current_num_tasks(self):
        return self.num_tasks[self.stagex]

    def no_label_updates(self):
        """Return how many times transcription must be updated."""
        return (self.iter // self.iteration_per_task[-1]) + 1

    def set_iteration_no(self, iter_no):
        self.iter = iter_no

    def step(self):
        print_rank("Iter={}: #tasks {} at stage {}".format(self.iter, self.current_num_tasks(), self.stagex))
        local_iter = self.iter % self.iteration_per_task[-1]
        if local_iter == 0:
            self.stagex = 0
        elif local_iter >= self.iteration_per_task[self.stagex]:
            self.stagex += 1

        self.iter += 1


# Logging and write-to-disk utilities

def init_logging(log_dir, loglevel=logging.DEBUG):
    """Initialize logging"""
    
    os.makedirs(log_dir, exist_ok=True)    
    log_file = os.path.join(log_dir, "log.out")
    logging.basicConfig(filename=log_file,
                        level=loglevel)
    handler = logging.StreamHandler(stream=sys.stdout)
    logging.getLogger().addHandler(handler)


def print_cuda_stats():
    if torch.cuda.is_available():
        print_rank("torch.cuda.memory_allocated(): {}".format(torch.cuda.memory_allocated()))
        print_rank("torch.cuda.memory_cached(): {}".format(torch.cuda.memory_cached()))
        print_rank("torch.cuda.synchronize(): {}".format(torch.cuda.synchronize()))
    else:
        print_rank("No CUDA GPU available")


def print_rank(str, loglevel=logging.INFO):

    str = "{} : {}".format(time.ctime(), str)
    logging.log(loglevel, str)

def print_profiler(profiler, loglevel=logging.INFO):
    memfile = io.StringIO()
    pstats.Stats(profiler, stream=memfile) \
        .strip_dirs() \
        .sort_stats(pstats.SortKey.CUMULATIVE) \
        .print_stats(20)                    
    for l in memfile.getvalue().split('\n'):
        print_rank(l, loglevel=loglevel)
    memfile.close()


def write_yaml(save_path, config):
    with open(save_path, 'w', encoding='utf8') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

def torch_save(save_path, state_or_model):
    torch.save(state_or_model, save_path)

def write_tokens(save_path, token_list):
    with open(save_path, 'w', encoding='utf8') as token_fid:
        for w in token_list:
            token_fid.write(w + '\n')


def try_except_save(save_fn, **kwargs):
    """ Try to write it out 3 times."""

    max_attempts = 3
    for attempt in range(1, max_attempts+1):
        try:
            save_fn(**kwargs)
        except IOError:
            print_rank("Write operation failed on {} attempt".format(attempt))
        else:
            print_rank("Write operation succeeded in {} attempts".format(attempt))
            return


def write_nbest_jsonl(uttid2jsonl, uttid2hypos, uttid2scores, outputpath, nbest, orgpath="", newpath=""):
    """ Dump a json list file with n-best hypos."""

    newjsonl = []
    for uttid, jsonl in uttid2jsonl.items():
        if not uttid in uttid2hypos:
            print("Missing utterance {} in results".format(uttid))
            continue
        hypos  = uttid2hypos[uttid]
        if nbest > 1:
            # re-normalize the probablity from N-best: ignoring the events out of the N-best hypos
            weights = uttid2scores[uttid]
            if len(weights) < nbest:
                for n in range(len(weights), nbest):
                    print_rank("Mising {}-th best result in {}. Appending {}".format(n, uttid, weights[0]))
                    weights = np.append(weights, np.array(weights[0]))

            weights = softmax(weights[0:nbest]) if uttid in uttid2scores else np.ones(nbest) / nbest
            # Filling the missing hypos with the 1st best candidate
            for n in range(min(nbest, len(hypos))):
                newjson = copy.deepcopy(jsonl)
                newjson["id"]   = "{}-{}".format(uttid, n)
                newjson["text"] = " ".join(hypos[n])
                newjson["loss_weight"] = weights[n]
        else:
            newjson = copy.deepcopy(jsonl)
            newjson["id"]   = uttid
            newjson["text"] = " ".join(hypos[0])

        newjsonl.append(newjson)

    with open(outputpath, 'w') as ofp:
        for jsonl in newjsonl:
            jsonl["wav"] = jsonl["wav"].replace(orgpath, newpath)
            ofp.write("{}\n".format(json.dumps(jsonl)))

    return True


def write_multitask_jsonl(uttid2jsonl, uttid2hypos, uttid2scores, outputpath, nbest, orgpath="", newpath=""):
    """ Dump a json list file with n-best hypos."""

    if nbest==1:
        return write_nbest_jsonl(uttid2jsonl, uttid2hypos, uttid2scores, outputpath, nbest, orgpath, newpath)

    newjsonl = []
    for uttid, jsonl in uttid2jsonl.items():
        if not uttid in uttid2hypos:
            print_rank("Missing utterance {} in results".format(uttid))
            continue
        hypos  = uttid2hypos[uttid]
        # re-normalize the probablity from N-best: ignoring the events out of the N-best hypos
        weights = uttid2scores[uttid]
        if len(weights) < nbest:
            for n in range(len(weights), nbest):
                print_rank("Mising {}-th best result in {}. Appending {}".format(n, uttid, weights[0]))
                weights = np.append(weights, np.array(weights[0]))

        weights = softmax(weights[0:nbest]) if uttid in uttid2scores else np.ones(nbest) / nbest
        newjson = jsonl
        newjson["task_weights"] = weights.tolist()
        assert len(weights) == nbest, "{}: Weight length does not match: {} != {}".format(uttid, len(weights), nbest)
        newjson["text"] = " ".join(hypos[0])
        newjson["subtextl"] = []
        all_null_results = newjson["text"] == ""
        for n in range(1, nbest):
            if n < len(hypos):
                newjson["subtextl"].append(" ".join(hypos[n]))
            else:
                print_rank("Mising {}-th best result in {}".format(n, uttid))
                newjson["subtextl"].append(" ".join(hypos[0]))
            if all_null_results is True:
                all_null_results = newjson["subtextl"][n-1] == ""

        assert len(newjson["subtextl"]) == nbest-1, "#sub-rec results does not match: {} != {}".format(len(newjson["subtextl"]), nbest-1)
        # take meaningful results only and ignore null string
        if all_null_results is False:
            newjsonl.append(newjson)
        else:
            print_rank("Skip {}: Invalid result '{}'".format(uttid, newjson["text"]))

    with open(outputpath, 'w') as ofp:
        for jsonl in newjsonl:
            jsonl["wav"] = jsonl["wav"].replace(orgpath, newpath)
            ofp.write("{}\n".format(json.dumps(jsonl)))

    return True


def load_eval_result_jsonl(resultjsonl, uttid2hypos=OrderedDict(), uttid2scores=OrderedDict(), dumpfp=None, dump_msg="RESULT: "):
    """Load the result JSON list file dumped by Evaluator().

    Args:

    resultjsonl (str): input JSON list file
    uttid2hypos: (dict): maps the utterance ID to text, [uttid] = hypothesis text
    uttid2scores (dict): maps the utterance ID to a confidence score, [uttid] = confidence score(s)
    dumpfp (file): pointer where the WERs will be written out
    dump_msg (str): message string before the WER result
    """
    total_weighted_best_wer   = 0
    total_weighted_oracle_wer = 0
    total_length              = 0
    with open(resultjsonl) as resultfp:
        for line in resultfp:
            elems = json.loads(line.strip())
            if "hypothesis" in elems:
                uttid = elems["utt_id"]
                params = list(elems["hypothesis"].keys())
                uttid2hypos[uttid] = elems["hypothesis"][params[0]]
                if "nbest_model_scores" in elems:
                    uttid2scores[uttid] = np.array(elems["nbest_model_scores"][params[0]])
            else:
                print_rank("Result: {}".format(line.strip()))
                if dumpfp is not None:
                    dumpfp.write("{}{}\n".format(dump_msg, line.strip()))
                params = list(elems["wer-"].keys())
                total_weighted_best_wer   += elems["wer-"][params[0]]["best_wer"] * elems["wer-"][params[0]]["total_length"]
                total_weighted_oracle_wer += elems["wer-"][params[0]]["oracle_wer"] * elems["wer-"][params[0]]["total_length"]
                total_length += elems["wer-"][params[0]]["total_length"]

    return uttid2hypos, uttid2scores, total_weighted_best_wer, total_weighted_oracle_wer, total_length


def find_pretrained_model(model_path, config):
    """"Load a a pre-trained/seed model if provided in config file."""
    output_file=None

    if config.get("pretrained_model_path", None):
        output_file=config["pretrained_model_path"]

    print_rank('Loading Model from: {}'.format(output_file), loglevel=logging.INFO)
    return output_file


def flatten_grads_model(learner) -> np.ndarray:
    """Given a model flatten all params and return as np array."""

    return np.concatenate([w.grad.detach().clone().cpu().numpy().flatten() for w in learner.parameters()])

def flatten_grads_array(param_array)->np.array:
    """Given a model flatten all params and return as np array."""

    N=len(param_array)
    tmp_array=[]
    for i in range(N):
        tmp_array.append(np.concatenate([w.detach().clone().cpu().numpy().flatten() for w in param_array[i]]))
    return np.array(tmp_array)

def dist_weights_to_model(weights, parameters):
    """Updates the model parameters with the supplied weights."""

    offset = 0
    for param in parameters:
        new_size = functools.reduce(lambda x, y: x*y, param.shape)
        current_data = weights[offset:offset + new_size]
        param.data[:] = torch.from_numpy(current_data.reshape(param.shape)).to(param.data)
        offset += new_size

def dist_params_to_model(grads, model):
    """Updates the model gradients (Corresponding to each param) with the supplied grads."""

    offset = 0
    for p in model:
        new_size = functools.reduce(lambda x, y: x*y, p.data.shape)
        current_data = torch.from_numpy(grads[offset:offset + new_size].reshape(p.data.shape)).type(p.data.dtype).to(p)
        p.grad = current_data if p.grad==None else p.grad+current_data
        offset += new_size
        
def reshape_params_to_model(grads, model):
    """ Given Gradients and a model architecture this method updates the model gradients (Corresponding to each param)
    with the supplied grads """
    offset = 0
    reshaped_grads=[]
    for p in model:
        new_size = functools.reduce(lambda x, y: x*y, p.shape)
        current_data = torch.from_numpy(grads[offset:offset + new_size].reshape(p.shape)).type(p.dtype).to(p)
        reshaped_grads.append(current_data)
        offset += new_size
    return reshaped_grads

def to_device(x):
    return x.cuda() if torch.cuda.is_available() else x

def update_json_log(log_path, status_info):
    """Update J-son elements"""
    
    elems = {}
    if os.path.exists(log_path):
        with open(log_path, 'r') as logfp: 
            elems = json.load(logfp)
            print_rank("Loaded status info: {}".format(elems))

    for k, v in status_info.items():
        elems[k] = v

    with open(log_path, 'w') as logfp:
        json.dump(elems, logfp)
        print_rank("Updated status info: {}".format(elems))


def scrub_empty_clients(data_strct):
    """ Clean empty clients in the data structure"""

    users_out = []
    user_data_out = {}
    num_samples_out = []
    if 'user_data_label' in data_strct.keys():
        user_data_label_out = {}
    for ix, user in enumerate(data_strct['users']):
        if data_strct['num_samples'][ix] > 0:
            users_out.append(user)
            user_data_out[user] = data_strct['user_data'][user]
            num_samples_out.append(data_strct['num_samples'][ix])
            if 'user_data_label' in data_strct.keys():
                user_data_label_out[user] = data_strct['user_data_label'][user]

    if ('user_data_label' in data_strct.keys()):
        return edict({'users': users_out, 'user_data': user_data_out, 'num_samples': num_samples_out, 'user_data_label': user_data_label_out})
    else:
        return edict({'users': users_out, 'user_data': user_data_out, 'num_samples': num_samples_out})


def compute_grad_cosines(grads, model_grad):
    def compute_cosine(g, m):
        tot = 0
        g2 = 0
        m2 = 0
        for p1, p2 in zip(g, m):
            tot += torch.mul(p1, p2.to('cpu')).sum().item()
            g2 += torch.mul(p1, p1).sum().item()
            m2 += torch.mul(p2, p2).sum().item()
        return tot / (np.sqrt(g2) * np.sqrt(m2)) if g2 > 0 and m2 > 0 else 0
    return [compute_cosine(g, model_grad) for g in grads]
