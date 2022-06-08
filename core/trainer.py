# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
import re

import numpy as np
import torch
import torch.nn as nn

from core.metrics import Metrics
from utils import \
    get_lr, \
    get_lr_all, \
    make_optimizer, \
    make_lr_scheduler, \
    print_rank, \
    torch_save, \
    try_except_save, \
    write_yaml
from utils.utils import to_device


class TrainerBase:
    """Abstract class defining Trainer objects' common interface.

    Args:
        model (torch.nn.Module): model to be trained.
        train_dataloader (torch.utils.data.DataLoader): dataloader that
            provides the training data.
        optimizer: (torch.optim.Optimizer): optimizer that will be used to
            update the model.
        max_grad_norm (float): if not None, avg gradients are clipped to this
            norm; defaults to None.
        ignore_subtask (bool): ignore subtasks, defaults to True.
        model_type (str): what kind of model is used, defaults to
            :code:`LanguageModel`.
        decoder_config (dict or None): config for decoder, defaults to None.
    """

    def __init__(
        self,
        model,
        train_dataloader,
        optimizer,
        max_grad_norm=None,
        ignore_subtask=True,
        model_type="LanguageModel",
        decoder_config=None
    ):

        self.model = model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.max_grad_norm = max_grad_norm
        self.model_type = model_type
        self.decoder_config = decoder_config

        self.step = 0  # count how many batches are processed
        self.ignore_subtask = ignore_subtask  # ignore subtasks even if there are multiple task branches

    def epoch_boundary(self):
        '''Check if we are at the end of any given epoch.'''
        return self.step % len(self.train_dataloader.create_loader()) == 0 and self.step != 0

    def train_desired_samples(self, desired_max_samples, apply_privacy_metrics):
        pass

    def save(self):
        pass

    def load(self):
        pass


class ModelUpdater(TrainerBase):
    """Update the model, given the already computed gradient.

    This is a special kind of trainer, that actually does not use any data.

    Args:
        model (torch.nn.Module): model to be updated.
        optimizer (torch.optim.Optimizer): optimizer that will be used to
            update the model.
        ss_scheduler: scheduled sampler.
        train_dataloader: train dataloader, this is not actually used.
        val_dataloader: val dataloader, this is not actually used.
        max_grad_norm (float): avg gradients are clipped to this norm.
        anneal_config (dict): annealing configuration.
        model_type (str): what kind of model is used, defaults to
            :code:`LanguageModel`.
        decoder_config (dict): config for decoder, defaults to None.
    """

    def __init__(
        self,
        model,
        optimizer,
        ss_scheduler,
        train_dataloader,
        val_dataloader,
        max_grad_norm,
        anneal_config,
        model_type="LanguageModel",
        decoder_config=None
    ):
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            model_type=model_type,
            decoder_config=decoder_config
        )

        self.val_dataloader = val_dataloader
        self.annealing_type = anneal_config["type"] if anneal_config is not None else None
        self.lr_scheduler = make_lr_scheduler(anneal_config, self.optimizer)
        self.ss_scheduler = ss_scheduler

    def update_model(self):
        """Update model parameters using pre-computed gradients."""

        # Apply gradient clipping
        if self.max_grad_norm is not None:
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            print_rank(f"clipped norm: {grad_norm} to {min(grad_norm,self.max_grad_norm)}", logging.DEBUG)

        # Do optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

    def run_lr_scheduler(self, force_run_val=False):
        """Update learning rate using scheduler."""

        val_loss = val_acc = None
        if force_run_val is True or self.annealing_type == "val_loss":
            _, val_loss, val_acc = run_validation_generic(self.model, self.val_dataloader)

        # Do LR scheduling
        print_rank(f"LR all: {list(get_lr_all(self.optimizer))}", loglevel=logging.DEBUG)
        print_rank("LR BEFORE lr_scheduler step: {}".format(get_lr(self.optimizer)))
        if self.annealing_type == "val_loss":
            self.lr_scheduler.step(val_loss)
        else:
            self.lr_scheduler.step()
        print_rank("LR AFTER lr_scheduler step: {}".format(get_lr(self.optimizer)), loglevel=logging.DEBUG)

        return (val_loss, val_acc)

    def run_ss_scheduler(self):
        """Do scheduled sampling."""

        if self.ss_scheduler is not None:
            self.ss_scheduler.step()

    def save(self, model_path, token=None, config=None):
        """Save model to disk."""

        save_model(
            model_path=model_path,
            config=config,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            ss_scheduler=self.ss_scheduler,
            token=token
        )

    def load(self, save_path, update_lr_scheduler, update_ss_scheduler):
        """Load model from disk.

        If save_path is given, load from there. If not, then resume training
        from current model dir.  If at any point the save_path is not present on
        the disk, it won't be loaded.
        """

        if os.path.isfile(save_path):
            print_rank("Loading checkpoint: {}".format(save_path))
            checkpoint = torch.load(save_path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            anl_st_dict = checkpoint.get("lr_scheduler_state_dict")
            if anl_st_dict and self.lr_scheduler is not None and update_lr_scheduler is True:
                self.lr_scheduler.load_state_dict(anl_st_dict)

            sss_st_dict = checkpoint.get("ss_scheduler_state_dict")
            if sss_st_dict and self.ss_scheduler is not None and update_lr_scheduler is True:
                self.ss_scheduler.load_state_dict(sss_st_dict)


class Trainer(TrainerBase):
    """Perform training step for any given client.

    The main method to be called for triggering a training step is
    :code:`train_desired_samples`, which on its turn relies on
    :code:`run_train_epoch`.

    Args:
        model (torch.nn.Module): model to be trained.
        ss_scheduler: scheduled sampler.
        train_dataloader (torch.data.utils.DataLoader): dataloader that
            provides the training data.
        server_replay_config (dict or None): config for replaying training;
            defaults to None, in which case no replaying happens.
        optimizer (torch.optim.Optimizer or None): optimizer that will be used
            to update the model. If :code:`None`, skip optimization.
        max_grad_norm (float or None): if not None, avg gradients are clipped
            to this norm; defaults to None.
        anneal_config (dict or None): annealing configuration.
        num_skips_threshold (int): previously used to skip users, deprecated.
        ignore_subtask (bool): ignore subtasks, defaults to True.
    """

    def __init__(
        self,
        model,
        ss_scheduler,
        train_dataloader,
        server_replay_config=None,
        optimizer=None,
        max_grad_norm=None,
        anneal_config=None,
        num_skips_threshold=-1,
        ignore_subtask=True
    ):
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            optimizer=optimizer,
            max_grad_norm=max_grad_norm,
            ignore_subtask=ignore_subtask
        )

        self.server_replay_config=None
        if server_replay_config is not None:
            self.server_replay_config = server_replay_config

        self.anneal_config=None
        if anneal_config is not None:
            self.anneal_config = anneal_config

        self.lr_scheduler = None
        if self.optimizer is None and self.server_replay_config is not None and "optimizer" in self.server_replay_config:
            self.optimizer = make_optimizer(self.server_replay_config["optimizer_config"], model)

        if self.optimizer is not None and self.anneal_config is not None:
            self.lr_scheduler = make_lr_scheduler(
                                                self.anneal_config,
                                                self.optimizer)

        self.cached_batches = []
        self.ss_scheduler = ss_scheduler

    def reset_gradient_power(self):
        """Reset the sum of gradient power.

        This is used to compute statistics about the gradients.
        """

        self.sum_grad = self.sum_grad2 = self.counter = 0

    def accumulate_gradient_power(self):
        """Compute sum of gradient power.

        This is used to compute statistics about the gradients.
        """

        for p in self.model.parameters():
            if p.grad is None:
                continue

            grad = p.grad.detach().clone().cpu().numpy()
            p1 = np.sum(grad)
            p2 = np.sum(grad ** 2)
            n = p.grad.numel()

            self.sum_grad += p1
            self.sum_grad2 += p2
            self.counter += n

        print_rank("Magn. Grad. Squared: {}".format(self.sum_grad2), loglevel=logging.DEBUG)
        print_rank("Magn. Grad.: {}".format(self.sum_grad), loglevel=logging.DEBUG)
        return self.sum_grad, self.sum_grad2, self.counter

    def estimate_sufficient_stats(self):
        """Compute statistics about the gradients."""

        sum_mean_grad, sum_mean_grad2, n = self.accumulate_gradient_power()

        mean_grad = sum_mean_grad / n
        mag_grad = np.sqrt(sum_mean_grad2 / n)
        var_grad = sum_mean_grad2 / n - mag_grad**2
        norm_grad = np.sqrt(sum_mean_grad2)

        self.sufficient_stats = {
            "n": n,
            "sum": sum_mean_grad,
            "sq_sum": sum_mean_grad2,
            "var": var_grad,
            "mean": mean_grad,
            "mag": mag_grad,
            "norm": norm_grad
        }

    def train_desired_samples(self, desired_max_samples=None, apply_privacy_metrics=False):
        """Triggers training step.

        Args:
            desired_max_samples (int): number of samples that you would like to process.
            apply_privacy_metrics (bool): whether to save the batches used for the round for privacy metrics evaluation.

        Returns:
            2-tuple of (float, int): total training loss and number of processed samples.
        """

        num_samples = 0
        total_train_loss = 0

        num_samples_per_epoch, train_loss_per_epoch = self.run_train_epoch(desired_max_samples, apply_privacy_metrics)

        num_samples += num_samples_per_epoch
        total_train_loss += train_loss_per_epoch

        return total_train_loss, num_samples

    def run_train_epoch(self, desired_max_samples=None, apply_privacy_metrics=False):
        """Implementation example for training the model.

        The training process should stop after the desired number of samples is processed.

        Args:
            desired_max_samples (int): number of samples that you would like to process.
            apply_privacy_metrics (bool): whether to save the batches used for the round for privacy metrics evaluation.

        Returns:
            2-tuple of (int, float): number of processed samples and total training loss.
        """

        sum_train_loss = 0.0
        num_samples = 0
        self.reset_gradient_power()

        # Reset gradient just in case
        self.model.zero_grad()

        train_loader = self.train_dataloader.create_loader()
        for batch in train_loader:
            if desired_max_samples is not None and num_samples >= desired_max_samples:
                break

            # Compute loss
            if self.optimizer is not None:
                self.optimizer.zero_grad()

            if self.ignore_subtask is True:
                loss = self.model.single_task_loss(batch)
            else:
                if apply_privacy_metrics:
                    if "x" in batch:
                        indices = to_device(batch["x"])
                    elif "input_ids" in batch:
                        indices = to_device(batch["input_ids"])
                    self.cached_batches.append(indices)
                loss = self.model.loss(batch)
            loss.backward()

            # Apply gradient clipping
            if self.max_grad_norm is not None:
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Sum up the gradient power
            self.estimate_sufficient_stats()

            # Now that the gradients have been scaled, we can apply them
            if self.optimizer is not None:
                self.optimizer.step()

            print_rank("step: {}, loss: {}".format(self.step, loss.item()), loglevel=logging.DEBUG)

            # Post-processing in this loop
            # Sum up the loss
            sum_train_loss += loss.item()

            # Increment the number of frames processed already
            if "attention_mask" in batch:
                num_samples += torch.sum(batch["attention_mask"].detach().cpu() == 1).item()
            elif "total_frames" in batch:
                num_samples += batch["total_frames"]
            else:
                num_samples += len(batch["x"])

            # Update the counters
            self.step += 1

        # Take a step in lr_scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return num_samples, sum_train_loss

    def prepare_iteration(self, model=None):
        """Steps to run before iteration begins."""

        if model is not None:
            self.model.load_state_dict(model.state_dict())

            self.lr_scheduler = None
            if self.optimizer is None and self.server_replay_config is not None and \
                    "optimizer_config" in self.server_replay_config:
                print_rank("Creating server-side replay training optimizer", loglevel=logging.DEBUG)
                self.optimizer = make_optimizer(self.server_replay_config["optimizer_config"], self.model)

            if self.optimizer is not None and self.anneal_config is not None:
                print_rank("Creating server-side replay-training lr_scheduler", loglevel=logging.DEBUG)
                self.lr_scheduler = make_lr_scheduler(self.anneal_config, self.optimizer)

    def reset_optimizer(self, optimizer_state_dict, annealing_config=None):
        """Re-load optimizer."""

        assert self.optimizer is not None, "This trainer does not have an optimizer"

        # Load optimizer on state dict
        self.optimizer.load_state_dict(optimizer_state_dict)

        # Set learning rate scheduler
        self.lr_scheduler = None
        if annealing_config is not None:
            self.lr_scheduler = make_lr_scheduler(annealing_config, self.optimizer)


def run_validation_generic(model, val_dataloader):
    """Perform a validation step.

    Args:
        model (torch.nn.Module): model to be validated.
        val_dataloader (torch.data.utils.DataLoader): provides val data.

    Returns:
        Average validation loss.
    """

    print_rank("run_validation_generic", loglevel=logging.DEBUG)
    model.set_eval()
    print_rank("set_eval", loglevel=logging.DEBUG)

    # Initialize dataloader etc.
    val_loader = val_dataloader.create_loader()
    print_rank(
        f"created loader {val_loader.num_workers}, " + \
        f"users: {len(val_dataloader.dataset.user_list)} " + \
        f"examples: {sum(val_dataloader.dataset.num_samples)} " + \
        f"lendata: {len(val_loader)} ",
        loglevel=logging.DEBUG
    )

    print_rank(
        f"drop_last: {val_loader.drop_last} " + \
        f"len_sampler: {len(val_loader._index_sampler)}",
        loglevel=logging.DEBUG
    )

    print_rank("Loading metrics ...")
    metrics_cl = Metrics()
    return metrics_cl.compute_metrics(dataloader=val_loader, model=model)

def set_component_wise_lr(model, optimizer_config, updatable_names):
    """Set zero learning rate for layers in order to freeze the update.

    Args:
        model (torch.nn.Module):
        optimizer_config (string):
        updatable_names (list): ["^dec_rnn", "^fc"]
    """

    def name_matched(name, updatable_names):
        for updatable_name in updatable_names:
            if re.match(updatable_name, name) is not None:
                return True

        return False

    # Set learning rate to zero in layers which name does not follow regex
    parameters = []
    for name, params in model.named_parameters():
        if name_matched(name, updatable_names) is True:
            print_rank("updating {} with lr = {}".format(name, optimizer_config["lr"]))
            parameters.append({"params": params, "lr":optimizer_config["lr"]})
        else:
            print_rank("freezing {}".format(name))
            parameters.append({"params": params, "lr": 0.0})

    return parameters

def save_model(model_path, config, model, optimizer, lr_scheduler, ss_scheduler, token=None):
    """Save a model as well as training information."""

    save_state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "lr_scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None
    }
    if ss_scheduler is not None:
        save_state["ss_scheduler_state_dict"] = ss_scheduler.state_dict()

    if token:  # just save as "best" and return
        save_path = os.path.join(model_path, "{}_model.tar".format(token))
    else:
        save_path = os.path.join(model_path, "model.tar")

    print_rank("Saving model to: {}".format(save_path))
    try_except_save(torch_save, state_or_model=save_state, save_path=save_path)

    # Write out the config to model_dir
    if config is not None:
        try_except_save(write_yaml, config=config,
                save_path=os.path.join(model_path, "config.yaml"))