# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
In this file we define the wrapper class for 
implementing metrics.
'''
from abc import ABC
import logging
from unittest import result

import numpy as np
import torch
import torch.nn as nn

from utils import print_rank

class Metrics(ABC):

    def __init__(self):
        super().__init__()

    def compute_metrics(self,dataloader, model):
        '''This function is called by ´run_validation_generic´ function 
        inside trainer.py .
        
        This is just a helper function that computes loss and accuracy  
        metrics that will be used for all experiments. This function will
        concatenate and return the basic_metrics dict + customized_metrics 
        dict.
        '''

        print_rank("Computing metrics")
        output_to, metrics, inf_results = self.basic_metrics(dataloader,model)
        try:
            metrics.update(self.customized_metrics(inf_results=inf_results))
        except:
            print_rank("File custom_metrics.py not found")
        
        return output_to, metrics

    def basic_metrics(self, dataloader, model):
        val_losses, val_accuracies = list(), list()
        output_tot = {"probabilities": [], "predictions": [], "labels":[]}
        counter = 0

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                val_loss = model.loss(batch).item()
                inf_results = model.inference(batch)
                output = inf_results['output']
                val_acc = inf_results['val_acc']
                batch_size = inf_results['batch_size']

                if isinstance(output, dict):
                    output_tot["probabilities"].append(output["probabilities"])
                    output_tot["predictions"].append(output["predictions"])
                    output_tot["labels"].append(output["labels"])

                val_losses.append(val_loss * batch_size)
                val_accuracies.append(val_acc * batch_size)
                counter += batch_size

        output_tot["probabilities"] = np.concatenate(output_tot["probabilities"]) if output_tot["probabilities"] else []
        output_tot["predictions"] = np.concatenate(output_tot["predictions"]) if output_tot["predictions"] else []
        output_tot["labels"] = np.concatenate(output_tot["labels"]) if output_tot["labels"] else []

        # Post-processing of metrics
        print_rank(f"validation complete {counter}", loglevel=logging.DEBUG)
        model.set_train()
        avg_val_loss = sum(val_losses) / counter
        avg_val_acc = sum(val_accuracies) / counter
        print_rank(f"validation examples {counter}", loglevel=logging.DEBUG)

        # Create metrics dict
        metrics = {'loss': {'value':avg_val_loss,'higher_is_better': False},
                    'acc': {'value':avg_val_acc,'higher_is_better': True}}
                    
        return output_tot, metrics, inf_results

    def customized_metrics(self, inf_results):
        pass