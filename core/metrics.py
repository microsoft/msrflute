# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
In this file we define the wrapper class for 
implementing metrics.
'''
import logging

import numpy as np
import torch

from utils import print_rank

class Metrics():

    def __init__(self):
        super().__init__()

    def compute_metrics(self,dataloader, model):
        '''This method is called by ´run_validation_generic´ function 
        inside trainer.py .
        
        This is just a helper function that computes the metrics returned 
        in the inference function inside ´model.py´.
        '''
        print_rank("Computing metrics")
        return self.call_inference(dataloader,model)

    def call_inference(self, dataloader, model):
        
        metrics, sum_metrics = dict(), dict()
        output_tot = {"probabilities": [], "predictions": [], "labels":[]}
        counter = 0

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                val_loss = model.loss(batch).item()
                inf_results = model.inference(batch)
                inf_results ['loss'] = {'value': val_loss,'higher_is_better': False}
                output = inf_results.pop('output')
                batch_size = inf_results.pop('batch_size')

                for key in inf_results.keys():
                    if not isinstance(inf_results[key], dict):
                        inf_results[key] = {'value':inf_results[key],'higher_is_better': True}
                    sum_metrics[key] = [] if not key in sum_metrics else sum_metrics[key]

                if isinstance(output, dict):
                    output_tot["probabilities"].append(output["probabilities"])
                    output_tot["predictions"].append(output["predictions"])
                    output_tot["labels"].append(output["labels"])

                for q in inf_results.keys():
                    sum_metrics[q].append(inf_results[q]['value']* batch_size)
                counter += batch_size

        output_tot["probabilities"] = np.concatenate(output_tot["probabilities"]) if output_tot["probabilities"] else []
        output_tot["predictions"] = np.concatenate(output_tot["predictions"]) if output_tot["predictions"] else []
        output_tot["labels"] = np.concatenate(output_tot["labels"]) if output_tot["labels"] else []

        # Post-processing of metrics
        print_rank(f"validation complete {counter}", loglevel=logging.DEBUG)
        model.set_train()

        for k in inf_results.keys():
            metrics[k] = inf_results[k]
            metrics[k]['value'] = sum(sum_metrics[k])/counter

        print_rank(f"validation examples {counter}", loglevel=logging.DEBUG)
                    
        return output_tot, metrics
