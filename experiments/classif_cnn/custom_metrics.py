# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
In this file we define the new metrics to 
implement in any experiment.
'''

from core.metrics import Metrics

class CustomMetrics(Metrics):
    def __init__(self):
        super().__init__()
    
    def customized_metrics(self, inf_results):
        '''This function is called by ´compute_metrics´ function inside 
        metrics.py .
        
        This is just a helper function that computes and fetches customized  
        metrics that will be used for any experiment. This function receives 
        the loss and accuracy computed previously, so they can be used for 
        computing customized metrics. It should return a dictionary where the 
        keys are the name of the metric to be logged, with the following form:
        
        metrics = {'metric_name_1': {'value':metric_value_1,
                                     'higher_is_better': False},
                   'metric_name_2': {'value':metric_value_2,
                                     'higher_is_better': True}
                  }

        Args:
            acc_and_loss (dict): Computations from 'basic_metrics', the only keys
                                 inside the dict are 'acc' and 'loss'.
        '''

        customized = dict()
        customized = {'f1_score': {'value':inf_results['f1_score'],'higher_is_better': True}}

        return customized