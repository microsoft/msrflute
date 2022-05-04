# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch as T
from abc import ABC, abstractmethod

class BaseModel(ABC, T.nn.Module):
    '''This is a wrapper class for PyTorch models.'''

    @abstractmethod
    def __init__(self,**kwargs):
        super(BaseModel, self).__init__()
        
    @abstractmethod
    def loss(self, input):
        '''Performs forward step and computes the loss

        Returns:
            torch: Computed loss.
        '''
        pass
    
    @abstractmethod
    def inference(self, input):
        '''Performs forward step and computes metrics
             
        Returns:
            dict: The metrics to be computed. The following keys are
            the minimum required by FLUTE during evaluations rounds: 
                - output
                - acc
                - batch_size

            More metrics can be computed by adding the key with a
            dictionary that includes the fields ´value´ and 
            ´higher_is_better´ as follows:

            {'output':output, 
             'acc': accuracy, 
             'batch_size': n_samples, 
             'f1_score': {'value':f1,'higher_is_better': True}}
        '''
        pass

    def set_eval(self):
        '''Bring the model into evaluation mode'''
        self.eval()

    def set_train(self):
        '''Bring the model into training mode'''
        self.train()
