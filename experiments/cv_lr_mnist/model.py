import torch
from torch import nn
from torch.nn import functional as F
from core.model import BaseModel

''' 
    The LogisticRegression model is taken from FedML repository. For more information regarding this model, 
    please refer to https://github.com/FedML-AI/FedML/blob/master/python/fedml/model/linear/lr.py.
'''


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        o = self.linear(x.view(-1,28*28))
        outputs = torch.sigmoid(o)
        #outputs = torch.sigmoid(self.linear(x))
        return outputs

class LR(BaseModel):
    '''This is a PyTorch model with some extra methods'''

    def __init__(self, model_config):
        super().__init__()
        self.net = LogisticRegression(model_config['input_dim'], model_config['output_dim'])

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        '''Performs forward step and computes the loss'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)
        criterion = nn.CrossEntropyLoss().to(device)
        return criterion(output, labels.long())

    def inference(self, input):
        '''Performs forward step and computes metrics'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)

        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output, dim=1) == labels).float()).item()

        return {'output':output, 'acc': accuracy, 'batch_size': n_samples} 


        