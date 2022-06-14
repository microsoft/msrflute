# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
Modified from https://github.com/pytorch/vision.git

The torchvision package consists of popular datasets, model architectures, 
and common image transformations for computer vision.
'''
import math
import torch as T
import torch.nn as nn
import numpy as np
import logging

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.DEBUG)

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, vgg, num_class, topK_results=None):
        super(VGG, self).__init__()

        self.topK_results = num_class if topK_results is None else topK_results
        self.vgg = vgg
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_class),
        )
        if 0:
            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    m.bias.data.zero_()

    def forward(self, inputs):
        inputs = inputs['x'].cuda() if T.cuda.is_available() else inputs['x']
        x = self.vgg(inputs.view(-1,3,32,32))
        x = T.flatten(x, 1)
        x = self.classifier(x)
        return x


    def loss(self, inputs):
        targets = inputs['y'].cuda() if T.cuda.is_available() else inputs['y']
        # Run the forward pass
        output = self(inputs)
        loss = T.nn.functional.cross_entropy(output, targets)

        return loss


    def inference(self, inputs):
        targets = inputs['y'].cuda() if T.cuda.is_available() else inputs['y']

        # Run the forward pass
        output = self(inputs)

        # accuracy
        accuracy = T.mean((T.argmax(output, dim=1) == targets).float()).item()

        output = {'probabilities': output.cpu().detach().numpy(),
                      'predictions': np.arange(0, targets.shape[0]),
                      'labels': targets.cpu().numpy()}

        return {'output':output, 'val_acc': accuracy, 'batch_size': targets.shape[0]}

    def get_logit(self, inputs = None, evalis = True, logmax=False):
        data, targets = inputs

        if logmax:
            Softmax = T.nn.LogSoftmax(dim=1)
        else:
            Softmax = T.nn.Softmax(dim=1)

        data = data.cuda() if T.cuda.is_available() else data

        if evalis:
            self.eval()
            with T.no_grad():
                # Run the forward pass
                output = self.forward(data)
                logits = Softmax(output)
        else:
            self.train()
            output = self.forward(data)
            logits = Softmax(output)

        loss = T.nn.functional.cross_entropy(output, targets)

        return logits.cpu(), targets.cpu(), loss.cpu()

    def copy_state_dict(self, state_dict):
        self.state_dict=state_dict.clone()

    def set_eval(self):
        """
        Bring the model into evaluation mode
        """
        self.eval()

    def set_train(self):
        """
        Bring the model into train mode
        """
        self.train()


def make_layers(cfg, n_channels=3, batch_norm=True):
    layers = []
    in_channels = n_channels
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(config):
    """VGG 11-layer model (configuration "A")"""
    num_class = config['num_classes']
    return VGG(make_layers(cfg['A'], batch_norm=False),num_class)


def vgg11_bn(config):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    num_class = config['num_classes']
    return VGG(make_layers(cfg['A'], batch_norm=True),num_class)


def vgg13(config):
    """VGG 13-layer model (configuration "B")"""
    num_class = config['num_classes']
    return VGG(make_layers(cfg['B'], batch_norm=False),num_class)


def vgg13_bn(config):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    num_class=config['num_classes']
    return VGG(make_layers(cfg['B'], batch_norm=True),num_class)


def vgg16(config):
    """VGG 16-layer model (configuration "D")"""
    num_class = config['num_classes']
    return VGG(make_layers(cfg['D'], batch_norm=False),num_class)


def vgg16_bn(config):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    num_class = config['num_classes']
    return VGG(make_layers(cfg['D'], batch_norm=True),num_class)


def vgg19(config):
    """VGG 19-layer model (configuration "E")"""
    num_class=config['num_classes']
    return VGG(make_layers(cfg['E'], batch_norm=False),num_class)


def vgg19_bn(config):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    num_class=config['num_classes']
    return VGG(make_layers(cfg['E'], batch_norm=True),num_class)