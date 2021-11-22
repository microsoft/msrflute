# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Utilities for the Trainer and TFTrainer class. Should be independent from PyTorch and TensorFlow.
"""

import random
from typing import Any, Dict, NamedTuple, Optional, Tuple, Union
import numpy as np
import torch
import logging

from utils import print_rank


def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).
    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available


class EvalPrediction(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (:obj:`np.ndarray`): Predictions of the model.
        label_ids (:obj:`np.ndarray`): Targets to be matched.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: np.ndarray


class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class ComputeMetrics:
    def __init__(self, p: EvalPrediction, mask=None):
        self.EvalPrediction = EvalPrediction
        self.compute_metrics( self.EvalPrediction)

    @staticmethod
    def compute_metrics(p: EvalPrediction, mask=None):
        print_rank('Prediction Block Size: {}'.format(p.predictions.size()), loglevel=logging.DEBUG)
        if len(list(p.predictions.size()))<3:
            if len(list(p.predictions.size()))<2:
                print_rank('There is something REALLY wrong with prediction tensor:'.format(p.predictions.size()), loglevel=logging.INFO)
                return {'acc': torch.tensor(0.0)}
            print_rank('There is something wrong with prediction tensor:'.format(p.predictions.size()), loglevel=logging.INFO)
            preds = np.argmax(p.predictions, axis=1)
        else:
            preds = np.argmax(p.predictions, axis=2)

        if mask is None:
            return {'acc': (preds == p.label_ids).float().mean()}
        else:
            #valid = preds >1  # reject oov predictions even if they're correct.
            valid = mask==1
            return {'acc': (preds.eq(p.label_ids.cpu()) * valid.cpu()).float().mean()}
