from __future__ import annotations
from dataclasses import dataclass
import sys
sys.path.append('../../')
from core.config import ModelConfig, from_dict


@dataclass
class GRUConfig(ModelConfig):
    """nlg_gru configuration

The model configuration specifies model architecture, parameters, and initialization settings.

Attributes:
    embed_dim (int): specific to GRU models, embedding dimension.

    vocab_size (int): specific to GRU models, the vocabulary size.

    hidden_dim (int): specific to GRU models, the hidden size.

    weight_init (str): ``default``, or ``xavier_normal``, indicating how to randomly initialize the model weights.

    OOV_correct (bool): whether OOV predictions are evaluated as correct, or ignored.
"""
    embed_dim: int | None = None
    vocab_size: int | None = None
    hidden_dim: int | None = None
    weight_init: str = None
    OOV_correct: bool = False
    
    @staticmethod
    def from_dict(config) -> GRUConfig:
        return from_dict(GRUConfig, config)
