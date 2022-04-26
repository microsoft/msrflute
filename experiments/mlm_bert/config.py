from __future__ import annotations
from dataclasses import dataclass
import sys
sys.path.append('../../')
from core.config import ModelConfig, Config, from_dict


@dataclass
class BERTModelConfig(Config):
    """BERT model configuration

The BERT configuration specifies huggingface-specific BERT model settings.

Attributes:
    model_name (str): The name of the BERT model.  eg bert-base-uncased.

    cache_dir (str): Tokenizer cache directory, will be created if it doesn't exist.

    use_fast_tokenizer (bool): Whether to use the fast tokenizer.

    mask_token (str): special token to use for masking.

    task (str): The task to use for BERT.  eg mlm.

    past_index (int): The index of the past state in the BERT model's state dict.

    prediction_loss_only (bool): if False, also produce metrics for predictions and labels.

    process_line_by_line (bool): if True, process the input line-by-line.

ToDo:
    * check how cache_dir is used- there's a risk of multiple processes reading/writing at the same time.
    * verify the meaning of past_index (thanks copilot)
    * document the difference when process_line_by_line is True vs False

    """
    model_name: str = None
    cache_dir: str = None
    use_fast_tokenizer: bool = False
    mask_token: str = '<mask>'
    task: str = 'mlm'
    past_index: int | None = -2
    prediction_loss_only: bool = False
    process_line_by_line: bool = False

    @staticmethod
    def from_dict(config) -> BERTModelConfig:
        return from_dict(BERTModelConfig, config)


@dataclass
class BERTTrainingConfig(Config):
    """BERT training configuration

    Configuration settings for BERT training.

    Attributes:
        seed (int): random seed for reproducibility.

        label_smoothing_factor (float): label smoothing factor.  Applied label smoothing when the factor is non-zero.

        batch_size (int): batch size.

        max_seq_length (int): maximum input sequence length.
    """
    seed: int | None = None
    label_smoothing_factor: float | None = None
    batch_size: int | None = None
    max_seq_length: int | None = None

    @staticmethod
    def from_dict(config) -> BERTTrainingConfig:
        return from_dict(BERTTrainingConfig, config)


@dataclass
class BERTSpecificConfig(Config):
    """BERT configuration
    Specifies the model and training configuration for huggingface modeling scenarios.

    Attributes:
        loader_type (str): loader type hint. eg 'text'

        model (BERTModelConfig): BERT model configuration.

        training (BERTTrainingConfig): BERT training configuration.
    """
    loader_type: str = None
    model: BERTModelConfig = None
    training: BERTTrainingConfig = None

    @staticmethod
    def from_dict(config) -> BERTSpecificConfig:
        result = BERTSpecificConfig()
        for k in config:
            if k == 'model':
                result.model = BERTModelConfig.from_dict(config[k])
            elif k == 'training':
                result.training = BERTTrainingConfig.from_dict(config[k])
            else:
                setattr(result, k, config[k])
        return result


@dataclass
class BERTConfig(ModelConfig):
    """
    Expected MLM config wraps the BERTSpecificConfig as a sub-field of the ModelConfig.
    """
    BERT: BERTSpecificConfig = None

    @staticmethod
    def from_dict(config) -> ModelConfig:
        result = BERTConfig()
        for k in config:
            if k=="BERT":
                result.BERT = BERTConfig.from_dict(config[k])
            else:
                setattr(result, k, config[k])
        return result
