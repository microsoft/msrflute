# Note this import requires python 3.7+
# Do we want to commit to this?
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import MutableMapping


# TODO everywhere: choose reasonable defaults.
# TODO: decide where task should live as a setting, maybe its own TaskConfig
# TODO: docstrings everywhere

# TODO: Make ModelConfig a base class that different models inherit from
# We could specify the modelconfig class in the config file,
# like we do for model.py.  The current implementation mixes NLG and BERT

# TODO: DatasetConfig needs to be teased apart.
# The main issue is we have *_data, list_of_train_data, train_data_server.
# They all essentially perform the same function in different contexts.
# also some no-longer-used parameters are still present.

# TODO: it's not clear what MutableMapping methods need overrides- we
# could probably just use the default implementation.

# TODO: not all pytorch optimizers can handle amsgrad - we should
# have distinct subclasses for the different optimizers

def from_dict(cls, config):
    """
    Helper function to convert a dict to a class
    """
    return cls(**config)


class Config(MutableMapping):
    def get(self, k: str, default=None):
        result = getattr(self, k, default)
        if result is None:
            return default
        return result

    def lookup(self, s: str, default=None):
        toks = s.split('.')
        child = getattr(self, toks[0], default)
        if len(toks) == 1:
            return child if child is not None else default
        elif isinstance(child, Config):
            return child.lookup('.'.join(toks[1:]), default)
        else:
            return default

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    def __delitem__(self, k):
        delattr(self, k)

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __contains__(self, k):
        return getattr(self, k, None) is not None

    def pop(self, k, default=None):
        result = self.get(k, default)
        if k in self:
            delattr(self, k)
        return result


@dataclass
class ModelConfig(Config):

    model_type: str = None
    model_folder: str = None
    pretrained_model_path: str = None
    embed_dim: int | None = None
    vocab_size: int | None = None
    hidden_dim: int | None = None
    weight_init: str = None
    OOV_correct: bool = False
    BERT: BERTConfig = None

    @staticmethod
    def from_dict(config) -> ModelConfig:
        result = ModelConfig()
        for k in config:
            if k == 'BERT':
                result.BERT = BERTConfig.from_dict(config[k])
            else:
                setattr(result, k, config[k])
        return result


@dataclass
class BERTModelConfig(Config):
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
    seed: int | None = None
    label_smoothing_factor: float | None = None
    batch_size: int | None = None
    max_seq_length: int | None = None

    @staticmethod
    def from_dict(config) -> BERTTrainingConfig:
        return from_dict(BERTTrainingConfig, config)


@dataclass
class BERTConfig(Config):
    loader_type: str = None
    model: BERTModelConfig = None
    training: BERTTrainingConfig = None

    @staticmethod
    def from_dict(config) -> BERTConfig:
        result = BERTConfig()
        for k in config:
            if k == 'model':
                result.model = BERTModelConfig.from_dict(config[k])
            elif k == 'training':
                result.training = BERTTrainingConfig.from_dict(config[k])
            else:
                setattr(result, k, config[k])
        return result


@dataclass
class PrivacyConfig(Config):
    enable_local_dp: bool = False
    enable_global_dp: bool = False
    eps: float | None = None
    delta: float | None = None
    global_sigma: float | None = None
    max_grad: float | None = None
    max_weight: float | None = None
    weight_scaler: float | None = None
    min_weight: float | None = None

    @staticmethod
    def from_dict(config) -> PrivacyConfig:
        return from_dict(PrivacyConfig, config)


@dataclass
class PrivacyMetricsConfig(Config):
    apply_metrics: bool = False
    apply_indices_extraction: bool = False
    allowed_word_rank: int | None = None
    apply_leakage_metric: bool = False
    max_leakage: float | None = None
    max_allowed_leakage: float | None = None
    adaptive_leakage_threshold: float | None = None
    is_leakage_weighted: bool = False
    attacker_optimizer_config: OptimizerConfig = None

    @staticmethod
    def from_dict(config) -> PrivacyMetricsConfig:
        result = PrivacyMetricsConfig()
        for k in config:
            if k == 'attacker_optimizer_config':
                result.attacker_optimizer_config = \
                    OptimizerConfig.from_dict(config[k])
            else:
                setattr(result, k, config[k])
        return result


@dataclass
class OptimizerConfig(Config):
    type: str = None
    # Leave this open for any keyword arguments, so we don't break torch constructors
    # In the future we can limit keywords to torch-specific ones.
    # lr: float = 0.0
    # weight_decay: float = 0.0
    # amsgrad: bool = False

    @staticmethod
    def from_dict(config) -> OptimizerConfig:
        # needs its own from_dict so we can accomodate any fields
        result = OptimizerConfig()
        assert 'type' in config
        for k in config:
            setattr(result, k, config[k])
        return result


@dataclass
class AnnealingConfig(Config):
    type: str = None
    step_interval: str = None
    gamma: float | None = None
    step_size: int | None = None

    @staticmethod
    def from_dict(config) -> AnnealingConfig:
        return from_dict(AnnealingConfig, config)


@dataclass
class DatasetConfig(Config):
    # Common to all text (NLG, MLM) dataloaders
    batch_size: int | None = None
    loader_type: str = None
    prepend_datapath: bool = False
    num_workers: int | None = None
    desired_max_samples: int | None = None

    # Common to all client.train dataloaders
    list_of_train_data: str = None
    max_grad_norm: float | None = None  # propose moving max_grad_norm to client config

    # Common to all server.train dataloaders. What is the difference?
    train_data: str = None
    train_data_server: str = None

    # Common to server.test dataloaders
    test_data: str = None

    # Common to server.val dataloaders
    val_data: str = None

    # Specific to NLG dataloaders
    tokenizer_type: str = None  # Note tokenizer_type appears in NLG configs, but always set to 'not applicable'
    vocab_dict: str = None
    pin_memory: bool = False
    num_frames: int | None = None  # num_frames is missing from NLG server.test dataloader
    max_batch_size: int | None = None
    max_num_words: int | None = None
    unsorted_batch: int | None = None
    utterance_mvn: bool = False  # only present on NLG client.train dataloader

    # Specific to MLM dataloaders
    task: str = None
    mlm_probability: float | None = None
    tokenizer_type_fast: bool = False
    max_seq_length: int | None = None
    min_words_per_utt: int | None = None
    max_samples_per_user: int | None = None
    mask_token: str = None
    cache_dir: str = None

    @staticmethod
    def from_dict(config) -> DatasetConfig:
        return from_dict(DatasetConfig, config)


@dataclass
class DataConfig(Config):
    train: DatasetConfig = None
    val: DatasetConfig = None
    test: DatasetConfig = None

    @staticmethod
    def from_dict(config) -> DataConfig:
        train = DatasetConfig.from_dict(config['train']) if 'train' in config else None
        val = DatasetConfig.from_dict(config['val']) if 'val' in config else None
        test = DatasetConfig.from_dict(config['test']) if 'test' in config else None

        return DataConfig(
            train, val, test
        )


@dataclass
class ServerReplayConfig(Config):
    server_iterations: int
    optimizer_config: OptimizerConfig

    @staticmethod
    def from_dict(config) -> ServerReplayConfig:
        return ServerReplayConfig(
            config['server_iterations'],
            OptimizerConfig.from_dict(config['optimizer_config'])
        )


@dataclass
class RLConfig(Config):
    marginal_update_RL: bool = False
    RL_path: str = None
    model_descriptor_RL: str = None
    network_params: list = None
    initial_epsilon: float | None = None
    final_epsilon: float | None = None
    epsilon_gamma: float | None = None
    max_replay_memory_size: int | None = None
    minibatch_size: int | None = None
    gamma: float | None = None
    optimizer_config: OptimizerConfig = None
    annealing_config: AnnealingConfig = None

    @staticmethod
    def from_dict(config) -> RLConfig:
        result = RLConfig()
        for k in config:
            if k == 'optimizer_config':
                result.optimizer_config = OptimizerConfig.from_dict(config[k])
            elif k == 'annealing_config':
                result.annealing_config = AnnealingConfig.from_dict(config[k])
            else:
                setattr(result, k, config[k])
        return result


@dataclass
class ServerConfig(Config):
    resume_from_checkpoint: bool = False
    max_iteration: int | None = None
    num_clients_per_iteration: int | None = None
    optimizer_config: OptimizerConfig = None
    annealing_config: AnnealingConfig = None
    val_freq: int | None = None
    rec_freq: int | None = None
    initial_val: bool = False
    initial_req: bool = False
    wantRL: bool = False
    data_config: DataConfig = None
    type: str = None
    aggregate_median: str = None
    weight_train_loss: str = None
    softmax_beta: float | None = None
    initial_lr_client: float | None = None
    lr_delay_factor: float | None = None
    best_model_criterion: str = None
    server_replay_config: ServerReplayConfig = None

    @staticmethod
    def from_dict(config) -> ServerConfig:
        result = ServerConfig()

        for k in config:
            if k == 'optimizer_config':
                result.optimizer_config = \
                    OptimizerConfig.from_dict(config[k])
            elif k == 'annealing_config':
                result.annealing_config = \
                    AnnealingConfig.from_dict(config[k])
            elif k == 'data_config':
                result.data_config = \
                    DataConfig.from_dict(config[k])
            elif k == 'server_replay_config':
                result.server_replay_config = \
                    ServerReplayConfig.from_dict(config[k])
            else:
                setattr(result, k, config[k])
        return result


@dataclass
class ClientConfig(Config):
    meta_learning: str = None
    stats_on_smooth_grad: bool = False
    ignore_subtask: bool = False
    num_skips_threshold: int | None = None
    copying_train_data: bool = False
    do_profiling: bool = False
    data_config: DataConfig = None
    type: str = None
    meta_optimizer_config: OptimizerConfig = None
    optimizer_config: OptimizerConfig = None
    annealing_config: AnnealingConfig = None

    @staticmethod
    def from_dict(config) -> ClientConfig:
        result = ClientConfig()
        for k in config:
            if k == 'data_config':
                result.data_config = DataConfig.from_dict(config[k])
            elif k == 'meta_optimizer_config':
                result.meta_optimizer_config = \
                    OptimizerConfig.from_dict(config[k])
            elif k == 'optimizer_config':
                result.optimizer_config = \
                    OptimizerConfig.from_dict(config[k])
            elif k == 'annealing_config':
                result.annealing_config = \
                    AnnealingConfig.from_dict(config[k])
            else:
                setattr(result, k, config[k])
        return result


@dataclass
class FLUTEConfig(Config):
    model_config: ModelConfig = None
    dp_config: PrivacyConfig = None
    privacy_metrics_config: PrivacyMetricsConfig = None
    server_config: ServerConfig = None
    client_config: ClientConfig = None

    @staticmethod
    def from_dict(config) -> FLUTEConfig:
        dp_config = \
            PrivacyConfig.from_dict(config['dp_config']) \
            if 'dp_config' in config else None

        priv_metrics_config = \
            PrivacyMetricsConfig.from_dict(config['privacy_metrics_config']) \
            if 'privacy_metrics_config' in config else None

        return FLUTEConfig(
            ModelConfig.from_dict(config['model_config']),
            dp_config, priv_metrics_config,
            ServerConfig.from_dict(config['server_config']),
            ClientConfig.from_dict(config['client_config'])
        )
