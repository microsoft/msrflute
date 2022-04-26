# Note this import requires python 3.7+
# Do we want to commit to this?
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import MutableMapping
from cerberus import Validator
from importlib.machinery import SourceFileLoader
from utils.utils import print_rank
from importlib.machinery import SourceFileLoader
import os


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
    """Base class for configuration classes."""
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
    """Base class for Model configurations

The model configuration specifies model architecture, parameters, and initialization settings.

Attributes:
    model_type (str): The class name of the model to instantiate. eg GRU.

    model_folder (str): The relative path to the model.py file where model_type is defined. eg experiments/nlg_gru/model.py

    pretrained_model_path (str): The path to the pretrained model.  If None, the model will be randomly initialized using the method defined in weight_init.

"""
    model_type: str = None
    model_folder: str = None
    pretrained_model_path: str = None

    @staticmethod
    def from_dict(config) -> ModelConfig:
        """Searches the model folder for config.py and if it is found the model config 
        is initialized from the class [model_type]Config"""
        cfg_path = os.path.dirname("./" + str(config['model_folder'])) + '/config.py'
        if os.path.exists(cfg_path):
            loader = SourceFileLoader('config', cfg_path).load_module()
            config_class = config['model_type'] + 'Config'
            try:
                config_type = getattr(loader, config_class)
                return from_dict(config_type, config)
            except AttributeError:
                print_rank(f"Config class {config_class} not found in {cfg_path}")
                raise
        else:
            print_rank(f"Warning: couldn't find {cfg_path}, falling back to dictionary.")
            return config
            

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
class BERTConfig(Config):
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
    """Privacy configuration

    The privacy configuration specified differential privacy settings for the model.
    The user can choose between local or global DP.  When local DP is enabled, a global
    epsilon can be computed by applying the RDP accountant (see extensions/privacy).
    The `eps` parameter is used to specify the privacy budget for local DP.  Conversely, when
    global DP is enabled, `eps` is ignored and `global_sigma` directly specifies the global
    Gaussian noise.   `max_grad` specifies the clipping parameter for local or global DP,
    `max_weight` specifies the clipping parameter for the local gradient aggregation weight
    (applies to softmax aggregation), and `weight_scaler` indicates how the aggregation weight
    is scaled before noise addition, and unscaled afterward. This enables a single eps/sigma
    parameter for both the gradient and its weight.

    Example:
       This example applies local DP with eps=1000. The global epsilon will be computing using Renyi DP accounting.

       .. code-block:: yaml

            dp_config:
                # Local dp clips and adds noise on the client and centrally accumulates the privacy budget.
                enable_local_dp: true
                eps: 100 # epsilon
                max_grad: 0.008  # max gradient
                # The max_weight and min_weight should be already scaled by weight_scaler
                # Because we scale down the weight using weight_scalar -> clip -> add noise -> scale back up.
                max_weight: 0.0001
                weight_scaler: 0.0001
                min_weight: 0.00009


    Attributes:
        enable_local_dp (bool): whether to enable local DP.

        enable_global_dp (bool): whether to enable global DP.

        eps (float): the privacy budget for local DP.

        delta (float): the privacy delta parameter for local DP.

        global_sigma (float): the global Gaussian noise for global DP.

        max_grad (float): the gradient clipping parameter.

        max_weight (float): the aggregation weight clipping parameter.

        weight_scaler (float): the aggregation weight scaling parameter.

        min_weight (float): the minimum per-gradient aggregation weight.

    """
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
    """Privacy metrics configuration

    This optional feature computes local privacy metrics for computed gradients,
    and optionally filters gradients based on estimated privacy loss.

    Attributes:
        apply_metrics (bool): whether to compute privacy metrics.

        apply_indices_extraction (bool): whether to attempt local data reconstruction.

        allowed_word_rank (int): threshold for successful reconstruction.

        apply_leakage_metric (bool): whether to compute a privacy leakage metric based on the ratio of perplexities before and after local training.

        max_leakage (float): the maximum allowed privacy leakage before filtering

        adaptive_leakage_threshold (float): if non-zero, compute an adaptive leakage threshold based on the previous round of training.  For example at 0.95, the max_leakage will be adjusted to reject 5% of gradients, based on the previous round of training.

        is_leakage_weighted (bool): scales the leakage by the maximum likelihood of the pre- and post- likelihood tensors. ie the worst-case leakage is weighted by the worst-case likelihood that we might encounter it.

        attacker_optimizer_config (OptimizerConfig): the optimizer configuration for the reconstruction attack.
    """
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
    """Optimizer configuration

    Pass any pytorch-supported optimizer configuration. The object should include
    a `type` field which indicates the pytorch optimizer type that should be invoked.
    This will be stripped from the object before being passed to the Optimizer's init.
    """
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
    """Learning rate annealing configuration


    Attributes:
        type (str): the type of annealing. Supported methods: :code:`step_lr`, :code:`multi_step_lr`, :code:`rampup-keep-expdecay-keep`, :code:`val_loss`.

        step_interval (str): the interval at which to step the learning rate. Supported intevals: :code:`epoch`, :code:`batch`.

        gamma (float): the learning rate decay factor.

        step_size (int): the interval between annealing operations.
    """
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
    """Data configurations

    Client and server configs may each contain a data config, consisting of train, test, and validate datasets.
    A typical configuration will define test and validate in the server data config, while the training data is defined in the client config.
    Optionally, the server can have a training config which defines server-side training data.


    Attributes:
        train (DatasetConfig): the training dataset configuration.

        val (DatasetConfig): the validation dataset configuration.

        test (DatasetConfig): the test dataset configuration.
    """
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
    """Server replay configuration

    When server-side training data is defined, this config defines how it is applied after each client training round.

    Attributes:
        server_iterations (int): the number of iterations to run over server-side training data for.

        optimizer_config (OptimizerConfig): the optimizer configuration to use for the server.
    """
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
    """Reinforcement learning configuration

    RL can be applied during dynamic gradient aggregation to speed up convergence. This configuration defines the settings for server-side RL to train the model for DGA.

    Attributes:
        marginal_update_RL (bool): whether to update the RL model when the loss is small.

        RL_path (str): the path to the RL model to train.

        RL_path_global (bool): whether the global training output path should be prepended to RL_path.

        model_descriptor_RL (str): string to append to the model filename.

        network_params (list): List of layer widths in the RL network. eg: 300,128,128,128,64,100

        initial_epsilon (float): the initial epsilon value for the epsilon-greedy policy.

        final_epsilon (float): the final epsilon value for the epsilon-greedy policy.

        epsilon_gamma (float): the decay rate for the epsilon-greedy policy.

        max_replay_memorize_size (int): the maximum number of samples to store in the replay memory.

        minibatch_size (int): the size of the minibatch to use for training.

        gamma (float): the discount factor for the RL model.

        optimizer_config (OptimizerConfig): the optimizer configuration to use for the RL model.

        annealing_config (AnnealingConfig): the annealing configuration to use for the RL model.


    """
    marginal_update_RL: bool = False
    RL_path: str = None
    RL_path_global: bool = False
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
    """Server configuration

    The server configuration defines the server-side settings.

    Attributes:
        resume_from_checkpoint (bool): whether to resume training from a checkpoint.

        max_iterations (int): the maximum number of iterations (federated training rounds) to run.

        num_clients (int): the number of clients to use per training round.

        optimizer_config (OptimizerConfig): the optimizer configuration to use server-side.

        annealing_config (AnnealingConfig): the learning rate annealing configuration to use server-side.

        val_freq (int): the number of iterations between validation evaluation runs.

        rec_freq (int): the number of iterations between test evaluation runs.

        initial_val (bool): whether to run validation before initiating training.

        initial_rec (bool): whether to run test before initiating training.

        wantRL (bool): whether to train the RL model.

        RL (RLConfig): the RL configuration to use if wantRL is True.

        data_config (DataConfig): the data configuration to use server-side.

        type (str): the type of server. Currently this parameter is ignored and OptimizationServer is always used. However there is some validation code that checks for one of the following values:

            - model_averaging
            - optimization
            - model_optimization
            - cluster_finetuning
            - cluster_parallel

        aggregate_median (str): the aggregation method to use (DGA softmax, or mean). Note that this only applies when the global aggregation strategy is DGA.

        weight_train_loss (str): when softmax DGA is enabled, what metric to use for weighting. One of

            - train_loss
            - mag_var_loss
            - mag_mean_loss

        softmax_beta (float): the beta value to use for the softmax DGA.

        max_weight (float): the maximum allowed client weight.

        initial_lr_client (float): the initial learning rate for each client.

        lr_decay_factor (float): the client learning rate decay factor.

        best_model_criterion (str): The metric to choose when resetting to the best model so far.

        server_replay_config (ServerReplayConfig): the server replay configuration to use for any server-side training.

    """
    resume_from_checkpoint: bool = False
    max_iteration: int | None = None
    num_clients_per_iteration: int | None = None
    optimizer_config: OptimizerConfig = None
    annealing_config: AnnealingConfig = None
    val_freq: int | None = None
    rec_freq: int | None = None
    initial_val: bool = True
    initial_rec: bool = True
    wantRL: bool = False
    RL: RLConfig = None
    data_config: DataConfig = None
    type: str = None
    aggregate_median: str = None
    weight_train_loss: str = None
    softmax_beta: float | None = None
    max_weight: float | None = None
    initial_lr_client: float | None = None
    lr_delay_factor: float | None = None
    best_model_criterion: str = 'loss'
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
            elif k == 'RL':
                result.RL = \
                    RLConfig.from_dict(config[k])
            else:
                setattr(result, k, config[k])
        return result


@dataclass
class ClientConfig(Config):
    """
    Client configuration

    The client configuration defines the client-side settings.

    Attributes:
        meta_learning (str): Set to 'basic'.  Currently ignored.

        stats_on_smooth_grad (bool): When true, gradient statistics are reset each round. Currently, it appears these statistics aren't used.

        ignore_subtask (bool): Used to determine which model loss to use. In most cases just set to False.

        num_skips_threshold (int): previously used to skip users, deprecated.

        copying_train_data (bool): has no effect.

        do_profiling (bool): whether to enable client-side profiling.

        data_config (DataConfig): the data configuration to use client-side.

        type (str): the type of client. Currently this parameter is ignored?

        meta_optimizer_config (OptimizerConfig): the optimizer configuration to use for meta-learning.

        optimizer_config (OptimizerConfig): the optimizer configuration to use for client-side training.

        annealing_config (AnnealingConfig): the learning rate annealing configuration to use client-side.
    """
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
    """
    FLUTEConfig represents the global configuration for a training job.

    Attributes:
        model_config (ModelConfig): the model configuration to use.

        dp_config (PrivacyConfig): differential privacy configuration.

        strategy (str): Aggregation strategy, eg DGA or FedAvg.

        server_config (ServerConfig): the server configuration to use.

        client_config (ClientConfig): the client configuration to use.

    """
    model_config: ModelConfig = None
    dp_config: PrivacyConfig = None
    privacy_metrics_config: PrivacyMetricsConfig = None
    strategy: str = None
    server_config: ServerConfig = None
    client_config: ClientConfig = None

    def validate(config):

        # Join paths in config file
        if config["server_config"]["wantRL"]:
            rl_path = config["server_config"]["RL"]["RL_path"]
            rl_path = os.path.join(config["output_path"],rl_path) if config["server_config"]["RL"].get("RL_path_global", True) \
                                                            else os.path.join(config["output_path"], config["experiment_name"],rl_path)

        if "pretrained_model_path" in config["model_config"]:
            config["model_config"]["pretrained_model_path"] = os.path.join(config["data_path"], config["model_config"]["pretrained_model_path"])

        for section in ["server_config", "client_config"]:
            for mode in ['test','val','train']:
                if mode in config[section]["data_config"] and "vocab_dict" in config[section]["data_config"][mode]:
                    config[section]["data_config"][mode]["vocab_dict"] = os.path.join(config['data_path'], config[section]["data_config"][mode]["vocab_dict"])
                
                # TODO: Remove BERT specific parameters
                if 'BERT' in config['model_config']:
                    if mode!= 'train':
                        config['server_config']['data_config'][mode]['model_name_or_path'] = config['model_config']['BERT']['model']['model_name']
                        config['server_config']['data_config'][mode]['process_line_by_line'] = config['model_config']['BERT']['model']['process_line_by_line']
                    else:
                        config['client_config']['data_config'][mode]['model_name_or_path'] = config['model_config']['BERT']['model']['model_name']
                        config['client_config']['data_config'][mode]['process_line_by_line'] = config['model_config']['BERT']['model']['process_line_by_line']
        return config

    @staticmethod
    def from_dict(config) -> FLUTEConfig:

        # Validate schema in config file
        schema = eval(open('./core/schema.py', 'r').read())
        v = Validator(schema)
        if not v.validate(config,schema):
            raise ValueError('Missing {} argumment in config file '.format(v.errors))
        
        # Normalize default values
        original_config = config
        config = v.normalized(config)

        for section in ['server_config', 'client_config']:
            for mode in config[section]['data_config'].keys():
                diff = config[section]['data_config'][mode].keys() - original_config[section]['data_config'][mode].keys()
                if len(diff) > 0:
                    print_rank("Assigning default values for: {} in [{}][{}][data_config]".format(diff, section, mode))
        
        dp_config = \
            PrivacyConfig.from_dict(config['dp_config']) \
            if 'dp_config' in config else None

        priv_metrics_config = \
            PrivacyMetricsConfig.from_dict(config['privacy_metrics_config']) \
            if 'privacy_metrics_config' in config else None

        strategy = config.get('strategy', 'DGA')

        return FLUTEConfig(
            ModelConfig.from_dict(config['model_config']),
            dp_config, priv_metrics_config, strategy,
            ServerConfig.from_dict(config['server_config']),
            ClientConfig.from_dict(config['client_config'])
        )
