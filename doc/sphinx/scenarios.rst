Adding New Scenarios
====================

Data Preparation
------------

At this moment FLUTE only allows JSON and HDF5 files, and requires an specific formatting for the training data. Here is a sample data blob for language model training.

.. code:: json

    {
        "users": ["bert","elmo"],
        "user_data": {
            "bert": {"x": ["my name is Bert.", "I live with Ernie."]},
            "elmo": {"x": ["Big Bird is my friend."]}
        },
        "num_samples": [2, 1]
    }

The blob consists of three fields:

    * ``users``: indicates a unique id for each user in the training data.  Users are sampled uniformly to create client tasks during training. There could be many more users than client tasks per round or even over all client tasks over all rounds. 
    * ``num_samples`` : indicates the number of samples for each user, in the same order as ``users`` list.  That is, for any index ``i`` in ``range(len(data['users']))``: 
    * ``user_data``: contains user-indexed training data. Each user's data is a dictionary of the form ``{"x": [list of examples]}``.  

If labels are needed by the task, ``user_data_label`` will be required by FLUTE with the user-indexed labels. The format should be similar to ``user_data`` where each user's label is a dictionary of the form ``{"x": [list of labels]}`` as follows:

.. code:: json

    "user_data_label": {
        "bert": {"x": [ 0 , 1 ]},
        "elmo": {"x": [ 0 ]}
        }

.. note::

    Test and validation data is formatted similarly.

.. note::

    Test/validate data is dispatched to workers by partitioning on users. If your test data isn't user-partitioned, we recommend partitioning it uniformly using some dummy user ids.

Add the model to FLUTE
--------------

FLUTE requires the model declaration framed in PyTorch, with the following functions:

    * __init__: model definition
    * loss: computes the loss used for training rounds
    * inference: computes the metrics used during evaluation rounds
    * set_eval: brings the model into evaluation mode
    * set_train: brings the model into training mode

Please see the example provided below:

.. code:: python

    class CNN(nn.Module):
    '''This is a PyTorch model with some extra methods'''

    def __init__(self, model_config):
        super().__init__()
        self.net = Net()

    def loss(self, input: torch.Tensor) -> torch.Tensor:
        '''Performs forward step and computes the loss'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)
        return F.cross_entropy(output, labels.long())

    def inference(self, input):
        '''Performs forward step and computes metrics'''
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        features, labels = input['x'].to(device), input['y'].to(device)
        output = self.net.forward(features)

        n_samples = features.shape[0]
        accuracy = torch.mean((torch.argmax(output, dim=1) == labels).float()).item()
        f1 = f1_score(labels.cpu(), torch.argmax(output, dim=1).cpu(), average='micro')

        return {'output':output, 'val_acc': accuracy, 'batch_size': n_samples, 'f1_score':f1}
        
    def set_eval(self):
        '''Bring the model into evaluation mode'''
        self.eval()

    def set_train(self):
        '''Bring the model into training mode'''
        self.train()

The Inference function must return a dictionary with the metrics that will be computed, as follows:

    .. code:: bash
        
        { "output": loss, "val_acc": accuracy, "batch_size": batch_size}

.. note:: FLUTE requires at least loss, accuracy and batch size for the dictionary returned by inference(). More metrics can be added just by includding a new key in the same dictionary.

Once the model is ready, all mandatory files must be in a single folder inside /experiments. Please adjust your files with the following naming structure so FLUTE can be able to find all the scripts needed.

.. code-block:: bash

    task_name
        |---- dataloaders
              |---- text_dataloader.py
        |---- utils
              |---- utils.py
        |---- model.py
        |---- config.yaml
        |---- custom_metrics.py (optional)
        |---- README.txt

.. note:: In case you need to import a module that has not been considered in FLUTE, this can be added in requirements.txt 

.. note:: All files must contain only absolute imports, in order to avoid issues when running.

Create the configuration file
---------------------------------

The configuration file will allow you to specify the setup in your experiment, such as the optimizer, learning rate, number of clients and so on. FLUTE requires the following 5 sections:

    * model_config: path an parameters (if needed) to initialize the model.
    * dp_config: differential privacy setup.
    * privacy_metrics_config: for cache data to compute additional metrics.
    * server_config: determines all the server-side settings.
    * client_config: dictates the learning parameters for client-side model updates. 

The blob below indicates the basic parameters required by FLUTE to run an experiment:

.. code:: yaml 

    model_config:
        model_type: CNN                                    # Class name in model.py 
        model_folder: experiments/classif_cnn/model.py     # Relative path to the model declaration

    dp_config:
        enable_local_dp: false                             # DP disabled

    privacy_metrics_config:
        apply_metrics: false                               # Privacy metrics disabled

    strategy: DGA                                          # Federated optimizar (DGA or FedAvg)

    server_config:   
        wantRL: false                                      # Whether to use RL-based meta-optimizers
        resume_from_checkpoint: false                      # Restart from checkpoint if file exists
        do_profiling: false                                # Run profiler and compute runtime metrics
        optimizer_config:                                  # Optimizer used to update the global model
            type: sgd
            lr: 1.0
        annealing_config:                                  # Annealer for the learning rate
            type: step_lr
            step_interval: epoch
            gamma: 1.0
            step_size: 100
        val_freq: 50                                       # Validation rounds frequency
        rec_freq: 100                                      # Testing rounds frequency
        initial_val: true                                  # Enable initial validation round
        initial_rec: true                                  # Enable initial testing round
        max_iteration: 2000                                # Total of iteration rounds
        num_clients_per_iteration: 10                      # Clients per interation
        data_config:                                       # Information for the test/val dataloaders
            val:
                batch_size: 10000
                loader_type: text
                val_data: test_data.hdf5
            test:
                batch_size: 10000
                loader_type: text
                test_data: test_data.hdf5
        type: model_optimization                           # Server type (model_optimization is the only available for now)
        aggregate_median: softmax                          # How aggregations weights are computed
        initial_lr_client: 0.001                           # Learning rate used on optimizer
        lr_decay_factor: 1.0                               # Decay factor for LR
        weight_train_loss: train_loss                      # Determines how each client's weight is computed (e.g. grad_mean_loss, train_loss)
        best_model_criterion: f1_score                     # Determines the best model based on minimal loss, for checkpointing
        fall_back_to_best_model: false                     # If a model degrades, use the previous best model
        softmax_beta: 1.0                                  # Beta value to use for the softmax DGA

    client_config:
        do_profiling: false                                # Run profiling and compute runtime metrics
        ignore_subtask: false                              # Determines which model loss to use. In most cases just set to False.
        data_config:                                       # Information for the train dataloader
            train:
                batch_size: 4
                loader_type: text
                list_of_train_data: train_data.hdf5
                desired_max_samples: 50000
        optimizer_config:                                  # Optimizer used by the client
            type: sgd
            lr: 0.001                                      # This is overridden by `initial_lr_client`
            momentum: 0.9
        type: optimization                                 # The type of client (always set "optimization for now")

.. note:: Documented templates for all the options available in the configuration files are provided inside configs folder.
