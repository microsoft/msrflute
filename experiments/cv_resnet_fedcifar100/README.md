## FedML Benchmark

### Examples

The example in this folder was taken from [FedML](https://github.com/FedML-AI/FedML/tree/master/python/examples/simulation/mpi_fedavg_datasets_and_models_example) repository on its release 0.7.300, using the configuration suggested on their
[benchmarking results](https://doc.fedml.ai/simulation/benchmark/BENCHMARK_MPI.html) for MPI-Based Federated Learning (fastest on this version).

### Data

FLUTE will automatically download the data used for this example, otherwise you can use the scripts provided [here](https://github.com/FedML-AI/FedML/tree/master/python/fedml/data) for each independent dataset in the FedML GitHub repository. 

### Run

If you downloaded the data manually, make sure that the variable `data_cache_dir` has been updated inside `preprocess.py`. Later, you can run the experiment as follows:

```code

    python -m torch.distributed.run  --nproc_per_node=4  e2e_trainer.py -dataPath ~/data -outputPath ~/outputTest  -config ./experiments/cv_resnet_fedcifar100/config.yaml -task cv_resnet_fedcifar100 -backend nccl
```

### Results

This comparison was carried out using Parrot (Simulator) on version 0.7.303 at commit ID [8f7f261f](https://github.com/FedML-AI/FedML/tree/8f7f261f44e58d0cb5a416b0d6fa270b42a91049). 
```
 _____________________________________________________________________________
|                    |   FedML (MPI) - Fastest   |   FLUTE (NCCL)  - Fastest  |
| Task               | Acc | Time     | GPU Mem  | Acc | Time     | GPU Mem   |
|--------------------|-----|----------|----------|-----|----------|-----------|
| LR_MNIST           | ~81 | 00:03:09 | ~3060 MB | ~81 | 00:01:35 | ~1060 MB  |
| CNN_FEMNIST        | ~83 | 05:49:52 | ~5180 MB | ~83 | 00:08:22 | ~1770 MB  |
| RESNET_FEDCIFAR100 | ~34 | 15:55:36 | ~5530 MB | ~33 | 01:42:01 | ~1900 MB  |
| RNN_FEDSHAKESPEARE | ~57 | 06:46:21 | ~3690 MB | ~57 | 00:21:50 | ~1270 MB  |
 -----------------------------------------------------------------------------
```
### FedML Configuration file

In order to reproduce this experiment in FedML please use the setup below. 

```yaml

common_args:
  training_type: "simulation"
  random_seed: 0

data_args:
  dataset: "fed_cifar100"
  data_cache_dir: ~/fedml_data
  partition_method: "hetero"
  partition_alpha: 0.5

model_args:
  model: "resnet18_gn"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list: "[]"
  client_num_in_total: 500
  client_num_per_round: 10
  comm_round: 4000
  epochs: 1
  batch_size: 20
  client_optimizer: sgd
  learning_rate: 0.1
  weight_decay: 0.001

validation_args:
  frequency_of_the_test: 50

device_args:
  worker_num: 10
  using_gpu: true
  gpu_mapping_file: config/fedcifar100_resnet18/gpu_mapping.yaml
  gpu_mapping_key: mapping_default # [3, 3, 3, 2]

comm_args:
  backend: "MPI"
  is_mobile: 0

```