# FLUTE

Welcome to FLUTE (Federated Learning Utilities for Testing and Experimentation), a platform for conducting high-performance federated learning simulations.

## Features

FLUTE is a pytorch-based orchestration environment enabling GPU or CPU-based FL simulations.  The primary goal of FLUTE is to enable researchers to rapidly prototype and validate their ideas.  Features include:

- large scale simulation (millions of clients, sampling tens of thousands per round)
- multi-GPU and multi-node orchestration
- local or global differential privacy
- model quantization
- a variety of standard optimizers and aggregation methods
- most model types including CNNs, RNNs, and Huggingface Transformers.
- extensibility, enabling new models, dataloaders, optimizers, and aggregators.
- local or cloud-based job staging using AzureML

## Benchmarking

The following common tasks were used to evaluate the performance in speed/memory utilization of FLUTE compared with the most representative simulation platforms based on their number of starts on GitHub: FedML 0.7.303 and Flower 1.0.0. 

|Task|Data Set|Model|Algorithm|# Clients|Clients per round|Batch Size|Client Optimizer|lr|Epochs|# Rounds|Test Freq|
|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|:----|
|CV|MNIST|LR|FedAvg|1000|10|10|SGD|0.03|1|100|20|
|CV|Federated EMNIST|CNN (2 Conv + 2 FC)|FedAvg|3400|10|20|SGD|0.1|1|1500|50|
|CV|FED_CIFAR-100|ResNet-18+group normalization|FedAvg|500|10|20|SGD|0.1|1|4000|50|
|NLP|Shakespeare|RNN (2 LSTM + 1 FC)|FedAvg|715|10|4|SGD|0.8|1|1200|50|

### FedML Comparison

This comparison was carried out using Parrot (Simulator) on version 0.7.303 at commit ID [8f7f261f](https://github.com/FedML-AI/FedML/tree/8f7f261f44e58d0cb5a416b0d6fa270b42a91049). Showing that in some cases FLUTE can outperform 43x faster.

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

You can find the examples above in [experiments](experiments).

### Flower Comparison

This comparison was carried out using Flower (Simulator) on version 1.0.0 at commit ID [4e7fad9](https://github.com/adap/flower/tree/4e7fad99389a5ee511730841b61f279e3359cb16) with the [lr_mnist](experiments/cv_lr_mnist/) task. Showing that in some cases FLUTE can outperform 53x faster.

```
 ________________________________________________
|        |    Flower (Ray)   | FLUTE (NCCL/Gloo) |
|        | Acc |    Time     | Acc |    Time     |
|--------|-----|-------------|-----|-------------|
| CPU    | ~80 |   00:30:14  | ~80 |   00:03:20  |
| GPU 2x | ~80 |   01:21:44  | ~80 |   00:01:31  |
| GPU 4x | ~79 |   00:56:45  | ~81 |   00:01:26  |
 ------------------------------------------------
```

You can find the example above in the [cv_lr_mnist](experiments/cv_lr_mnist/) folder.

## Quick Start

Install the requirements stated inside of `requirements.txt`. Ideally this sould be done inside of a virtual environment, for instance, using Anaconda.

```
conda create -n FLUTE python==3.8
pip install -r requirements.txt
```

FLUTE uses torch.distributed API as its main communication backbone, supporting three built-in backends. For more information please refer to [Distributed Communication Package](https://pytorch.org/docs/stable/distributed.html). Therefore, we highly suggest to use NCCL backend for distributed GPU training and Gloo for distributed CPU training. There is no `setup.py` as FLUTE is not currently distributed as a package, but instead meant to run from the root of the repository.

After this initial setup, you can use the data created for the integration test for a first local run. Note that this data needs to be download manually inside the `testing` folder, for more instructions please look at [the README file inside `testing`](testing/README.md).

```
python -m torch.distributed.run --nproc_per_node=3 e2e_trainer.py -dataPath ./testing -outputPath scratch -config testing/hello_world_nlg_gru.yaml -task nlg_gru -backend nccl
```

This config uses 1 node with 3 workers (1 server, 2 clients). The config file `testing/hello_world_nlg_gru.yaml` has some comments explaining the major sections and some important details; essentially, it consists in a very short experiment where a couple of iterations are done for just a few clients. A `scratch` folder will be created containing detailed logs.

## Documentation

Online documentation is available at https://microsoft.github.io/msrflute/

Locally, the documentation is inside the `doc/sphinx` folder. To build the docs on Linux:

```
$ pip install sphinx
$ cd doc/sphinx
$ make html
```

On Windows, you can use the `make.bat` script.  It may be necessary to `export PYTHONPATH=../../` for sphinx to find the code.

## Architecture

The core client/server training code is inside the `core` folder.

- Server-side federation and global DP application takes place in `server.py`, more specifically in the `OptimizationServer.train()` method.
- Client-side training updates take place in the static method `Client.process_round()`, inside `client.py`.

General FL orchestration code is in `federated.py`, but for most hub and spoke federation scenarios you won't need to touch this (unless you want to invest in optimizing server-client calls, which would be great!). Note that FLUTE does not implement secure aggregation since this is primarily a security feature for production scenarios; contributors are invited to add it for experimentation purposes.

The primary entry point for an experiment is in the script `e2e_trainer.py`. Primary config scripts for experiments are in `configs`. For instance, a basic training scenario for a next-word prediction task is set up in `hello_world_nlg_gru_json.yaml`.

Privacy accounting is expensive so the main parameters are logged and the actual accounting can be done offline. RDP privacy accounting is in `extensions/privacy/analysis.py`. A better accounting method is in the `dp-accountant` submodule.

## Customization

See `experiments` folder for illustrations of how dataloaders and models are customized. In order to in include a new experiment, the new scenario must be added following the same folder structure as `nlg_gru` and `mlm_bert`, naming the folder with the task.

## Experiments

Experiments are defined by YAML files, examples are provided in the `configs` folder. These can be run either locally or on AzureML.

For running experiments on AzureML, the CLI can help. You should first [install the CLI](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli) (make sure you have v2) and [create a resource group and workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace-cli?tabs=createnewresources%2Cvnetpleconfigurationsv1cli). You can then create a compute cluster, type `az ml compute create -h` for more info. Afterwards, you should write an YAML file with instructions for the job; we provide a simple example below

```yaml
experiment_name: basic_example
description: Basic example of AML config for submitting FLUTE jobs
code:
  local_path: .
compute: azureml:Test
environment:
  image: pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel
inputs:
  data:
    folder: azureml://datastores/data/paths/cifar
    mode: rw_mount
command: >
  apt -y update &&
  apt -y install openmpi-bin libopenmpi-dev openssh-client &&
  python3 -m pip install --upgrade pip &&
  python3 -m pip install -r requirements.txt &&
  python -m torch.distributed.run --nproc_per_node=4 e2e_trainer.py
  -outputPath=./outputs
  -dataPath={inputs.data}
  -task=classif_cnn
  -config=./experiments/classif_cnn/config.yaml
  -backend=nccl
```

You should replace `compute` with the name of the one you created before, and adjust the path of the datastore containing the data -- in the example above, we created a datastore called `data` and added to it a folder called `cifar`, which contained the two HDF5 files. The command passed above will install dependencies and then launch a distributed job with 4 threads, for the experiment defined in `experiments/classif_cnn`. Details on how to run a job using the AzureML CLI are given [in its documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli), but typically it suffices to set up the environment and type `az ml job create -f <name-of-the-yaml-file>`. In the same page of the documentation, you can also find more info about how to set up the YAML file above, in case other changes are needed.

Note that the `local_path` above is relative to the location of the YAML file, so setting it to `.` assumes it is in the same folder as `e2e_trainer.py`. All files on this folder will be uploaded to Azure, including hidden folders such as `.git`, so make sure to temporarily get rid of large files and folders that are not needed.

After launching the experiment, you can follow it on AzureML Studio, which prints logs, plots metrics and makes the output easily available after the experiment is finished.

## Privacy Accounting

Accounting is expensive, so we log all the privacy parameters so that accounting can be run offline. Best run on a Linux box with a GPU.
In particular, we use a DP accountant from another Microsoft repository, which is included in ours as a submodule. For using this accountant, just follow the instructions below:

```
$ git submodule update --init --recursive
$ cd utils
$ cd dp-accountant
$ python setup.py install
$ ./bin/compute-dp-epsilon --help
usage: compute-dp-epsilon [-h] -p SAMPLING_PROBABILITY -s NOISE_MULTIPLIER -i ITERATIONS -d DELTA
```
## Third Party Notice

This software includes the files listed below from the Huggingface/Transformers Library (https://github.com/huggingface/transformers) as part of task performance and preprocessing pretrained models.

    experiments/mlm_bert
    └── utils
        ├── trainer_pt_utils.py
        └── trainer_utils.py

This software includes the file extensions/privacy/analysis.py from the Tensorflow/Privacy Library (https://github.com/tensorflow/privacy) as part of Renyi Differential Privacy implementation.

This software includes the script testing/build_vocab.py from LEAF Library (https://github.com/TalwalkarLab/leaf) to create the vocabulary needed to run a testing job. 

This software includes the model implementation of the example ECG Classification | CNN LSTM Attention Mechanism from Kaggle Competition (https://www.kaggle.com/polomarco/ecg-classification-cnn-lstm-attention-mechanism) to reproduce the [ecg_cnn](experiments/ecg_cnn/model.py) experiment.

This software includes the model implementation of the FedNewsRec repository (https://github.com/taoqi98/FedNewsRec)| Code from the paper "Privacy-Preserving News Recommendation Model Learning" (https://arxiv.org/abs/2003.09592) ported to PyTorch framework to reproduce the [fednewsrec](experiments/fednewsrec/model.py) experiment.
For more information about third-party OSS licence, please refer to [NOTICE.txt](NOTICE.txt).

This software includes the Data Augmentation scripts of the Fast AutoAugment repository (https://github.com/kakaobrain/fast-autoaugment) to preprocess the data used in the [semisupervision](experiments/semisupervision/dataloaders/cifar_dataset.py) experiment.
## Support

You are welcome to open issues on this repository related to bug reports and feature requests.

## Contributing

Contributions are welcomed and encouraged. For details on how to contribute, please see [CONTRIBUTING.md](CONTRIBUTING.md).


