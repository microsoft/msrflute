# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2021-11-22

We're super excited to announce FLUTE: Federated Learning Utilities for Testing and Experimentation, a platform for conducting high-performance federated learning simulations!

This first release fully focuses on implementing fast prototyping to validate different CL scenarios 
in an Federated environment.

### Features

- large scale simulation (millions of clients, sampling tens of thousands per round).
- multi-GPU and multi-node orchestration backed up by MPI.
- local or global differential privacy.
- model quantization.
- a variety of standard optimizers and aggregation methods.
- most model types including CNNs, RNNs, and Huggingface Transformers.
- extensibility, enabling new models, dataloaders, optimizers, and aggregators.
- local or cloud-based job staging using AzureML.


## [1.0.0] - 2022-08-29

This release contain major changes in the communication backbone , in order
to run previous experiments you have already integrated in FLUTE, please make sure
to use `torch.distributed` instead of `MPI `to launch the jobs. For more documentation
about the new command, please refer to the [README](README.md).


### New features

- 🏎 Better performance: Support for NCCL and Gloo as backend communication protocols. 
  - Improvements in GPU utilization and overall communication speed (on the order of minutes!) for projects with huge models and datasets.
- 🌟 Remove file type dependency on client.py, now FLUTE can receive any kind of dataset and even download the data on-the-fly. The data intantiation is completely under control of each task dataset.
  - In older versions FLUTE only allowed `json` and `hdf5` files, so the client could recognize it.
- 🌟 Abstract classes for new models/dataloaders.
- 🌟 Allows Federated Learning with Personalization. 
  - Personalization allows you to leverage each client local data to obtain models that are better adjusted to their own data distribution. You can run the `cv` task in order to try out this feature.
