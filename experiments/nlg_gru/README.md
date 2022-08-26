# Simple example of a NLG task on Reddit Dataset

Instructions on how to run the experiment, given below.

## Preparing the data

For this experiment, we can create a dummy dataset by running the 
script located in `testing/create_data.py` as follows:

```code
    python create_data.py -e nlg
```

A couple of scripts are provided in `utils/preprocessing` for preprocessing .tsv files
in case you want to use your own data.

## Creating a config file

All the parameters of the experiment are passed in a YAML file. An basic example is 
provided in `configs/hello_world_nlg_gru_json.yaml` with the suggested 
parameters for local runs. 

The example provided above is for running json files. If you want to try with HDF5 files
make sure to use the script `utils/preprocessing/from_json_to_hdf5.py` to convert the mock
data to HDF5 format.

## Running the experiment

Finally, to launch the experiment locally , it suffices to launch the `e2e_trainer.py`
script using torch.distributed , you can use as example the following line:

```code
    python -m torch.distributed.run --nproc_per_node=3 e2e_trainer.py -dataPath .\testing\mockup\ -outputPath scratch -config .\testing\configs\hello_world_nlg_gru.yaml -task nlg_gru -backend nccl
```

For submitting jobs in Azure ML, we have included the instructions in the `Experiments` 
section of the main `README.md`.