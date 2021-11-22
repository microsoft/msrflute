# Simple example of a NLG task on Reddit Dataset

Instructions on how to run the experiment, given below.

## Preparing the data

Right now FLUTE expects data to be provided either in JSON or HDF5 formats. It
should be made data-agnostic in the near future, but at this moment we need to do some 
preprocessing before handling the data on the model. For this experiment, we can run the 
script located in `testing/create_data.py` as follows:

```code
    python create_data.py -e nlg
```
to download mock data already preprocessed. A new folder `mockup` will be generated 
inside `testing` with all data needed for a local run.

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
script using MPI, you can use as example the following line:

```code
    mpiexec -n 3 python e2e_trainer.py -dataPath .\testing\mockup\ -outputPath scratch -config .\testing\configs\hello_world_local.yaml -task nlg_gru
```

For submitting jobs in Azure ML, we have included the instructions in the `Experiments` 
section of the main `README.md`.