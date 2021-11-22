# Simple example of a MLM task on Reddit Dataset

Instructions on how to run the experiment, given below.

## Preparing the data

Right now FLUTE expects data to be provided either in JSON or HDF5 formats. It
should be made data-agnostic in the near future, but at this moment we need to do some 
preprocessing before handling the data on the model. For this experiment, we can run the 
script located in `testing/create_data.py` as follows:

```code
    python create_data.py -e mlm
```
to download mock data already preprocessed. A new folder `mockup` will be generated 
inside `testing` with all data needed for a local run.

A couple of scripts are provided in `utils/preprocessing` for preprocessing .tsv files
in case you want to use your own data.

## Creating a config file

All the parameters of the experiment are passed in a YAML file. An example is
provided in `configs/hello_world_mlm_bert_json.yaml` with the suggested parameters
to do a simple run for this experiment. Make sure to point your training files at
the fields: train_data, test_data and val_data inside the config file.

## Running the experiment

For submitting jobs in Azure ML, we have included the instructions in the `Experiments` 
section of the main `README.md`.