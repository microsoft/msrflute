# Simple example of a MLM task on Reddit Dataset

Instructions on how to run the experiment, given below.

## Preparing the data

For this experiment, we can create a dummy dataset by running the 
script located in `testing/create_data.py` as follows:

```code
    python create_data.py -e mlm
```

A couple of scripts are provided in `utils/preprocessing` for preprocessing .tsv files
in case you want to use your own data.

## Creating a config file

All the parameters of the experiment are passed in a YAML file. An example is
provided in `configs/hello_world_mlm_bert_json.yaml` with the suggested parameters
to do a simple run for this experiment. Make sure to point your training files at
the fields: list_of_train_data, test_data and val_data inside the config file.

## Running the experiment locally

Finally, to launch the experiment, it suffices to launch the `e2e_trainer.py`
script using torch.distributed:

```code
    python -m torch.distributed.run --nproc_per_node=2 .\e2e_trainer.py -dataPath data_folder -outputPath scratch -config configs\hello_world_mlm_bert_json.yaml -task mlm_bert -backend nccl
```

For submitting jobs in Azure ML, we have included the instructions in the `Experiments` 
section of the main `README.md`.