### Data

In order to run this experiment, you need to previously download the MIND dataset [here](https://msnews.github.io/index.html) and the glove.840B.300d embbeding vector [here](https://nlp.stanford.edu/projects/glove/). Once you have the data, make sure to replace the `root_data_path` and `embedding_path` parameters inside [dataset.py](dataloaders/dataset.py) and [configuration file](config.yaml). The preprocessing steps will be done automatically by FLUTE once the jobs is launched.

### Run

Once the paths for the dataset and embedding have been updated, you can run the experiment as follows:

```code

    python -m torch.distributed.run  --nproc_per_node=4  e2e_trainer.py -dataPath ~/data -outputPath ~/outputTest  -config ./experiments/fednewsrec/config.yaml -task fednewsrec -backend nccl
    
```
### Results

- MIND_Large, 1500 rounds, 6 clients per round:

|Platform|AUC|MRR|nDCG5|nDCG10|
|:----|:----|:----|:----|:----|
|FedNews|0.54|0.23|0.25|0.32|
|FLUTE|0.58|0.24|0.26|0.33| 

