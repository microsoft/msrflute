### Data

In order to run this experiment, you need to previously run the script [cifar_dataset.py](dataloaders/cifar_dataset.py) in order to download and preprocess the CIFAR100 dataset needed for this task. 

```code

    python experiments/semisupervision/dataloaders/cifar_dataset.py
    
```
### Run

Once the data has been downloaded, you can run the experiment as follows:

```code

    python -m torch.distributed.run --nproc_per_node=2  e2e_trainer.py -dataPath ~/data -outputPath ~/outputTest  -config ./experiments/semisupervision/config.yaml -task semisupervision -backend nccl
    
```
