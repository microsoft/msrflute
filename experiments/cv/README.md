# Simple example of ResNet model using personalization

Our objective here is to bring a simple experiment of Computer Vision task,
and convert it to FLUTE using the personalization feature. Instructions on 
how to do this are given below.

## Preparing the data

In this experiment we are making use of the CIFAR10 Dataset from torchvision, 
initializated in `data.py`, which is wrapped by FLUTE Base Dataset.

## Specifying the model

Next, we prepare the model. The `model.py` file contains different classes than
can be used for this experiment. However, for this example we are using the
`ResNet` class . Importantly, the `ResNet` class inheeits from `Base Model` 
declared in `core/model.py` and defines two methods: `loss` and `inference`; 
both perform forward steps and then perform additional computations, in particular, 
the former executes the loss' evaluation, and the latter the metrics' computation. 
The format of the inputs and outputs should be the same as in this example.

## Specifying dataset and dataloaders

Inside the `dataloaders` folder, there are two files: `dataset.py` and
`dataloader.py`. Both inherit from the base classes declared in `core`
folder, that under the hood inhereit from Pytorch classes with same name.

The dataset should be able to access all the data, and store it in the
attributes `user_list`, `user_data`, `user_data_labels` and `num_samples` (user
names, user features, user labels if the problem is supervised, and number of
samples for each user, respectively). These attributes are required to have
these exact names. Otherwise, it should also be able to access the examples of a
specific user, which id is passed during initialization via the `user_idx`
argument.

The dataloader is simpler, and essentially just instantiates the dataset and
creates batches with a specific format.

## Creating a config file

All the parameters of the experiment are passed in a YAML file. A documented
example is provided in `config.yaml`.

## Running the experiment

Finally, to launch the experiment, it suffices to launch the `e2e_trainer.py`
script using torch.distributed.

```
python -m torch.distributed.run --nproc_per_node=4 e2e_trainer.py -dataPath ./ -outputPath scratch -config experiments/classif_cnn/config.yaml -task cv -backend gloo
```

The `dataPath`, `outputPath` and `config` arguments should just specify the
respective files or folders, as in the example above -- in this case, `dataPath` 
can be any path given that data is being downloaded on-the.fly. A folder
called `scratch` will be created containing logs and checkpoints. The task
should be the name of the folder insider `experiments`.
