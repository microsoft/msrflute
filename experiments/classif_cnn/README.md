# Simple example of a CNN on CIFAR-10

Our objective here is to bring a simple experiment from the Pytorch tutorials,
more specifically the one in https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py,
and convert it to FLUTE. Instructions on how to do this are given below.

An adapted version of the tutorial above is provided in the
`utils/centralized_training.py` script.

## Preparing the data

Right now FLUTE expects data to be provided either in JSON or HDF5 formats. It
should be made data-agnostic in the near future, but right now we need to
convert the data to either of these formats. In our case, we can use the script
`utils/download_and_convert_data.py` to do that for us; a HDF5 file will be
generated.

## Specifying the model

Next, we prepare the model. The `model.py` file contains two classes: one is the
`Net` class already contained in the original script, and the other, a class
called `CNN` which effectively wraps `Net`. Importantly, the `CNN` class defines
two methods: `loss` and `inference`; both perform forward steps and then perform
additional computations, in particular, the former executes the loss' evaluation,
and the latter the metrics' computation. The format of the inputs and outputs
should be the same as in this example.

## Specifying dataset and dataloaders

Inside the `dataloaders` folder, there are two files: `text_dataset.py` and
`text_dataloader.py` (the word "text" is used to mimic the other datasets, even
though in practice this loads images -- this will be changed in the future).
Both inherit from the Pytorch classes with same name.

The dataset should be able to access all the data, which is stored in the
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
script using MPI (don't forget to first run
`utils/download_and_convert_data.py`):

```
mpiexec -n 4 python e2e_trainer.py -dataPath experiments/classif_cnn/utils/data -outputPath scratch -config experiments/classif_cnn/config.yaml -task classif_cnn
```

The `dataPath`, `outputPath` and `config` arguments should just specify the
respective files or folders, as in the example above -- in this case, a folder
called `scratch` will be created containing logs and checkpoints. The task
should be the name of the folder insider `experiments`.

Following what is specified in the config file, the experiment will run for
2000 rounds, and during each of them 10 clients will be selected at random,
each of whom has 50 samples. It is more or less the same, then, as the 2
epochs in the centralized training, except that clients are selected at
random so we might not see all of them.