## Setup Instructions for Pytest

1. In order to run test_e2e_trainer.py, we need a dataset for test, train an validation. For demonstrative purposes, we are using as example the Reddit dataset already processed by LEAF, that can be downloaded here: https://github.com/TalwalkarLab/leaf/tree/master/data/reddit (Setup instructions, point I)
2. Create the following folder structure mockup/data inside /testing. Make sure that inside /data the files needed are divided by test, train and val folders.
3. Run ```python create_data.py``` to adjust the data as per FLUTE requirements.
4. Run ```pytest -v``` to test the program.

## Troubleshooting

In case you encounter any issue while running test_e2e_trainer.py, please check the following points:

1. The file structure matches the path provided in testing/configs/hello_world_local.yaml
2. Timeout in test_e2e_trainer.py is proportional to the amount of data using for the training.
3. Command line used in test_e2e_trainer.py is commented according to the OS in use.