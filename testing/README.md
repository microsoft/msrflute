## Information

The tests are designed to evaluate the operation of the tasks, not the performance. Therefore, we are using dummy data to run all tasks. In order to have ralistic results about the behaviour of each experiment, please follow the instructions provided in the README.md  file inside each experiment folder, for downloading the recommended datasets. 

## Setup Instructions for Pytest
1. Run create_data.py in order to download and preprocess the dummy training and testing datasets that will be used. Make sure to indicate the task name. The example below shows how to create the data for the ```nlg_gru``` task.

``` python
    python create_data.py --task nlg_gru
```
2. The script ```test_e2e_trainer.py``` is designed to run the test over all tasks, therefore you need to run Step 1 for each experiment first).
3. Run ```pytest -v -s``` to perfor the local test.
