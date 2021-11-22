Adding New Scenarios
====================

Requirements
------------

Before adding a new scenario in FLUTE, make sure that your files comply with following:

* The model class has declared the functions: loss(), inference(), set_eval() and set_train()
* Inference function is used for testing and must return loss, accuracy and batch size.
* Raw data input must be stored in JSON or HDF5 files
* FLUTE assumes the following format from the text data
    
.. code-block:: bash

            {"num_samples": [sample_1, ......, sample_n],
              "users":[user_1, ......, user_n],
              "user_data": {
                              "user_1":{
                                          "x":[ .. data..,
                                              .....data_n..]
                                       },
                                       .......
                              "user_n":{
                                          "x":[ .. data..,
                                              .....data_n..]
                                       }
        
                           }
            }

.. note:: The list 'x' inside of the dictionary with the name of the user, can contain data or arrays.
Alternatively, instead of using a single-key dictionary, a list of lists might be assigned to each user.

Copy the files 
------------

All mandatory files must be inside a folder with the same name as the model in /models. Please adjust your files with the following 
naming structure so FLUTE can be able to find all the scripts needed.

.. code-block:: bash

    model_name
        |---- dataloaders
              |---- text_dataloader.py
        |---- utils
              |---- utils.py
        |---- model.py
        |---- README.txt

In case you need to import a module that has not been considered in FLUTE, this can be added in requirements.txt 

.. note:: All files must contain only absolute imports, in order to avoid issues when running your job.

Create a model configuration file
------------
Once your model has been added into FLUTE, it is necessary to create a configuration file (in case you haven't already), specifiying all the parameters
for the model. A template has been provided for this in  ./configs/hello_world_local_nlg_gru_json.yaml

Troubleshooting 
------------
* If a module is not being recognized by Python, verify that this module has been previously installed or is included in requirements.txt
* If the model class is not being detected, make sure the name of the model class is the same as specified in the yaml configuration file (case sensitive)
* If the dataloader type is not being detected, make sure that field 'loader_type' has been declared in the yaml configuration file.