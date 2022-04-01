# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
# Macro variable that sets which distributed trainig framework is used (e.g. mpi, syft, horovod)
TRAINING_FRAMEWORK_TYPE = 'mpi'
logging_level = logging.INFO  # DEBUG | INFO
file_type = None
task = None


def define_file_type (data_path,config, exp_folder):
    global file_type
    global task

    filename = os.path.join(data_path, config["client_config"]["data_config"]["train"]["list_of_train_data"])
    arr_filename = filename.split(".")
    file_type = arr_filename[-1]
    print(" File_type has ben assigned to: {}".format(file_type))
    task = exp_folder