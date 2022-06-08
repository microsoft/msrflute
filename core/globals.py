# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

# Macro variable that sets which distributed trainig framework is used (e.g. mpi, syft, horovod)
TRAINING_FRAMEWORK_TYPE = 'mpi'
logging_level = logging.INFO  # DEBUG | INFO