Option Reference
================

Command Line Arguments
----------------------

YAML Configuration
------------------

FLUTE yaml files consist of three main sections, and a few optional sections. The `model_config` specifies model architecture and pretrained model setup path. The `server_config` section defines server settings such as total training rounds, aggregation method, optimizer settings, learning rate schedule, and any server-side training data.  The `client_config` section specifies client optimizer settings and the client-side training data.

.. note:: Training data is loaded by the server and dispatched to the clients. The configuration settings for this data are specified in the `client_config`.


model_config
~~~~~~~~~~~~

server_config
~~~~~~~~~~~~~

client_config
~~~~~~~~~~~~~

Optional Sections
-----------------
In addition to the main sections, some optional sections may be specified to control privacy settings, specifically a `dp_config` section for differential privacy settings, and `privacy_metrics_config` for applying privacy metrics.


dp_config
~~~~~~~~~

privacy_metrics_config
~~~~~~~~~~~~~~~~~~~~~~