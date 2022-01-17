FLUTE Overview
==============

FLUTE uses a distributed processing architecture backed by OpenMPI.  An FLUTE job consists of one or more nodes (physical or virtual machines) executing a total of K workers (independent OS-level processes).

Worker 0 acts as a central orchestrator, maintaining and distributing a central model to workers, and subsequently distributing client tasks to the workers. 

Each worker>0 processes client tasks sequentially, consisting of data encoding and one or more batch updates to the central model (note the central model is reset to its original state for each client task). As each client task completes, the model delta, aka the pseudo-gradient is sent back to the orchestrator for federation into a new central model.

Execution runs for up to N training rounds.  In each round the orchestrator may sample a subset of clients, and may also randomly delay pseudo-gradient updates from some clients to future rounds. The orchestrator will also periodically distribute evaluation tasks to determine model quality on validation and test data.

.. note:: AzureML generally expects there will be one worker per GPU on each node.
.. note:: Due to networking overhead, it is often faster to run jobs on a single node with 8 or 16 GPUs, rather than on multiple nodes.

Architecture
------------

.. figure:: img/concepts.png
    :width: 400

    An FLUTE job consists of one or more independent nodes (multi-GPU VMs) executing up to K workers.  

.. figure:: img/client-server.png
    :width: 600

    On each training round the orchestrator (Worker 0) dispatches the central model to the rest of the workers, and then queues up client tasks for workers to execute. Workers receive client tasks (client training data and training config) and execute SGD on the central model using their client's training data, sending the model delta (pseudo-gradient) back to the orchestrator. 
