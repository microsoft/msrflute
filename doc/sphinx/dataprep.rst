Data Preparation
================

TODO: formatting for other data loaders.

Here is a sample data blob for language model training.

.. code:: json

    {
        "users": ["bert","elmo"],
        "user_data": {
            "bert": {"x": ["my name is Bert.", "I live with Ernie."]},
            "elmo": {"x": ["Big Bird is my friend."]}
        },
        "num_samples": [2, 1]
    }

The blob consists of three fields.  The ``users`` field indicates a unique id for each user in the training data.  Users are sampled uniformly to create client tasks during training. There could be many more users than client tasks per round or even over all client tasks over all rounds. The ``user_data`` field contains user-indexed training data. Each user's data is a dictionary of the form ``{"x": [list of examples]}``.  Finally, the ``num_samples`` field indicates the number of samples for each user, in order of the ``users`` list.  That is, for any index ``i`` in ``range(len(data['users']))``: 

.. code:: python
    
    data['num_samples'][i] == len(data['user_data'][data['users'][i]]['x'])


Test and validation data is formatted similarly.

.. note::

    Test/validate data is dispatched to workers by partitioning on users. If your test data isn't user-partitioned, we recommend partitioning it uniformly using some dummy user ids.