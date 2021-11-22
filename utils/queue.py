# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from mpi4py import MPI
from concurrent.futures import ThreadPoolExecutor
from utils import print_rank


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

""" Here we have classes and functions that allow one to send and process multiple
messages in parallel on MPI. """

def process_in_parallel(client_fn, client_data, server_data, models, data_path):
    """ Process multiple orders in parallel
    
    Parameters
    ----------
    client_fn: callback
        Function we want to call.
    client_data: list of tuples
        Arguments that will be passed to function.
    server_data: tuple
        Data passed from server to update model parameters.
    models: torch.nn.Module
        Models we will send to the clients.
    data_path: str
        Path to data.

    Returns
    -------
    list
        Output of each callback in the list passed as input.
    """
    with ThreadPoolExecutor(max_workers=len(client_data)) as pool:
        requests = []
        for k, args in enumerate(client_data):
            requests.append(pool.submit(client_fn, args, server_data, models[k], data_path))
        
        results = [request.result() for request in requests]
    print_rank(f'finished processing batch of size {len(client_data)}')

    return results