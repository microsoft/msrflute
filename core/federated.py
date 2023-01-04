# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import cProfile
import logging

import torch
import torch.distributed as dist
import numpy as np

from core.client import Client
from utils import (
    print_rank,
    print_profiler,
    to_device,
)

COMMAND_UPDATE = 0
COMMAND_TRAIN = 1
COMMAND_TERMINATE = 10
COMMAND_TESTVAL = 11
COMMAND_SYNC_NODES = 9

def encode_string(word, string_to_int = True):
    """ Encodes/Decodes the dictionary keys into an array of integers to be sent 
    as tensors of the same shape during NCCL/Gloo P2P communication.
    
    Args:
            word (string/array): key to be encoded/decoded.
            string_to_int (bool): flag that indicates which action to perform.
    """

    if string_to_int: # encode
        word = word.ljust(8, ' ') if len(word) < 8 else word # padding -- 8 is max length, all tensors must have the same size during communication
        word_encoded = [letter for letter in word.encode()]
        return word_encoded
    else: #decode
        cleanup_array = [letter for letter in word if letter!= 32] # Remove padding
        word_decoded = bytes(cleanup_array).decode()
        return word_decoded

def rank():
    """ Return rank of node. """
    return int(os.environ['RANK'])

def local_rank():
    """ Return local rank of node. """
    return int(os.environ['LOCAL_RANK'])

def size():
    """ Returns number of nodes in the distributed group, including server. """
    return int(os.environ['WORLD_SIZE'])

def _recv(x, src=0):
    """ Receives tensors with a single element or a list of tensors 
    with the same shape during distributed communication. """

    x = torch.tensor(x) if torch.is_tensor(x) == False else x
    x = to_device(x)
    dist.recv(tensor=x, src=src)
    x.to('cpu')
    
    try:
        return x.item() # single element
    except:
        return x.tolist() # list of tensors

def _recv_gradients(src):
    """ Receives a list of tensors with different shape during 
    distributed communication. """

    n, n_dimensions, grads = 0, 0, [] # tensors intialization -- required by torch.
    n = _recv(n,src)
    for i in range(n):
        n_dimensions = _recv(n_dimensions,src)
        dimensions = [0 for i in range(n_dimensions)]
        dimensions = _recv(dimensions, src)
        print_rank(f"Received dimensions {dimensions}", loglevel=logging.DEBUG)
        param = to_device(torch.zeros(dimensions))
        print_rank(f"Shape assigned {param.shape}", loglevel=logging.DEBUG)
        dist.recv(param,src)
        grads.append(param.detach().cpu())
    torch.cuda.empty_cache() 
    return grads

def _send(x, dst=0):
    """ Send tensors with a single element or a list of tensors 
    with the same shape during distributed communication. """
    x = torch.tensor(x)
    x = to_device(x)
    dist.send(x, dst)
    del x 
    torch.cuda.empty_cache()

def _send_metrics(output):
    """ Organize the keys and values from the resulting dictionary 
    from test/val rounds into arrays that are sent as independent 
    tensors during distributed communication. """

    keys = [encode_string(key) for key in output.keys()]
    values = [float(output[key]['value']) for key in output.keys()]
    higher_is_better = [int(output[key]['higher_is_better']) for key in output.keys()] # send the boolean as int

    _send(len(keys),0) 
    _send(keys)
    _send(values)
    _send(higher_is_better)

def _send_gradients(gradients, dst):
    """ Send a list of tensors with different shape during 
    distributed communication. """

    _send(len(gradients), dst)
    for i in gradients:
        dimensions = [int(d) for d in i.shape]
        _send(len(dimensions),dst)
        _send(dimensions,dst)
        param = to_device(i)
        dist.send(param,dst)
        del param 
        torch.cuda.empty_cache()

def _send_train_output(output):
    """ Organize the keys and values from the the returning ´client_output´ 
    dictionary in ´Client.proces_round()´ function during training rounds,
    into arrays that are sent as independent tensors during distributed 
    communication. """

    cs_values = [float(cs_v) for cs_v in output['cs'].values()] # cs dict -- values are flatten in 1d array
    pl_values = [float(output['pl']['weight'])] # pl dict
    gradients = output['pl']['gradients'] # gradients are sent independently
    values = cs_values + [float(output[key]) for key in output.keys() if key not in ['cs','pl']] + pl_values # reorganizing values in the order expected by the Server
    
    # Send data
    _send(values, 0)
    _send_gradients(gradients, 0)

def build_grads_dict(node):
    """ Reconstruct the dictionary ´client_output´ returned by 
    ´Client.proces_round()´ function on the Server side during 
    distributed communication. """

    # Initialize tensors
    keys = ['cs','tl','mg','vg','ng','rg','ns','ts','pl']
    values = [0.0 for i in range(11)] # initializing tensor shape -- 11 is fixed number of keys expected
    
    # Read data
    values = _recv(values,node)
    grads = _recv_gradients(node)
    
    # Rebuilding original dictionary
    cs_values = [{key: values.pop(0) for key in ['setup','training','full cost']}] # recreating cs dict
    pl_values = [{'weight':values.pop(), 'gradients': grads}] # recreating pl dict
    values_list = cs_values + [values.pop(0) for i in range(7)] + pl_values # 7 is fixed length for remaining items
    result = dict(zip(keys,values_list))

    # Cast values to original type
    for key in ['mg','vg','ng','rg']:
        result[key] = np.float32(result[key])
    result['ns'] = int(result['ns'] )
                
    return result

def build_metrics_dict(node):
    """ Reconstruct the dictionary returned during test/val rounds
    on the Server side during distributed communication. """

    # Initialize tensors
    n = 0
    n = _recv(n,node)
    keys = [[0 for j in range(8)] for i in range(n)] # max_seq_len for metric name is 8
    values = [0.0 for i in range(n)]
    higher_is_better = [0 for i in range(n)]

    # Read data
    keys = _recv(keys,node)
    values = _recv(values,node)
    higher_is_better = _recv(higher_is_better,node)

    # Reorganize output + decode dict keys
    orig_keys = [encode_string(key, string_to_int=False) for key in keys]
    values_dict = [{'value': float(v), 'higher_is_better': bool(higher_is_better[i])} for i, v in enumerate(values)]
    metrics = dict(zip(orig_keys,values_dict))
    num_instances = int(metrics.pop('num')['value'])

    result = None, metrics, num_instances
            
    return result

def receive_workers_output(node_request_map, results_list, free_nodes, command, idle_nodes):
    """ Receives the clients output on the Server side in async/sync mode. 
    Asynchronous mode is only enabled when using NCCL backend given that Gloo 
    does not provide native non-blocking implementation to check if the operation 
    has been completed during distributed training"""

    if dist.get_backend() == "nccl": # Async
        for node, req in node_request_map:
            if req.is_completed():
                result = build_metrics_dict(node) if command == COMMAND_TESTVAL else build_grads_dict(node)
                results_list.append(result)
                free_nodes.append(node)
                node_request_map.remove((node,req))
                print_rank(f"Finished releasing the nodes {free_nodes}", loglevel=logging.DEBUG)
    else: # Sync
        print_rank(f"Waiting for a workers", loglevel=logging.DEBUG)
        gather_objects = [(None,None,None) for i in range(size())]
        output = [None for _ in gather_objects]
        dist.all_gather_object(output, gather_objects[rank()])
        print_rank(f" All workers have finished ... taking the remaining clients {len(output)}", loglevel=logging.DEBUG)
        output = [e for i,e in enumerate(output) if i not in idle_nodes ] # Cleanup for idle workers
        results_list = results_list + output[1:]
        free_nodes = list(range(1, size()))
    
    return node_request_map, results_list, free_nodes

def append_async_requests(node_request_map, node):
    """ Appends the asynchronous request sent to each worker during 
    asynchronous training. """

    ack = to_device(torch.tensor(1))
    req = dist.irecv(tensor=ack, src=node)
    node_request_map.append((node,req))
    return node_request_map

def sync_idle_nodes(client_queue, free_nodes):
    """ Request dummy outputs to the odd (idle) nodes during synchronous training
    to prevent them to get trapped in the state of the previous iterations """

    idle_nodes = []
    if len(client_queue) == 0:
        print_rank(f"Free idle nodes {len(free_nodes)}", loglevel=logging.DEBUG)
        while len(free_nodes) > 0:
            node = free_nodes.pop()
            idle_nodes.append(node)
            _send(COMMAND_SYNC_NODES, node)
    return idle_nodes

class Server:
    """Server object responsible for orchestration and aggregation.

    The Server is one of the two objects that may exist inside of a thread, all
    throughout its execution (the other being the Worker). At every round, the
    Server samples clients ids and send their data for an available Worker to process.
    The Workers then each produce a new model, and all models are sent to the Server
    for aggregation.

    The methods defined here are related to orchestration only, the aggregation
    will be done by a different object which inherits from this one.

    Notes:
        This class has no :code`__init__` method, and all its methods are static.
        It thus only serves the purpose of grouping the methods, but nothing
        is actually stored inside of the object.
    """
    @staticmethod
    def dispatch_clients(clients, server_data, command, mode=None, do_profiling=False):
        """Perform the orchestration between Clients and Workers.

        This function does the following:
            1. It sends the server_data to all workers
            2. For each available Worker:
                2a. It sends the index of the client to instantiate
                2c. It triggers the execution of the command on the
                    Client.
            3. Collect and return all client outputs.

        Notes:
            This function yields the gradients of different clients
            as they are received. Therefore, the order of the results generally
            does not correspond to the order of the clients.

            All commands used during Server-Worker communication must be 
            float/integers given that torch.distributed only allows to
            send/recv tensors.

        Args:
            clients (list): list of clients to be processed.
            server_data (dict): server data sent to the workers and passed to
                clients, typically includes the global model at that step.
            command (int): instruction for worker to execute on the Client.
            mode (int): test/val only provided during evaluation rounds.
            do_profiling (bool): enables profiler during comunication.
        
        Returns:
            Generator of results.
        """

        # Some cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

        # Initialize communication profiler
        profiler = None
        if do_profiling:
            profiler = cProfile.Profile()
            profiler.enable()

        # Update lr + model parameters each round for all workers
        lr, model_params, nround = server_data
        for worker_rank in range(1, size()):
            _send(COMMAND_UPDATE, worker_rank)
            _send(lr,worker_rank)
            _send_gradients(model_params, worker_rank)
            _send(float(nround),worker_rank)
            print_rank(f"Finished sending lr {lr} and n_params {len(model_params)} to worker {worker_rank} - round {nround}", loglevel=logging.DEBUG)

        print_rank(f"Finished sending server_data to workers", loglevel=logging.DEBUG)

        client_queue = clients.copy()
        print_rank(f"Clients queue: {client_queue}", loglevel=logging.DEBUG)
        free_nodes = list(range(1, size()))
        results_list, node_request_map = [], []

        # Initiate computation for all clients
        while client_queue:
            print_rank(f"Clients queue: {client_queue}", loglevel=logging.DEBUG)
            assert len(free_nodes) > 0
            node = free_nodes.pop()
            index = len(client_queue)-1
            client_to_process = client_queue.pop(index) 
            print_rank(f"Sending client {index} to worker {node}", loglevel=logging.DEBUG)
            _send(command, node) # The command should indicate the worker which function to run on the client

            if command == COMMAND_TESTVAL:
                _send(mode,node) # Only for test/val has a value
                _send(index, node) # Worker receives the index of the client to pop
            elif command == COMMAND_TRAIN:
                _send(client_to_process, node)
            print_rank(f"Finished assigning worker {node}, free nodes {free_nodes}", loglevel=logging.DEBUG)

            if dist.get_backend() == "nccl":
                append_async_requests(node_request_map, node)
                idle_nodes = None
            else:
                idle_nodes = sync_idle_nodes(client_queue, free_nodes)
   
            # Waits until receive the output from all ranks
            if not free_nodes:
                print_rank(f"Waiting for a workers, free nodes {free_nodes}, reqs_lst {node_request_map}", loglevel=logging.DEBUG)
                while len(free_nodes) == 0:
                    node_request_map, results_list, free_nodes = receive_workers_output(node_request_map, results_list, free_nodes, command, idle_nodes)
                    for output in results_list:
                        yield output
                    results_list = []

        # Wait for all workers to finish
        while (len(node_request_map)) != 0:
            node_request_map, results_list, free_nodes = receive_workers_output(node_request_map, results_list, free_nodes, command, idle_nodes)

            for output in results_list:
                yield output
            results_list = []
        
        if do_profiling:
            profiler.disable()
            print_profiler(profiler)

        # Some cleanup
        torch.cuda.empty_cache()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    @staticmethod
    def process_clients(clients, server_data):
        """Ask workers to perform training on Clients.

        Args:
            clients (list): list of clients indexes sampled by ´Server.py´ 
                            object per iteration.
            server_data (dict): dictionary containing model.

        Returns:
            Generator of results.
        """
        return Server.dispatch_clients(clients, server_data, COMMAND_TRAIN)

    @staticmethod
    def process_testvalidate(clients, server_data, mode):
        """Ask workers to perform test/val on Clients.

        Args:
            clients (list): list of clients indexes for test/val rounds.
            server_data (dict): dictionary containing model.
            mode (str): test/val.

        Returns:
            Generator of results.
        """

        mode = [-2] if mode == "test" else [2]
        return Server.dispatch_clients(clients, server_data, COMMAND_TESTVAL, mode)

    @staticmethod
    def terminate_workers(terminate=True):
        """Terminate the execution of the workers."""

        if terminate:
            print_rank("Terminating worker processes")
            for worker_rank in range(1, size()):
                _send(COMMAND_TERMINATE, worker_rank)

class Worker:
    """Worker object responsible for instantiate Clients based on incoming data
    from the Server and perform train/eval functions on it.

    Each worker lives on a different NCCL/Gloo thread and is assigned to a different
    GPU. Via the :code:`dispatch_clients` function, the Server passes to the
    Worker specific instructions to process clients' data, typically in order
    to generate a new model or to compute metrics.

    Attributes:
        model (torch.nn.Module): model being trained.
        data_path (str): path where all clients' data is located.
        do_profiling (bool): if True, analyzes execution in depth.
        val_clients (list): clients list for validation rounds.
        test_clients (list): clients list for testing rounds.
        config (dict): clients configuration.
        val_dataset (torch.utils.data.Dataset): validation dataset.
        test_dataset (torch.utils.data.Dataset): testing dataset.
    """
    def __init__(self, model=None, data_path=None, do_profiling=False, val_clients= None, \
                test_clients=None, config=None, val_dataset = None, test_dataset = None):

        self.model = model
        self.data_path = data_path
        self.do_profiling = do_profiling
        self.config = config
        self.val_clients = val_clients
        self.test_clients = test_clients
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

    def run(self):
        """Main loop executed by worker nodes.
        
        This method handles the NCCL/Gloo communication between the worker and
        the server. It keeps listening for commands from the Server,
        and performs different actions on the Client assigned depending on 
        the command received.
        """
        
        while True:  # keeps listening for incoming server calls

            # Initialize tensors -- required by torch.distributed
            command, client_idx, mode = 0, 0, 0  # int
            lr, nround = torch.zeros(1), torch.zeros(1) # float

            # Read command
            command = _recv(command)
            print_rank(f"Command received {command} on worker {rank()}", loglevel=logging.DEBUG)

            # Receive server data -- lr, model_params
            if command == COMMAND_UPDATE:
                print_rank(f"COMMMAND_UPDATE received {rank()}", loglevel=logging.DEBUG)                
                lr = _recv(lr, 0)
                model_params = _recv_gradients(0)
                nround = _recv(nround, 0)
                server_data = (lr, model_params, int(nround))
                print_rank(f"Received lr: {lr} and n_params: {len(model_params)} - round {nround}", loglevel=logging.DEBUG)
                
            elif command == COMMAND_TRAIN:
                print_rank(f"COMMMAND_TRAIN received {rank()}", loglevel=logging.DEBUG)
                
                # Init profiler in training worker
                profiler = None
                if self.do_profiling:
                    profiler = cProfile.Profile()
                    profiler.enable()
                                
                # Receive client id from Server
                client_idx = _recv(client_idx)
                print_rank(f"Cliend idx received from Server: {client_idx}", loglevel=logging.DEBUG)

                # Instantiate client
                client_to_process = Client(
                        [client_idx],
                        self.config,
                        self.config['client_config']['type'] == 'optimization') 
                
                # Execute Client.get_data()
                client_data = client_to_process.get_client_data()

                # Execute Client.process_round()
                output = client_to_process.process_round(client_data, server_data, self.model, self.data_path)

                # Send output back to Server
                if dist.get_backend() == "nccl":
                    # ASYNC mode -- enabled only for nccl backend
                    ack = to_device(torch.tensor(1))
                    dist.isend(tensor=ack, dst=0)
                    _send_train_output(output)
                else:
                    # SYNC mode -- gloo backend does not have a non-blocking way to check if the operation is completed
                    gather_objects = [output for i in range(size())]
                    output = [None for _ in gather_objects]
                    dist.all_gather_object(output, gather_objects[rank()])

                # Some cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize() if torch.cuda.is_available() else None

                if self.do_profiling:
                    profiler.disable()
                    print_profiler(profiler)

            elif command == COMMAND_TESTVAL:
                print_rank(f"COMMMAND_TESTVAL received {rank()}", loglevel=logging.DEBUG)

                # Init profiler in validation worker
                profiler = None
                if self.do_profiling:
                    profiler = cProfile.Profile()
                    profiler.enable()
                
                # Receive mode and client id from Server
                mode = _recv(mode)
                mode = "test" if mode == -2 else "val"
                client_idx = _recv(client_idx)
                print_rank(f"Client idx received from Server: {client_idx}, {mode}", loglevel=logging.DEBUG)
                
               # Get client and dataset
                clients = self.val_clients if mode == "val" else self.test_clients
                dataset = self.val_dataset if mode == "val" else self.test_dataset
                clients_queue = clients.copy()
                assert 0 <= client_idx < len(clients_queue)
                client_to_process = clients_queue.pop(client_idx)

                # Execute Client.get_data()
                client_data = client_to_process.get_client_data(dataset)
   
                # Execute Client.run_testvalidate()
                output = client_to_process.run_testvalidate(client_data, server_data, mode, self.model)

                # Send output back to Server
                if dist.get_backend() == "nccl":
                    # ASYNC mode -- enabled only for nccl backend
                    _, metrics, num_instances = output
                    metrics['num']= {'value': float(num_instances), 'higher_is_better': False}
                    output = metrics
                    print_rank(f"Worker {rank()} output {output}", loglevel=logging.DEBUG)
                    ack = to_device(torch.tensor(1))
                    dist.isend(tensor=ack, dst=0)
                    _send_metrics(output)
                else:
                    # SYNC mode -- gloo backend does not have a non-blocking way to check if the operation is completed
                    gather_objects = [output for i in range(size())]
                    output = [None for _ in gather_objects]
                    dist.all_gather_object(output, gather_objects[rank()])
                    print_rank(f"Worker {rank()} sent output back", loglevel=logging.DEBUG)

                # Some cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize() if torch.cuda.is_available() else None

                if self.do_profiling:
                    profiler.disable()
                    print_profiler(profiler)

            elif command == COMMAND_TERMINATE:
                print_rank(f"COMMMAND_TERMINATE received {rank()}", loglevel=logging.DEBUG)

                # Some cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                return

            elif command == COMMAND_SYNC_NODES: # Only for sync calls
                print_rank(f"COMMMAND_SYNC_NODES received {rank()}", loglevel=logging.DEBUG)

                gather_objects = [None for i in range(size())]
                output = [None for _ in gather_objects]
                dist.all_gather_object(output, gather_objects[rank()])
                print_rank(f"Worker IDLE {rank()} sent dummy output back", loglevel=logging.DEBUG)

                # Some cleanup
                torch.cuda.empty_cache()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            else:
                assert False, "unknown command"
