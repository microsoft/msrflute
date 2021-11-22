# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import cProfile
import gc
import os
import pickle
import logging

import torch
from mpi4py import MPI

from utils import (
    print_rank,
    print_profiler
)
from utils.queue import process_in_parallel


SPLIT_SIZE = 512 * 1024 * 1024  # messages above this size (in bytes) are split

COMMAND_UPDATE = "update"
COMMAND_TRAIN = "train"
COMMAND_TERMINATE = "terminate"
COMMAND_TESTVAL = "testvalidate"


def rank():
    """Return rank of node"""
    return MPI.COMM_WORLD.Get_rank()

def local_rank():
    """Return local rank of MPI node"""
    assert (
        "OMPI_COMM_WORLD_LOCAL_RANK" in os.environ
    ), "local rank can only be determined when using OpenMPI"
    return int(os.environ["OMPI_COMM_WORLD_LOCAL_RANK"])

def size():
    """Returns number of MPI nodes including server"""
    return MPI.COMM_WORLD.Get_size()


class Server:
    """Server object responsible for orchestration and aggregation.

    The Server is one of the two objects that may exist inside of a thread, all
    throughout its execution (the other being the Worker). At every round, the
    Server samples clients and sends their data for an available Worker to process.
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
    def dispatch_clients(clients, server_data, payload_fn, clients_in_parallel=None):
        """Perform execution of client code on the worker nodes.

        This function does the following:
            1. It sends the server_data to all workers
            2. For each client:
                2a. It sends the function process_round of the client
                    to a free worker.
                2b. It calls get_client_data on the client.
                2c. It triggers the execution of the payload_fn on the
                    worker with parameters server_data and client_data.

        Notes:
            This function yields the gradients of different clients
            as they are received. Therefore, the order of the results generally
            does not correspond to the order of the clients.

        Args:
            clients (list): list of clients to be processed.
            server_data (dict): server data sent to the workers and passed to
                clients, typically includes the global model at that step.
            payload_fn (callback): instructions for worker to execute.
            clients_in_parallel (int or None): how many threads will be used for 
                processing clients, defaults to None in which case all of them
                are processed on the same thread.

        Returns:
            Generator of results sent by server via MPI.
        """
        # Send args to workers
        data_pickled = pickle.dumps(server_data)  # pickle once
        for worker_rank in range(1, MPI.COMM_WORLD.Get_size()):
            MPI.COMM_WORLD.send(COMMAND_UPDATE, worker_rank)
            _send(data_pickled, worker_rank, pickled=True)

        # Perform payload_fn on clients
        client_queue = clients.copy()
        free_nodes = list(range(1, MPI.COMM_WORLD.Get_size()))
        node_request_map = []

        # Initiate computation for all clients
        while client_queue:
            if clients_in_parallel is not None:
                clients_to_process = [client_queue.pop() for _ in range(clients_in_parallel) if len(client_queue) > 0]
            else:
                clients_to_process = client_queue.pop()

            print_rank(f"Queueing {clients_to_process}, {len(client_queue)} remaining", loglevel=logging.DEBUG)

            # Wait for free worker node
            if not free_nodes:
                print_rank(f"Waiting for a worker", loglevel=logging.DEBUG)
                assert(len(node_request_map) > 0)
                status = MPI.Status()                
                ix, _ = MPI.Request.waitany(node_request_map, status=status)

                # Collects worker output after processing has finished
                output = _recv(status.source)
                if isinstance(output, list):
                    yield from output
                else:
                    yield output

                free_nodes.append(status.source)                
                print_rank(f"Found free worker {ix}:{status.source}", loglevel=logging.DEBUG)
                node_request_map.pop(ix)

            # Run client computation on free worker node
            assert len(free_nodes) > 0
            node = free_nodes.pop()
            print_rank(f"Sending to worker {node}", loglevel=logging.DEBUG)
            payload_fn(clients_to_process, node)
            print_rank(f"Payload sent. Queueing irecv on {node}", loglevel=logging.DEBUG)
            node_request_map.append(MPI.COMM_WORLD.irecv(source=node))
            print_rank(f"Queued irecv for {node}", loglevel=logging.DEBUG)

        print_rank(f"Done queuing clients. Waiting on workers")

        # Wait for all workers to finish
        for i, request in enumerate(node_request_map):
            status = MPI.Status()            
            request.wait(status)
            print_rank(f"Result for item {i}: source: {status.source}", loglevel=logging.DEBUG)

            print_rank(f"Calling _recv for {status.source}", loglevel=logging.DEBUG)
            output = _recv(status.source)
            if isinstance(output, list):
                yield from output
            else:
                yield output

    @staticmethod
    def process_clients(clients, server_data, clients_in_parallel):
        """Ask workers to process client data.

        The payload function defined below will send a free worker instructions
        on how to process the data of one or more clients. This payload function 
        is then passed to :code:`dispatch_clients`, which continuously looks for
        free workers and sends them more clients to process.

        Args:
            clients (list): list of client.Client objects.
            server_data (dict): dictionary containing model.
            clients_in_parallel (None or int): how many threads to use for
                processing the clients on a given worker.

        Returns:
            Generator of results sent by server via MPI.
        """

        def payload_fn(clients, node):
            """Payload function for a training round."""

            # Send command for training and function to process round
            MPI.COMM_WORLD.send(COMMAND_TRAIN, node)

            # Loop through clients and send their data
            if clients_in_parallel is None:
                MPI.COMM_WORLD.send(clients.process_round, node)
                MPI.COMM_WORLD.send(clients.get_client_data(), node)
            else:
                MPI.COMM_WORLD.send(clients[0].process_round, node)  # clients is a list
                MPI.COMM_WORLD.send(len(clients), node)
                for client in clients:
                    MPI.COMM_WORLD.send(client.get_client_data(), node)

        return Server.dispatch_clients(clients, server_data, payload_fn, clients_in_parallel=clients_in_parallel)

    @staticmethod
    def process_testvalidate(clients, server_data, mode):
        """Ask workers to use clients data to compute metrics.

        Similar to :code:`process_round` but asks workers to
        compute metrics instead, by using a different payload function.

        Args:
            clients (list): list of client.Client objects.
            server_data (dict): dictionary containing model.
            mode(str): whether to :code:`test` or :code:`validate`.

        Returns:
            Generator of results sent by server via MPI.
        """

        def payload_fn(client, node):
            """Payload function for a test/validation round."""

            MPI.COMM_WORLD.send(COMMAND_TESTVAL, node)
            MPI.COMM_WORLD.send(client.run_testvalidate, node)
            MPI.COMM_WORLD.send(client.get_client_data(), node)
            MPI.COMM_WORLD.send(mode, node)

        return Server.dispatch_clients(clients, server_data, payload_fn)

    @staticmethod
    def terminate_workers(terminate=True):
        """Terminate the execution of the workers."""

        if terminate:
            print_rank("Terminating worker processes")
            for worker_rank in range(1, MPI.COMM_WORLD.Get_size()):
                MPI.COMM_WORLD.send(COMMAND_TERMINATE, worker_rank)


class Worker:
    """Worker object responsible for processing clients' data.

    Each worker lives on a different MPI thread and is assigned to a different
    GPU. Via the :code:`dispatch_clients` function, the Server passes the
    Worker specific instructions to process clients' data, typically in order
    to generate a new model or to compute metrics.

    Attributes:
        model (torch.nn.Module): model being trained.
        data_path (str): path where all clients' data is located.
        do_profiling (bool): if True, analyzes execution in depth.
        clients_in_parallel (None or int): if not None, processes clients in
            threads during training round.
        server_data (dict): stores data received from Server when an update
            command is received.
    """

    def __init__(self, model=None, data_path=None, do_profiling=False, clients_in_parallel=None):
        """
        Set the GPU workspace for the model to be exchanged between the server and clients
        This prevents a model instance from being created on the GPU worker many time

        Args:
            model (torch.nn.Module, optional): model being trained, defaults to None.
            data_path (str, optional): path where all clients' data is located,
                defaults to None.
            do_profiling (bool, optional): if True, analyzes execution in depth; defaults
                to False.
            clients_in_parallel (None or int, optional): if not None, processes clients in
                threads during training round. Defaults to None.
        """
        self.model = model
        self.data_path = data_path
        self.do_profiling = do_profiling
        self.clients_in_parallel = clients_in_parallel

        self.server_data = None

        # For processing in different threads, we need copies of the model
        if clients_in_parallel is not None:
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
            self.model_copies = [copy.deepcopy(model).to(device) for _ in range(clients_in_parallel)]

    def run(self):
        """Main loop executed by worker nodes.
        
        This method triggers the MPI communication between the worker and
        the server. It keeps listening for commands from the Server,
        and performs different actions depending on the command received.
        """

        while True:  # keeps listening for commands on MPI
            command = MPI.COMM_WORLD.recv()
            assert isinstance(command, str)

            if command == COMMAND_UPDATE:
                self.server_data = _recv(0)

            elif command == COMMAND_TRAIN:
                profiler = None
                if self.do_profiling:
                    profiler = cProfile.Profile()
                    profiler.enable()

                client_fn = MPI.COMM_WORLD.recv()  # NOTE: assumes function is same for all clients

                # Pick whether to do processing in batches or not
                if self.clients_in_parallel is None:
                    client_data = MPI.COMM_WORLD.recv()

                    torch.cuda.empty_cache()
                    output = client_fn(client_data, self.server_data, self.model, self.data_path)
                else:
                    n_clients = MPI.COMM_WORLD.recv()
                    client_data = [MPI.COMM_WORLD.recv() for _ in range(n_clients)]

                    torch.cuda.empty_cache()
                    output = process_in_parallel(client_fn, client_data, self.server_data, self.model_copies, self.data_path)
                    print_rank(f"Processed batch of size {len(client_data)}, got {len(output)} outputs", loglevel=logging.DEBUG)

                # Wait for server to be available and send output(s)
                MPI.COMM_WORLD.isend(None, 0).wait()
                _send(output, 0)

                # Make sure that memory is cleaned up
                if self.clients_in_parallel is not None:
                    for args in client_data:
                        del args
                    del client_fn, client_data, output

                torch.cuda.empty_cache()
                torch.cuda.synchronize() if torch.cuda.is_available() else None

                if self.do_profiling:
                    profiler.disable()
                    print_profiler(profiler)
     
            elif command == COMMAND_TESTVAL:
                profiler = None
                if self.do_profiling:
                    profiler = cProfile.Profile()
                    profiler.enable()

                client_fn = MPI.COMM_WORLD.recv()
                client_data = MPI.COMM_WORLD.recv()
                client_mode = MPI.COMM_WORLD.recv()

                # Clean up memory before client processing
                torch.cuda.empty_cache()

                try:
                    output = client_fn(client_data, self.server_data, client_mode, self.model)
                except RuntimeError as e:
                    _dump_tensors(gpu_only=True)
                    raise RuntimeError("Federated Error: {}".format(str(e)))

                MPI.COMM_WORLD.isend(None, 0).wait()
                _send(output, 0)

                # Make sure that memory is cleaned up
                del client_fn, client_data, output
                torch.cuda.empty_cache()
                torch.cuda.synchronize() if torch.cuda.is_available() else None

                if self.do_profiling:
                    profiler.disable()
                    print_profiler(profiler)

            elif command == COMMAND_TERMINATE:
                return

            else:
                assert False, "unknown command"


def _send(data, rank, pickled=False, verbose=False):
    """Send large object by chunking it into multiple MPI messages."""

    # Pickle data
    data_pickled = data
    if not pickled:
        data_pickled = pickle.dumps(data_pickled)

    # Compute in how many chunks data will be sent
    num_chunks = len(data_pickled) // SPLIT_SIZE + 1
    if verbose:
        print_rank(f"_send data_pickled size: {len(data_pickled)}, {num_chunks} chunks")

    # Send data in chunks
    MPI.COMM_WORLD.send(num_chunks, rank)

    ix = 0
    while len(data_pickled) - ix > SPLIT_SIZE:
        MPI.COMM_WORLD.send(data_pickled[ix:ix+SPLIT_SIZE], rank)
        ix += SPLIT_SIZE
    MPI.COMM_WORLD.send(data_pickled[ix:], rank)
    
def _recv(rank):
    """Receive large object by chunking it into multiple MPI messages."""

    num_chunks = MPI.COMM_WORLD.recv(source=rank)
    pickled_chunks = []
    for _ in range(num_chunks):
        pickled_chunks.append(MPI.COMM_WORLD.recv(source=rank))        
    data_pickled = b"".join(pickled_chunks)
    return pickle.loads(data_pickled)

def _dump_tensors(gpu_only=True):
    """Print a list of the Tensors being tracked by the garbage collector."""

    def pretty_size(size):
        """Pretty prints a torch.Size object."""
        assert(isinstance(size, torch.Size))
        return " × ".join(map(str, size))

    print_rank("Dump memory allocated")
    print_rank(torch.cuda.memory_allocated())
    print_rank("Dump max memory allocated")
    print_rank(torch.cuda.max_memory_allocated())
    print_rank("Dump memory cached")
    print_rank(torch.cuda.memory_cached())
    print_rank("Dump max memory cached")
    print_rank(torch.cuda.max_memory_cached())

    total_size = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if not gpu_only or obj.is_cuda:
                    print("%s:%s%s %s" % (type(obj).__name__,
                                            " GPU" if obj.is_cuda else "",
                                            " pinned" if obj.is_pinned else "",
                                            pretty_size(obj.size())))
                    total_size += obj.numel()
            elif hasattr(obj, "data") and torch.is_tensor(obj.data):
                if not gpu_only or obj.is_cuda:
                    print("%s -> %s:%s%s%s%s %s" % (type(obj).__name__,
                                                    type(obj.data).__name__,
                                                    " GPU" if obj.is_cuda else "",
                                                    " pinned" if obj.data.is_pinned else "",
                                                    " grad" if obj.requires_grad else "",
                                                    " volatile" if obj.volatile else "",
                                                    pretty_size(obj.data.size())))
                    total_size += obj.data.numel()
        except Exception as e:
            pass

    print_rank("Total size: {}".format(total_size))