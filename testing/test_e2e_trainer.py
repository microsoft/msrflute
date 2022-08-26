# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import os
import platform
import pytest 

xfail = pytest.mark.xfail

def get_info(task):

    data_path=r'./testing/'
    output_path=r'./testing/outputs/'

    if task == 'nlg_gru':
        config_path=r'./testing/hello_world_nlg_gru.yaml'
    elif task == "classif_cnn":
        config_path=r'./testing/hello_world_classif_cnn.yaml'
    elif task == "ecg_cnn":
        config_path=r'./testing/hello_world_ecg_cnn.yaml'
    elif task == "mlm_bert":
        config_path=r'./testing/hello_world_mlm_bert.yaml'

    return data_path, output_path, config_path

def run_pipeline(data_path, output_path, config_path, task):

    print("Testing {} task".format(task))

    # Adjust command to the task and OS
    sym = "&" if platform.system() == "Windows" else ";" 
    command = 'cd .. '+ sym +' python '+'-m '+'torch.distributed.run '+ '--nproc_per_node=2 '+ 'e2e_trainer.py '+ \
            '-dataPath '+ data_path+' -outputPath '+output_path+' -config ' +config_path +\
            ' -task '+ task + ' -backend '+ 'nccl'

    # Execute e2e_trainer + stores the exit code
    with open('logs.txt','w') as f:                      
        process= subprocess.run(command, shell=True,stdout=f,text=True,timeout=900)
    return_code=process.returncode
    
    # Print logs
    os.system("ls")
    os.system("less logs.txt")
    print(process.stderr)
    print("Finished running {} task".format(task))

    return return_code

def test_nlg_gru():  
    
    task = 'nlg_gru'
    data_path, output_path, config_path = get_info(task)
    assert run_pipeline(data_path, output_path, config_path, task)==0

def test_ecg_cnn():  
    
    task = 'ecg_cnn'
    data_path, output_path, config_path = get_info(task)
    assert run_pipeline(data_path, output_path, config_path, task)==0
    
@pytest.mark.xfail
def test_mlm_bert():  
    
    task = 'mlm_bert'
    data_path, output_path, config_path = get_info(task)
    assert run_pipeline(data_path, output_path, config_path, task)==0
    print("PASSED")

@pytest.mark.xfail
def test_classif_cnn():  
    
    task = 'classif_cnn'
    data_path, output_path, config_path = get_info(task)
    assert run_pipeline(data_path, output_path, config_path, task)==0
    print("PASSED")
