# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import subprocess
import os.path
import os
import platform

launcher_path='e2e_trainer.py'
data_path=r'./testing/mockup/'
output_path=r'./testing/outputs'
output_folder='./testing/outputs'
config_path=r'./testing/configs/hello_world_local.yaml'


def test_e2e_trainer():  

    try:
        #Verify complete script execution
        os.system("mkdir -p "+ output_folder)

        command = ['mpiexec', '-np', '2', 'python', launcher_path,\
                '-dataPath',data_path,'-outputPath',output_path,'-config',config_path,\
                '-task','nlg_gru']
        
        command_string = ""
        for elem in command:
            command_string = " ".join([command_string, str(elem)])
        
        if platform.system() == "Windows":
            command_string = "cd .. &" + command_string
        else:
            command_string = "cd .. ;" + command_string # For Linux users

        with open('logs.txt','w') as f:                      
            process= subprocess.run(command_string, shell=True,stdout=f,text=True,timeout=420)
            
        return_code=process.returncode
        print(process.stderr)
        assert return_code==0

        #Verify output files
        directory=len(os.listdir('./outputs'))
        assert directory > 0

        #Verify logs for config file
        config_exists=False
        config_file='Copy created'
        logs=open('logs.txt','r')
        readLogs=logs.read()
        if config_file in readLogs: 
            config_exists=True
        assert config_exists


    except Exception as e:
        print("Encountered an exception: {}".format(e))
        raise e
    