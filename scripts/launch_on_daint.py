# This script launches all files lying around with daint srun commands

import os
import subprocess

path = './launch/'

if __name__ == "__main__":
    os.system('source scripts/piz_daint_cpu.sh')
    for launch_file in os.listdir(path):
        if "conflux" or "confchox" in launch_file:
            os.system('chmod +x %s' %(path + launch_file))
            os.system("sbatch " + path + launch_file)
        
