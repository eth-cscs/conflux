import os
import subprocess

path = './launch/'




if __name__ == "__main__":

    for launch_file in os.listdir(path):
        if "psychol" in launch_file or "scalapack" in launch_file:
            os.system('source scripts/piz_daint_cpu.sh')
            os.system('chmod +x %s' %(path + launch_file))
            os.system("sbatch " + path + launch_file)
        
