import os
import general_utils.environment_variables as env_vars
import subprocess

def main():
    command = 'python -m scripts.move_dataset'
    slurm_nodes = env_vars.get_slurm_nodes()
    procs = []
    for slurm_node in slurm_nodes:
        logname = 'logs/log_{}_move'.format(slurm_node)
        node_command = 'srun -o {} -e {} --constraint={} '.format(logname+'.out', logname+'.err', slurm_node) + command
        print(node_command)
        proc = subprocess.Popen(node_command, shell=True)
        procs.append(proc)

    for proc in procs:
        proc.wait()
        print('proc wait done')

if __name__ == '__main__':
    main()
