import argparse
import os
import subprocess

def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--run_name', type=str, default='dummy')
    arg_parser.add_argument('config_file', type=str)
    args = arg_parser.parse_args()
    run_name = args.run_name
    config_file = args.config_file
    return config_file, run_name



def main():
    config_file, run_name = parse_arguments()
    procs = []
    num_runs = 5
    for run_index in range(num_runs):
        conf_string = '-o logs/log_%J_slurm.out -e logs/log_%J_slurm.err --gres=gpu:1 --mem=10000 --cpus-per-task 4'
        script_string = 'python -m scripts.train {} --run_name {} --run_index {}'.format(config_file, run_name, run_index)
        node_command = 'srun ' + conf_string + ' ' + script_string
        print(node_command)
        proc = subprocess.Popen(node_command, shell=True)
        procs.append(proc)

    for proc in procs:
        proc.wait()
        print('proc wait done')

if __name__ == '__main__':
    main()
