import argparse
import os
import subprocess

def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--run_name', type=str, default='dummy')
    arg_parser.add_argument('--training_type', type=str)
    arg_parser.add_argument('--epoch', type=str)
    args = arg_parser.parse_args()
    return args.run_name, args.training_type, args.epoch

def main():
    run_name, training_type, epoch = parse_arguments()
    procs = []
    num_runs = 5
    for run_index in range(num_runs):
        conf_string = '-o logs/log_%J_slurm.out -e logs/log_%J_slurm.err --gres=gpu:1 --mem=10000 --cpus-per-task 4'
        script_string = 'python -m scripts.compute_fusions --run_name {} --training_type {} --epoch {} --run_index {}'.format(run_name, training_type, epoch, run_index)
        node_command = 'srun ' + conf_string + ' ' + script_string
        print(node_command)
        proc = subprocess.Popen(node_command, shell=True)
        procs.append(proc)

    for proc in procs:
        proc.wait()
        print('proc wait done')

if __name__ == '__main__':
    main()
