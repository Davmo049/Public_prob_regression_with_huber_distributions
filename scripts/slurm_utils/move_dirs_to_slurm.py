import os
import subprocess

from general_utils.environment_variables import get_slurm_nodes, SLAVE_DATASET_DIR, DATASET_DIR

def main():
    nodes = get_slurm_nodes()
    print(nodes)

    for node in nodes:
        slave_pos = os.environ[SLAVE_DATASET_DIR]
        out_dir = slave_pos
        command = ['srun', '--constraint={}'.format(node), 'mkdir', out_dir]
        print(command)
        subprocess.call(command)
    for node in nodes:
        dataset_pos = os.environ[DATASET_DIR]
        slave_pos = os.environ[SLAVE_DATASET_DIR]
        in_directory = os.path.join(dataset_pos, 'COCO_keypoint_preproc')
        out_dir = slave_pos

        command = ['srun', '--constraint={}'.format(node), 'cp', '-r', in_directory, out_dir]
        print(command)
        subprocess.Popen(command)
        exit(0)

if __name__ == '__main__':
    main()
