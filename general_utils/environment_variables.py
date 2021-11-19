import os
import subprocess
import socket

def is_slurm_installed():
    try:
        ret = subprocess.run(['sinfo'], capture_output=True)
        # use knowledge of stdout format
        return ret.stdout[:9].decode('utf-8') == 'PARTITION'
    except FileNotFoundError:
        return False

def get_slurm_nodes():
    process_result = subprocess.run(['sinfo', '-o', '%n %a'], capture_output=True)
    stdout = process_result.stdout.decode('utf-8')
    lines = stdout.split('\n')
    nodes = []
    for line in lines:
        elements = line.split(' ')
        if len(elements) != 2:
            continue
        hostname, status = elements
        if status == 'up':
            nodes.append(hostname)
    return nodes

def is_slave():
    hostname = socket.gethostname()
    hostname = hostname.split('.')[0]
    return is_slurm_installed() and (hostname in get_slurm_nodes())


SLAVE_DATASET_DIR = 'SLAVE_DATASET_DIR'
DATASET_DIR = 'DATASET_DIR'

def get_slave_dataset_dir():
    return os.environ[SLAVE_DATASET_DIR]

def get_cluster_dataset_dir():
    return os.environ[DATASET_DIR]

def all_environment_variables_set():
    try:
        os.environ[DATASET_DIR]
        if is_slave():
            os.environ[SLAVE_DATASET_DIR]
    except KeyError as e:
        return False
    return True

def get_dataset_dir():
    if is_slave():
        ret = os.environ[SLAVE_DATASET_DIR]
    else:
        ret = os.environ[DATASET_DIR]
    print(ret)
    return ret
