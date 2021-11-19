import os
import general_utils.environment_variables as env_vars
import shutil

def main():
    dataset_name = 'WFLW_keypoint_preproc_jpeg'
    assert(env_vars.is_slave())
    slave_dir = env_vars.get_slave_dataset_dir()
    cluster_dir = env_vars.get_cluster_dataset_dir()
    target_dst = os.path.join(slave_dir, dataset_name)
    target_src = os.path.join(cluster_dir, dataset_name)
    if not os.path.exists(target_dst):
        shutil.copytree(target_src, target_dst)

if __name__ == '__main__':
    main()
