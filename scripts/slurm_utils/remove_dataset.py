import os
import general_utils.environment_variables as env_vars
import shutil

def main():
    dataset_name = 'mpii_pose_preprocessed'
    assert(env_vars.is_slave())
    slave_dir = env_vars.get_slave_dataset_dir()
    target = os.path.join(slave_dir, dataset_name)
    shutil.rmtree(target)

if __name__ == '__main__':
    main()
