import numpy as np
import matplotlib.pyplot as plt
import Mpii_pose.Dataset as Dataset
from matplotlib.pyplot import cm
import os
import general_utils.environment_variables
import tqdm
import scipy.linalg

def visualize_raw_mpii():
    ds = Dataset.RawDataset()
    c_train = 0
    for ann in ds.image_annos_train:
        c_train += len(ann['samples'])
    print(c_train)
    c_test = 0
    for ann in ds.image_annos_test:
        c_test += len(ann['samples'])
    print(c_test)
    for im, ann in ds:
        plt.imshow(im)
        color=cm.rainbow(np.linspace(0,1,len(ann)))
        for c, a in zip(color, ann):
            top = a['y_box']-100*a['box_scale']
            bottom = a['y_box']+100*a['box_scale']
            left = a['x_box']-100*a['box_scale']
            right = a['x_box']+100*a['box_scale']
            sp = a['single_person']
            dash = '-' if sp else '--'
            plt.plot([left, right, right, left, left], [top, top, bottom, bottom, top], dash, color=c)
            for pos, weight, vis in zip(a['joints']['pos'], a['joints']['weights'], a['joints']['is_visible']):
                if weight > 0:
                    if vis:
                        plt.plot([pos[0]], [pos[1]], 'o', color=c)
                    else:
                        plt.plot([pos[0]], [pos[1]], 'x', color=c)
        plt.show()

def preprocess_mpii():
    ds = Dataset.RawDataset()
    Dataset.preprocess_dataset(ds, os.path.join(general_utils.environment_variables.get_dataset_dir(), 'mpii_pose_preprocessed'))

def visualize_preproc():
    ds = Dataset.PreprocessedMpiiDataset()
    train_ds = ds.get_train()
    for im, points, _, weights in train_ds:
        im = im.transpose(1,2,0)
        plt.imshow(im)
        for p, w in zip(points, weights):
            if w != 0.0:
                plt.plot([p[0]], [p[1]], 'xb')
        plt.show()

def compute_normailization_stats():
    ds = Dataset.PreprocessedMpiiDataset()
    train_ds = ds.get_train()
    N_per_kp = np.zeros((16))
    sumX = np.zeros((16, 2))
    sumX2 = np.zeros((16, 2, 2))
    for im, points, _, weights in tqdm.tqdm(train_ds):
        mask = weights > 0
        N_per_kp[mask] += 1
        sumX[mask] += points[mask]
        sumX2[mask] += points[mask].reshape(-1, 2, 1)*points[mask].reshape(-1, 1, 2)

    meanX = sumX/N_per_kp.reshape(-1, 1)
    estVarX = (sumX2-N_per_kp.reshape(-1, 1,1)*(meanX.reshape(-1, 2,1)*meanX.reshape(-1, 1, 2)))/(N_per_kp.reshape(-1,1,1)-1)
    print('mean')
    print(meanX)
    print('est_half_prec_X')
    estStd = list(map(scipy.linalg.sqrtm, estVarX))
    for e in estStd:
        print(np.linalg.inv(e))


if __name__ == '__main__':
    visualize_raw_mpii()
    preprocess_mpii()
    visualize_preproc()
    compute_normailization_stats()
