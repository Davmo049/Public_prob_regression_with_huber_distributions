import CocoKeypoints
import torch
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import DS_300W_LP.Dataset

def verify_norm():
    dataset = CocoKeypoints.PreprocessedCocoKeypoints(split=CocoKeypoints.SplitType.DEVELOP)
    N_keypoints = 17
    train_dataset = dataset.get_train()
    point_sum = np.zeros((N_keypoints,2))
    point_square_sum = np.zeros((N_keypoints,2, 2))
    point_num = np.zeros((N_keypoints))
    i = 0
    kp_normalizer = CocoKeypoints.TorchNormalizer('cpu')
    points_x = list(map(lambda x: list(), range(N_keypoints)))
    points_y = list(map(lambda x: list(), range(N_keypoints)))
    for im, keypoints, weights, keypoint_type in train_dataset:
        keypoints = kp_normalizer.normalize(torch.tensor(keypoints).view(1, N_keypoints, 2)).view(N_keypoints,2).numpy()
        if i % 10 == 0:
            print(i)
        if i > 1000:
            break
        i += 1
        mask=keypoint_type != 0
        point_num += mask
        point_sum += keypoints*mask.reshape(-1, 1)
        point_square_sum += (keypoints*mask.reshape(-1, 1)).reshape(-1, 2, 1)*keypoints.reshape(-1, 1, 2)
        for j in range(N_keypoints):
            if mask[j] != 0:
                points_x[j].append(keypoints[j,0])
                points_y[j].append(keypoints[j,1])
    eX = point_sum/point_num.reshape(-1, 1)
    vX = point_square_sum/(point_num-1).reshape(-1, 1, 1) - eX.reshape(-1, 2, 1)*eX.reshape(-1, 1, 2)
    hpX = np.array(list(map(lambda x: np.linalg.inv(scipy.linalg.sqrtm(x)), vX)))
    for j in range(N_keypoints):
        plt.scatter(points_x[j], points_y[j])
        plt.show()

    print('eX')
    print(eX)
    print('hpX')
    print(hpX)


def compute_means():
    # dataset = CocoKeypoints.PreprocessedCocoKeypoints(split=CocoKeypoints.SplitType.DEVELOP)
    dataset = DS_300W_LP.Dataset.Preprocessed300WLPKeypointsDataset(split=CocoKeypoints.SplitType.DEVELOP)
    N_keypoints = 68
    train_dataset = dataset.get_train()
    point_sum = np.zeros((N_keypoints,2))
    point_square_sum = np.zeros((N_keypoints,2, 2))
    point_num = np.zeros((N_keypoints))
    for im, keypoints, weights, keypoint_type in train_dataset:
        mask=keypoint_type != 0
        point_num += mask
        point_sum += keypoints*mask.reshape(-1, 1)
        point_square_sum += (keypoints*mask.reshape(-1, 1)).reshape(-1, 2, 1)*keypoints.reshape(-1, 1, 2)

    eX = point_sum/point_num.reshape(-1, 1)
    vX = point_square_sum/(point_num-1).reshape(-1, 1, 1) - eX.reshape(-1, 2, 1)*eX.reshape(-1, 1, 2)
    hpX = np.array(list(map(lambda x: np.linalg.inv(scipy.linalg.sqrtm(x)), vX)))

    print('eX')
    print(eX)
    print('hpX')
    print(hpX)
    # print(np.array(list(map(np.linalg.inv, hpX))))

def main():
    compute_means()


if __name__ == '__main__':
    main()
