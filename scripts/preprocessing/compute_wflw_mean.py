import WFLW.WFLW as WFLW
from WFLW.wflw_normalizer import WflwTorchNormalizer
import matplotlib.pyplot as plt
import tqdm
import numpy as np
import torch
import scipy.linalg

def main():
    ds = WFLW.PreprocessedWflwfKeypointsDataset()
    train = ds.get_train()
    points_sum_X = 0
    points_sum_XX = 0
    N = 0
    normalizer = WflwTorchNormalizer('cpu')
    check_normalizer = True
    for im, points in tqdm.tqdm(train):
        im = im.transpose(1,2,0)
        if check_normalizer:
            points = normalizer.normalize(torch.tensor(points)).numpy()
        points = points.astype(np.float64)
        points_sum_X += points
        points_sum_XX += points.reshape(98, -1,1)*points.reshape(98, 1,-1)
        N += 1
        if N > 10:
            break
    mean = points_sum_X/N
    var = (points_sum_XX-N*mean.reshape(98, -1,1)*mean.reshape(98, 1,-1))/(N-1)
    half_prec = []
    for A in var:
        half_prec.append(np.linalg.inv(scipy.linalg.sqrtm(A)))
    half_prec = np.array(half_prec)

    print('mean')
    print(print_copyable_mat(mean))
    print('half_prec')
    print(print_copyable_mat(half_prec))


def print_copyable_mat(a, depth=0):
    if isinstance(a, np.ndarray):
        if len(a.shape) == 1:
            return '[' +', '.join(map(str, a)) + ']'
        rows = []
        for r in a:
            rows.append(print_copyable_mat(r, depth=depth+1))
        s = '['+(','+'\n'+'  '*(depth)).join(rows)+']'
        return s
    else:
        return str(a)

if __name__ == '__main__':
    main()
