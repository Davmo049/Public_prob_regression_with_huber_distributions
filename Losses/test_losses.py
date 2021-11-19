import unittest
import mathlib.torch_math as torch_math
from Losses.Losses import ProbHuberLoss
import torch
import numpy as np


def get_loss_func_to_diff(loss_func, gt_points):
    def retfun(x):
        losses, _, _ = loss_func(x, gt_points, 1.0)
        return torch.sum(losses)
    return retfun

def main():
    np.random.seed(9009)
    loss_func = ProbHuberLoss(1.0, 50.0)
    N = 1
    B = 1
    out = np.random.normal(size=(N,B,5))
    gt_pos = np.random.normal(size=(N,B,2))
    out = torch.tensor(out, requires_grad=True)
    gt_pos = torch.tensor(gt_pos)
    loss_func_to_diff = get_loss_func_to_diff(loss_func, gt_pos)
    loss = loss_func_to_diff(out)
    loss.backward()
    grad = torch_math.numdiff(loss_func_to_diff, out)
    print(grad)
    print(out.grad)

# def main():
#     np.random.seed(9001)
#     loss_func = torch_math.create_posdef_symeig_remap(1.0, 0.0, 1.0, 23.5)
#     N = 1
#     B = 2
#     out = np.random.normal(size=(N,B,5))
#     gt_pos = np.random.normal(size=(N,B,2))
#     out = torch.tensor(out, requires_grad=True)
#     gt_pos = torch.tensor(gt_pos)
#     loss_func_to_diff = get_loss_func_to_diff(loss_func, gt_pos)
#     grad = torch_math.numdiff(loss_func_to_diff, out)
#     loss = loss_func_to_diff(out)
#     loss.backward()
#     print(grad)
#     print(out.grad)



if __name__ == '__main__':
    main()
