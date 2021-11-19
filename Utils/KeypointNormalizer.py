import numpy as np
import torch

class TorchNormalizer(torch.nn.Module):
    def __init__(self, mean, half_prec, dtype=torch.float32):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor(mean, dtype=dtype).reshape(1,-1,2), requires_grad=False)
        self.half_prec = torch.nn.Parameter(torch.tensor(half_prec, dtype=dtype), requires_grad=False)
        np_std = np.array(list(map(np.linalg.inv, half_prec)))
        self.std = torch.nn.Parameter(torch.tensor(np_std, dtype=dtype), requires_grad=False)
        self.num_keypoints= mean.shape[0]

    def normalize(self, vals):
        # vals is Bx17x2
        N = self.num_keypoints
        vals = vals - self.mean
        vals = torch.matmul(self.half_prec.view(-1, N, 2,2), vals.view(-1, N, 2, 1))
        vals = vals.view(-1,N,2)
        return vals

    def denormalize(self, vals):
        # vals is Bx17x2
        N = self.num_keypoints
        vals = torch.matmul(self.std.view(-1, N, 2,2), vals.view(-1, N, 2, 1))
        vals = vals.view(-1, N, 2)+self.mean
        return vals

    def denormalize_prec(self, precs):
        # hps is Bx17x2
        N = self.num_keypoints
        return torch.matmul(self.half_prec.view(-1,N,2,2), torch.matmul(precs, self.half_prec.view(-1, N, 2,2)))
