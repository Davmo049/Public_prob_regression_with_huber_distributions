import torch

import mathlib.torch_math as tm
import numpy as np
import matplotlib.pyplot as plt


with torch.no_grad():
    r = 0.0
    rs = []
    r2s = []
    for _ in range(1000):
        A = np.random.normal(size=(32,2,2))
        A[:,0,1]=A[:,1,0]
        A = torch.tensor(A, dtype=torch.float32)
        ev1, evec1 = tm.symeig_2x2(A)
        Ar = torch.bmm(evec1, ev1.view(-1, 2, 1)*evec1.transpose(1,2))
        r = torch.sum((Ar-A).view(-1, 4)**2, dim=1)
        ev2, evec2 = torch.symeig(A, upper=True, eigenvectors=True)
        Arr = torch.bmm(evec2, ev2.view(-1, 2, 1)*evec2.transpose(1,2))
        r2=torch.sum((Arr-A).view(-1, 4)**2, dim=1)
        rs += list(r.numpy())
        r2s += list(r2.numpy())
    print(np.mean(rs))
    print(np.max(rs))
    plt.hist(rs, 1000)
    plt.show()
    print(np.mean(r2s))
    print(np.max(r2s))
    plt.hist(r2s, 1000)
    plt.show()
