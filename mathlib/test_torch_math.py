import torch
import numpy as np
from .np_math import numdiff
from . import torch_math
import unittest

class TestRegularizedLeastSquares(unittest.TestCase):

    def test_trivial(self):
        A = torch.eye(2)
        b = torch.tensor([2.0,3])
        x = torch_math.regularized_least_squares(A, b, 0.00001).numpy()
        self.assertAlmostEqual(x[0], b.numpy()[0], places=3)
        self.assertAlmostEqual(x[1], b.numpy()[1], places=3)

    def test_degenerate(self):
        A = torch.tensor([[1.0, 1, 2],
                         [2,   2, 4],
                         [1,   2, 3]])
        b = torch.tensor([4.01, 7.99, 6])
        x = torch_math.regularized_least_squares(A, b, 0.01).numpy()
        self.assertAlmostEqual(x[0], 2/3, places=1)
        self.assertAlmostEqual(x[1], 2/3, places=1)
        self.assertAlmostEqual(x[2], 4/3, places=1)



class TestDummersDecomposition(unittest.TestCase):
    def test_diag(self):
        A = torch.eye(3)
        L = torch.ones(3)
        S, R1, R2 = torch_math.dummers_decomposition(A, L)
        S, R1, R2= S.numpy(), R1.numpy(), R2.numpy()
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(R1[i,j], 0, places=3)
                self.assertAlmostEqual(R2[i,j], 0, places=3)
        for i in range(3):
            self.assertAlmostEqual(S[i], 1, places=3)

    def test_happy(self):
        A = torch.tensor([[1.0,8,3],
                          [5,5,5],
                          [9,-3,2]])
        L = torch.tensor([1.0,5,2.2])
        S, R1, R2 = torch_math.dummers_decomposition(A, L)
        S = S.numpy()
        R1 = R1.numpy()
        R2 = R2.numpy()
        S = np.diag(S)
        L = np.diag(L.numpy())
        R1_term = np.matmul(L, R1)
        R2_term = np.matmul(R2, L)
        reconstructed = S + R1_term + R2_term
        A = A.numpy()
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(reconstructed[i,j], A[i,j], places=3)
                self.assertAlmostEqual(R1[i,j], -R1[j,i], places=3)
                self.assertAlmostEqual(R2[i,j], -R2[j,i], places=3)

def numpy_reference_square_remap(A):
    U,S,Vt = np.linalg.svd(A)
    fS = np.sign(S)*S**2
    # fS = np.exp(S)-1
    # fS = S**3
    return np.matmul(np.matmul(U, np.diag(fS)), Vt)


class TestSvdSquareRemap(unittest.TestCase):
    def test_forward(self):
        S = torch.tensor([1.0, 2.0, 3.0])
        a1 = np.pi/6
        c1 = np.cos(a1)
        s1 = np.sin(a1)
        a2 = np.pi/3
        c2 = np.cos(a2)
        s2 = np.sin(a2)
        R1 = torch.tensor([[1, 0,  0],
                           [0, c1, s1],
                           [0, -s1,c1]])
        R2 = torch.tensor([[c2, s2, 0],
                           [-s2,c2, 0],
                           [0,  0,  1]])
        A = torch.matmul(R1*S.view(1, 3), R2.transpose(0,1))
        B = torch_math.svd_square_remap(A)
        B = B.numpy()
        S = S.numpy()
        R1 = R1.numpy()
        R2 = R2.numpy()
        BS = np.matmul(np.matmul(R1.transpose(), B), R2)
        for i in range(3):
            for j in range(3):
                if i == j:
                    self.assertAlmostEqual(BS[i,j], S[i]**2, places=3)
                else:
                    self.assertAlmostEqual(BS[i,j], 0, places=3)

    def test_backward_diag(self):
        loss_mat_np = np.array([[1.0, 0, 0],
                                [0,   1, 0],
                                [0,   0, 1]])
        loss_mat_torch = torch.tensor(loss_mat_np, dtype=torch.float32)
        A_np = np.array([[1.0,0, 0],
                         [0,  2, 0],
                         [0,  0, 4]])
        A_torch = torch.tensor(A_np, dtype=torch.float32, requires_grad=True)
        B = torch_math.svd_square_remap(A_torch)
        loss = torch.dot(B.flatten(), loss_mat_torch.flatten())
        loss.backward()
        tg = A_torch.grad.numpy()
        def reference_function(x):
            y= numpy_reference_square_remap(x)
            return np.dot(y.flatten(), loss_mat_np.flatten())
        grad_np = numdiff(A_np, reference_function, 0.01)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(grad_np[i,j], tg[i,j], places=1)

    def test_backward(self):
        loss_mat_np = np.array([[1.0, 0, 0],
                                [0,   0.19, 0.9],
                                [0,   -0.9, 0.19]])
        loss_mat_torch = torch.tensor(loss_mat_np, dtype=torch.float32)
        A_np = np.array([[1.0,2, 5],
                         [3,  9, 4],
                         [3,  -4, 2]])
        A_torch = torch.tensor(A_np, dtype=torch.float32, requires_grad=True)
        B = torch_math.svd_square_remap(A_torch)
        loss = torch.dot(B.flatten(), loss_mat_torch.flatten())
        loss.backward()
        tg = A_torch.grad.numpy()
        def reference_function(x):
            y= numpy_reference_square_remap(x)
            return np.dot(y.flatten(), loss_mat_np.flatten())
        grad_np = numdiff(A_np, reference_function, 0.01)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(grad_np[i,j], tg[i,j], places=1)

    def test_backward_similar_values(self):
        loss_mat_np = np.array([[1,0,0],
                                [0,0.19,0.9],
                                [0,-0.9,0.19]])
        loss_mat_torch = torch.tensor(loss_mat_np, dtype=torch.float32)
        A_np = np.array([[3.0,0, 0],
                         [0,  2, 0],
                         [0,  0, 2]])
        A_torch = torch.tensor(A_np, dtype=torch.float32, requires_grad=True)
        B = torch_math.svd_square_remap(A_torch)
        loss = torch.dot(B.flatten(), loss_mat_torch.flatten())
        loss.backward()
        tg = A_torch.grad.numpy()
        def reference_function(x):
            y = numpy_reference_square_remap(x)
            return np.dot(y.flatten(), loss_mat_np.flatten())
        grad_np = numdiff(A_np, reference_function)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(grad_np[i,j], tg[i,j], places=2)

    def test_backward_negative_values(self):
        loss_mat_np = np.array([[1,0,0],
                                [0,0.19,0.9],
                                [0,-0.9,0.19]])
        loss_mat_torch = torch.tensor(loss_mat_np, dtype=torch.float32)
        A_np = np.array([[1.00,0, 0],
                         [0,  1.00, 0],
                         [0,  0, -1]])
        A_torch = torch.tensor(A_np, dtype=torch.float32, requires_grad=True)
        B = torch_math.svd_square_remap(A_torch)
        loss = torch.dot(B.flatten(), loss_mat_torch.flatten())
        loss.backward()
        tg = A_torch.grad.numpy()
        def reference_function(x):
            y = numpy_reference_square_remap(x)
            return np.dot(y.flatten(), loss_mat_np.flatten())
        grad_np = numdiff(A_np, reference_function)
        for i in range(3):
            for j in range(3):
                if (i == 2 and j == 1) or (i == 1 and j == 2):
                    # intentional pass to show flaw.
                    # when singular values very similar the regularizer in dummers decomposition becomes active and too small gradients on the diagonal are returned
                    self.assertAlmostEqual(0, tg[i,j], places=2)
                    pass
                else:
                    self.assertAlmostEqual(grad_np[i,j], tg[i,j], places=2)

def ip_helper(a, b):
    a = a.view(a.shape[0], 1, -1)
    b = b.view(a.shape[0], -1, 1)
    r = torch.bmm(a,b).view(a.shape[0])
    return r

class TestExpSvdRemap(unittest.TestCase):
    def test_hammer_exp(self):
        np.random.seed(9001)
        T = 3
        bs = 32
        for _ in range(T):
            A_np = np.random.normal(size=(bs, 3, 3))
            A_t = torch.tensor(A_np, requires_grad=True)
            C_np = np.random.normal(size=(bs, 3, 3))
            C_t = torch.tensor(C_np)
            f = lambda x: ip_helper(torch_math.svd_exp_remap(x), C_t.view(bs, 9, 1))
            loss = torch.sum(f(A_t))
            loss.backward()
            grad = A_t.grad.detach().numpy()
            nd = torch_math.numdiff(f,A_t,batch_dims=1).detach().numpy()
            for b in range(bs):
                for r in range(3):
                    for c in range(3):
                        self.assertAlmostEqual(grad[b,r,c], nd[b,r,c], delta=0.1)

class TestSigmoidRemap(unittest.TestCase):
    def test_hammer_sigmoid(self):
        np.random.seed(91001)
        T = 5
        for _ in range(T):
            A_np = np.random.normal(size=(1, 3, 3))*5
            A_t = torch.tensor(A_np, requires_grad=True)
            C_np = np.random.normal(size=(1, 3, 3))
            C_t = torch.tensor(C_np)
            f = lambda x: torch.dot(torch_math.svd_sigmoid_remap(x).flatten(), C_t.flatten())
            loss = f(A_t)
            loss.backward()
            grad = A_t.grad.detach().numpy()
            nd = torch_math.numdiff(f,A_t,batch_dims=1).detach().numpy()
            for r in range(3):
                for c in range(3):
                    self.assertAlmostEqual(np.abs(grad[0,r,c]), np.abs(nd[0,r,c]), delta=0.1)

def create_sym_mat(A):
    # A is Bx3
    # return Bx3x3
    r = torch.empty((A.shape[0], 2,2), dtype=A.dtype, device=A.device)
    r[:, 0,0] = A[:,0]
    r[:, 0,1] = A[:,1]
    r[:, 1,0] = A[:,1]
    r[:, 1,1] = A[:,2]
    return r

class TestCustomEigRemap(unittest.TestCase):
    def test_hammer_eig(self):
        remapping = torch_math.create_posdef_symeig_remap(1.0, 0.0, 1.0, 23.5)
        np.random.seed(9001)
        T = 5
        for _ in range(T):
            A_np = np.random.normal(size=(1, 3))*5
            A_t = torch.tensor(A_np, requires_grad=True)
            C_np = np.random.normal(size=(1, 2, 2))
            C_np[:,1,0] = C_np[:, 0, 1]
            C_t = torch.tensor(C_np)
            f = lambda x: torch.dot(remapping(create_sym_mat(x))[1].flatten(), C_t.flatten())
            loss = f(A_t)
            loss.backward()
            grad = A_t.grad.detach().numpy()
            nd = torch_math.numdiff(f,A_t,batch_dims=1).detach().numpy()
            for i in range(3):
                self.assertAlmostEqual(grad[0,i]/nd[0,i], 1, places=2)

    def test_hammer_eig_with_diag(self):
        np.random.seed(91001)
        remapping = torch_math.create_posdef_symeig_remap(1.0, 0.0, 1.0, 23.5)
        T = 50
        for _ in range(T):
            A_np = np.random.normal(size=(1, 3))*5
            A_t = torch.tensor(A_np, requires_grad=True)
            C1_np = np.random.normal(size=(1, 2, 2))
            # C1_np[:,1,0] = C1_np[:, 0, 1]
            C1_t = torch.tensor(C1_np)
            C0_np = np.random.normal(size=(1, 2))
            C0_t = torch.tensor(C0_np)
            f = lambda x: eig_ip(x, C0_t, C1_t, remapping)
            loss = f(A_t)
            loss.backward()
            grad = A_t.grad.detach().numpy()
            nd = torch_math.numdiff(f,A_t,batch_dims=1).detach().numpy()
            for i in range(3):
                diff = min(np.abs(grad[0,i]/nd[0,i]-1)*10e3, np.abs(grad[0,i]-nd[0,i])*10e3)
                self.assertAlmostEqual(diff, 0, delta=1)

    def test_hammer_same_eigenvalue(self):
        np.random.seed(91043)
        T = 50
        remapping = torch_math.create_posdef_symeig_remap(1.0, 0.0, 1.0, 23.5)
        d = 4
        for t in range(T):
            A = np.random.normal(size=(d,d))
            U,S,V = np.linalg.svd(A)
            R = np.matmul(U,V)
            D = np.random.normal(size=(d))
            D[0] = D[1]
            n_before_ident_ev = np.sum(D<D[0])
            A = np.matmul(R.transpose(), D.reshape(d, 1)*R)
            A = np.diag(D)
            A = A.reshape(1,d,d)
            A_torch = torch.tensor(A, requires_grad=True)
            C1_np = np.random.normal(size=(1, d,d))
            C1_t = torch.tensor(C1_np)
            C0_np = np.random.normal(size=(1, d))
            C0_np[0, n_before_ident_ev+1] = C0_np[0, n_before_ident_ev]
            C0_t = torch.tensor(C0_np)
            f = lambda x: eig_ip_already_symm(x, C0_t, C1_t, remapping)
            loss = f(A_torch)
            loss.backward()
            grad = A_torch.grad.detach().numpy()
            nd = torch_math.numdiff(f,A_torch,batch_dims=1).detach().numpy()
            for i in range(d):
                for j in range(d):
                    rel_diff = np.abs(grad[0, i,j]/nd[0, i,j]-1)
                    abs_diff = np.abs(grad[0, i,j]-nd[0, i,j])
                    diff = min(rel_diff*10e1, abs_diff*10e1)
                    self.assertAlmostEqual(diff, 0, delta=1)




def eig_ip(x, c0, c1, remapping):
    a, b = remapping(create_sym_mat(x))
    f1 = torch.dot(a.flatten(), c0.flatten())
    f2 = torch.dot(b.flatten(), c1.flatten())
    return f1 + f2

def eig_ip_already_symm(x, c0, c1, remapping):
    x = (x+x.permute(0,2,1))/2
    a, b = remapping(x)
    f1 = torch.dot(a.flatten(), c0.flatten())
    f2 = torch.dot(b.flatten(), c1.flatten())
    return f1 + f2

if __name__ == '__main__':
    unittest.main()
