import torch
import numpy as np

def numdiff(f, X, eps=10e-5, batch_dims=1):
    # f is function
    # X is input
    # eps is step to use
    # batch dim is number of first dimensions to ignore as part of a batch (independent)
    X = X.detach()
    bs = np.prod(X.shape[:batch_dims])
    dims_per_batch = np.prod(X.shape[batch_dims:])
    X_like = X.view(bs, dims_per_batch)
    fx = f(X).view(bs)
    diff = torch.empty((bs, dims_per_batch), dtype=X.dtype, device=X.device)
    for i in range(dims_per_batch):
        X_eps = X_like.clone()
        X_eps[:, i] += eps
        diff[:, i] = (f(X_eps.view(*X.shape)).view(bs)-fx)/eps
    return diff.view(*X.shape)


def regularized_least_squares(A_in, b_in, eps):
    # A is * x n x m
    # b is * x n
    # eps is real > 0 return * x m vector corresponding to min_x ||Ax-b||_2^2 + eps x'x
    # note if eps is 0 use torch.solve instead
    extra_dims = A_in.shape[:-2]
    A = A_in.view(-1, A_in.shape[-2], A_in.shape[-1])
    b = b_in.view(-1, b_in.shape[-1])
    fac1 = torch.matmul(A.transpose(1,2),A)
    fac1.view(-1, A.shape[1]*A.shape[2])[:, ::(A.shape[-1]+1)] += eps # add eps to diagonal
    fac2 = torch.matmul(A.transpose(1,2),b.view(-1, b.shape[1], 1))
    ret = torch.solve(fac2, fac1)[0] # second return is LU of A
    return ret.view(list(extra_dims) + [A.shape[1]])

def dummers_decomposition(A_in, L_in):
    # A is * x m x m
    # L is * x m
    # creates a decomposition like A \approx diag(S) + diag(L)R_1 +  R_2diag(L)
    # where R_1 and R_2 have R^T = -R
    # return S, R_1, R_2
    extra_dims = A_in.shape[:-2]
    m = A_in.shape[-2]
    A = A_in.view(-1, m, m)
    L = L_in.view(-1, m)
    S = A.view(-1, m*m)[:, ::(m+1)]
    triu = np.triu_indices(m, 1)
    Lin_comb = torch.empty((A.shape[0], len(triu[0]), 2,2), dtype=A.dtype, device=A.device)
    b = torch.empty((A.shape[0], len(triu[0]), 2), dtype=A.dtype, device=A.device)
    for vec_idx, (r,c) in enumerate(zip(*triu)):
        Lin_comb[:, vec_idx, 0, 0] = L[:, r]
        Lin_comb[:, vec_idx, 0, 1] = L[:, c]
        Lin_comb[:, vec_idx, 1, 0] = -L[:, c]
        Lin_comb[:, vec_idx, 1, 1] = -L[:, r]
        b[:, vec_idx, 0] = A[:, r, c]
        b[:, vec_idx, 1] = A[:, c, r]
    solution = regularized_least_squares(Lin_comb, b, 10e-5)
    R1 = torch.zeros((A.shape[0], m, m), dtype=A.dtype, device=A.device)
    R2 = torch.zeros((A.shape[0], m, m), dtype=A.dtype, device=A.device)
    for vec_idx, (r,c) in enumerate(zip(*triu)):
        R1[:, r,c] = solution[:, vec_idx, 0]
        R1[:, c,r] = -solution[:, vec_idx, 0]
        R2[:, r,c] = solution[:, vec_idx, 1]
        R2[:, c,r] = -solution[:, vec_idx, 1]
    return S.view(list(extra_dims) + [m]), R1.view(list(extra_dims) + [m,m]), R2.view(list(extra_dims) + [m,m])

def sym_dummers_decomposition(A, S):
    # A is Bxnxn s.t. A = A^T
    # return L, R such that L+R*S + S*R = A
    # where L is diagonal and R = -R^T
    B = A.shape[0]
    n = A.shape[1]
    L = A.view(A.shape[0], -1)[:,::n+1]
    eig_diff = S.view(B, n, 1) - S.view(B, 1, n)
    R = A / eig_diff
    return L, R


def svd_forward(ctx, A_in, f):
    m = A_in.shape[-1]
    A = A_in.view(-1, m, m)
    U,S,V = torch.svd(A)
    ctx.save_for_backward(U,S,V)
    f_S = f(S)
    # f_S = torch.sign(S)*(torch.exp(torch.abs(S))-1)
    Uxf_s = U*f_S.view(-1, 1, m)
    Uxf_sxVt = torch.matmul(Uxf_s, V.transpose(1,2))
    return Uxf_sxVt.view(A_in.shape)

def svd_backward(ctx, grad_output_in, f, df):
    m = grad_output_in.shape[-1]
    grad_output = grad_output_in.view(-1, m, m)
    U,S,V = ctx.saved_tensors
    m = S.shape[1]
    compensated_grad = torch.matmul(torch.matmul(U.transpose(1,2), grad_output), V)
    # note method is general just replace f_S and df_ds with other expression
    df_ds =  df(S)
    f_S = f(S)
    # example f_S = S**3
    # df_ds = S**2*3
    # f_S = S**3
    # below is example for activation f_S = sign(S)*(exp(abs(S))-1)
    # df_ds = torch.sign(S)*torch.exp(torch.abs(S))
    # f_S = torch.sign(S)*(torch.exp(torch.abs(S))-1)
    # TODO redo math check this shouldn't S be f_S?
    # it seems to work with several different activations so it's probably correct
    lam, dJ_dVt, dJ_dU = dummers_decomposition(compensated_grad, S)
    dJ_dA_comp = dJ_dU * f_S.view(-1, 1, m) + dJ_dVt * f_S.view(-1, m, 1)
    dJ_ds = lam * df_ds
    dJ_dA_comp.view(-1, m*m)[:, ::m+1] += dJ_ds # add to diagonal
    return torch.matmul(torch.matmul(U, dJ_dA_comp), V.transpose(1,2)).view(grad_output_in.shape)

def symeig_2x2(A):
    # default is slow https://github.com/pytorch/pytorch/issues/22573
    # we solve our special case here
    # A = Mx[[a, c], [c, b]]
    # return same as torch.symeig(upper=True, eigenvectors=True)
    M = A.shape[0]
    a = A[:,0,0]
    b = A[:,1,1]
    c = A[:,0,1]
    m = (a+b)/2
    c2 = c**2
    diff_v = (a-b)/2
    diff_sq = diff_v**2
    sq = torch.sqrt(diff_sq+c2)
    eigvals = torch.empty((M, 2), dtype=A.dtype, device=A.device)
    eigvals[:, 0].copy_(m+sq)
    eigvals[:, 1].copy_(m-sq)
    eigvecs = torch.empty((M, 2, 2), dtype=A.dtype, device=A.device)
    mask0 = sq < 1e-5 # both values very similar, avoid numerical instability
    a1 = a - eigvals[:, 0]
    b1 = b - eigvals[:, 0]
    a2 = a - eigvals[:, 1]
    b2 = b - eigvals[:, 1]
    mask1 = torch.abs(a1) < torch.abs(b1)
    mask2 = torch.abs(a2) < torch.abs(b2)
    # default case (neither mask0/1/2)
    eigvecs[:,0,0] = -c
    eigvecs[:,1,0] = a1
    eigvecs[:,0,1] = -c
    eigvecs[:,1,1] = a2

    # mask1/2 case, but not mask 0
    eigvecs[mask1,0,0] = b1[mask1]
    eigvecs[mask1,1,0] = -c[mask1]
    eigvecs[mask2,0,1] = b2[mask2]
    eigvecs[mask2,1,1] = -c[mask2]

    # mask0 case
    eigvecs[mask0,0,0] = 1
    eigvecs[mask0,0,1] = 0
    eigvecs[mask0,1,0] = 0
    eigvecs[mask0,1,1] = 1
    norm = torch.norm(eigvecs, dim=1)
    eigvecs = eigvecs / norm.view(M,2,1)
    return eigvals, eigvecs

def symeig_forward(ctx, A_in, f):
    m = A_in.shape[-1]
    A = A_in.view(-1, m, m)
    D, V = symeig_2x2(A)
    # D, V = torch.symeig(A, upper=True, eigenvectors=True)
    f_D = f(D)
    ctx.save_for_backward(D,V, f_D)
    Vxf_D = V*f_D.view(-1, 1,m)
    Vxf_DxVt = torch.bmm(Vxf_D, V.transpose(1,2))
    return f_D, Vxf_DxVt.view(A_in.shape)

def symeig_backward_old(ctx, grad_eigen_in, grad_output_in, f, df):
    symeig_backward2(ctx, grad_eigen_in, grad_output_in, f, df)
    m = grad_output_in.shape[-1]
    grad_output = grad_output_in.view(-1, m, m)
    grad_output = (grad_output+grad_output.transpose(1,2))/2
    S,V,f_S = ctx.saved_tensors
    compensated_grad = torch.matmul(torch.matmul(V.transpose(1,2), grad_output), V)
    compensated_grad.view(-1, m*m)[:, ::m+1] += grad_eigen_in.view(-1, m)
    df_ds =  df(S)
    lam, dJ_dVt = sym_dummers_decomposition(compensated_grad, S)
    dJ_dV = dJ_dVt.transpose(1,2)

    dJ_dA_comp = dJ_dV * f_S.view(-1, 1, m) + dJ_dVt * f_S.view(-1, m, 1)
    close_vals = torch.abs(S.view(-1, 1, m)-S.view(-1, m, 1)) < 10e-3
    close_replace = df_ds.view(-1, m, 1).repeat(1,1,m)
    close_replace.view(-1, m*m)[:, ::m+1] = 0
    dJ_dA_comp[close_vals] = close_replace[close_vals]
    dJ_ds = lam * df_ds
    dJ_dA_comp = torch.clone(dJ_dA_comp)
    dJ_dA_comp.view(-1, m*m)[:, ::m+1] += dJ_ds # add to diagonal
    return torch.matmul(torch.matmul(V, dJ_dA_comp), V.transpose(1,2)).view(grad_output_in.shape)

def symeig_backward(ctx, grad_eigen_in, grad_output_in, f, df):
    m = grad_output_in.shape[-1]
    grad_output = grad_output_in.view(-1, m, m)
    grad_output = (grad_output+grad_output.transpose(1,2))/2
    S, V, f_S = ctx.saved_tensors
    compensated_grad = torch.matmul(torch.matmul(V.transpose(1,2), grad_output), V)
    df_ds =  df(S)

    # note when you have identical eigen values you would hope grad_eigen_in is the same for all identical eigenvalues
    # if not the following line is a bug since the eigenvalues shuffle around when increasing any lambda
    compensated_grad.view(-1, m*m)[:,::m+1] += grad_eigen_in
    eig_diff = S.view(-1, m, 1)-S.view(-1, 1, m)
    close_vals = torch.abs(eig_diff) < 10e-3
    incline = (f_S.view(-1, m, 1) - f_S.view(-1, 1, m)) / eig_diff
    close_val_backup = df_ds.view(-1, m, 1).repeat(1,1,m)
    incline[close_vals] = close_val_backup[close_vals]
    dJ_dA_comp = compensated_grad * incline

    return torch.matmul(torch.matmul(V, dJ_dA_comp), V.transpose(1,2)).view(grad_output_in.shape)

class class_svd_square_remap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_in):
        return svd_forward(ctx, A_in, class_svd_square_remap.f)

    @staticmethod
    def backward(ctx, grad_output_in):
        return svd_backward(ctx, grad_output_in, class_svd_square_remap.f, class_svd_square_remap.df)

    @staticmethod
    def f(s):
        return torch.sign(s)*s**2

    @staticmethod
    def df(s):
        return torch.sign(s)*s*2
svd_square_remap = class_svd_square_remap.apply

class class_svd_exp_remap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_in):
        return svd_forward(ctx, A_in, class_svd_square_remap.f)

    @staticmethod
    def backward(ctx, grad_output_in):
        return svd_backward(ctx, grad_output_in, class_svd_square_remap.f, class_svd_square_remap.df)

    @staticmethod
    def f(s):
        return torch.exp(s)-1

    @staticmethod
    def df(s):
        return torch.exp(s)
svd_exp_remap = class_svd_exp_remap.apply


class class_svd_sigmoid_remap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_in):
        return svd_forward(ctx, A_in, class_svd_square_remap.f)

    @staticmethod
    def backward(ctx, grad_output_in):
        return svd_backward(ctx, grad_output_in, class_svd_square_remap.f, class_svd_square_remap.df)

    @staticmethod
    def f(s):
        return 1/(1+torch.exp(-s))

    @staticmethod
    def df(s):
        return 1/((1+torch.exp(s))*(1+torch.exp(-s)))
svd_sigmoid_remap = class_svd_sigmoid_remap.apply

class PosdefSymeigRemap():
    # this function has deep rooted problems with regards to how it interfaces with torch.
    # Modules have state but not custom backward
    # Functions have custom backward but not state
    # I want both, so I create a class refering to variables in a local scope instead.
    # That is kind of fucked up
    def __init__(self,a,b,c,d):
        self.a = a
        self.b = b+2
        self.c = c
        self.d = d

    def __call__(self, x):
        D = x.shape[-1]
        eye = torch.eye(D, dtype=x.dtype, device=x.device).view(-1, D, D)
        x = (x*self.c)+(self.d*eye)
        eigs, x = linexp_symm(x)
        x = x*self.a+self.b*eye
        eigs = eigs*self.a+self.b
        return eigs, x

class LinExpSymm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_in):
        return symeig_forward(ctx, A_in, LinExpSymm.f)

    @staticmethod
    def backward(ctx, grad_eigen_in, grad_output_in):
        return symeig_backward(ctx, grad_eigen_in, grad_output_in, LinExpSymm.f, LinExpSymm.df)

    @staticmethod
    def f(s):
        mask = s < -1
        ret = torch.clone(s)
        ret[mask] = torch.exp(s[mask]+1)-2
        return ret

    def df(s):
        mask = s < -1
        ret = torch.ones_like(s)
        ret[mask] = torch.exp(s[mask]+1)
        return ret
linexp_symm = LinExpSymm.apply

class IdentitySymm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_in):
        return symeig_forward(ctx, A_in, IdentitySymm.f)

    @staticmethod
    def backward(ctx, grad_eigen_in, grad_output_in):
        return symeig_backward(ctx, grad_eigen_in, grad_output_in, IdentitySymm.f, IdentitySymm.df)

    @staticmethod
    def f(s):
        return s

    def df(s):
        return torch.ones(s.shape, dtype=s.dtype, device=s.device)
identity_symm = IdentitySymm.apply



class PosdefSymeigRemap(torch.nn.Module):
    def __init__(self, a,b,c,d):
        super().__init__()
        self.a = torch.Parameter(a)
        self.b = torch.Parameter(b)
        self.c = torch.Parameter(c)
        self.d = torch.Parameter(d)

    def forward(self, s):
        s = c*s+d
        mask = s < -1
        ret = 2+torch.clone(s)
        ret[mask] = torch.exp(s[mask]+1)
        ret = a*ret+b
        return ret

    def df(s):
        s = c*s+d
        mask = s < -1
        ret = torch.ones_like(s)
        ret[mask] = torch.exp(s[mask]+1)
        ret = a*c*ret
        return ret



def create_posdef_symeig_remap(a,b,c,d):
    # this function has deep rooted problems with regards to how it interfaces with torch.
    # Modules have state but not custom backward
    # Functions have custom backward but not state
    # I want both, so I create a class refering to variables in a local scope instead.
    # That is kind of fucked up

    def f(s):
        s = c*s+d
        mask = s < -1
        ret = 2+torch.clone(s)
        ret[mask] = torch.exp(s[mask]+1)
        ret = a*ret+b
        return ret

    def df(s):
        s = c*s+d
        mask = s < -1
        ret = torch.ones_like(s)
        ret[mask] = torch.exp(s[mask]+1)
        ret = a*c*ret
        return ret

    class TMPCLASS(torch.autograd.Function):
        @staticmethod
        def forward(ctx, A_in):
            return symeig_forward(ctx, A_in, f)

        @staticmethod
        def backward(ctx, grad_eigen_in, grad_output_in):
            return symeig_backward(ctx, grad_eigen_in, grad_output_in, f, df)

    return TMPCLASS.apply

class class_symeig_exp_remap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A_in):
        return symeig_forward(ctx, A_in, class_symeig_exp_remap.f)

    @staticmethod
    def backward(ctx, grad_eigen_in, grad_output_in):
        return symeig_backward(ctx, grad_eigen_in, grad_output_in, class_symeig_exp_remap.f, class_symeig_exp_remap.df)

    @staticmethod
    def f(s):
        return torch.exp(s)

    @staticmethod
    def df(s):
        return torch.exp(s)
eig_exp_remap = class_symeig_exp_remap.apply
