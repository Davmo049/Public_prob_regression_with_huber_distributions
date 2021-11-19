import torch
from mathlib import torch_math as torch_math

class LossType(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def deserialize(dic):
        if dic['_type'] == 'SimpleHuberLoss':
            return LossSimpleHuber.deserialize(dic)
        elif dic['_type'] == 'ProbHuberLoss':
            return ProbHuberLoss.deserialize(dic)
        elif dic['_type'] == 'ProbGaussLoss':
            return ProbGaussLoss.deserialize(dic)
        elif dic['_type'] == 'ProbCharbonnierLoss':
            return ProbCharbonnier.deserialize(dic)
        elif dic['_type'] == 'L2Loss':
            return L2Loss.deserialize(dic)
        elif dic['_type'] == 'L1Loss':
            return L1Loss.deserialize(dic)
        elif dic['_type'] == 'CharbonnierLoss':
            return CharbonnierLoss.deserialize(dic)


class LossSimpleHuber(LossType):
    def __init__(self, cutoff_mag):
        super().__init__()
        self.cutoff_mag = cutoff_mag

    def __call__(self, values, targets, weights):
        B = values.shape[0]
        N = values.shape[1]

        pred = values[:, :, :2].view(-1, 2)
        gt = targets.view(-1,2)
        norm = torch.norm(gt-pred, dim=1)
        norm2 = norm**2
        losses_per_kp = norm * self.cutoff_mag
        mask = norm < self.cutoff_mag
        losses_per_kp[mask] = norm2[mask]
        losses = torch.sum(losses_per_kp.view(B, N)*weights, dim=1)
        half_prec = torch.eye(2).view(1,1,2,2).repeat(B,N,1,1).to(values.device)
        return losses, half_prec, pred

    def serialize(self):
        return {'_type': 'SimpleHuberLoss',
                'cutoff_magnitude': self.cutoff_mag}

    @staticmethod
    def deserialize(dic):
        cutoff_mag = dic['cutoff_magnitude']
        return LossSimpleHuber(cutoff_mag)


def dbgprint(*x):
    pass
    # print(*x)

def plain_huber_loss(values, targets, weights, cutoff_mag=1.0):
    B = values.shape[0]
    N = values.shape[1]

    pred = values[:, :, :2].view(-1, 2)
    gt = targets.view(-1,2)
    norm = torch.norm(gt-pred, dim=1)
    norm2 = norm**2
    losses_per_kp = norm * cutoff_mag
    mask = norm < cutoff_mag
    losses_per_kp[mask] = norm2[mask]
    losses = torch.sum(losses_per_kp.view(B, N)*weights, dim=1)
    half_prec = torch.eye(2).view(1,1,2,2).repeat(B,N,1,1).to(values.device)
    return losses, half_prec, pred 


def huber_term(A, x, mu, mu_direct, delta):
        A_shape_in = A.shape
        D = A_shape_in[-1]
        A = A.view(-1, D, D)
        N = A.shape[0]
        x = x.view(N, D, 1)
        mu = mu.view(N, D, 1)
        if mu_direct:
            diff = torch.bmm(A,(x-mu)).view(N,D)
        else:
            diff = (torch.bmm(A,x)-mu).view(N,D)
        term = torch.matmul(diff.view(N,1,D), diff.view(N,D,1)).view(N)/2
        if delta > 0:
            const_factor = 1/delta
        else:
            const_factor = 1.0 # not important since mask will make sure this case is not covered
        const_term = -delta/2
        mask = term > delta
        term = term*const_factor
        term[mask] = torch.norm(diff[mask], dim=1)+const_term
        return term.view(A_shape_in[:-2])

def gauss_term(A, x, mu, mu_direct):
        A_shape_in = A.shape
        D = A_shape_in[-1]
        A = A.view(-1, D, D)
        N = A.shape[0]
        x = x.view(N, D, 1)
        mu = mu.view(N, D, 1)
        if mu_direct:
            diff = torch.bmm(A,(x-mu)).view(N,D)
        else:
            diff = (torch.bmm(A,x)-mu).view(N,D)
        term = torch.matmul(diff.view(N,1,D), diff.view(N,D,1)).view(N)/2
        return term.view(A_shape_in[:-2])


def charbonnier_term(A, x, mu, mu_direct):
        A_shape_in = A.shape
        D = A_shape_in[-1]
        A = A.view(-1, D, D)
        N = A.shape[0]
        x = x.view(N, D, 1)
        mu = mu.view(N, D, 1)
        if mu_direct:
            diff = torch.bmm(A,(x-mu)).view(N,D)
        else:
            diff = (torch.bmm(A,x)-mu).view(N,D)
        term = torch.matmul(diff.view(N,1,D), diff.view(N,D,1)).view(N)/2
        term = torch.sqrt((term+1))-1
        return term.view(A_shape_in[:-2])


class ProbCharbonnier(LossType):
    def __init__(self, min_reasonable_half_prec, max_reasonable_half_prec, mu_direct, only_diagonal=False):
        super().__init__()
        self.min_reasonable = min_reasonable_half_prec
        self.max_reasonable = max_reasonable_half_prec
        min_reasonable = float(min_reasonable_half_prec)
        max_reasonable = float(max_reasonable_half_prec)
        min_allowed = 0.0
        a = min_reasonable - min_allowed
        b = min_allowed
        c = 1/a
        d = (max_reasonable-min_reasonable)/(2*a)-1
        self.mu_direct = mu_direct
        self.remapping = torch_math.create_posdef_symeig_remap(a,b,c,d)
        self.only_diagonal = only_diagonal
        # self.remapping = torch_math.PosdefSymeigRemap(a,b,c,d)


    def __call__(self, values, targets, weights):
        # values is BxNx5
        # targets is BxNx2
        # weights is BxN

        # return loss, mean (BxNx2), precision (BxNx2x2)
        # mean output is detached
        B = values.shape[0]
        N = values.shape[1]
        mu = values[:, :, :2].view(-1, 2)
        targets = targets.view(-1, 2)
        A = torch.empty((B, N, 2, 2), dtype=values.dtype, device=values.device)
        A_params = values[:, :, 2:]
        A[:, :, 0,0] = A_params[:, :, 0]
        if not self.only_diagonal:
            A[:, :, 0,1] = A_params[:, :, 1] * 0.707106
            A[:, :, 1,0] = A_params[:, :, 1] * 0.707106 # unnecessary symeig only cares about upper region (parameter upper=True)
        else:
            A[:, :, 0,1] = 0 
            A[:, :, 1,0] = 0  # unnecessary symeig only cares about upper region (parameter upper=True)
        A[:, :, 1,1] = A_params[:, :, 2]
        A = A.view(-1, 2,2)
        eigs, half_prec = self.remapping(A)
        reg_loss = charbonnier_term(half_prec, targets, mu, self.mu_direct)
        reg_loss_per_kp = weights*reg_loss.view(B, N)
        reg_loss = torch.sum(reg_loss_per_kp, dim=1)
        tr_per_mat = -torch.sum(torch.log(eigs), dim=1)
        det_loss_per_kp = weights*(tr_per_mat.view(B,N))
        det_loss = torch.sum(det_loss_per_kp, dim=1)
        if self.mu_direct:
            modes = mu.detach()
        else:
            try:
               modes = torch.solve(mu.view(B*N, 2, 1), half_prec)[0].detach()
            except RuntimeError:
                modes = mu*0.0
        return reg_loss+det_loss, half_prec.view(B,N,2,2), modes.view(B,N,2)

    def serialize(self):
        return {'_type': 'ProbCharbonnierLoss',
                'min_reasonable': self.min_reasonable,
                'max_reasonable': self.max_reasonable,
                'mu_direct': self.mu_direct,
                'only_diagonal': self.only_diagonal
                }

    @staticmethod
    def deserialize(dic):
        min_reasonable = dic['min_reasonable']
        max_reasonable = dic['max_reasonable']
        only_diagonal = dic.get('only_diagonal', False)
        if 'mu_direct' not in dic:
            print('WARNING FORGOT TO SET MU_DIRECT in loss')
            mu_direct = False
        else:
            mu_direct = dic['mu_direct']
        return ProbCharbonnier(min_reasonable, max_reasonable, mu_direct, only_diagonal)


class ProbGaussLoss(LossType):
    def __init__(self, min_reasonable_half_prec, max_reasonable_half_prec, mu_direct, only_diagonal=False):
        super().__init__()
        self.min_reasonable = min_reasonable_half_prec
        self.max_reasonable = max_reasonable_half_prec
        min_reasonable = float(min_reasonable_half_prec)
        max_reasonable = float(max_reasonable_half_prec)
        min_allowed = 0.0
        a = min_reasonable - min_allowed
        b = min_allowed
        c = 1/a
        d = (max_reasonable-min_reasonable)/(2*a)-1
        self.mu_direct = mu_direct
        self.remapping = torch_math.create_posdef_symeig_remap(a,b,c,d)
        self.only_diagonal = only_diagonal
        # self.remapping = torch_math.PosdefSymeigRemap(a,b,c,d)


    def __call__(self, values, targets, weights):
        # values is BxNx5
        # targets is BxNx2
        # weights is BxN

        # return loss, mean (BxNx2), precision (BxNx2x2)
        # mean output is detached
        B = values.shape[0]
        N = values.shape[1]
        mu = values[:, :, :2].view(-1, 2)
        targets = targets.view(-1, 2)
        A = torch.empty((B, N, 2, 2), dtype=values.dtype, device=values.device)
        A_params = values[:, :, 2:]
        A[:, :, 0,0] = A_params[:, :, 0]
        if not self.only_diagonal:
            A[:, :, 0,1] = A_params[:, :, 1] * 0.707106
            A[:, :, 1,0] = A_params[:, :, 1] * 0.707106 # unnecessary symeig only cares about upper region (parameter upper=True)
        else:
            A[:, :, 0,1] = 0 
            A[:, :, 1,0] = 0  # unnecessary symeig only cares about upper region (parameter upper=True)
        A[:, :, 1,1] = A_params[:, :, 2]
        A = A.view(-1, 2,2)
        eigs, half_prec = self.remapping(A)
        reg_loss = gauss_term(half_prec, targets, mu, self.mu_direct)
        reg_loss_per_kp = weights*reg_loss.view(B, N)
        reg_loss = torch.sum(reg_loss_per_kp, dim=1)
        tr_per_mat = -torch.sum(torch.log(eigs), dim=1)
        det_loss_per_kp = weights*(tr_per_mat.view(B,N))
        det_loss = torch.sum(det_loss_per_kp, dim=1)
        if self.mu_direct:
            modes = mu.detach()
        else:
            try:
               modes = torch.solve(mu.view(B*N, 2, 1), half_prec)[0].detach()
            except RuntimeError:
                modes = mu*0.0
        return reg_loss+det_loss, half_prec.view(B,N,2,2), modes.view(B,N,2)

    def serialize(self):
        return {'_type': 'ProbGaussLoss',
                'min_reasonable': self.min_reasonable,
                'max_reasonable': self.max_reasonable,
                'mu_direct': self.mu_direct,
                'only_diagonal': self.only_diagonal
                }

    @staticmethod
    def deserialize(dic):
        min_reasonable = dic['min_reasonable']
        max_reasonable = dic['max_reasonable']
        only_diagonal = dic.get('only_diagonal', False)
        if 'mu_direct' not in dic:
            print('WARNING FORGOT TO SET MU_DIRECT in loss')
            mu_direct = False
        else:
            mu_direct = dic['mu_direct']
        return ProbGaussLoss(min_reasonable, max_reasonable, mu_direct, only_diagonal)



class ProbHuberLoss(LossType):
    def __init__(self, min_reasonable_half_prec, max_reasonable_half_prec, mu_direct, only_diagonal=False, delta=1.0):
        super().__init__()
        self.min_reasonable = min_reasonable_half_prec
        self.max_reasonable = max_reasonable_half_prec
        min_reasonable = float(min_reasonable_half_prec)
        max_reasonable = float(max_reasonable_half_prec)
        min_allowed = 0.0
        a = min_reasonable - min_allowed
        b = min_allowed
        c = 1/a
        d = (max_reasonable-min_reasonable)/(2*a)-1
        self.mu_direct = mu_direct
        self.remapping = torch_math.create_posdef_symeig_remap(a,b,c,d)
        self.only_diagonal = only_diagonal
        self.delta = delta
        # self.remapping = torch_math.PosdefSymeigRemap(a,b,c,d)


    def __call__(self, values, targets, weights):
        # values is BxNx5
        # targets is BxNx2
        # weights is BxN

        # return loss, mean (BxNx2), precision (BxNx2x2)
        # mean output is detached
        B = values.shape[0]
        N = values.shape[1]
        mu = values[:, :, :2].view(-1, 2)
        targets = targets.view(-1, 2)
        A = torch.empty((B, N, 2, 2), dtype=values.dtype, device=values.device)
        A_params = values[:, :, 2:]
        A[:, :, 0,0] = A_params[:, :, 0]
        if not self.only_diagonal:
            A[:, :, 0,1] = A_params[:, :, 1] * 0.707106
            A[:, :, 1,0] = A_params[:, :, 1] * 0.707106 # unnecessary symeig only cares about upper region (parameter upper=True)
        else:
            A[:, :, 0,1] = 0 
            A[:, :, 1,0] = 0  # unnecessary symeig only cares about upper region (parameter upper=True)
        A[:, :, 1,1] = A_params[:, :, 2]
        A = A.view(-1, 2,2)
        eigs, half_prec = self.remapping(A)
        reg_loss = huber_term(half_prec, targets, mu, self.mu_direct, self.delta)
        reg_loss_per_kp = weights*reg_loss.view(B, N)
        reg_loss = torch.sum(reg_loss_per_kp, dim=1)
        tr_per_mat = -torch.sum(torch.log(eigs), dim=1)
        det_loss_per_kp = weights*(tr_per_mat.view(B,N))
        det_loss = torch.sum(det_loss_per_kp, dim=1)
        if self.mu_direct:
            modes = mu.detach()
        else:
            try:
               modes = torch.solve(mu.view(B*N, 2, 1), half_prec)[0].detach()
            except RuntimeError:
                modes = mu*0.0
        return reg_loss+det_loss, half_prec.view(B,N,2,2), modes.view(B,N,2)

    def serialize(self):
        return {'_type': 'ProbHuberLoss',
                'min_reasonable': self.min_reasonable,
                'max_reasonable': self.max_reasonable,
                'mu_direct': self.mu_direct,
                'only_diagonal': self.only_diagonal,
                'delta': self.delta
                }

    @staticmethod
    def deserialize(dic):
        min_reasonable = dic['min_reasonable']
        max_reasonable = dic['max_reasonable']
        delta = dic['delta']
        only_diagonal = dic.get('only_diagonal', False)
        if 'mu_direct' not in dic:
            print('WARNING FORGOT TO SET MU_DIRECT in loss')
            mu_direct = False
        else:
            mu_direct = dic['mu_direct']
        return ProbHuberLoss(min_reasonable, max_reasonable, mu_direct, only_diagonal, delta)

class L2Loss(LossType):
    def __init__(self):
        pass

    def __call__(self, values, targets, weights):
        B = values.shape[0]
        N = values.shape[1]
        mu = values[:,:,:2]
        diff = torch.sum((targets-mu)**2, dim=2)
        losses = torch.sum(diff * weights, dim=1)
        half_prec = torch.eye(2).view(1,1,2,2).repeat(B,N,1,1).to(values.device)
        return losses, half_prec, mu

    def serialize(self):
        return {'_type': 'L2Loss'
                }

    @staticmethod
    def deserialize(dic):
        return L2Loss()


class L1Loss(LossType):
    def __init__(self):
        pass

    def __call__(self, values, targets, weights):
        B = values.shape[0]
        N = values.shape[1]
        mu = values[:,:,:2]
        diff = torch.norm(targets-mu, dim=2)
        losses = torch.sum(diff * weights, dim=1)
        half_prec = torch.eye(2).view(1,1,2,2).repeat(B,N,1,1).to(values.device)
        return losses, half_prec, mu

    def serialize(self):
        return {'_type': 'L1Loss'
                }

    @staticmethod
    def deserialize(dic):
        return L1Loss()

class CharbonnierLoss(LossType):
    def __init__(self):
        pass

    def __call__(self, values, targets, weights):
        B = values.shape[0]
        N = values.shape[1]
        mu = values[:,:,:2]
        diff = torch.norm(targets-mu, dim=2)
        loss_val = torch.sqrt(1+diff**2)
        losses = torch.sum(loss_val * weights, dim=1)
        half_prec = torch.eye(2).view(1,1,2,2).repeat(B,N,1,1).to(values.device)
        return losses, half_prec, mu

    def serialize(self):
        return {'_type': 'CharbonnierLoss'
                }

    @staticmethod
    def deserialize(dic):
        return CharbonnierLoss()
