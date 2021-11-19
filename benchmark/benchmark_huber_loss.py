from Losses.Losses import ProbHuberLoss
import time
import torch
import cProfile, pstats, io
from pstats import SortKey


def main_gpu():
    T = 1000
    y = torch.zeros((32, 98, 2))
    loss_function = ProbHuberLoss(1.0, 50.0)
    device = torch.device('cuda')
    m_all = torch.rand((T,32, 98, 5)).to(device)
    x = torch.zeros((32, 98, 5), requires_grad=True)
    x = x.to(device)
    y = y.to(device)
    loss_function = loss_function.to(device)
    pr = cProfile.Profile()
    pr.enable()
    for t in range(T):
        x_e = m_all[t] + x
        losses, _, _ = loss_function(x_e, y, 1.0)
        loss = torch.mean(losses)
        loss.backward()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


def main_cpu():
    T = 1000
    y = torch.zeros((32, 98, 2))
    loss_function = ProbHuberLoss(1.0, 50.0)
    device = torch.device('cuda')
    m_all = torch.rand((T,32, 98, 5)).to(device)
    x = torch.zeros((32, 98, 5), requires_grad=True)
    x = x.to(device)
    pr = cProfile.Profile()
    pr.enable()
    for t in range(T):
        x_e = m_all[t] + x
        x_e = x_e.cpu()
        losses, _, _ = loss_function(x_e, y, 1.0)
        loss = torch.mean(losses)
        loss.backward()
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())


def main():
    main_gpu()


if __name__ == '__main__':
    main()
