import time
import torch
import cProfile, pstats, io
from pstats import SortKey
from ConfigParser.ConfigParser import SplitType
import WFLW.WFLW
import numpy as np

def main_wflw():
    dataset = WFLW.WFLW.PreprocessedWflwfKeypointsDataset(split=SplitType.EVAL)
    t_ds = dataset.get_train()
    print('len training set {}'.format(len(t_ds)))

    pr = cProfile.Profile()
    pr.enable()
    for t in t_ds:
        pass
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

def main_wflw_batched():
    batch_size=32
    dataset = WFLW.WFLW.PreprocessedWflwfKeypointsDataset(split=SplitType.EVAL)
    t_ds = dataset.get_train()
    dataloader_train = torch.utils.data.DataLoader(
        t_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        worker_init_fn=lambda _: np.random.seed(torch.utils.data.get_worker_info().seed % (2**32)),
        pin_memory=True,
        drop_last=True)
    start = time.time()
    for t in dataloader_train:
        pass
    end = time.time()
    print('wflw batched_time: {}'.format(end-start))

def main_coco():
    pass

def main():
    main_wflw()
    # main_wflw_batched()

if __name__ == '__main__':
    main()
