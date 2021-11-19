import WFLW.WFLW as WFLW
import PIL.Image
import json
from ConfigParser.ConfigParser import SplitType
import os
import time

import numpy as np

import cProfile, pstats, io
from pstats import SortKey

def main():
    ims = []
    anns = []
    num_ims = 10
    idxs = list(range(num_ims))
    dataset = WFLW.PreprocessedWflwfKeypointsDataset(split=SplitType.DEVELOP)
    t_ds = dataset.get_train()
    for idx in idxs:
        im_path = os.path.join(t_ds.image_dir, '{}.png'.format(idx))
        ann_path = os.path.join(t_ds.ann_dir, '{}.json'.format(idx))

        img_PIL = PIL.Image.open(im_path)

        data = img_PIL.getdata()
        full_im = np.asarray(data).reshape(img_PIL.size[1], img_PIL.size[0], 3)
        with open(ann_path, 'r') as f:
            ann = json.load(f)
            ann = WFLW.PreprocessesedWflwAnnotation.deserialize(ann)
        ims.append(full_im)
        anns.append(ann)

    preprocess = WFLW.WflwPreprocessingInstance.randomly_generate_augmentation(224)
    T = 20
    start = time.time()
    pr = cProfile.Profile()
    pr.enable()
    for t in range(T):
        for c_im, c_ann in zip(ims, anns):
         preprocess.apply(full_im, ann)
    pr.disable()
    s = io.StringIO()
    sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())

    end = time.time()
    print('time per aug {}'.format((end-start)/(T*num_ims)))

if __name__ == '__main__':
    main()
