import io
import cv2
import numpy as np

class ImageBuffer():
    def __init__(self, filenames, keys):
        data, index_table = self.prepare(filenames, keys)
        self.data = data
        self.index_table = index_table

    def __getitem__(self, idx):
        offset,end = self.index_table[idx]
        data_at_idx = self.data[offset:end]
        im = cv2.imdecode(data_at_idx, cv2.IMREAD_COLOR)
        return im[:,:,::-1]

    def __len__(self):
        return len(index_table)

    @staticmethod
    def prepare(filenames, keys):
        data = []
        cumlen = 0
        for filename in filenames:
            with open(filename, 'rb') as f:
                cur_data = f.read()
                data.append(cur_data)
                cumlen += len(cur_data)
        buf = np.empty((cumlen), dtype=np.uint8)
        curidx = 0
        index_table={}
        for key, d in zip(keys, data):
            index_table[key] = (curidx, curidx+len(d))
            buf[curidx:curidx+len(d)] = np.frombuffer(d, dtype=np.uint8)
            curidx += len(d)
        return buf, index_table

# Things to do
# preprocess again with 'jpegtran' -crop -perfect
# Use Imagebuffer to load everything into RAM during training (Partially Done)
#   Implelent ImageBuffer (Done)
#   Integrate ImageBuffer
# implement 'stop caring about early losses' scheduler
#  I.e. train using only last loss at the end of training.
# Verify preprocessing is not broken now (np_mirror_idx optimization)
