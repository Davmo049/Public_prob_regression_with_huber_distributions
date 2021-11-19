import ctypes
import os
import numpy as np

path_this = os.path.abspath(__file__)
dir_this = '/'+'/'.join(path_this.split('/')[:-1])
so_path = os.path.join(dir_this, 'build/library.so')
loaded_library = ctypes.CDLL(so_path)

c_welzl = loaded_library.welzl_plain

def welzl_c(points):
    points = points.astype(np.float64)
    assert(len(points.shape) == 2)
    N = points.shape[0]
    d = points.shape[1]
    center = np.empty((d), dtype=np.float64)
    radius = np.empty((1), dtype=np.float64)
    pointer = (ctypes.c_double*(N*d)).from_buffer(points.data)
    center_p = (ctypes.c_double*d).from_buffer(center.data)
    radius_p = (ctypes.c_double*1).from_buffer(radius.data)
    c_welzl(center_p, radius_p, pointer, N, d);
    return center, np.sqrt(radius[0])
