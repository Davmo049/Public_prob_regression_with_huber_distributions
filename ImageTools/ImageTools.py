import numpy as np
import torch
from ImageTools.c_interface import interp_c

class NpAffineTransforms():
    def __init__(self, A):
        self.A = A

    def to_torch(self):
        assert(False)

    def __call__(self, e):
        tmp = np.empty((3, *e.shape[1:]), dtype=np.float)
        tmp[:2] = e
        tmp[2] = 1
        tmp = np.matmul(self.A, tmp.reshape(3, -1))
        normalized = (tmp[:2] / tmp[2].reshape(1,-1)).reshape(e.shape)
        return normalized[:2].reshape(e.shape)

    def Jacobian(self, e):
        edim = np.prod(e.shape[1:]).astype(np.int)
        eshape = e.shape[1:]
        e = e.reshape(-1, edim)
        d = e.shape[0]
        x = np.empty((3, edim), dtype=np.float)
        x[:2] = e
        x[2] = 1
        Ax = np.matmul(self.A, x)
        ret = np.empty((2,2,x.shape[1]))
        ret[0,0] = (self.A[0,0]*Ax[2, :]-self.A[2,0]*Ax[0,:])/(Ax[2, :]**2)
        ret[1,0] = (self.A[1,0]*Ax[2, :]-self.A[2,0]*Ax[1,:])/(Ax[2, :]**2)
        ret[0,1] = (self.A[0,1]*Ax[2, :]-self.A[2,1]*Ax[0,:])/(Ax[2, :]**2)
        ret[1,1] = (self.A[1,1]*Ax[2, :]-self.A[2,1]*Ax[1,:])/(Ax[2, :]**2)
        if len(eshape) == 0:
             return ret.reshape(d,d)
        else:
             return ret.reshape(d,d,eshape)


def fliplr_as_affine(imsize):
    A = np.array([[-1, 0, imsize[1]],
                  [0, 1, 0],
                  [0, 0, 1]])
    return NpAffineTransforms(A)

def rotate_as_affine(angle, center=np.array([0.0, 0.0])):
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    tx = (1-cos_angle)*center[0]+sin_angle*center[1]
    ty = (1-cos_angle)*center[1]-sin_angle*center[0]
    A = np.array([[cos_angle, -sin_angle, tx],
                  [sin_angle, cos_angle, ty],
                  [0, 0, 1]])
    return NpAffineTransforms(A)

def scale_as_affine(angle, scale, scale_center=np.array([0.0, 0.0])):
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    R = np.array([[cos_angle, sin_angle], [-sin_angle, cos_angle]])
    S = np.array([[scale[0], 0], [0, scale[1]]])
    V = np.matmul(np.matmul(R.transpose(), S), R)
    t = scale_center - np.matmul(V, scale_center)
    A = np.eye(3, dtype=np.float)
    A[:2, :2] = V
    A[2,:2] = t
    return NpAffineTransforms(A)

def translation_as_affine(t):
    A = np.array([[1.0, 0, t[0]],
                  [0, 1, t[1]],
                  [0, 0, 1]])
    return NpAffineTransforms(A)

def perspective_as_affine(angle, strength):
    ca = np.cos(angle)
    sa = np.sin(angle)
    A = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [strength*ca, strength*sa, 1]])
    return NpAffineTransforms(A)

def stack_affine_transforms(affine_list):
    cur_trans = np.array([[1,0,0], [0,1,0], [0,0,1.0]])
    for at in affine_list:
        cur_trans = np.matmul(at.A, cur_trans)
    return NpAffineTransforms(cur_trans)

import matplotlib.pyplot as plt
def np_warp_im(im, transform, new_im_size):
    rNew = np.arange(new_im_size[0]).reshape(-1,1).repeat(new_im_size[1], 1)
    cNew = np.arange(new_im_size[1]).reshape(1,-1).repeat(new_im_size[0], 0)
    inv_mapping = np.linalg.inv(transform.A)
    trans_inv = NpAffineTransforms(inv_mapping)

    rNew = rNew.astype(np.float)
    cNew = cNew.astype(np.float)
    coords = np.stack([cNew, rNew])
    old_coords = trans_inv(coords)
    old_r = old_coords[1]
    old_c = old_coords[0]
    pixel_vals = interp_c(im, old_r.flatten(), old_c.flatten())
    im_c = pixel_vals.reshape(list(new_im_size)+ [im.shape[-1]])
    # im_np =  np_interpolate_image(im, old_r, old_c)
    return im_c


def interp_factor(t):
    return t
    # return 6*t**5 - 15*t**4 + 10*t**3

def np_mirror_indices(values, max_size):
    values = np.abs(values)
    num_wraps = values // max_size
    odd_wraps = num_wraps % 2 == 1
    values = values % max_size
    values[odd_wraps] = max_size - values[odd_wraps]
    return values

def np_interpolate_image(im, rNew, cNew):
    # I am fairly certain this method has minor issues with frequency aliasing
    R, C, ch = im.shape[0], im.shape[1], im.shape[2]
    rows_return, cols_return = rNew.shape[0], rNew.shape[1]
    rNew = rNew.flatten()
    cNew = cNew.flatten()
    # mirror
    rNew0 = np_mirror_indices(rNew, R)
    rNew0 = np.clip(rNew0, 0, R-1)
    rNew1 = np.minimum(rNew0+1, R-1)
    cNew0 = np_mirror_indices(cNew, C)
    cNew0 = np.clip(cNew0, 0, C-1)
    cNew1 = np.minimum(cNew0+1, C-1)

    points0 = np.stack([rNew0, cNew0])
    points1 = np.stack([rNew1, cNew1])
    rem_points = points0 % 1
    int_points0 = points0.astype(np.int)
    int_points1 = points1.astype(np.int)
    i00 = im[int_points0[0], int_points0[1]]
    i01 = im[int_points0[0], int_points1[1]]
    i10 = im[int_points1[0], int_points0[1]]
    i11 = im[int_points1[0], int_points1[1]]

    interp_val = interp_factor(rem_points)

    i0 = i00 + (i10-i00)*interp_val[0].reshape(-1, 1)
    i1 = i01 + (i11-i01)*interp_val[0].reshape(-1, 1)

    r = (i0+(i1-i0)*interp_val[1].reshape(-1, 1))
    return r.reshape(rows_return,cols_return,ch)
