import numpy as np
from . import c_interface

def min_sphere(points):
    if len(points.shape) != 2:
        print('invalid shape {}'.format(points.shape))
        return None
    return c_interface.welzl_c(points)

def minimum_covering_sphere(points):
    # points = nxd array
    # welzl's algorithm
    points = np.copy(points)
    np.random.shuffle(points)
    n, d = points.shape
    buf = np.empty((d+1,d), dtype=np.float)
    return welzl_recursive(points, buf, 0)

def minimum_covering_sphere_iterative(points):
    n, d = points.shape
    points = np.copy(points)
    np.random.shuffle(points)
    c_idx = len(points)
    included_indices = np.empty((d+1),dtype=np.int)
    included_points = np.empty((d+1, d),dtype=points.dtype)
    inc_len = 0
    cand_r = 0
    cand_center = np.zeros((d), dtype=points.dtype)
    if n == 0:
        return cand_center, cand_r
    # 3 modes
    # 1) back up check if points are included
    # 2) if limit of indices are used go back and pop until unincluded index found
    # 3) if limit of indices are not used remove indices after this one and add this one after.
    eps = 10e-5
    while True:
        if c_idx == len(points):
            cand_center, cand_r = welzl_trivial(included_points[:inc_len])
            c_idx -= 1

        if np.linalg.norm(points[c_idx] - cand_center) < cand_r+eps:
            # mode 1
            if c_idx == 0:
                return cand_center, cand_r
            else:
                c_idx -= 1
        else:
            for i in range(inc_len):
                if included_indices[i] > c_idx:
                    inc_len = i
                    break
            if inc_len == 0 or included_indices[inc_len-1] < c_idx:
                included_indices[inc_len] = c_idx
                included_points[inc_len] = points[c_idx]
                inc_len += 1
                if inc_len == d+1:
                    cand_center, cand_r = welzl_trivial(included_points[:inc_len])
                    c_idx -= 1
                else:
                    c_idx = len(points)
            else:
                # included_indices[inc_len-1] == c_idx, since we removed all other occurences
                while included_indices[inc_len-1] == c_idx:
                    inc_len -= 1
                    c_idx -= 1
                included_indices[inc_len] = c_idx
                included_points[inc_len] = points[c_idx]
                inc_len += 1
                c_idx = len(points)


def welzl_recursive(points, included_points, num_included_points):
    if len(points) == 0 or num_included_points == 3:
        return welzl_trivial(included_points[:num_included_points])
    else:
        p = points[0]
        rem = points[1:]
        cand_mid, cand_rad = welzl_recursive(rem, included_points, num_included_points)
        if np.linalg.norm(p-cand_mid) < cand_rad:
            return cand_mid, cand_rad
        included_points[num_included_points] = p
        ret = welzl_recursive(rem, included_points, num_included_points+1)
        return ret


def welzl_trivial(points):
    d = points.shape[1]
    assert(d==2) # fix general case later, right now we only need d==2
    if points.shape[0] == 0:
        return np.zeros((d)), 0.0
    elif points.shape[0] == 1:
        return points[0], 0.0
    elif points.shape[0] == 2:
        midpoint = (points[0] + points[1])/2
    else:
        midpoint = center_for_points_on_circle(points)
    distances = np.linalg.norm(points-midpoint.reshape(1,d), axis=1)
    r = np.max(distances)
    return midpoint, r

def center_for_points_on_circle(points):
    b = np.linalg.norm(points, axis=1)**2/2
    b = b[:2]-b[2]
    A = np.copy(points)
    A = A[:2] - A[2].reshape(1,2)
    solution = np.linalg.solve(A, b)
    return solution


def geometric_median(points, weights=None, eps=10e-5, halt_eps=10e-5):
    # reference https://www.ncbi.nlm.nih.gov/pmc/articles/PMC26449/pdf/pq001423.pdf
    # weights[i] = $\eta_i$ in paper
    # a point is close (ref eq 2.5 in paper) if ||x-y|| < eps
    # stop if ||y_{k+1}-y_k||_2 < eps
    n, d = points.shape
    if weights is None:
        weights = np.ones((n))
    else:
        assert(weights.shape[0] == n)
    y_k = np.mean(points, axis=0)
    while True:
        close = np.linalg.norm(y_k.reshape(1, d)-points,axis=1) < eps
        close_weight = np.sum(weights[close])
        not_close = np.logical_not(close)
        eta = weights[not_close]
        x = points[not_close]
        diff = x-y_k.reshape(1, d)
        factor = (eta / np.linalg.norm(diff,axis=1)).reshape(x.shape[0], 1)
        R_tilde = np.sum(diff * factor, axis=0)
        r = np.linalg.norm(R_tilde)
        if r == 0: # updates will stop
            return y_k
        T_tilde = np.sum(x * factor, axis=0) / np.sum(factor)
        T_factor = max(1 - close_weight/r, 0)
        R_factor = min(1, close_weight/r)
        y_old = y_k
        y_k = T_factor * T_tilde + R_factor*y_k
        if np.linalg.norm(y_k - y_old) < eps:
            return y_k
