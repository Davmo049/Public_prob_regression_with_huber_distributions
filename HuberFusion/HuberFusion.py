import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.linalg

def get_quadratic_overest(prec, v, x):
    diff = x-v
    mah_norm2 = np.matmul(diff.reshape(1,-1), np.matmul(prec, diff.reshape(-1, 1))).reshape(1)
    if mah_norm2 < 1:
        A = prec/2
        c = mah_norm2/2
    else:
        d = np.sqrt(mah_norm2)
        A = prec/(2*d)
        c = d-1/2
    b = -2*np.matmul(A, v.reshape(-1, 1)).reshape(-1)
    c = c-np.matmul(x.reshape(1,-1), np.matmul(A, x.reshape(-1, 1))).reshape(1)-np.dot(x,b)
    return A, b ,c

def eval_huber(prec, v, x):
    diff = x-v
    mah_norm2 = np.matmul(diff.reshape(1,-1), np.matmul(prec, diff.reshape(-1, 1))).reshape(1)
    if mah_norm2 < 1:
        return mah_norm2/2
    else:
        return np.sqrt(mah_norm2)-0.5

def eval_quadratic(A, b, c, x):
    quadterm = np.matmul(np.matmul(x.reshape(1, -1), A), x.reshape(-1, 1)).reshape(1)
    linterm = np.dot(b,x)
    return c+linterm + quadterm

def pointwise_apply(f, xs):
    return np.array(list(map(f, xs)))

def plot_huber():
    plot_start = np.array([0.0, 0.0])
    plot_direction = np.array([1.0, 0.0])
    prec = np.array([[2.0, 1.0],[1.0, 3.0]])
    v = np.array([0.0, 1.0])
    plot_steps = np.arange(0, 100)*10.0/99-2.0
    plot_xs = plot_start.reshape(1, -1) + plot_direction.reshape(1,-1)*plot_steps.reshape(-1, 1)
    ys = pointwise_apply(lambda x: eval_huber(prec, v, x), plot_xs)
    plt.plot(plot_steps,ys)
    quad_start = np.array([-1.0, 1])
    quad_direction = np.array([1.0, 1])
    steps = np.arange(0, 5)
    for ss in steps:
        approx_pos = quad_start+quad_direction*ss
        A,b,c = get_quadratic_overest(prec, v, approx_pos)
        y_quad = pointwise_apply(lambda x: eval_quadratic(A,b,c,x), plot_xs)
        plt.plot(plot_steps, y_quad, '--')
    plt.show()

def generate_random_matrix():
    eigs = np.exp(np.random.uniform(size=(2)))-1
    ang = np.random.uniform()*2*np.pi
    ca = np.cos(ang)
    sa = np.sin(ang)
    R = np.array([[ca, sa], [-sa, ca]])
    return np.matmul(R.transpose(), eigs.reshape(2,1)*R)


def ML_huber(vs, precs):
    x = np.copy(vs[0])
    T = 20
    for t in range(T):
        Acum = np.zeros((2,2))
        bcum = np.zeros((2))
        for v, p in zip(vs, precs):
            A,b,c=get_quadratic_overest(p, v, x)
            Acum += A
            bcum += b
        x = np.linalg.solve(Acum, -bcum/2)
    return x

def combine_hubers(modes, precs):
    # modes are Nx17x2
    # precs are Nx17x2x2
    N_keypoints = modes.shape[1]
    ret_modes = []
    for kp_idx in range(N_keypoints):
        mode = ML_huber(modes[:,kp_idx], precs[:,kp_idx])
        ret_modes.append(mode)
    return np.array(ret_modes), precs[0]

def demo_fusion_huber():
    N = 32
    D = 2
    vs = np.random.normal(size=(N, D))*2-22
    vs[-1] = np.array([101, -44])

    precs = np.array(list(map(lambda x: generate_random_matrix(), range(N))))
    point = ML_huber(vs, precs)
    plt.plot(vs[:, 0], vs[:, 1], 'rx')
    plt.plot(point[0], point[1], 'bo')
    plt.show()


def main():
    # plot_huber()
    demo_fusion_huber()

if __name__ == '__main__':
    main()
