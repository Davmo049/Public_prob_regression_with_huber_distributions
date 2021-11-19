from . import covers
from . import c_interface
import numpy as np
import matplotlib.patches as patches
import time
import matplotlib.pyplot as plt

def happy_test():
    points = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]])
    plt.scatter(points[:, 0], points[:, 1])
    center, r = c_interface.welzl_c(points)
    circle = patches.Circle(center,r, fill=False)
    plt.gca().add_patch(circle)
    plt.show()

def visualize_welzl():
    N = 100000
    points = np.random.laplace(size=(N, 2))
    plt.scatter(points[:, 0], points[:, 1])
    center, r = c_interface.welzl_c(points)
    circle = patches.Circle(center,r, fill=False)
    plt.gca().add_patch(circle)
    plt.show()

def time_welzl():
    N = 2**np.arange(3, 25)
    # T = map(lambda x: int(2**18/x), N)
    T = map(lambda x: 10, N)
    NnT = list(zip(N,T))
    time_rec = []
    # for n,times in NnT:
    #     times=1
    #     print(n)
    #     if n > 500:
    #         time_rec.append(None)
    #         continue
    #     points = np.random.laplace(size=(times, n, 2))
    #     t0 = time.time()
    #     for t in range(times):
    #         covers.minimum_covering_sphere(points[t])
    #     time_rec.append((time.time() - t0)/times)
    # time_it = []
    # for n,times in NnT:
    #     print(n)
    #     points = np.random.laplace(size=(times, n, 2))
    #     t0 = time.time()
    #     for t in range(times):
    #         covers.minimum_covering_sphere_iterative(points[t])
    #     time_it.append((time.time() - t0)/times)
    time_c = []
    for n,times in NnT:
        print(n)
        dim = 3
        points = 2*np.random.uniform(size=(times, n, dim))-1
        if False:
            len_mod = 1-0.01*(np.random.uniform(size=(times, n))*2-1)
            factor = (len_mod/np.linalg.norm(points, axis=2)).reshape(times, n, 1)
            points *= factor

        t0 = time.time()
        for t in range(times):
            center, r = c_interface.welzl_c(points[t])
            # np.sort(points[t,:,0]) # just for timing reference
        time_c.append((time.time() - t0)/times)
    # print(time_it)
    print(time_c)
    # plt.plot(N, time_rec, 'b')
    # plt.plot(N, time_it, 'r')
    plt.plot(N, time_c, 'g')
    plt.show()

def check_welzl_multidim():
    # points = np.array([[-0.2, -1.6, -0.7], [-1.1, 1.0, -1.5], [1.0, 1.2, -0.2], [1.2, 0.5, 1.1]])
    n = 2000
    d = 30
    points = np.random.laplace(size=(n,d))
    center, r = c_interface.welzl_c(points)
    print(center)
    print(r)
    dist = np.linalg.norm(points-center.reshape(1, d), axis=1)
    print(dist)



if __name__ == '__main__':
    # happy_test()
    for _ in range(10):
        visualize_welzl()
    # time_welzl()
    # check_welzl_multidim()
