import mathlib.covers as covers
import numpy as np

import matplotlib.pyplot as plt

def visualize_welzl():
    N = 50
    points = np.random.laplace(size=(N, 2))
    plt.scatter(points[:, 0], points[:, 1])
    center, r = covers.minimum_covering_sphere_iterative(points)
    circle=patches.Circle((center[0],center[1]),r,facecolor='red',
              edgecolor='blue',linestyle='dotted',linewidth='2.2')
    plt.gca().add_patch(circle)
    plt.plot(circle)
    plt.show()

if __name__ == '__main__':
    visualize_welzl()
