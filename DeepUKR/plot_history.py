import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist

from data import gen_saddle_shape
from visualizer import visualize_history

sigma = 0.5
kernel = lambda Z1, Z2: np.exp(-dist.cdist(Z1, Z2)**2 / (2 * sigma**2))


def estimate_f(X, Z1, Z2=None):
    Z2 = np.copy(Z1) if Z2 is None else Z2
    kernels = kernel(Z1, Z2)
    R = kernels / np.sum(kernels, axis=1, keepdims=True)
    return R @ X


def make_grid2d(resolution, bounds=(-1, +1)):
    mesh, step = np.linspace(bounds[0],
                             bounds[1],
                             resolution,
                             endpoint=False,
                             retstep=True)
    mesh += step / 2.0
    grid = np.meshgrid(mesh, mesh)
    return np.dstack(grid).reshape(-1, 2)


with open("X.pickle", 'rb') as f:
    X = pickle.load(f)
with open("Y_history.pickle", 'rb') as f:
    Y_history = pickle.load(f)
with open("Z_history.pickle", 'rb') as f:
    Z_history = pickle.load(f)

resolution = 10
f_history = np.zeros((Y_history.shape[0], resolution**2, 3))
for i, Z in enumerate(Z_history):
    Zeta = make_grid2d(resolution, (Z.min(), Z.max()))
    f = estimate_f(X, Zeta, Z)
    f_history[i] = f

visualize_history(X, f_history, Z_history, True)
