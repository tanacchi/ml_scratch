import pickle
import matplotlib.pyplot as plt
from data import gen_saddle_shape
from visualizer import visualize_history


def estimate_f(self, X, Z1, Z2=None):
    Z2 = np.copy(Z1) if Z2 is None else Z2
    kernels = self.kernel(Z1, Z2)
    R = kernels / np.sum(kernels, axis=1, keepdims=True)
    return R @ X, R

def make_grid2d(resolution, bounds=(-1, +1)):
    mesh, step = np.linspace(bounds[0], bounds[1], resolution,
                             endpoint=False, retstep=True)
    mesh += step / 2.0
    grid = np.meshgrid(mesh, mesh)
    return np.dstack(grid).reshape(-1, 2)


X = gen_saddle_shape(100)

with open("Y_history.pickle", 'rb') as f:
    Y_history = pickle.load(f)
with open("Z_history.pickle", 'rb') as f:
    Z_history = pickle.load(f)

visualize_history(X, Y_history, Z_history, True)
