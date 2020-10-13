import numpy as np
from tqdm import tqdm
from scipy.spatial import distance as dist
from collections import namedtuple
import torch


History = namedtuple('History', ['Y', 'f', 'Z'])

class UKR(object):
    def __init__(self, latent_dim, eta):
        self.L = latent_dim
        self.η = eta
        self.σ = 1
        self.kernel = lambda Z1, Z2: torch.exp(- torch.cdist(Z1, Z2)**2 / (2 * self.σ ** 2))

    def fit(self, X, num_epoch=50):
        N, D = X.shape
        X = torch.from_numpy(X).float()
        f_resolution = 10
        Z = torch.normal(mean=0.0, std=0.1, size=(N, self.L))
        Z.requires_grad = True
        history = History(np.zeros((num_epoch, N, D)),
                          np.zeros((num_epoch, f_resolution**self.L, D)),
                          np.zeros((num_epoch, N, self.L)))

        for epoch in tqdm(range(num_epoch)):
            Y, R = self.estimate_f(X, Z)
            Z = self.estimate_e(X, Y, Z, R)

            Z_new = make_grid2d(f_resolution, bounds=(torch.min(Z.detach()), torch.max(Z.detach())))
            f, _ = self.estimate_f(X, Z_new, Z)

            history.Y[epoch] = Y.detach().numpy()
            history.f[epoch] = f.detach().numpy()
            history.Z[epoch] = Z.detach().numpy()

        return history

    def estimate_f(self, X, Z1, Z2=None):
        Z2 = Z1.clone() if Z2 is None else Z2
        kernels = self.kernel(Z1, Z2)
        R = kernels / torch.sum(kernels, axis=1, keepdims=True)
        return R @ X, R

    def estimate_e(self, X, Y, Z, R):
        E = torch.sum((Y - X)**2) / 2.0
        E.backward()
        Z.requires_grad = False
        Z = Z - self.η * Z.grad
        Z.requires_grad = True
        return Z


def make_grid2d(resolution, bounds=(-1, +1)):
    mesh, step = np.linspace(bounds[0], bounds[1], resolution,
                             endpoint=False, retstep=True)
    mesh += step / 2.0
    grid = np.meshgrid(mesh, mesh)
    Zeta = np.dstack(grid).reshape(-1, 2)
    return torch.from_numpy(Zeta).float()


if __name__ == '__main__':
    from data import gen_saddle_shape
    from visualizer import visualize_history

    X = gen_saddle_shape(100, noise_scale=0.0)
    ukr = UKR(latent_dim=2, eta=0.8)
    history = ukr.fit(X, num_epoch=50)
    visualize_history(X, history.f, history.Z)
