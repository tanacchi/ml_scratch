import numpy as np
import torch
import torch.nn as nn


class UKR(nn.Module):
    def __init__(self, data_num, latent_dim, sigma=1):
        super().__init__()
        self.kernel = lambda Z1, Z2: torch.exp(-torch.cdist(Z1, Z2)**2 /
                                               (2 * sigma**2))
        self.Z = nn.Parameter(torch.randn(data_num, latent_dim) / 10.)

    def forward(self, X):
        kernels = self.kernel(self.Z, self.Z)
        R = kernels / torch.sum(kernels, axis=1, keepdims=True)
        Y = R @ X
        return Y


def make_grid2d(resolution, bounds=(-1, +1)):
    mesh, step = np.linspace(bounds[0],
                             bounds[1],
                             resolution,
                             endpoint=False,
                             retstep=True)
    mesh += step / 2.0
    grid = np.meshgrid(mesh, mesh)
    return np.dstack(grid).reshape(-1, 2)


class UKRNet(nn.Module):
    def __init__(self, N):
        super(UKRNet, self).__init__()
        self.layer = UKR(N, latent_dim=2, sigma=2)

    def forward(self, x):
        return self.layer(x)
