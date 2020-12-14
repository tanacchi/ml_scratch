import numpy as np
import torch
import torch.nn as nn


class UKR(nn.Module):
    def __init__(self, data_num, latent_dim, sigma=1, random_seed=0):
        super().__init__()
        self.kernel = lambda Z1, Z2: torch.exp(-torch.cdist(Z1, Z2)**2 /
                                               (2 * sigma**2))
        torch.manual_seed(random_seed)
        self.Z = nn.Parameter(torch.randn(data_num, latent_dim) / 10.)

    def forward(self, X):
        kernels = self.kernel(self.Z, self.Z)
        R = kernels / torch.sum(kernels, axis=1, keepdims=True)
        Y = R @ X
        return Y


class UKRNet(nn.Module):
    def __init__(self, N, latent_dim=2, sigma=2):
        super(UKRNet, self).__init__()
        self.layer = UKR(N, latent_dim, sigma)

    def forward(self, x):
        return self.layer(x)
