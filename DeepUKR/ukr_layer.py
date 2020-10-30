import torch
import torch.nn as nn


class UKR(nn.Module):
    def __init__(self, data_num, latent_dim, sigma=1, eta=0.5):
        super().__init__()
        self.kernel = lambda Z1, Z2: torch.exp(- torch.cdist(Z1, Z2)**2 / (2 * sigma ** 2))
        self.Z = nn.Parameter(torch.randn(data_num, latent_dim))

    def forward(self, X):
        # Estimate f
        kernels = self.kernel(self.Z, self.Z)
        R = kernels / torch.sum(kernels, axis=1, keepdims=True)
        Y = R @ X
        # Estimate Z
        #  d_ii = Y - X
        #  d_in = Y[:, None, :] - X[None, :, :]
        #  d_ni = Y[None, :, :] - X[:, None, :]
        #  δ_ni = self.Z[None, :, :] - self.Z[:, None, :]
        #  δ_in = self.Z[:, None, :] - self.Z[None, :, :]
        #  diff_left = torch.einsum("ni,nd,ind,inl->nl", R, d_ii, d_ni, δ_ni)
        #  diff_right = torch.einsum("ni,id,ind,inl->nl", R.T, d_ii, d_in, δ_in)
        #  diff = 2 * (diff_left - diff_right) / X.shape[0]
        #  return self.Z - self.η * diff
        return Y
