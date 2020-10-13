import numpy as np
from tqdm import tqdm
from collections import namedtuple
import torch


History = namedtuple('History', ['Y', 'F', 'f', 'Zeta', 'Z'])

class UKR(object):
    def __init__(self, latent_dim, eta):
        self.L = latent_dim
        self.η = eta
        self.σ = 1
        self.kernel = lambda Z1, Z2: torch.exp(- torch.cdist(Z1, Z2)**2 / (2 * self.σ ** 2))

    def fit(self, X, num_epoch=50):
        N, D = X.shape
        X = torch.from_numpy(X).float()
        landmarks_resolution = 5
        K = landmarks_resolution**2
        f_resolution = 10
        Z = torch.normal(mean=0.0, std=0.1, size=(N, self.L))
        Z.requires_grad = True
        Zeta = make_grid2d(landmarks_resolution, (-1, 1))
        history = History(np.zeros((num_epoch, K, D)),
                          np.zeros((num_epoch, N, D)),
                          np.zeros((num_epoch, f_resolution**self.L, D)),
                          np.zeros((num_epoch, K, self.L)),
                          np.zeros((num_epoch, N, self.L)))

        for epoch in tqdm(range(num_epoch)):
            Y, P_k = self.estimate_lp(X, Z, Zeta)
            F = self.estimate_F(Y, P_k, Z, Zeta)
            Z = self.estimate_e(X, F, Z, Zeta)

            Z_new = make_grid2d(f_resolution, bounds=(torch.min(Z.detach()), torch.max(Z.detach())))
            f = self.estimate_F(Y, P_k, Z_new, Zeta)

            history.Y[epoch] = Y.detach().numpy()
            history.F[epoch] = F.detach().numpy()
            history.f[epoch] = f.detach().numpy()
            history.Zeta[epoch] = Zeta.numpy()
            history.Z[epoch] = Z.detach().numpy()

        return history

    def estimate_lp(self, X, Z, Zeta):
        g_ik = self.kernel(Z, Zeta)
        G_i = torch.sum(g_ik, axis=1)
        p_ki = g_ik.T / G_i
        P_k = torch.sum(p_ki, axis=1)
        q_ik = p_ki.T / P_k
        return q_ik.T @ X, P_k

    def estimate_F(self, Y, P_k, Z, Zeta):
        h_ik = self.kernel(Z, Zeta)
        H_i = h_ik @ P_k
        N, K = h_ik.shape
        F2 = torch.zeros((N, 3))
        for i in range(N):
            tmp = 0.0
            for k in range(K):
                F2[i] += h_ik[i, k] * P_k[k] * Y[k]
                weight = float(h_ik[i, k] * P_k[k])
                if weight < 0.0:
                    exit(0)
                tmp += weight
            if tmp / H_i[i] > 1.1:
                print(tmp / H_i[i])
                exit(0)
            F2[i] /= H_i[i]
        #  P_k = P_k.repeat(3, 1)
        #  print(h_ik.shape)
        #  print((Y@P_k).shape)
        #  F = (h_ik @ (Y @ P_k)) / H_i
        #  print(F == F2)
        return F2

    def estimate_f(self, X, Z1, Z2=None):
        Z2 = Z1.clone() if Z2 is None else Z2
        kernels = self.kernel(Z1, Z2)
        R = kernels / torch.sum(kernels, axis=1, keepdims=True)
        return R @ X

    def estimate_e(self, X, F, Z, Zeta):
        E = torch.sum((F - X)**2) / 2.0
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
    import sys
    from data import gen_saddle_shape
    from visualizer_lp import visualize_history

    X = gen_saddle_shape(100, noise_scale=0.0)
    ukr = UKR(latent_dim=2, eta=0.8)
    history = ukr.fit(X, num_epoch=100)
    visualize_history(X, history, save_gif=(len(sys.argv) == 2))
