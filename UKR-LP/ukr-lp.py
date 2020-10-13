import numpy as np
from tqdm import tqdm
from scipy.spatial import distance as dist
from collections import namedtuple


History = namedtuple('History', ['F', 'f', 'Z'])

class UKR(object):
    def __init__(self, latent_dim, eta):
        self.L = latent_dim
        self.η = eta
        self.σ = 1
        self.kernel = lambda Z1, Z2: np.exp(- dist.cdist(Z1, Z2)**2 / (2 * self.σ ** 2))

    def fit(self, X, num_epoch=50):
        N, D = X.shape
        landmarks_resolution = 5
        f_resolution = 10
        Z = np.random.normal(scale=0.1, size=(N, self.L))
        Zeta = make_grid2d(landmarks_resolution, (-1, 1))
        history = History(np.zeros((num_epoch, N, D)),
                          np.zeros((num_epoch, f_resolution**self.L, D)),
                          np.zeros((num_epoch, N, self.L)))

        for epoch in tqdm(range(num_epoch)):
            Y, P_k = self.estimate_lp(X, Z, Zeta)
            F = self.estimate_f(X, Z)
            Z = self.estimate_e(X, F, Z, Zeta)

            #  Z_new = make_grid2d(f_resolution, bounds=(np.min(Z), np.max(Z)))
            #  f, _ = self.estimate_f(X, Z_new, Z)

            history.F[epoch] = F
            history.f[epoch] = f
            history.Z[epoch] = Z

        return history

    def estimate_lp(self, X, Z, Zeta):
        N, _ = X.shape

        g_ik = self.kernel(Z, Zeta)
        G_i = np.sum(g_ik, axis=1)
        p_ki = g_ik.T / G_i
        P_k = np.sum(p_ki, axis=1)
        q_ik = p_ki.T / P_k
        return q_ik.T @ X, P_k

    def estimate_F(self, Y, P_k, Z, Zeta):
        h_ik = self.kernel(Z, Zeta)
        H_i = h_ik @ P_k
        F = (h_ik @ (P_k * Y)) / H_i
        return F

    def estimate_f(self, X, Z1, Z2=None):
        Z2 = np.copy(Z1) if Z2 is None else Z2
        kernels = self.kernel(Z1, Z2)
        R = kernels / np.sum(kernels, axis=1, keepdims=True)
        return R @ X

    def estimate_e(self, X, F, Z, Zeta):
        d_ii = F - X
        d_in = F[:, np.newaxis, :] - X[np.newaxis, :, :]
        d_ni = F[np.newaxis, :, :] - X[:, np.newaxis, :]
        δ_ni = Z[np.newaxis, :, :] - Z[:, np.newaxis, :]
        δ_in = Z[:, np.newaxis, :] - Z[np.newaxis, :, :]
        kernels = self.kernel(Z, Zeta)
        R = kernels / np.sum(kernels, axis=1, keepdims=True)

        diff_left = np.einsum("ni,nd,ind,inl->nl", R, d_ii, d_ni, δ_ni)
        diff_right = np.einsum("ni,id,ind,inl->nl", R.T, d_ii, d_in, δ_in)
        diff = 2 * (diff_left - diff_right) / X.shape[0]
        return Z - self.η * diff


def make_grid2d(resolution, bounds=(-1, +1)):
    mesh, step = np.linspace(bounds[0], bounds[1], resolution,
                             endpoint=False, retstep=True)
    mesh += step / 2.0
    grid = np.meshgrid(mesh, mesh)
    return np.dstack(grid).reshape(-1, 2)


if __name__ == '__main__':
    from data import gen_saddle_shape
    from visualizer import visualize_history

    X = gen_saddle_shape(100, noise_scale=0.0)
    ukr = UKR(latent_dim=2, eta=100)
    history = ukr.fit(X, num_epoch=50)
    visualize_history(X, history.f, history.Z)
