from collections import namedtuple
import numpy as np
from scipy.spatial import distance as dist
from data import gen_saddle_shape
from tqdm import tqdm


History = namedtuple('History', ['U1', 'U2', 'Y', 'Z1', 'Z2'])


class TSOM(object):
    def __init__(self):
        self.L1 = 1
        self.L2 = 1
        self.K1 = 10
        self.K2 = 10
        self.mode1_sigma_max = 2.2
        self.mode2_sigma_max = 2.2
        self.mode1_sigma_min = 0.1
        self.mode2_sigma_min = 0.1
        self.mode1_tau = 40
        self.mode2_tau = 40

    def fit(self, X, num_epoch=50):
        N1, N2, D = X.shape
        self.N1 = N1
        self.N2 = N2
        self.D = D
        Zeta1 = np.linspace(-1, 1, self.K1).reshape(-1, 1)
        Zeta2 = np.linspace(-1, 1, self.K2).reshape(-1, 1)

        U1 = np.random.normal(scale=0.1, size=(N1, self.K2, D))
        U2 = np.random.normal(scale=0.1, size=(self.K1, N2, D))
        Y = np.random.normal(scale=0.1, size=(self.K1, self.K2, D))

        history = History(np.zeros(((num_epoch,) + U1.shape)),
                          np.zeros(((num_epoch,) + U2.shape)),
                          np.zeros(((num_epoch,) + Y.shape)),
                          np.zeros(((num_epoch, N1, self.L1))),
                          np.zeros(((num_epoch, N2, self.L2))))

        for epoch in tqdm(range(num_epoch)):
            Z1 = self.mode1_competitive(U1, Y, Zeta1)
            Z2 = self.mode2_competitive(U2, Y, Zeta2)

            mode1_sigma = calc_sigma(self.mode1_sigma_max, self.mode1_sigma_min, self.mode1_tau, epoch)
            mode2_sigma = calc_sigma(self.mode2_sigma_max, self.mode2_sigma_min, self.mode2_tau, epoch)
            R1 = self.cooperative(Z1, Zeta1, mode1_sigma)
            R2 = self.cooperative(Z2, Zeta2, mode2_sigma)

            U1 = self.estimate_1st_model1(X, R2)
            U2 = self.estimate_1st_model2(X, R1)
            Y = self.estimate_2nd_model(X, R1, R2, U1)

            history.U1[epoch] = U1
            history.U2[epoch] = U2
            history.Y[epoch]  = Y
            history.Z1[epoch] = Z1
            history.Z2[epoch] = Z2

        return history, Zeta1, Zeta2

    def mode1_competitive(self, U, Y, Zeta):
        N1, K2, D = U.shape
        dists = np.zeros((N1, self.K1))
        for n1 in range(N1):
            for k1 in range(self.K1):
                for k2 in range(self.K2):
                    for d in range(D):
                        dists[n1, k1] += (U[n1, k2, d] - Y[k1, k2, d])**2
        #  dists = dist.cdist(U, Y)
        bmus = np.argmin(dists, axis=1)
        return Zeta[bmus, :]


    def mode2_competitive(self, U, Y, Zeta):
        K1, N2, D = U.shape
        dists = np.zeros((N2, self.K2))
        for n2 in range(N2):
            for k2 in range(self.K2):
                for k1 in range(K1):
                    for d in range(D):
                        dists[n2, k2] += (U[k1, n2, d] - Y[k1, k2, d])**2
        #  dists = dist.cdist(U, Y)
        bmus = np.argmin(dists, axis=1)
        return Zeta[bmus, :]

    def estimate_1st_model1(self, X, R2):
        g = np.sum(R2, axis=1)
        U = np.empty((self.N1, self.K2, self.D))
        for n1 in range(self.N1):
            for k2 in range(self.K2):
                for d in range(self.D):
                    U[n1, k2, d] = sum([R2[k2, n2]*X[n1, n2, d] for n2 in range(self.N2)]) / g[k2]
        return U


    def estimate_1st_model2(self, X, R1):
        g = np.sum(R1, axis=1)
        U = np.empty((self.K1, self.N2, self.D))
        for k1 in range(self.K1):
            for n2 in range(self.N2):
                for d in range(self.D):
                    U[k1, n2, d] = sum([R1[k1, n1]*X[n1, n2, d] for n1 in range(self.N1)]) / g[k1]
        return U


    def estimate_2nd_model(self, X, R1, R2, U1):
        g1 = np.sum(R1, axis=1)
        g2 = np.sum(R2, axis=1)
        Y = np.empty((self.K1, self.K2, self.D))
        for k1 in range(self.K1):
            for k2 in range(self.K2):
                for d in range(self.D):
                    Y[k1, k2, d] = sum([sum([R1[k1, n1] * R2[k2, n2] * X[n1, n2, d] for n2 in range(self.N2)]) for n1 in range(self.N1)]) / (g1[k1] * g2[k2])
        return Y


    def cooperative(self, Z, Zeta, sigma):
        N, L = Z.shape
        K, _ = Zeta.shape
        R = np.empty((K, N))
        for k in range(K):
            for n in range(N):
                R[k, n] = sum([np.exp(- (Zeta[k, l] - Z[n, l])**2 / (2*sigma**2)) for l in range(L)])
        return R


def calc_sigma(sigma_max, sigma_min, tau, epoch):
    return max(sigma_min, sigma_min + (sigma_max - sigma_min) * (1 - (epoch / tau)))


if __name__ == '__main__':
    X = gen_saddle_shape(50)
    tsom = TSOM()
    tsom.fit(X, num_epoch=100)
