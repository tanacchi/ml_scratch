from collections import namedtuple
import numpy as np
from scipy.spatial import distance as dist
from tqdm import tqdm


History = namedtuple('History', ['U1', 'U2', 'Y', 'Z1', 'Z2'])


class TSOM(object):
    def __init__(self, L1, L2, K1, K2, sigma1_max, sigma2_max, sigma1_min, sigma2_min, tau1, tau2):
        self.L1 = L1
        self.L2 = L2
        self.K1 = K1
        self.K2 = K2
        self.mode1_sigma_max = sigma1_max
        self.mode2_sigma_max = sigma2_max
        self.mode1_sigma_min = sigma1_min
        self.mode2_sigma_min = sigma2_min
        self.mode1_tau = tau1
        self.mode2_tau = tau2

    def fit(self, X, num_epoch=50, init=None):
        if X.ndim == 2:
            X = X[:, :, np.newaxis]
        N1, N2, D = X.shape

        # Initialize Zeta
        Zeta1, step = np.linspace(-1, 1, self.K1, endpoint=(not self.L1 == 2), retstep=True)
        if self.L1 == 2:
            Zeta1 += step / 2
            Zeta1 = np.meshgrid(Zeta1, Zeta1)
            Zeta1 = np.dstack(Zeta1)
        Zeta1 = Zeta1.reshape(-1, self.L1)

        Zeta2, step = np.linspace(-1, 1, self.K2, endpoint=(not self.L2 == 2), retstep=True)
        if self.L2 == 2:
            Zeta2 += step / 2
            Zeta2 = np.meshgrid(Zeta2, Zeta2)
            Zeta2 = np.dstack(Zeta2)
        Zeta2 = Zeta2.reshape(-1, self.L2)

        if not init:
            Z1 = np.random.rand(N1, self.L1)
            Z2 = np.random.rand(N2, self.L2)
        else:
            Z1 = init[0].copy()
            Z2 = init[1].copy()

        calc_sigma1 = gen_sigma_calculator(self.mode1_sigma_max, self.mode1_sigma_min, self.mode1_tau)
        calc_sigma2 = gen_sigma_calculator(self.mode2_sigma_max, self.mode2_sigma_min, self.mode2_tau)

        history = History(np.zeros((num_epoch, N1, self.K2**self.L2, D)),
                          np.zeros((num_epoch, self.K1**self.L1, N2, D)),
                          np.zeros((num_epoch, self.K1**self.L1, self.K2**self.L2, D)),
                          np.zeros((num_epoch, N1, self.L1)),
                          np.zeros((num_epoch, N2, self.L2)))

        for epoch in tqdm(range(num_epoch)):
            sigma1, sigma2 = calc_sigma1(epoch), calc_sigma2(epoch)
            U1, U2, Y = m_step(X, Z1, Z2, Zeta1, Zeta2, sigma1, sigma2)
            Z1, Z2 = e_step(U1, U2, Y, Zeta1, Zeta2)

            history.U1[epoch] = U1
            history.U2[epoch] = U2
            history.Y[epoch]  = Y
            history.Z1[epoch] = Z1
            history.Z2[epoch] = Z2

        return history, Zeta1, Zeta2


def e_step(U1, U2, Y, Zeta1, Zeta2):
    dists1 = np.sum((U1[:, None, :, :] - Y[None, :, :, :])**2, axis=(2, 3))
    bmus1 = np.argmin(dists1, axis=1)
    dists2 = np.sum((U2[:, :, None, :] - Y[:, None, :, :])**2, axis=(0, 3))
    bmus2 = np.argmin(dists2, axis=1)
    return Zeta1[bmus1, :], Zeta2[bmus2, :]


def m_step(X, Z1, Z2, Zeta1, Zeta2, sigma1, sigma2):
    R1 = cooperative(Z1, Zeta1, sigma1)
    R2 = cooperative(Z2, Zeta2, sigma2)

    U1 = estimate_1st_model1(X, R2)
    U2 = estimate_1st_model2(X, R1)
    Y = estimate_2nd_model(X, R1, R2, U1, Z1, Zeta1, sigma1)
    return U1, U2, Y


def estimate_1st_model1(X, R2):
    g = np.sum(R2, axis=1)
    U = np.einsum("km,nmd->nkd", R2, X)
    return U / g[:, None]


def estimate_1st_model2(X, R1):
    g = np.sum(R1, axis=1)
    U = np.einsum("km,mnd->knd", R1, X)
    return U / g[:, None, None]


def estimate_2nd_model(X, R1, R2, U1, Z1, Zeta1, sigma1):
    N1, N2, D = X.shape
    K1, _ = R1.shape
    #  K2, _ = R2.shape
    #  g1 = np.sum(R1, axis=1)
    #  g2 = np.sum(R2, axis=1)
    #  Y = np.zeros((K1, K2, D))
    #  for k1 in range(K1):
        #  for k2 in range(K2):
            #  for d in range(D):
                #  for n1 in range(N1):
                    #  for n2 in range(N2):
                        #  Y[k1, k2, d] = (R1[k1, n1] * R2[k2, n2] * X[n1, n2, d]) / (g1[k1] * g2[k2])

    #  Y = np.einsum("kn,lm,nmd->kld", R1, R2, X)
    #  return Y / (g1[:, None] * g2[:, None, None])

    distance1 = dist.cdist(Zeta1, Z1, 'sqeuclidean')  # 距離行列をつくるDはN*K行列
    H1 = np.exp(-distance1 / (2 * pow(sigma1, 2)))  # かっこに気を付ける
    G1 = np.sum(H1, axis=1)  # Gは行ごとの和をとったベクトル
    R1 = (H1.T / G1).T  # 行列の計算なので.Tで転置を行う
    Y = np.einsum('Nkd,KN->Kkd', U1 ,R1)
    # n1 => n
    # k1 => l
    # k2 => k
    #  Y = np.einsum("nl,ln->lk", U1, R1)
    return Y


def cooperative(Z, Zeta, sigma):
    dists = dist.cdist(Zeta, Z, 'sqeuclidean')
    R = np.exp(- dists / (2.0 * sigma**2))
    return R


def gen_sigma_calculator(sigma_max, sigma_min, tau):
    return lambda epoch: max(sigma_min, sigma_min + (sigma_max - sigma_min) * (1 - (epoch / tau)))
