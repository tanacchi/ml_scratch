import numpy as np


def gen_saddle_shape(resolution, random_seed=None, noise_scale=0.1):
    np.random.seed(random_seed)

    z1 = np.random.rand(resolution) * 2.0 - 1.0
    z2 = np.random.rand(resolution) * 2.0 - 1.0

    X = np.empty((resolution, 3))
    X[:, 0] = z1
    X[:, 1] = z2
    X[:, 2] = z1**2 - z2**2
    X += np.random.normal(loc=0, scale=noise_scale, size=X.shape)

    return X
