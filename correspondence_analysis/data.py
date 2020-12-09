import numpy as np

__all__ = ['gen_saddle_shape']


def gen_saddle_shape(resolution, random_seed=None, noise_std=0.1):
    demention = 3
    np.random.seed(random_seed)

    Z1 = np.linspace(-1, 1, resolution)
    Z2 = np.linspace(-1, 1, resolution*2)

    X = np.zeros((resolution, resolution*2, demention))
    for i, z1 in enumerate(Z1):
        for j, z2 in enumerate(Z2):
            X[i, j, 0] = z1
            X[i, j, 1] = z2
            X[i, j, 2] = z1**2 - z2**2
    X += np.random.normal(0, noise_std, X.shape)

    return X


if __name__ == '__main__':
    X = gen_saddle_shape(10, noise_std=0)
    print(X)
