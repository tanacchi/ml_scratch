import numpy as np


def gen_saddle_shape(resolution, random_seed=0, noise_scale=0.1):
    np.random.seed(random_seed)

    z1 = np.random.rand(resolution) * 2.0 - 1.0
    z2 = np.random.rand(resolution) * 2.0 - 1.0

    X = np.empty((resolution, 3))
    X[:, 0] = z1
    X[:, 1] = z2
    X[:, 2] = z1**2 - z2**2
    X += np.random.normal(loc=0, scale=noise_scale, size=X.shape)

    return X


if __name__ == '__main__':
    ## gif を生成するコード

    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation

    def update_graph(angle, X, fig, ax):
        ax.cla()
        ax.view_init(azim=angle, elev=30)
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 0])

    X = gen_saddle_shape(200)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ani = FuncAnimation(fig,
                        update_graph,
                        frames=360,
                        interval=30,
                        repeat=True,
                        fargs=(X, fig, ax))
    plt.show()
    ani.save("tmp.gif", writer='pillow')
