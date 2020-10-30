import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_history(X, Y_history, Z_history, save_gif=False):
    input_dim, latent_dim = X.shape[1], Z_history[0].shape[1]
    input_projection_type = '3d' if input_dim > 2 else 'rectilinear'

    fig = plt.figure(figsize=(10, 5))
    input_ax = fig.add_subplot(1, 2, 1, projection=input_projection_type)
    latent_ax = fig.add_subplot(1, 2, 2)
    num_epoch = len(Y_history)

    if input_dim == 3 and latent_dim == 2:
        _, K, _ = Y_history.shape
        reso = int(np.sqrt(K))
        Y_history = np.array(Y_history).reshape((num_epoch, reso, reso, input_dim))

    observable_drawer = [None, None, draw_observable_2D,
                         draw_observable_3D][input_dim]
    latent_drawer = [None, draw_latent_1D, draw_latent_2D][latent_dim]

    print("X: {}, Y: {}, Z:{}".format(X.shape, Y_history[0].shape,
                                      Z_history[0].shape))
    ani = FuncAnimation(fig,
                        update_graph,
                        frames=num_epoch,
                        repeat=True, interval=5,
                        fargs=(observable_drawer, latent_drawer, X, Y_history,
                               Z_history, fig, input_ax, latent_ax, num_epoch))
    plt.show()
    if save_gif:
        ani.save("tmp.gif", writer='pillow')


def update_graph(epoch, observable_drawer, latent_drawer, X, Y_history,
                 Z_history, fig, input_ax, latent_ax, num_epoch):
    fig.suptitle(f"epoch: {epoch}")
    input_ax.cla()
    #  input_ax.view_init(azim=(epoch * 400 / num_epoch), elev=30)
    latent_ax.cla()

    Y, Z = Y_history[epoch], Z_history[epoch]
    colormap = X[:, 0]

    observable_drawer(input_ax, X, Y, colormap)
    latent_drawer(latent_ax, Z, colormap)


def draw_observable_3D(ax, X, Y, colormap):
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colormap)
    #  ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='b', alpha=0.5, s=30)
    if len(Y.shape) == 3:
        ax.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black', alpha=0.5)


#     else:
#         ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='black')
# ax.plot(Y[:, 0], Y[:, 1], Y[:, 2], color='black')
# ax.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')


def draw_observable_2D(ax, X, Y, colormap):
    ax.scatter(X[:, 0], X[:, 1], c=colormap)
    ax.plot(Y[:, 0], Y[:, 1], c='k')


def draw_latent_2D(ax, Z, colormap):
    # ax.set_xlim(-5, 5)
    # ax.set_ylim(-5, 5)
    ax.scatter(Z[:, 0], Z[:, 1], c=colormap)


def draw_latent_1D(ax, Z, colormap):
    ax.scatter(Z, np.zeros(Z.shape), c=colormap)
    ax.set_ylim(-1, 1)
