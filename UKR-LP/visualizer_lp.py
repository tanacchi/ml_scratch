import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_history(X, history, save_gif=False):
    Y_history = history.Y
    F_history = history.F
    f_history = history.f
    Zeta_history = history.Zeta
    Z_history = history.Z
    input_dim, latent_dim  = X.shape[1], Z_history[0].shape[1]
    input_projection_type = '3d' if input_dim > 2 else 'rectilinear'

    fig = plt.figure(figsize=(10, 5))
    input_ax = fig.add_subplot(1, 2, 1, projection=input_projection_type)
    latent_ax = fig.add_subplot(1, 2, 2)
    num_epoch = len(F_history)

    if input_dim  == 3 and latent_dim == 2:
        F_history = np.array(F_history).reshape((num_epoch, 10, 10, input_dim))
        f_history = np.array(f_history).reshape((num_epoch, 10, 10, input_dim))

    observable_drawer = [None, None, draw_observable_2D, draw_observable_3D][input_dim]
    latent_drawer     = [None, draw_latent_1D, draw_latent_2D][latent_dim]

    ani = FuncAnimation(fig, update_graph,
                        frames=num_epoch, repeat=True,
                        fargs=(observable_drawer, latent_drawer,
                               X, Y_history, F_history, f_history, Zeta_history, Z_history,
                               fig, input_ax, latent_ax, num_epoch))
    plt.show()
    if save_gif:
        ani.save("tmp.gif", writer='pillow')


def update_graph(epoch, observable_drawer, latent_drawer,
                 X, Y_history, F_history, f_history, Zeta_history, Z_history,
                 fig, input_ax, latent_ax, num_epoch):
    fig.suptitle(f"epoch: {epoch}")
    input_ax.cla()
    input_ax.view_init(azim=(epoch*400 / num_epoch), elev=30)
    latent_ax.cla()

    f, F, Z = f_history[epoch], F_history[epoch], Z_history[epoch]
    Y, Zeta = Y_history[epoch], Zeta_history[epoch]
    colormap = X[:, 0]

    observable_drawer(input_ax, X, Y, f, colormap)
    latent_drawer(latent_ax, Zeta, Z, colormap)


def draw_observable_3D(ax, X, Y, f, colormap):
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colormap)
    ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2], c='red', alpha=1, s=50)
    if len(f.shape) == 3:
        ax.plot_wireframe(f[:, :, 0], f[:, :, 1], f[:, :, 2], color='black')
#     else:
#         ax.plot(F[:, 0], F[:, 1], F[:, 2], color='black')
    # ax.plot(F[:, 0], F[:, 1], F[:, 2], color='black')
    # ax.plot_wireframe(F[:, :, 0], F[:, :, 1], F[:, :, 2], color='black')


def draw_observable_2D(ax, X, F, colormap):
    ax.scatter(X[:, 0], X[:, 1], c=colormap)
    ax.plot(F[:, 0], F[:, 1], c='black')


def draw_latent_2D(ax, Zeta, Z, colormap):
    #  ax.set_xlim(-5, 5)
    #  ax.set_ylim(-5, 5)
    ax.scatter(Zeta[:, 0], Zeta[:, 1], c='red')
    ax.scatter(Z[:, 0], Z[:, 1], c=colormap)


def draw_latent_1D(ax, Z, colormap):
    ax.scatter(Z, np.zeros(Z.shape), c=colormap)
    ax.set_ylim(-1, 1)
