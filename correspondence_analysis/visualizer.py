import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_history(X, Y_history, U1_history, U2_history, Z1_history, Z2_history, Zeta1, Zeta2, save_gif=True):
    fig = plt.figure(figsize=(10, 5))
    input_ax = fig.add_subplot(1, 2, 1, projection='3d')
    latent_ax = fig.add_subplot(1, 2, 2)
    num_epoch = len(Y_history)

    ani = FuncAnimation(fig, update_graph,
                        frames=num_epoch, repeat=True,
                        fargs=(X, Y_history, U1_history, U2_history,
                               Z1_history, Z2_history, Zeta1, Zeta2,
                               fig, input_ax, latent_ax, num_epoch))
    plt.show()
    if save_gif:
        ani.save("tmp.gif", writer='imagemagick')


def update_graph(epoch, X, Y_history, U1_history, U2_history,
                 Z1_history, Z2_history, Zeta1, Zeta2,
                 fig, input_ax, latent_ax, num_epoch):
    fig.suptitle(f"epoch: {epoch}")
    input_ax.cla()
    latent_ax.cla()

    Y = Y_history[epoch]
    U1 = U1_history[epoch]
    U2 = U2_history[epoch]
    Z1 = Z1_history[epoch]
    Z2 = Z2_history[epoch]
    colormap = 'green'
    # colormap = X[:, 0, 0]
    draw_X(input_ax, X, colormap)
    draw_U(input_ax, U1, 'red')
    draw_U(input_ax, U2, 'blue')
    draw_Y(input_ax, Y)
    # draw_latent(latent_ax, Z1, Zeta2, colormap)
    draw_latent(latent_ax, Z2, Zeta2, colormap)
    input_ax.set_xlim(-1.5, 1.5)
    input_ax.set_ylim(-1.5, 1.5)
    input_ax.set_zlim(-1.5, 1.5)


def draw_X(ax, X, colormap):
    for X_i in X:
        ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2], c=colormap)


def draw_U(ax, U, color):
    for U_i in U:
        ax.scatter(U_i[:, 0], U_i[:, 1], U_i[:, 2], color=color)


def draw_Y(ax, Y):
    ax.plot_wireframe(Y[:, :, 0], Y[:, :, 1], Y[:, :, 2], color='black')


def draw_latent(ax, Z, Zeta, colormap):
    ax.scatter(Zeta, np.zeros(Zeta.shape), c='gray', marker='x')
    ax.scatter(Z, np.zeros(Z.shape), c=colormap)
