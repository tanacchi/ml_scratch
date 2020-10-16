import numpy as np
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


def visualize_component_plane(X, history, Zeta1, Zeta2):
    Y = history.Y[-1]
    U1, U2 = history.U1[-1], history.U2[-1]
    Z1, Z2 = history.Z1[-1], history.Z2[-1]
    fig_mode1 = plt.figure(figsize=(5, 5))
    fig_mode2 = plt.figure(figsize=(5, 5))
    mode1_ax = fig_mode1.gca()
    mode2_ax = fig_mode2.gca()
    latent_dim = Zeta1.shape[1]

    cp = ComponentPlane(X, Y, U1, U2, Z1, Z2, Zeta1, Zeta2, fig_mode1, fig_mode2, mode1_ax, mode2_ax, latent_dim)
    cp.draw()
    fig_mode1.canvas.mpl_connect('button_press_event', cp.onclick_mode1)
    fig_mode2.canvas.mpl_connect('button_press_event', cp.onclick_mode2)
    plt.show()


class ComponentPlane(object):
    def __init__(self, X, Y, U1, U2, Z1, Z2, Zeta1, Zeta2,
               fig_mode1, fig_mode2, mode1_ax, mode2_ax, latent_dim):
        self.X = X
        self.Y = Y
        self.U1 = U1
        self.U2 = U2
        self.Z1 = Z1
        self.Z2 = Z2
        self.Zeta1 = Zeta1
        self.Zeta2 = Zeta2
        self.fig_mode1 = fig_mode1
        self.fig_mode2 = fig_mode2
        self.mode1_ax = mode1_ax
        self.mode2_ax = mode2_ax
        self.latent_dim = latent_dim

    def draw(self):
        self.fig_mode1.suptitle("Mode1")
        self.fig_mode2.suptitle("Mode2")
        self.mode1_ax.cla()
        self.mode2_ax.cla()

        self.onclick_mode1(event=None, clicked_point=[[0.5, 0.2]])
        self.onclick_mode2(event=None, clicked_point=[[0.1, 0.2]])

    def draw_latent(self, ax, Z, Zeta, colormap):
        ax.scatter(Zeta, np.zeros(Zeta.shape), c='gray', marker='x')
        ax.scatter(Z, np.zeros(Z.shape), c=colormap)


    def onclick_mode1(self, event, clicked_point=None):
        if not clicked_point:
            if event.xdata is None or event.ydata is None:
                return
            clicked_point = [[event.xdata, event.ydata]]
        clicked_point = np.array(clicked_point)
        unit = get_bmu(self.Zeta1, clicked_point)
        self.calc_component_mode1(unit)
        self.mode2_ax.cla()
        #  self.draw_latent(self.mode2_ax, self.Z2, self.Zeta2, colormap=self.Map2)
        if self.latent_dim == 1:
            self.mode2_ax.imshow(self.Map2[:, np.newaxis].T, interpolation='spline36',
                    extent=[0, self.Map2.shape[0] -1, 1, 0], cmap="bwr")
        else:
            self.mode2_ax.imshow(self.Map2[::], interpolation='spline36',
                    extent=[0, self.Map2.shape[0] -1, -self.Map2.shape[1] + 1, 0], cmap="bwr")
            self.mode1_ax.plot(self.Zeta1[unit][0], self.Zeta2[unit][1], ".", ms=20, color='black')
        #  self.mode2_ax.scatter(self.Zeta2, np.zeros(self.Map2.shape), c=self.Map2)
        #  cbar = self.fig_mode2.colorbar(im, ticks=[0,0.5, 1])
        #  cbar.ax.set_ylim(0,1)
        #  char.ax.set_yticklabels(['< 0', '0.5', '> 1'])
        self.fig_mode1.canvas.draw()
        self.fig_mode2.canvas.draw()

    def calc_component_mode1(self, clicked_unit):
        tmp = self.Y[clicked_unit, :, :]
        self.Map2 = np.sqrt(np.sum(tmp * tmp, axis=1)).reshape([10, 10])

    def onclick_mode2(self, event, clicked_point=None):
        if not clicked_point:
            if event.xdata is None or event.ydata is None:
                return
            clicked_point = [[event.xdata, event.ydata]]
        clicked_point = np.array(clicked_point)
        unit = get_bmu(self.Zeta2, clicked_point)
        self.calc_component_mode2(unit)
        self.mode1_ax.cla()
        if self.latent_dim == 1:
            self.mode1_ax.imshow(self.Map1[:, np.newaxis].T, interpolation='spline36',
                    extent=[0, self.Map2.shape[0] -1, 1, 0], cmap="bwr")
        else:
            self.mode1_ax.imshow(self.Map1[::], interpolation='spline36',
                    extent=[0, self.Map1.shape[0] -1, -self.Map1.shape[1] + 1, 0], cmap="bwr")
        self.fig_mode1.canvas.draw()
        self.fig_mode2.canvas.draw()

    def calc_component_mode2(self, clicked_unit):
        tmp = self.Y[:, clicked_unit, :]
        self.Map1 = np.sqrt(np.sum(tmp * tmp, axis=1)).reshape([10, 10])

def get_bmu(Zeta, clicked_point):
    dists = dist.cdist(Zeta, clicked_point)
    unit = np.argmin(dists, axis=0)
    return unit[0]
