from data import gen_saddle_shape
from tsom import TSOM
from visualizer import visualize_history

X = gen_saddle_shape(10)
tsom = TSOM()
history, Zeta1, Zeta2 = tsom.fit(X, num_epoch=50)
visualize_history(X, history.Y, history.U1, history.U2,
                  history.Z1, history.Z2, Zeta1, Zeta2, save_gif=False)
