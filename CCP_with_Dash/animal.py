import sys
sys.path.append("../..")

from datasets import load_data
from tsom import TSOM
from component_plane import visualize_component_plane


X, labels_animal, labels_feature = load_data(retlabel_animal=True, retlabel_feature=True)
tsom = TSOM(L1=2, L2=2, K1=10, K2=10,
            sigma1_max=2.2, sigma2_max=2.2,
            sigma1_min=0.2, sigma2_min=0.2,
            tau1=50, tau2=50)
history, Zeta1, Zeta2 = tsom.fit(X, num_epoch=50)
visualize_component_plane(X, history, Zeta1, Zeta2)
