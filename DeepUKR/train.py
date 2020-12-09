import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from tqdm import tqdm

from data import gen_saddle_shape
from ukr_layer import UKR

samples = 1000
N = 100
num_epoch = 500

X = torch.from_numpy(gen_saddle_shape(N).astype(np.float32))
X_train = X.repeat(samples, 1, 1)
train = torch.utils.data.TensorDataset(X_train, X_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = UKR(N, latent_dim=2, sigma=2)

    def forward(self, x):
        return self.layer(x)


model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

Y_history = np.zeros((num_epoch, N, 3))
Z_history = np.zeros((num_epoch, N, 2))

losses = []
with tqdm(range(num_epoch)) as pbar:
    for epoch in pbar:
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            outputs1, outputs2 = model(inputs)
            loss = criterion(outputs1, labels)
            loss = loss + loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        pbar.set_postfix(
            OrderedDict(epoch=f"{epoch + 1}", loss=f"{running_loss:.3f}"))
        Y, Z = model(X)
        Y_history[epoch] = Y.detach().numpy()
        Z_history[epoch] = model.layer.Z.detach().numpy()
        losses.append(running_loss)
        running_loss = 0.0

with open("./X.pickle", 'wb') as f:
    pickle.dump(X.detach().numpy(), f)
with open("./Y_history.pickle", 'wb') as f:
    pickle.dump(Y_history, f)
with open("./Z_history.pickle", 'wb') as f:
    pickle.dump(Z_history, f)

plt.plot(np.arange(num_epoch), np.array(losses))
plt.show()
