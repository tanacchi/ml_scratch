import pickle
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision
from torchvision import datasets, models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from data import gen_saddle_shape
from ukr_layer import UKR


samples = 1000
N = 100
num_epoch = 1000

X = torch.from_numpy(gen_saddle_shape(N).astype(np.float32))
X_train = X.repeat(samples, 1, 1)
train = torch.utils.data.TensorDataset(X_train, X_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer = UKR(N, latent_dim=2)

    def forward(self, x):
        return self.layer(x)

model = Net()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

Y_history = np.zeros((num_epoch, N, 3))
Z_history = np.zeros((num_epoch, N, 2))

losses = []
for epoch in tqdm(range(num_epoch)):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss))
    Y_history[epoch] = model(X).detach().numpy()
    Z_history[epoch] = model.layer.Z.detach().numpy()
    losses.append(running_loss)
    running_loss = 0.0

with open("./Y_history.pickle", 'wb') as f:
    pickle.dump(Y_history, f)
with open("./Z_history.pickle", 'wb') as f:
    pickle.dump(Z_history, f)

plt.plot(np.arange(num_epoch), np.array(losses))
plt.show()
