import pickle
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, models, transforms
from tqdm import tqdm

from data import gen_saddle_shape
from ukr import UKRNet

# プロセッサの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データの準備
X = torch.from_numpy(gen_saddle_shape(N := 100).astype(np.float32)).to(device)
X_train = X.repeat(samples := 1000, 1, 1)
train = torch.utils.data.TensorDataset(X_train, X_train)
trainloader = torch.utils.data.DataLoader(train, batch_size=1, shuffle=True)

# モデル，学習の設定
model = UKRNet(N).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(),
                      lr=0.01,
                      momentum=0.9,
                      weight_decay=1e-4)

# 学習結果，loss 保存用の変数
num_epoch = 200
Z_history = np.zeros((num_epoch, N, 2))
losses = []

# 学習ループ
with tqdm(range(num_epoch)) as pbar:
    for epoch in pbar:
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, targets = data
            inputs, targets = Variable(inputs), Variable(targets)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 潜在変数の保存
        Z_history[epoch] = model.layer.Z.detach().cpu().numpy()
        # loss の値の保存
        losses.append(running_loss)
        # プログレスバーの表示
        pbar.set_postfix(OrderedDict(epoch=f"{epoch + 1}", loss=f"{running_loss:.3f}"))

# Loss の推移の描画
plt.plot(np.arange(num_epoch), np.array(losses))
plt.show()

# 学習結果を *.pickle で保存
with open("./X.pickle", 'wb') as f:
    pickle.dump(X.detach().cpu().numpy(), f)
with open("./Z_history.pickle", 'wb') as f:
    pickle.dump(Z_history, f)
