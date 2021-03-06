import torch
import torch.nn as nn
from torch import optim
from model.loss import AsymmetricLoss
from model.model import visionTransformer, device
from Data import CellTrainingDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

learning_rate = 0.001
training_epochs = 20
batch_size = 256

TrainData = CellTrainingDataset()

dataLoader = DataLoader(dataset=TrainData, 
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last = True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(visionTransformer.parameters(), lr=learning_rate)

total_batch = len(dataLoader)
print('총 배치의 수 : {}'.format(total_batch))

cnt = 0

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in tqdm(dataLoader): 
        if cnt == 0:
            print(X.shape)
        cnt += 1
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = visionTransformer(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

model_path = './DeepViTTorchModel.pt'
torch.save(visionTransformer.state_dict(), model_path)
