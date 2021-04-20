import torch
import torch.nn as nn
from torch import optim
from model.loss import AsymmetricLoss
from model.model import visionTransformer
from Data import CellTestDataset, CellTrainingDataset
from torch.utils.data import DataLoader

learning_rate = 0.001
training_epochs = 15
batch_size = 100

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777) 
# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

TrainData = CellTrainingDataset()
TestData = CellTestDataset().testImage

dataLoader = DataLoader(dataset=TrainData, 
                        batch_size=64,
                        shuffle=True,
                        drop_last = True)

TransformerModel = visionTransformer().to(device)

criterion = AsymmetricLoss()
optimizer = optim.Adam(TransformerModel.parameters(), lr=learning_rate)

total_batch = len(dataLoader)
print('총 배치의 수 : {}'.format(total_batch))

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in dataLoader: # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y느 ㄴ레이블.
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = TransformerModel(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))