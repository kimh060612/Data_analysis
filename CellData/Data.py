import torch
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import pandas as pd

imageType = ['red', 'green', 'blue', 'yellow']

def imageMerge(imageHash, ImagePath):
    imageList = []
    for i in range(4):
        imageName = imageHash + '_' + imagType[i] + '.png'
        imageArray = Image.open(ImagePath + '/' + imageName, cv2.IMREAD_UNCHANGED)
        imageList.append(imageArray)
    RGBYImage = np.transpose(np.array(imageList), (1, 2, 0))
    return RGBYImage

def oneHotEncoding(label):
    OH = np.zeros(19)
    for con in label:
        OH[con] = 1
    return OH

class CellTrainingDataset(Dataset):
    def __init__(self, img_dir):
        super().__init__()
        self.ImagePath = os.path.abspath('/data/CellData/train')
        self.dataList = os.listdir(self.ImagePath)
        self.GTList = pd.read_csv(self.ImagePath + '/train.csv')
        self.imgLabel = self.getEncodedDistribution()
        self.ImageTensor = self.getRGBYImages()

    def getEncodedDistribution(self):
        oneHotEmbedding = []
        for i in range(N):
            labels = self.GTList['Label'][i]
            labels = labels.split('|')
            labels = [ int(x) for x in labels ]
            labels = oneHotEncoding(labels)
            oneHotEmbedding.append(labels)
        oneHotEmbedding = np.array(oneHotEmbedding)
        return oneHotEmbedding

    def getRGBYImages(self):
        N = len(self.GTList)
        image2Train = []
        for i in range(N):
            imageHash = self.GTList['ID'][i]
            image2Load = imageMerge(imageHash, self.ImagePath)
            image2Train.append(image2Load)
        imageArr2Load = np.array(image2Train)
        print(imageArr2Load.shape)
        return imageArr2Load

    def __len__(self):
        return len(self.ImageTensor)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.ImageTensor[idx])
        y = torch.FloatTensor(self.imgLabel[idx])
        return x, y

class CellTestDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.testImageDir = os.path.abspath('/data/CellData/test')
        

    def getRGBYImage(self):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self):
        pass

