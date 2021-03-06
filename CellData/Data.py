import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import pandas as pd

imageType = ['red', 'green', 'blue', 'yellow']

def imageMerge(imageHash, ImagePath):
    imageList = []
    for i in range(4):
        imageName = imageHash + '_' + imageType[i] + '.png'
        imageArray = cv2.imread(ImagePath + '/' + imageName, cv2.IMREAD_UNCHANGED)
        imageList.append(imageArray)
    RGBYImage = np.transpose(np.array(imageList), (1, 2, 0))
    RGBYImage = np.transpose(RGBYImage, (2, 0, 1))
    return RGBYImage

def oneHotEncoding(label):
    OH = np.zeros(19)
    for con in label:
        OH[con] = 1
    return OH

class CellTrainingDataset(Dataset):
    def __init__(self, img_dir = '/data/CellData'):
        super().__init__()
        self.ImagePath = os.path.abspath(img_dir)
        self.dataList = os.listdir(self.ImagePath + '/train_resize')
        self.GTList = pd.read_csv(self.ImagePath + '/train.csv')
        self.imgLabel = self.getEncodedDistribution()
        self.ImageTensor = self.getRGBYImages()

    def getEncodedDistribution(self):
        oneHotEmbedding = []
        N = len(self.GTList)
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
            image2Load = imageMerge(imageHash, self.ImagePath + '/train_resize')
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

class CellTestDataset:
    def __init__(self):
        super().__init__()
        self.testImageDir = os.path.abspath('/data/CellData/test_resize')
        self.testList = os.listdir(self.testImageDir)
        self.testImage = self.getRGBYImage()
        self.ImageHash = self.getTestImageDict()
        self.ImgSize = self.getTestImageSize()

    def getTestImageDict(self):
        N = len(self.testList)
        imageType = ['red', 'green', 'blue', 'yellow']
        imageHashList = []
        imageDict = {}
        for i in range(N):
            imageHash = self.testList[i].split('_')[0]
            try:
                imageDict[imageHash].append(self.testList[i])
            except:
                imageDict[imageHash] = [self.testList[i]]
        for _, key in enumerate(imageDict):
            imageHashList.append(key)
        return imageHashList

    def getRGBYImage(self):
        imageHashList = self.getTestImageDict()
        N = len(imageHashList)
        image2Test = []
        for i in range(N):
            image2Load = imageMerge(imageHashList[i], self.testImageDir)
            image2Test.append(image2Load)
        image2Test = np.array(image2Test)
        return image2Test

    def getTestImageSize(self):
        N = len(self.ImageHash)
        ImgSizeHash = {}
        for i in range(N):
            img = cv2.imread('/data/CellData/test/' + self.ImageHash[i] + '_red.png')
            ImgSizeHash[self.ImageHash[i]] = (img.shape[0], img.shape[1])
        return ImgSizeHash
        