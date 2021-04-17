import torch
from torch.utils.data import Dataset

class CellDataset(Dataset):
    def __init__(self, img_dir):
        super().__init__()

    def getEncodedDistribution(self):
        pass

    def getRGBYImages(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass