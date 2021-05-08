import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
from Data import CellTestDataset
from vit_pytorch.deepvit import DeepViT

TestData = CellTestDataset()
TestData_Image = TestData.testImage
TestData_Hash = TestData.ImageHash
TestData_Size = TestData.ImgSize
model_path = './DeepViTTorchModel.pt'
ones = torch.ones(19)
zeros = torch.zeros(19)
submission_path = './submission.csv'
Submission = {}
device = torch.device('cuda')

Submission['ID'] = []
Submission['ImageWidth'] = []
Submission['ImageHeight'] = []
Submission['PredictionString'] = []

visionTransformer = DeepViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 19,
    dim = 2048,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.3,
    channels=4
)

visionTransformer.load_state_dict(torch.load(model_path))
visionTransformer.eval()

for i in tqdm(range(len(TestData_Hash))):
    testImg = TestData_Image[i].astype(np.float64)
    testImg = np.array([testImg])
    result = visionTransformer(torch.Tensor(testImg))
    result = torch.where(result > 0.4, ones, zeros).tolist()
    pred = [ j for j in range(len(result)) if result[j] == 1 ]
    prediction = ' '.join(pred)
    Submission['ID'].append(TestData_Hash[i])
    Submission['ImageWidth'].append(TestData_Size[TestData_Hash[i]][0])
    Submission['ImageHeight'].append(TestData_Size[TestData_Hash[i]][1])
    Submission['PredictionString'].append(prediction)

SubmissionCSV = pd.DataFrame(Submission)
SubmissionCSV.to_csv(submission_path)