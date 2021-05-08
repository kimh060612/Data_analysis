import torch
from tqdm import tqdm
import pandas as pd
from Data import CellTestDataset
from model.model import visionTransformer

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
ViT = visionTransformer.load_state_dict(torch.load(model_path))

for i in tqdm(range(len(TestData_Hash))):
    result = ViT(torch.Tensor(TestData_Image[i]))
    result = torch.where(result > 0.6, ones, zeros).tolist()
    pred = result.join(' ')
    Submission['ID'].append(TestData_Hash[i])
    Submission['ImageWidth'].append(TestData_Size[TestData_Hash[i]][0])
    Submission['ImageHeight'].append(TestData_Size[TestData_Hash[i]][1])
    Submission['PredictionString'].append(pred)

SubmissionCSV = pd.DataFrame(Submission)
SubmissionCSV.to_csv(submission_path)