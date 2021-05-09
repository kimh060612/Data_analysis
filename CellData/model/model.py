from vit_pytorch import ViT
from vit_pytorch.deepvit import DeepViT
import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777) 
# GPU 사용 가능일 경우 랜덤 시드 고정
if device == 'cuda':
    torch.cuda.manual_seed_all(777)

visionTransformer = nn.Sequential(
    DeepViT(
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
    ), 
    nn.Softmax()
).to(device)