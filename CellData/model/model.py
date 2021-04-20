from vit_pytorch import ViT

visionTransformer = ViT(
    image_size=256,
    patch_size=32,
    num_classes=19,
    channels=4,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)