from vit_pytorch import ViT

visionTransformer = ViT(
    image_size=256,
    patch_size=32,
    num_classes=19,
    dim=4,
    channels=4,
)