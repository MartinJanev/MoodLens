import torch.nn as nn

try:
    from torchvision.models import resnet18, ResNet18_Weights
except Exception:
    resnet18 = None
    ResNet18_Weights = None


def create_resnet18(num_classes: int = 7, pretrained: bool = True):
    if resnet18 is None:
        raise ImportError("torchvision is required for resnet18 (pip install torchvision).")
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = resnet18(weights=weights)
    in_feats = m.fc.in_features
    m.fc = nn.Linear(in_feats, num_classes)
    return m
