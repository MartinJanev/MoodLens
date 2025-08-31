from .CNN_class import EmotionCNN
import torch.nn as nn

# Optional torchvision (for ResNet18)
try:
    from torchvision.models import resnet18, ResNet18_Weights
except Exception:
    resnet18 = None
    ResNet18_Weights = None

def create_model(name: str = "cnn_small", num_classes: int = 7, pretrained: bool = True):
    """
    Factory for models used in this project.
    - "cnn_small" (EmotionCNN) : compact FER model for 48x48 grayscale
    - "resnet18"               : ImageNet-pretrained ResNet-18 with replaced FC for FER classes
    """
    name = name.lower()
    if name in ("cnn_small", "emotion_cnn", "default"):
        return EmotionCNN(num_classes=num_classes)

    if name in ("resnet18", "resnet"):
        if resnet18 is None:
            raise ImportError("torchvision is required for resnet18 (pip install torchvision).")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = resnet18(weights=weights)
        in_feats = m.fc.in_features
        m.fc = nn.Linear(in_feats, num_classes)
        return m

    raise ValueError(f"Unknown model: {name}")
