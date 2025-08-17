from .CNN_class import EmotionCNN

def create_model(name: str = "cnn_small", num_classes: int = 7):
    name = name.lower()
    if name in ("cnn_small", "emotioncnn", "default"):
        return EmotionCNN(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")
