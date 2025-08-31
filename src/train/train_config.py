# src/scripts/train_config.py
from src.train.config import default_config, with_overrides

# === OPTION A: Train YOUR CNN (same as before) ==========================
# CONFIG = with_overrides(
#     default_config(),
#     model_name="cnn_small",     # <— your model
#     out_dir="models",
#     train_csv="datasets/train.csv",
#     val_csv="datasets/val.csv",
#     epochs=40,
#     batch_size=128,
#     num_workers=4,
#     lr=1e-3,
#     weight_decay=1e-5,
#     use_clahe=True,
#     use_cache=True,
#     channels_last=True,
#     use_torch_compile=False,
#     show_progress=True,
# )

# === OPTION B: Train FAST ResNet-18 (toggle by commenting the block above
# and uncommenting this one) =============================================
CONFIG = with_overrides(
    default_config(),
    model_name="resnet18",     # <— ResNet path
    out_dir="models",
    train_csv="datasets/train.csv",
    val_csv="datasets/val.csv",
    epochs=30,
    batch_size=64,             # keep modest; ResNet uses bigger inputs
    lr=1e-3,
    weight_decay=1e-5,
    use_clahe=True,
    use_cache=True,
    channels_last=True,
    use_torch_compile=False,
    show_progress=True,
    # Fast-ResNet tweaks:
    use_amp=True,
    resnet_input_size=160,         # 160 is a good speed/acc trade-off
    resnet_freeze_backbone_epochs=3,
)
