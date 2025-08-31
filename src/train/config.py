# src/train/config.py
from dataclasses import dataclass, replace
import multiprocessing, torch
from typing import Optional, Any


def _auto_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _auto_workers() -> int:
    try:
        return max(2, min(12, multiprocessing.cpu_count() // 2)) if not torch.cuda.is_available() \
            else max(2, min(12, multiprocessing.cpu_count()))
    except Exception:
        return 2


def _auto_prefetch(num_workers: int) -> int:
    return 6 if num_workers >= 4 else 2


@dataclass
class TrainConfig:
    # --- data & model ---
    train_csv: str = "datasets/train.csv"
    val_csv: str = "datasets/val.csv"
    out_dir: str = "models"

    # Pick exactly one:
    #   "cnn_small"  -> your 48x48 grayscale CNN (same behavior as before)
    #   "resnet18"   -> ImageNet-pretrained ResNet-18 (fast path enabled below)
    model_name: str = "cnn_small"

    use_clahe: bool = True
    use_cache: bool = True
    seed: int = 1337

    # --- training ---
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    num_workers: int = _auto_workers()
    prefetch_factor: int = _auto_prefetch(_auto_workers())
    persistent_workers: bool = True
    device: str = _auto_device()
    channels_last: bool = True
    use_torch_compile: bool = False
    show_progress: bool = True

    # --- early stopping ---
    early_stop_patience: int = 6
    early_stop_delta: float = 1e-4

    # ---------- Fast-ResNet knobs (ignored for cnn_small) ----------
    use_amp: bool = True  # half-precision on CUDA
    resnet_input_size: int = 160  # 224→160 for ~2–3x speedup
    resnet_freeze_backbone_epochs: int = 3  # linear-probe warmup, then unfreeze


def default_config() -> TrainConfig:
    nw = _auto_workers()
    return TrainConfig(
        num_workers=nw,
        prefetch_factor=_auto_prefetch(nw),
        device=_auto_device(),
    )


def with_overrides(base: Optional[TrainConfig] = None, **over: Any) -> TrainConfig:
    cfg = base or default_config()
    return replace(cfg, **over)
