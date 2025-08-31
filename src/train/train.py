# src/scripts/train.py
import os, sys, multiprocessing, torch

HERE = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if SRC_ROOT not in sys.path: sys.path.insert(0, SRC_ROOT)

from src.data.fer2013 import FER2013Class, DEFAULT_CLASSES
from src.models.factory import create_model
from src.train.train_loop import TrainConfig, train_model
from src.utils.seed import fix_seed

TRAIN_CSV = os.path.join(PROJECT_ROOT, "datasets", "train.csv")
VAL_CSV = os.path.join(PROJECT_ROOT, "datasets", "val.csv")

# ======= Optimized defaults =======
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 30
BATCH_SIZE = 64
LR = 1e-3
NUM_WORKERS = max(2, min(8, multiprocessing.cpu_count() // 2))  # 2–8 workers is optimal for CPU
PREFETCH = 6  # number of prefetch batches in DataLoader
PERSISTENT = True  # keep DataLoader workers alive between epochs
WEIGHT_DECAY = 1e-5
OUT_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(OUT_DIR, exist_ok=True)
USE_CLAHE = True
USE_CACHE = True  # << key speedup
CHANNELS_LAST = True
USE_TORCH_COMPILE = False  # set True if trying PyTorch 2.0+ compile mode. Means you need PyTorch 2.0+ installed.
SEED = 1337

# ================================

if __name__ == "__main__":
    fix_seed(SEED)

    # Limit BLAS threads to avoid contention with DataLoader workers
    try:
        torch.set_num_threads(max(1, multiprocessing.cpu_count() // 2))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, multiprocessing.cpu_count() // 2)))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

    tr_ds = FER2013Class(TRAIN_CSV, classes=DEFAULT_CLASSES, augment=True, use_clahe=USE_CLAHE, cache=USE_CACHE)
    va_ds = FER2013Class(VAL_CSV, classes=DEFAULT_CLASSES, augment=False, use_clahe=USE_CLAHE, cache=USE_CACHE)

    model = create_model("cnn_small", num_classes=len(DEFAULT_CLASSES))

    cfg = TrainConfig(
        epochs=EPOCHS, lr=LR, weight_decay=WEIGHT_DECAY, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, prefetch_factor=PREFETCH, persistent_workers=PERSISTENT,
        device=DEVICE, out_dir=OUT_DIR, channels_last=CHANNELS_LAST,
        use_torch_compile=USE_TORCH_COMPILE, show_progress=True,  # << here
    )

    train_model(model, tr_ds, va_ds, DEFAULT_CLASSES, cfg)
