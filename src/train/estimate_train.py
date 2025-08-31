# Estimate training time for a model on CPU or GPU.
# Adjust batch size and number of workers as needed.

# We use this module to estimate training time for a model on CPU or GPU.
# Sometimes the training time can be long, and this helps to get a rough idea of how long it might take.

import os, sys, time, math, torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(__file__)
SRC_ROOT = os.path.abspath(os.path.join(HERE, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if SRC_ROOT not in sys.path: sys.path.insert(0, SRC_ROOT)


from src.data.fer2013 import FER2013Class, DEFAULT_CLASSES
from src.models.factory import create_model
from src.utils.time_measure import measure_time as format_time


TRAIN_CSV = os.path.join(PROJECT_ROOT, "datasets", "train.csv")
VAL_CSV = os.path.join(PROJECT_ROOT, "datasets", "val.csv")
MODEL_NAME = "cnn_small"
BATCH_SIZE = 128  # try 128–512 on CPU; adjust if RAM-bound - e.g. for 16GB RAM, use 64 or 128
NUM_WORKERS = 8
EPOCHS_TARGET = 30
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # "cpu" or "cuda"
USE_CLAHE = True
WARMUP_STEPS_TRAIN = 10
MEASURE_STEPS_TRAIN = 50
WARMUP_STEPS_VAL = 10
MEASURE_STEPS_VAL = 20

if __name__ == "__main__":
    device = torch.device(DEVICE)
    tr_ds = FER2013Class(TRAIN_CSV, classes=DEFAULT_CLASSES, augment=True, use_clahe=USE_CLAHE)
    va_ds = FER2013Class(VAL_CSV, classes=DEFAULT_CLASSES, augment=False, use_clahe=USE_CLAHE)
    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = create_model(MODEL_NAME, num_classes=len(DEFAULT_CLASSES)).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit = torch.nn.CrossEntropyLoss()


    def cycle(loader):
        """Cycle through the DataLoader indefinitely."""
        while True:
            for batch in loader: yield batch


    it = cycle(tr_loader)
    for _ in range(WARMUP_STEPS_TRAIN):
        xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        optim.step()
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEASURE_STEPS_TRAIN):
        xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward()
        optim.step()
    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()
    train_step_time = (t1 - t0) / MEASURE_STEPS_TRAIN
    itv = cycle(va_loader)

    with torch.no_grad():
        for _ in range(WARMUP_STEPS_VAL):
            xb, yb = next(itv)
            xb = xb.to(device)
            _ = model(xb)
    if device.type == "cuda": torch.cuda.synchronize()
    v0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(MEASURE_STEPS_VAL):
            xb, yb = next(itv)
            xb = xb.to(device)
            _ = model(xb)
    if device.type == "cuda": torch.cuda.synchronize()
    v1 = time.perf_counter()
    val_step_time = (v1 - v0) / MEASURE_STEPS_VAL


    def steps_per_epoch(n):
        """Calculate the number of steps per epoch based on dataset size and batch size."""
        return math.ceil(n / BATCH_SIZE)


    tr_steps, va_steps = steps_per_epoch(len(tr_ds)), steps_per_epoch(len(va_ds))
    epoch_train_time = tr_steps * train_step_time
    epoch_val_time = va_steps * val_step_time
    epoch_time = epoch_train_time + epoch_val_time
    total_time = epoch_time * EPOCHS_TARGET


    print("\n=== Timing Estimate ===")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE}, num_workers: {NUM_WORKERS}")
    print(f"Train step: {train_step_time:.4f} s  |  Val step: {val_step_time:.4f} s")
    print(f"Per-epoch: train ~ {format_time(epoch_train_time)}, val ~ {format_time(epoch_val_time)}, total ~ {format_time(epoch_time)}")
    print(f"Projected total for {EPOCHS_TARGET} epochs: {format_time(total_time)}\n")
