# src/train/train_loop.py
import time, math
from dataclasses import dataclass
from typing import List
import torch
from torch.utils.data import DataLoader
from .metrics import accuracy
from ..utils.io import save_checkpoint
from ..utils.time_measure import measure_time as format_time


try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # fallback to no bars if tqdm unavailable

@dataclass
class TrainConfig:
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    num_workers: int = 4
    prefetch_factor: int = 4
    persistent_workers: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "models/fer2013/cnn_small"
    # Early stopping parameters
    early_stop_patience: int = 6
    early_stop_delta: float = 1e-4
    channels_last: bool = True
    use_torch_compile: bool = False
    show_progress: bool = True  # << use tqdm bars


def train_model(model, train_ds, val_ds, class_names: List[str], cfg: TrainConfig):
    """
    Train a model with the given configuration and datasets.
    :param model: The PyTorch model to train.
    :param train_ds: Training dataset.
    :param val_ds: Validation dataset.
    :param class_names: List of class names for the dataset.
    :param cfg: Training configuration parameters.
    :return: None

    1. Initializes the model, optimizer, and loss function.
    2. Sets up data loaders for training and validation datasets.
    3. Computes class weights based on training dataset distribution.
    4. Trains the model for a specified number of epochs, with early stopping based on validation loss.
    5. Saves the best model checkpoint based on validation loss.
    6. Optionally uses PyTorch's channels_last memory format and compilation for performance.
    """
    device = torch.device(cfg.device)

    # Optional: channels_last memory format for better CPU throughput
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # Optional: compile model (PyTorch 2.x)
    if cfg.use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    model.to(device)

    pin_mem = (cfg.device == "cuda" and cfg.num_workers > 0)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    # class weights from train set
    import numpy as np
    ys = np.array([int(y) for _, y in train_ds])
    counts = np.bincount(ys, minlength=len(class_names)).astype("float32")
    weights = (counts.sum() / (counts + 1e-6))
    weights = weights / weights.mean()
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    crit  = torch.nn.CrossEntropyLoss(weight=weights_t)

    best_val = math.inf
    bad_epochs = 0
    overall_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()

        # ---- TRAIN ----
        model.train()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        train_iter = train_loader
        pbar_train = None
        if cfg.show_progress and tqdm is not None:
            pbar_train = tqdm(
                train_loader, total=len(train_loader),
                desc=f"Train {epoch}/{cfg.epochs}", dynamic_ncols=True, leave=False
            )
            train_iter = pbar_train

        for xb, yb in train_iter:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()

            total_loss += float(loss.detach())
            total_acc += accuracy(logits.detach(), yb)
            n_batches += 1

            if pbar_train and n_batches % 5 == 0:
                pbar_train.set_postfix(loss=total_loss / n_batches, acc=total_acc / n_batches)

        if pbar_train: pbar_train.close()

        train_loss = total_loss / max(1, n_batches)
        train_acc  = total_acc  / max(1, n_batches)

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 0

        val_iter = val_loader
        pbar_val = None
        if cfg.show_progress and tqdm is not None:
            pbar_val = tqdm(
                val_loader, total=len(val_loader),
                desc=f"Val   {epoch}/{cfg.epochs}", dynamic_ncols=True, leave=False
            )
            val_iter = pbar_val

        with torch.no_grad():
            for xb, yb in val_iter:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = crit(logits, yb)
                val_loss += float(loss)
                val_acc  += accuracy(logits, yb)
                n_val += 1

                if pbar_val and n_val % 5 == 0:
                    pbar_val.set_postfix(loss=val_loss / n_val, acc=val_acc / n_val)

        if pbar_val: pbar_val.close()

        val_loss /= max(1, n_val)
        val_acc  /= max(1, n_val)

        # ---- TIMING + EARLY STOP ----
        epoch_time = time.perf_counter() - epoch_start
        elapsed = time.perf_counter() - overall_start
        remaining = epoch_time * (cfg.epochs - epoch)

        # single clean summary line (doesn't mess with tqdm bars)
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train {train_loss:.4f}/{train_acc:.3f} | "
            f"val {val_loss:.4f}/{val_acc:.3f} | "
            f"time {format_time(epoch_time)} | ETA {format_time(remaining)} (elapsed {format_time(elapsed)})"
        )

        if val_loss + cfg.early_stop_delta < best_val:
            best_val = val_loss
            bad_epochs = 0
            save_checkpoint(
                f"{cfg.out_dir}/best.pt",
                {"model": model.state_dict(), "classes": class_names, "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}
            )
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.early_stop_patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    print("Training complete. Best checkpoint saved to", f"{cfg.out_dir}/best.pt")
