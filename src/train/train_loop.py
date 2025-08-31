# src/train/train_loop.py
import os, math, time, multiprocessing, csv
from typing import List, Tuple
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext

from ..utils.io import save_checkpoint, ensure_dir
from ..utils.time_measure import measure_time as format_time
from ..utils.seed import fix_seed
from ..data.fer2013 import FER2013Class, DEFAULT_CLASSES
from ..models.factory import create_model
from .config import TrainConfig

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


@torch.no_grad()
def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _setup_threads():
    try:
        torch.set_num_threads(max(1, multiprocessing.cpu_count() // 2))
    except Exception:
        pass
    os.environ.setdefault("OMP_NUM_THREADS", str(max(1, multiprocessing.cpu_count() // 2)))
    os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
    os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


# ---------- small helpers ----------
def _get_lr(optim: torch.optim.Optimizer) -> float:
    for g in optim.param_groups:
        return float(g.get("lr", 0.0))
    return 0.0


def _append_metrics_csv(path: str, row: dict):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _plot_curves(hist: dict, out_png: str):
    import matplotlib.pyplot as plt
    ensure_dir(os.path.dirname(out_png))
    ep = range(1, len(hist["train_loss"]) + 1)
    plt.figure(figsize=(9.5, 5))
    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(ep, hist["train_loss"], label="train loss")
    plt.plot(ep, hist["val_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.grid(True, alpha=0.25)
    plt.legend()
    # Val acc
    plt.subplot(1, 2, 2)
    plt.plot(ep, hist["val_acc"], label="val acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ---------- ResNet adapter ----------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to_resnet_input(x: torch.Tensor, size: int) -> torch.Tensor:
    # (B,1,48,48)->(B,3,size,size), normalize to ImageNet stats
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return (x - _IMAGENET_MEAN.to(x.device)) / _IMAGENET_STD.to(x.device)


def _freeze_resnet_backbone(m: torch.nn.Module, freeze: bool):
    # freeze all except final fc
    for name, p in m.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = not freeze


# ---------- Public entry ----------
def run_training(cfg: TrainConfig) -> Tuple[float, float]:
    fix_seed(cfg.seed)
    _setup_threads()
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Datasets / Loaders
    train_ds = FER2013Class(cfg.train_csv, classes=DEFAULT_CLASSES, augment=True,
                            use_clahe=cfg.use_clahe, cache=cfg.use_cache)
    val_ds = FER2013Class(cfg.val_csv, classes=DEFAULT_CLASSES, augment=False,
                          use_clahe=cfg.use_clahe, cache=cfg.use_cache)

    pin_mem = (cfg.device == "cuda" and cfg.num_workers > 0)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin_mem,
        persistent_workers=cfg.persistent_workers if cfg.num_workers > 0 else False,
        prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
    )

    # Model
    device = torch.device(cfg.device)
    is_resnet = "resnet" in cfg.model_name.lower()

    model = create_model(cfg.model_name, num_classes=len(DEFAULT_CLASSES))
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if cfg.use_torch_compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)  # type: ignore
        except Exception:
            pass
    model = model.to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # speedup

    # Loss / weights
    ys = np.array([int(y) for _, y in train_ds])
    counts = np.bincount(ys, minlength=len(DEFAULT_CLASSES)).astype("float32")
    counts = np.maximum(counts, 1.0)
    weights = (counts.sum() / counts)
    weights = weights / weights.mean()
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)
    crit = torch.nn.CrossEntropyLoss(weight=w_t)

    # Optimizer (handle optional warmup-freeze for ResNet)
    if is_resnet and cfg.resnet_freeze_backbone_epochs > 0:
        _freeze_resnet_backbone(model, True)
        opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                               lr=cfg.lr, weight_decay=cfg.weight_decay)
        frozen_now = True
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        frozen_now = False

    # AMP scaler (only helps on CUDA)
    use_amp = (device.type == "cuda") and bool(cfg.use_amp)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Output dirs/paths
    model_dir = os.path.join(cfg.out_dir, cfg.model_name)
    ensure_dir(model_dir)
    ckpt_path = os.path.join(model_dir, "best.pt")
    metrics_csv = os.path.join(model_dir, "metrics.csv")
    curves_png = os.path.join(model_dir, "curves.png")
    summary_txt = os.path.join(model_dir, "summary.txt")

    # Hist for plots
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [], "epoch_time_s": []}

    best_val = math.inf
    best_acc = 0.0
    best_epoch = 0
    bad_epochs = 0
    t0_all = time.perf_counter()

    try:
        for epoch in range(1, cfg.epochs + 1):
            # Unfreeze after warmup epochs for ResNet
            if is_resnet and frozen_now and epoch == (cfg.resnet_freeze_backbone_epochs + 1):
                _freeze_resnet_backbone(model, False)
                opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
                frozen_now = False

            # --------- TRAIN ---------
            model.train()
            sum_loss, sum_acc, n_batches = 0.0, 0.0, 0
            iterator = train_loader
            bar = None
            if cfg.show_progress and tqdm is not None:
                bar = tqdm(iterator, total=len(train_loader),
                           desc=f"Train {epoch}/{cfg.epochs}", dynamic_ncols=True, leave=False)
                iterator = bar

            last = time.perf_counter()
            for xb, yb in iterator:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                xb_in = _to_resnet_input(xb, size=cfg.resnet_input_size) if is_resnet else xb

                opt.zero_grad(set_to_none=True)
                amp_ctx = torch.amp.autocast('cuda', enabled=use_amp) if device.type == 'cuda' else nullcontext()
                with amp_ctx:
                    logits = model(xb_in)
                    loss = crit(logits, yb)

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                # stats
                sum_loss += loss.detach().item()
                sum_acc += accuracy(logits.detach(), yb)
                n_batches += 1

                if bar and n_batches % 5 == 0:
                    now = time.perf_counter()
                    dt = max(1e-9, now - last)
                    ips = xb.size(0) / dt
                    last = now
                    bar.set_postfix(loss=sum_loss / n_batches,
                                    acc=sum_acc / n_batches,
                                    ips=f"{ips:.0f}")

            if bar: bar.close()
            train_loss = sum_loss / max(1, n_batches)
            train_acc = sum_acc / max(1, n_batches)

            # --------- VALIDATION ---------
            model.eval()
            v_loss, v_acc, v_batches = 0.0, 0.0, 0
            iterator = val_loader
            bar = None
            if cfg.show_progress and tqdm is not None:
                bar = tqdm(iterator, total=len(val_loader),
                           desc=f"Val   {epoch}/{cfg.epochs}", dynamic_ncols=True, leave=False)
                iterator = bar

            with torch.no_grad():
                for xb, yb in iterator:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    xb_eval = _to_resnet_input(xb, size=cfg.resnet_input_size) if is_resnet else xb
                    logits = model(xb_eval)
                    loss = crit(logits, yb)
                    v_loss += float(loss)
                    v_acc += accuracy(logits, yb)
                    v_batches += 1
                    if bar and v_batches % 5 == 0:
                        bar.set_postfix(loss=v_loss / v_batches, acc=v_acc / v_batches)
            if bar: bar.close()

            val_loss = v_loss / max(1, v_batches)
            val_acc = v_acc / max(1, v_batches)

            # --------- LOG + SAVE ---------
            elapsed = time.perf_counter() - t0_all
            epoch_time = elapsed / epoch
            eta = epoch_time * (cfg.epochs - epoch)

            print(
                f"Epoch {epoch:02d}/{cfg.epochs} | "
                f"train {train_loss:.4f}/{train_acc:.3f} | "
                f"val {val_loss:.4f}/{val_acc:.3f} | "
                f"lr {_get_lr(opt):.1f} | "
                f"time {format_time(elapsed)} | ETA {format_time(eta)}"
            )

            # metrics.csv row
            row = dict(
                epoch=epoch,
                train_loss=round(train_loss, 6),
                train_acc=round(train_acc, 6),
                val_loss=round(val_loss, 6),
                val_acc=round(val_acc, 6),
                lr=_get_lr(opt),
                epoch_time_sec=round(epoch_time, 3),
                elapsed_sec=round(elapsed, 3),
                is_resnet=int(is_resnet),
                resnet_input=getattr(cfg, "resnet_input_size", 0),
                amp=int(use_amp),
                frozen_backbone=int(frozen_now),
            )
            _append_metrics_csv(metrics_csv, row)

            # keep in-memory history for curves.png
            hist["train_loss"].append(train_loss)
            hist["train_acc"].append(train_acc)
            hist["val_loss"].append(val_loss)
            hist["val_acc"].append(val_acc)
            hist["lr"].append(_get_lr(opt))
            hist["epoch_time_s"].append(epoch_time)

            # early stop + best
            improved = (val_loss + cfg.early_stop_delta < best_val)
            if improved:
                best_val = val_loss
                best_acc = val_acc
                best_epoch = epoch
                bad_epochs = 0
                save_checkpoint(
                    ckpt_path,
                    {"model": model.state_dict(), "classes": DEFAULT_CLASSES,
                     "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}
                )
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.early_stop_patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        # end for epochs
    except KeyboardInterrupt:
        print("\n[train] Interrupted — saving partial progress…")

    # Curves + summary
    _plot_curves(hist, curves_png)
    with open(summary_txt, "w") as f:
        f.write(
            f"best_epoch={best_epoch}\n"
            f"best_val_loss={best_val:.6f}\n"
            f"best_val_acc={best_acc:.6f}\n"
            f"ckpt={ckpt_path}\n"
        )

    print("Training complete. Best checkpoint:", ckpt_path)
    print("Metrics:", metrics_csv)
    print("Curves:", curves_png)
    print("Summary:", summary_txt)
    return float(best_acc), float("nan")
