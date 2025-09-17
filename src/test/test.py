# src/test/test.py
import os, sys, time, csv, glob
from typing import List, Tuple, Dict
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext
import matplotlib.pyplot as plt

# --------- Path bootstrap so "src.*" works when run as a module or file ---------
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # <project>
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")  # <project>/src
for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# --------- Imports from your project ---------
from src.data.fer2013 import FER2013Class as FERDS
from src.data.fer2013 import DEFAULT_CLASSES
from src.models.factory import create_model
from src.utils.io import load_checkpoint

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # progress bar optional

# --------- Config (no argparse) ---------
# switch this to "public_test.csv" if you want that split
TEST_CSV = os.path.join(PROJECT_ROOT, "datasets", "test_private.csv")
SPLIT_NAME = os.path.splitext(os.path.basename(TEST_CSV))[0]
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
OUT_CMP_DIR = os.path.join(MODELS_ROOT, "test_results")
os.makedirs(OUT_CMP_DIR, exist_ok=True)

BATCH_SIZE = 256
NUM_WORKERS = 0  # safe default on Windows; try 2-4 on Linux/mac
USE_CLAHE = True
USE_CACHE = True

# --------- Device selection (CUDA -> CPU) ---------
def pick_device():
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            print(f"[eval] Using CUDA GPU: {name}")
        except Exception:
            print("[eval] Using CUDA GPU")
        return torch.device("cuda"), True
    print("[eval] Using CPU")
    return torch.device("cpu"), False

DEVICE, IS_CUDA = pick_device()
PIN_MEMORY = bool(IS_CUDA)

# ---------- ResNet adapter ----------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def _to_resnet_input(x: torch.Tensor, size: int) -> torch.Tensor:
    # (B,1,48,48)->(B,3,size,size), normalize to ImageNet stats
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return (x - _IMAGENET_MEAN.to(x.device)) / _IMAGENET_STD.to(x.device)

# ---------- utils ----------
def _looks_like_resnet(state_keys) -> bool:
    return any(str(k).startswith("layer1.0.conv1.weight") for k in state_keys) or ("fc.weight" in state_keys)

def _infer_model_name(ckpt_path: str, state_keys) -> str:
    if _looks_like_resnet(state_keys):
        return "resnet18"
    base = os.path.basename(os.path.dirname(ckpt_path)).lower()
    return "resnet18" if "resnet" in base else "cnn_small"

def _metrics_resnet_input(model_dir: str, default_size: int = 160) -> int:
    mpath = os.path.join(model_dir, "metrics.csv")
    if not os.path.exists(mpath):
        return default_size
    size = default_size
    try:
        with open(mpath, newline="") as f:
            r = csv.DictReader(f)
            last = None
            for row in r:
                last = row
            if last and "resnet_input" in last and str(last["resnet_input"]).strip():
                size = int(float(last["resnet_input"]))
    except Exception:
        pass
    return size

@torch.no_grad()
def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def _scan_checkpoints(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "**", "*.pt"), recursive=True))

# -------------- evaluation over DataLoader --------------
def _evaluate_model_on_loader(model: torch.nn.Module, loader: DataLoader, device, is_resnet: bool, resnet_size: int) -> Tuple[float, Dict[int, Tuple[int,int]]]:
    model.eval().to(device)
    if IS_CUDA:
        torch.backends.cudnn.benchmark = True
    total_correct = 0
    total_seen = 0
    num_classes = len(DEFAULT_CLASSES)
    per_cls_correct = np.zeros(num_classes, dtype=np.int64)
    per_cls_total   = np.zeros(num_classes, dtype=np.int64)

    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, total=len(loader), desc="Evaluating", leave=False, dynamic_ncols=True)

    amp_ctx = torch.amp.autocast('cuda', enabled=IS_CUDA) if IS_CUDA else nullcontext()

    with amp_ctx:
        for xb, yb in iterator:
            xb = xb.to(device, non_blocking=IS_CUDA)
            if is_resnet:
                xb = _to_resnet_input(xb, size=resnet_size)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            yb_cpu = (yb if isinstance(yb, torch.Tensor) else torch.tensor(yb)).cpu().numpy()

            total_correct += int((preds == yb_cpu).sum())
            total_seen += int(yb_cpu.size)

            for c in range(num_classes):
                mask = (yb_cpu == c)
                per_cls_total[c] += int(mask.sum())
                if per_cls_total[c] > 0:
                    per_cls_correct[c] += int((preds[mask] == c).sum())

            if tqdm is not None:
                iterator.set_postfix(acc=f"{(total_correct / max(1, total_seen)):.3f}")

    overall_acc = total_correct / max(1, total_seen)
    per_cls = {c: (int(per_cls_correct[c]), int(per_cls_total[c])) for c in range(num_classes)}
    return overall_acc, per_cls

# -------------------------
# Main test evaluation
# -------------------------
def main():
    # --------- Dataset / loader ---------
    if not os.path.exists(TEST_CSV):
        raise FileNotFoundError(f"Test CSV not found at {TEST_CSV}")
    ds = FERDS(TEST_CSV, classes=DEFAULT_CLASSES, augment=False, use_clahe=USE_CLAHE, cache=USE_CACHE)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # --------- Find checkpoints ---------
    ckpts = _scan_checkpoints(MODELS_ROOT)
    if not ckpts:
        print(f"[eval] No .pt files found under {MODELS_ROOT}.")
        return

    # --------- Evaluate all ---------
    results = []
    t_all0 = time.perf_counter()

    for ckpt in ckpts:
        # Load state
        try:
            state = load_checkpoint(ckpt)
        except Exception as e:
            print(f"[eval] Skipping '{ckpt}': cannot load ({e})")
            continue

        sd_keys = state.get("model", {}).keys()
        if not sd_keys:
            print(f"[eval] Skipping '{ckpt}': missing 'model' state.")
            continue

        # Model name + config recovery
        model_name = _infer_model_name(ckpt, sd_keys)
        model_dir = os.path.dirname(ckpt)
        resnet_size = _metrics_resnet_input(model_dir, default_size=160) if model_name.startswith("resnet") else 0
        classes = state.get("classes", DEFAULT_CLASSES)
        num_classes = len(classes)

        # Create model + load weights
        try:
            model = create_model(model_name, num_classes=num_classes)
            model.load_state_dict(state["model"])
        except Exception as e:
            print(f"[eval] Skipping '{ckpt}': model load failed ({e})")
            continue

        # Count params
        params = sum(p.numel() for p in model.parameters())
        is_resnet = model_name.startswith("resnet")

        # Evaluate
        t0 = time.perf_counter()
        acc, per_cls = _evaluate_model_on_loader(model, dl, DEVICE, is_resnet=is_resnet, resnet_size=resnet_size)
        elapsed = time.perf_counter() - t0

        print(f"[eval] {model_name:<10} | acc {acc:.3%} | size {resnet_size if is_resnet else 48} | "
              f"params {params/1e6:.2f}M | ckpt: {os.path.relpath(ckpt, PROJECT_ROOT)} | {elapsed:.1f}s")

        results.append(dict(
            model_name=model_name,
            ckpt=os.path.relpath(ckpt, PROJECT_ROOT),
            acc=acc,
            params=params,
            eval_time_s=elapsed,
            resnet_input=resnet_size if is_resnet else 48,
        ))

    total_elapsed = time.perf_counter() - t_all0
    if not results:
        print("[eval] No results to plot.")
        return

    # --------- Save CSV summary ---------
    out_csv = os.path.join(OUT_CMP_DIR, f"test_{SPLIT_NAME}_results.csv")
    results_sorted = sorted(results, key=lambda r: (-r["acc"], r["model_name"], r["ckpt"]))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        w.writeheader()
        w.writerows(results_sorted)
    print(f"[eval] Wrote: {out_csv}")

    # --------- Bar chart of overall accuracy ---------
    labels = [f"{r['model_name']} \n({os.path.basename(os.path.dirname(r['ckpt']))})" for r in results_sorted]
    accs   = [r["acc"] for r in results_sorted]

    plt.figure(figsize=(max(8, 1.2 * len(labels)), 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    bars = plt.bar(range(len(labels)), accs, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel(f"Accuracy ({SPLIT_NAME})")
    title_split = SPLIT_NAME.replace("_", " ").title()
    plt.title(f"Model Accuracy on FER2013 {title_split} Set")
    plt.ylim(0.0, 1.0)
    for b, a, c in zip(bars, accs, colors):
        plt.text(b.get_x() + b.get_width()/2, a + 0.01, f"{a*100:.1f}%", ha="center", va="bottom", fontsize=9, color=c)
    plt.grid(axis="y", alpha=0.25)
    out_png = os.path.join(OUT_CMP_DIR, f"test_{SPLIT_NAME}_overall.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[eval] Wrote: {out_png}")

    print(f"[eval] Done. Total time: {total_elapsed:.1f}s for {len(results_sorted)} model(s).")

if __name__ == "__main__":
    main()
