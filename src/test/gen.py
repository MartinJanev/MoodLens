# scripts/make_paper_assets.py
# Generates paper-ready assets (figures + tables) for the FER2013 project.
# Usage: python -m scripts.make_paper_assets
import os, sys, csv, time, math, random, shutil, json
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score


# ---------- Path bootstrap: find project root robustly ----------


def _find_project_root(start_dir: str, anchors=("src", "datasets", "README.md")) -> str:
    """
    Walk upward from start_dir until we find a folder that looks like the repo root,
    identified by the presence of common anchors (a 'src' dir, a 'datasets' dir, or README).
    """
    cur = os.path.abspath(start_dir)
    for _ in range(8):  # go up at most 8 levels
        has = [os.path.exists(os.path.join(cur, a)) for a in anchors]
        if any(has):
            return cur
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt
    return os.path.abspath(start_dir)  # fallback


# Where is this file? (works when run as a module too)
HERE = os.path.abspath(os.path.dirname(__file__))

# Allow override via env var if you ever need it
if "ML_PROJECT_ROOT" in os.environ:
    PROJECT_ROOT = os.path.abspath(os.environ["ML_PROJECT_ROOT"])
else:
    # If this file lives at project root, PROJECT_ROOT == HERE;
    # if it lives under scripts/ or src/test, we'll climb up.
    PROJECT_ROOT = _find_project_root(HERE)

SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"[assets] Project root: {PROJECT_ROOT}")

# ---------- Project imports (already in your repo) ----------
from src.data.fer2013 import FER2013Class as FERDS
from src.data.fer2013 import DEFAULT_CLASSES
from src.models.factory import create_model
from src.utils.io import load_checkpoint, ensure_dir

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from contextlib import nullcontext

# ---------- Defaults & destinations ----------
TRAIN_CSV = os.path.join(PROJECT_ROOT, "datasets", "train.csv")
TEST_PRIVATE_CSV = os.path.join(PROJECT_ROOT, "datasets", "test_private.csv")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")
OUT_DIR = os.path.join(PROJECT_ROOT, "paper_assets")
os.makedirs(OUT_DIR, exist_ok=True)

# plotting constants
DPI = 160
WIDTH_PX = 1700  # ≥1600px
IN_W = WIDTH_PX / DPI
RAND_SEED = 1337

# dataloader
BATCH_SIZE = 256
NUM_WORKERS = 0
USE_CLAHE = True
USE_CACHE = True

# ---------- ResNet adapter (matches your code path) ----------
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _to_resnet_input(x: torch.Tensor, size: int) -> torch.Tensor:
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    x = x.repeat(1, 3, 1, 1)
    return (x - _IMAGENET_MEAN.to(x.device)) / _IMAGENET_STD.to(x.device)


def _looks_like_resnet(state_keys) -> bool:
    return any(str(k).startswith("layer1.0.conv1.weight") for k in state_keys) or ("fc.weight" in state_keys)


def _infer_model_name(ckpt_path: str, sd_keys) -> str:
    if _looks_like_resnet(sd_keys): return "resnet18"
    base = os.path.basename(os.path.dirname(ckpt_path)).lower()
    return "resnet18" if "resnet" in base else "cnn_small"


def _metrics_resnet_input(model_dir: str, default_size: int = 160) -> int:
    mpath = os.path.join(model_dir, "metrics.csv")
    if not os.path.exists(mpath): return default_size
    try:
        last = None
        with open(mpath, newline="") as f:
            for row in csv.DictReader(f):
                last = row
        if last and "resnet_input" in last and str(last["resnet_input"]).strip():
            return int(float(last["resnet_input"]))
    except Exception:
        pass
    return default_size


# ---------- Helpers ----------
def _savefig(path: str, fig=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig = fig or plt.gcf()
    # Do NOT resize here; rely on the figure's own figsize
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ROBUST pixel parsing (handles irregular whitespace/rows)
def _read_pixels_48x48(row_pixels: str) -> np.ndarray:
    arr = np.fromstring(row_pixels, sep=" ", dtype=np.float32)
    if arr.size != 48 * 48:
        # fallback if numpy fails due to strange spacing
        arr = np.array(row_pixels.split(), dtype=np.float32)
    return arr.reshape(48, 48)


def _device_pick():
    if torch.cuda.is_available():
        try:
            print("[assets] Using CUDA:", torch.cuda.get_device_name(0))
        except Exception:
            print("[assets] Using CUDA")
        return torch.device("cuda"), True
    print("[assets] Using CPU")
    return torch.device("cpu"), False


DEVICE, HAS_CUDA = _device_pick()
PIN_MEMORY = bool(HAS_CUDA)


# ---- AMP context that works on Torch 1.x and 2.x ----
def _autocast_ctx(device_type: str):
    # Enable AMP only on CUDA; disable on CPU to avoid bfloat16 tensors
    if device_type == "cuda":
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):  # Torch 2.x
            return torch.amp.autocast(device_type="cuda")
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):  # Torch 1.x
            return torch.cuda.amp.autocast()
    return nullcontext()


# ============================================================
# A) FIGURES
# ============================================================

def fig_samples_per_class(train_csv: str, out_png: str, n_per_class: int = 4, use_clahe: bool = False):
    """Grid of example faces per label (rows=classes, cols=n_per_class)."""
    random.seed(RAND_SEED)
    np.random.seed(RAND_SEED)
    df = pd.read_csv(train_csv)
    cls_idx = list(range(len(DEFAULT_CLASSES)))
    rows, cols = len(cls_idx), n_per_class
    fig_h = max(5.5, rows * 2.2)
    plt.figure(figsize=(IN_W * 1.2, fig_h))
    for r, c_idx in enumerate(cls_idx):
        sub = df[df["emotion"] == c_idx]
        take = sub.sample(n=min(cols, len(sub)), random_state=RAND_SEED)
        for i, (_, rec) in enumerate(take.iterrows()):
            g = _read_pixels_48x48(rec["pixels"])
            if use_clahe:
                import cv2
                g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g.astype("uint8")).astype("float32")
            ax = plt.subplot(rows, cols, r * cols + (i + 1))
            ax.imshow(g, cmap="gray", vmin=0, vmax=255)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(DEFAULT_CLASSES[r], fontsize=20, pad=2)
    plt.suptitle("Dataset samples per class", y=1.02, fontsize=25)
    plt.tight_layout()
    _savefig(out_png)


def fig_pipeline_diagram(out_path: str):
    """Simple block diagram: Input → Preprocess → Model → Postprocess → Output label."""
    from matplotlib.patches import FancyBboxPatch
    plt.figure(figsize=(IN_W, 2.2))
    ax = plt.gca()
    ax.set_axis_off()

    boxes = [
        ("Input", "face image"),
        ("Preprocess", "grayscale, CLAHE, resize"),
        ("Model", "CNN (48×48) or ResNet-18 (160×160)"),
        ("Postprocess", "softmax, EMA smoothing"),
        ("Output", "label (7 classes)"),
    ]
    x0, y0, w, h = 0.05, 0.35, 0.17, 0.35
    gap = 0.03

    centers = []
    for i, (title, desc) in enumerate(boxes):
        x = x0 + i * (w + gap)
        centers.append((x + w / 2, y0 + h / 2))
        bb = FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                            linewidth=1.2, edgecolor="black", facecolor="#f2f2f2")
        ax.add_patch(bb)
        ax.text(x + w / 2, y0 + h * 0.60, title, ha="center", va="center", fontsize=12, weight="bold")
        ax.text(x + w / 2, y0 + h * 0.33, desc, ha="center", va="center", fontsize=10)
    # arrows
    for (x1, y1), (x2, y2) in zip(centers[:-1], centers[1:]):
        ax.annotate("", xy=(x2 - w / 2 + 0.005, y2), xytext=(x1 + w / 2 - 0.005, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4))
    _savefig(out_path)


def _scan_all_ckpts(root: str) -> List[str]:
    out = []
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f.endswith(".pt"): out.append(os.path.join(dirpath, f))
    return sorted(out)


def _build_private_loader(csv_path: str) -> DataLoader:
    ds = FERDS(csv_path, classes=DEFAULT_CLASSES, augment=False, use_clahe=USE_CLAHE, cache=USE_CACHE)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)


def _predict_on_private(ckpt_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Returns (y_true, y_pred, probs[N,C], meta)."""
    state = load_checkpoint(ckpt_path)
    sd_keys = state.get("model", {}).keys()
    if not sd_keys: raise RuntimeError(f"Missing 'model' state in {ckpt_path}")
    model_name = _infer_model_name(ckpt_path, sd_keys)
    is_resnet = model_name.startswith("resnet")
    resnet_size = _metrics_resnet_input(os.path.dirname(ckpt_path), default_size=160) if is_resnet else 0

    model = create_model(model_name, num_classes=len(DEFAULT_CLASSES))
    model.load_state_dict(state["model"])
    model.eval().to(DEVICE)
    if DEVICE.type == "cuda": torch.backends.cudnn.benchmark = True

    loader = _build_private_loader(TEST_PRIVATE_CSV)
    true, pred = [], []
    probs_all = []

    # AMP-safe autocast (Torch 1.x and 2.x)
    amp_ctx = _autocast_ctx(DEVICE.type)
    with torch.no_grad(), amp_ctx:
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            if is_resnet: xb = _to_resnet_input(xb, size=resnet_size)
            logits = model(xb)
            p = F.softmax(logits, dim=1)
            probs_all.append(p.detach().float().cpu().numpy())
            pred.append(p.argmax(dim=1).cpu().numpy())
            true.append(yb.cpu().numpy())
    y_true = np.concatenate(true, axis=0)
    y_pred = np.concatenate(pred, axis=0)
    probs = np.concatenate(probs_all, axis=0)
    meta = dict(model_name=model_name, resnet_size=resnet_size, classes=DEFAULT_CLASSES)
    return y_true, y_pred, probs, meta


def fig_confusion_private(ckpt_path: str, out_png: str, out_pred_csv: str, out_probs_npz: str,
                          out_table1_csv: str):
    """Compute predictions on PrivateTest, plot normalized confusion, export raw preds + Table 1 CSV."""
    print(f"[assets] Predicting PrivateTest with: {ckpt_path}")
    y_true, y_pred, probs, meta = _predict_on_private(ckpt_path)

    # save raw artifacts (C)
    dfp = pd.DataFrame({"y_true": y_true.astype(int), "y_pred": y_pred.astype(int)})
    dfp.to_csv(out_pred_csv, index=False)
    np.savez_compressed(out_probs_npz, y_true=y_true, probs=probs)

    # confusion
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(DEFAULT_CLASSES))))
    cm_norm = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(IN_W, IN_W * 0.62))
    im = plt.imshow(cm_norm, cmap="Blues", vmin=0.05, vmax=1.0)
    plt.colorbar(im, fraction=0.046, pad=0.03)
    plt.xticks(range(len(DEFAULT_CLASSES)), DEFAULT_CLASSES, rotation=35, ha="right")
    plt.yticks(range(len(DEFAULT_CLASSES)), DEFAULT_CLASSES)
    plt.title("Confusion Matrix — PrivateTest (normalized)")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            v = cm_norm[i, j]
            if v >= 0.005:
                plt.text(j, i, f"{v * 100:.1f}", ha="center", va="center", fontsize=9,
                         color=("white" if v > 0.5 else "black"))
    plt.xlabel("Predicted");
    plt.ylabel("True")
    plt.tight_layout()
    _savefig(out_png)

    # Table 1 — per-class accuracy + overall, including class sizes
    class_sizes = cm.sum(axis=1).tolist()
    per_class_acc = (np.diag(cm) / np.maximum(cm.sum(axis=1), 1)).tolist()
    overall = float((y_pred == y_true).mean())
    table = pd.DataFrame({
        "class": DEFAULT_CLASSES,
        "size": class_sizes,
        "accuracy": [round(100.0 * a, 2) for a in per_class_acc],
    })
    table = pd.concat(
        [table, pd.DataFrame([{"class": "Overall", "size": sum(class_sizes), "accuracy": round(100.0 * overall, 2)}])],
        ignore_index=True)
    table.to_csv(out_table1_csv, index=False)
    print(f"[assets] Wrote Table 1 CSV → {out_table1_csv}")


def fig_class_distribution(train_csv: str, out_png: str):
    df = pd.read_csv(train_csv)
    counts = df["emotion"].value_counts().reindex(range(len(DEFAULT_CLASSES)), fill_value=0)
    labels = DEFAULT_CLASSES
    plt.figure(figsize=(IN_W, 4.5))
    bars = plt.bar(range(len(labels)), counts.values)
    plt.xticks(range(len(labels)), labels, rotation=15, ha="right")
    for b, c in zip(bars, counts.values):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height() + max(1, counts.max() * 0.01), str(int(c)),
                 ha="center", va="bottom", fontsize=9)
    plt.title("Class distribution — Training split")
    plt.ylabel("count")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    _savefig(out_png)


def _load_metrics_csv(model_dir: str) -> Optional[pd.DataFrame]:
    p = os.path.join(model_dir, "metrics.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        df = df.sort_values("epoch").drop_duplicates("epoch", keep="last")
        return df
    return None


def fig_train_and_val_curves(model_dir: str, out_train_png: str, out_val_png: str):
    """Reads models/<name>/metrics.csv and emits separate train/val figures (loss & acc)."""
    df = _load_metrics_csv(model_dir)
    if df is None:
        print(f"[assets] Missing metrics.csv in {model_dir} — skipping train/val curves.")
        return
    # Train curves
    plt.figure(figsize=(IN_W, 4.8))
    ax1 = plt.gca()
    ax1.plot(df["epoch"], df.get("train_loss", pd.Series([np.nan] * len(df))), label="train loss")
    ax1.set_xlabel("epoch");
    ax1.set_ylabel("loss");
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df.get("train_acc", pd.Series([np.nan] * len(df))), label="train acc", linestyle="--")
    ax2.set_ylabel("accuracy")
    plt.title("Training — Loss & Accuracy")
    _savefig(out_train_png)

    # Val curves
    plt.figure(figsize=(IN_W, 4.8))
    ax1 = plt.gca()
    ax1.plot(df["epoch"], df.get("val_loss", pd.Series([np.nan] * len(df))), label="val loss")
    ax1.set_xlabel("epoch");
    ax1.set_ylabel("loss");
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df.get("val_acc", pd.Series([np.nan] * len(df))), label="val acc", linestyle="--")
    ax2.set_ylabel("accuracy")
    plt.title("Validation — Loss & Accuracy")
    _savefig(out_val_png)


def fig_accuracy_comparison(models_root: str, out_png: str):
    """If models/compare/test_private_results.csv exists, plots it. Otherwise scans .pt files and computes."""
    compare_dir = os.path.join(models_root, "compare")
    prebuilt = os.path.join(compare_dir, "test_private_results.csv")
    if os.path.exists(prebuilt):
        df = pd.read_csv(prebuilt)
        labels = [f"{r.model_name}\n({os.path.basename(os.path.dirname(r.ckpt))})" for r in df.itertuples()]
        accs = df["acc"].values
    else:
        # lightweight: evaluate each checkpoint once (overall acc only)
        ckpts = _scan_all_ckpts(models_root)
        if not ckpts:
            print("[assets] No checkpoints found — skipping accuracy comparison.")
            return
        vals = []
        for ck in ckpts:
            y_true, y_pred, _, meta = _predict_on_private(ck)
            vals.append((f"{meta['model_name']}\n({os.path.basename(os.path.dirname(ck))})",
                         (y_true == y_pred).mean()))
        vals.sort(key=lambda t: -t[1])
        labels = [a for a, _ in vals]
        accs = np.array([b for _, b in vals], dtype=float)

    plt.figure(figsize=(max(IN_W, 0.9 * len(labels)), 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(labels)))
    bars = plt.bar(range(len(labels)), accs, color=colors)
    plt.xticks(range(len(labels)), labels, rotation=20, ha="right")
    plt.ylabel("Accuracy (PrivateTest)")
    plt.title("Model accuracy comparison")
    plt.ylim(0.0, 1.0)
    for b, a, c in zip(bars, accs, colors):
        plt.text(b.get_x() + b.get_width() / 2, a + 0.01, f"{a * 100:.1f}%", ha="center", va="bottom", fontsize=9,
                 color=c)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    _savefig(out_png)


# ============================================================
# B) TABLES
# ============================================================

import glob


def _csv_items_from_path(path_str: str):
    """Read CSV robustly (BOM, case-insensitive headers, header synonyms)."""
    if not os.path.exists(path_str):
        print(f"[assets][WARN] variants CSV not found: {path_str}")
        return []
    # Handle BOM on Windows
    df = pd.read_csv(path_str, encoding="utf-8-sig")
    # Normalize headers
    df.columns = [str(c).strip().lower() for c in df.columns]
    # Accept synonyms
    if "ckpt" not in df.columns:
        for alt in ("checkpoint", "weights", "path", "model", "checkpoint_path"):
            if alt in df.columns:
                df["ckpt"] = df[alt]
                break
    if "variant" not in df.columns:
        # Make a best-effort variant name from filename
        df["variant"] = df.get("name", df.get("model", df.get("ckpt", "variant")))

    if "ckpt" not in df.columns:
        print("[assets][ERROR] CSV must contain a 'ckpt' column (or a synonym like 'checkpoint').")
        return []

    if "notes" not in df.columns:
        df["notes"] = ""

    recs = df[["variant", "ckpt", "notes"]].to_dict("records")
    return recs


def _resolve_ckpt(ck: str) -> Optional[str]:
    """
    Resolve a checkpoint spec to an actual .pt file:
      - expand ~ and %VAR%
      - if dir → pick newest .pt in it
      - if contains wildcard → glob and pick newest .pt
      - if relative → try relative to PROJECT_ROOT
    """
    if not ck:
        return None
    ck = os.path.expandvars(os.path.expanduser(str(ck)))
    # Allow glob patterns
    if any(ch in ck for ch in ["*", "?", "["]):
        matches = sorted(glob.glob(ck, recursive=True), key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0)
        matches = [m for m in matches if m.lower().endswith(".pt")]
        if matches:
            return matches[-1]
    # If it's a directory, pick newest .pt inside
    if os.path.isdir(ck):
        pts = []
        for dp, _, files in os.walk(ck):
            for f in files:
                if f.lower().endswith(".pt"):
                    full = os.path.join(dp, f)
                    pts.append((os.path.getmtime(full), full))
        if pts:
            pts.sort()
            return pts[-1][1]
    # If it's a file and exists
    if os.path.isfile(ck):
        return ck
    # Try relative to project root
    rel = os.path.join(PROJECT_ROOT, ck)
    if os.path.isfile(rel):
        return rel
    print(f"[assets][WARN] checkpoint not found: {ck}")
    return None


def table2_ablation(variants_csv_or_list, out_csv: str):
    """
    Builds Table 2 with columns: variant, overall_acc, macro_f1, latency_ms, notes.
    - variants_csv_or_list can be:
        * path to CSV with columns [variant, ckpt, notes]  OR
        * list of dicts: [{"variant": "...", "ckpt": "...", "notes": "..."}]
    """
    # 1) Ingest items
    if isinstance(variants_csv_or_list, str):
        items = _csv_items_from_path(variants_csv_or_list)
    else:
        items = variants_csv_or_list or []
        # Normalize programmatic input, just in case
        normd = []
        for it in items:
            d = {k.strip().lower(): v for k, v in it.items()}
            if "ckpt" not in d:
                for alt in ("checkpoint", "weights", "path", "model", "checkpoint_path"):
                    if alt in d:
                        d["ckpt"] = d[alt]
                        break
            d.setdefault("variant", d.get("name", d.get("ckpt", "variant")))
            d.setdefault("notes", "")
            normd.append({"variant": d["variant"], "ckpt": d.get("ckpt"), "notes": d["notes"]})
        items = normd

    if not items:
        # No items at all → write template
        tmpl = pd.DataFrame([
            dict(variant="CNN-small + CLAHE", overall_acc="", macro_f1="", latency_ms="", notes="48x48, Adam 1e-3"),
            dict(variant="ResNet18 @160 + EMA", overall_acc="", macro_f1="", latency_ms="", notes="EMA=0.8, warmup=3"),
        ])
        tmpl.to_csv(out_csv, index=False)
        print(f"[assets] No variants provided — wrote a template at {out_csv}")
        return

    rows = []
    for it in items:
        ck_spec = it["ckpt"]
        ck = _resolve_ckpt(ck_spec)
        if not ck:
            print(f"[assets][SKIP] unresolved ckpt: {ck_spec}")
            continue

        # 2) Predictions on PrivateTest
        y_true, y_pred, probs, _ = _predict_on_private(ck)

        # 3) Metrics (robust)
        macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        overall = float((y_pred == y_true).mean())

        # 4) Latency (ms/frame) on a single batch
        loader = _build_private_loader(TEST_PRIVATE_CSV)
        xb, _ = next(iter(loader))
        xb = xb.to(DEVICE, non_blocking=True)

        state = load_checkpoint(ck)
        model_name = _infer_model_name(ck, state["model"].keys())
        model = create_model(model_name, num_classes=len(DEFAULT_CLASSES))
        model.load_state_dict(state["model"])
        model.eval().to(DEVICE)

        with torch.no_grad():
            if model_name.startswith("resnet"):
                xb = _to_resnet_input(xb, size=_metrics_resnet_input(os.path.dirname(ck), default_size=160))
            # warmup
            for _ in range(3):
                _ = model(xb)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(xb)
            if DEVICE.type == "cuda": torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        ms_per = (dt / xb.size(0)) * 1000.0

        row = dict(
            variant=it.get("variant", os.path.basename(os.path.dirname(ck))),
            overall_acc=round(100.0 * overall, 2),
            macro_f1=round(100.0 * float(macro), 2),
            latency_ms=round(float(ms_per), 2),
            notes=it.get("notes", "")
        )
        rows.append(row)
        print(
            f"[assets] {row['variant']}: acc={row['overall_acc']}%  macroF1={row['macro_f1']}%  latency={row['latency_ms']} ms")

    if not rows:
        # All rows were skipped due to unresolved ckpts → write a diagnostic template
        pd.DataFrame([{
            "variant": "NO VALID ROWS — check your ckpt paths",
            "overall_acc": "",
            "macro_f1": "",
            "latency_ms": "",
            "notes": "CSV headers must include 'ckpt'. Paths may be relative to project root; dirs/wildcards allowed."
        }]).to_csv(out_csv, index=False)
        print(f"[assets][ERROR] All rows skipped — wrote a diagnostic template at {out_csv}")
        return

    pd.DataFrame(rows, columns=["variant", "overall_acc", "macro_f1", "latency_ms", "notes"]).to_csv(out_csv,
                                                                                                     index=False)
    print(f"[assets] Wrote Table 2 CSV → {out_csv}")


# ============================================================
# C) RAW ARTIFACTS (already produced in fig_confusion_private)
# ============================================================

# ============================================================
# D) TEXT DETAILS TEMPLATE (you can paste into the paper)
# ============================================================

DETAILS_TEMPLATE = """# Training & Inference Details (paste bullets into paper)

## Training setup (per model)
- epochs: …
- early-stopping: (rule, patience, metric): …
- batch size: …
- optimizer: Adam (base LR …), weight decay …
- scheduler: … (milestones: …)
- input sizes: CNN=48×48; ResNet=160×160 (confirm)
- class weights: (formula or list) …
- seeds / #runs averaged: …

## Augmentation
- horizontal flip p=…
- rotation ±…°
- translation/zoom: …
- CLAHE: clip limit …, tile grid …
- normalization: grayscale to [0,1] (CNN) / ImageNet stats (ResNet)

## Hardware & timing
- CPU/GPU, VRAM/RAM: …
- typical epoch time: …
- total train time: …

## Real-time demo
- face detector interval: every N=…
- EMA coefficient: …
- smoothing/window: …
- average FPS: …
"""


def write_details_template(path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(DETAILS_TEMPLATE)
    print(f"[assets] Wrote details template → {path}")


# ============================================================
# E) MAP/REBUILD CURVE PLOTS TO YOUR FILENAMES
#    Fig. 3 → train_curves.png
#    Fig. 4 → val_curves.png
#    Fig. 6 → accuracy_comparison.png
# ============================================================

def build_all_assets():
    # A) Required figures
    fig_samples_per_class(TRAIN_CSV, os.path.join(OUT_DIR, "samples_per_class.png"), n_per_class=4, use_clahe=False)
    fig_pipeline_diagram(os.path.join(OUT_DIR, "pipeline_diagram.png"))

    # choose a checkpoint for confusion matrix: prefer newest under models/
    ckpts = _scan_all_ckpts(MODELS_ROOT)
    # Find best cnn_small and best resnet18 checkpoints by modification time
    cnn_ckpts = [ck for ck in ckpts if "cnn_small" in ck.lower()]
    resnet_ckpts = [ck for ck in ckpts if "resnet" in ck.lower()]

    if cnn_ckpts:
        best_cnn = max(cnn_ckpts, key=os.path.getmtime)
        fig_confusion_private(
            ckpt_path=best_cnn,
            out_png=os.path.join(OUT_DIR, "confusion_matrix_private_cnn_small.png"),
            out_pred_csv=os.path.join(OUT_DIR, "predictions_private_cnn_small.csv"),
            out_probs_npz=os.path.join(OUT_DIR, "predictions_private_probs_cnn_small.npz"),
            out_table1_csv=os.path.join(OUT_DIR, "table1_private_per_class_accuracy_cnn_small.csv"),
        )
    if resnet_ckpts:
        best_resnet = max(resnet_ckpts, key=os.path.getmtime)
        fig_confusion_private(
            ckpt_path=best_resnet,
            out_png=os.path.join(OUT_DIR, "confusion_matrix_private_resnet18.png"),
            out_pred_csv=os.path.join(OUT_DIR, "predictions_private_resnet18.csv"),
            out_probs_npz=os.path.join(OUT_DIR, "predictions_private_probs_resnet18.npz"),
            out_table1_csv=os.path.join(OUT_DIR, "table1_private_per_class_accuracy_resnet18.csv"),
        )
    else:
        print("[assets] No checkpoints found — skipping confusion matrix & Table 1.")

    # Optional but nice
    if os.path.exists(TRAIN_CSV):
        fig_class_distribution(TRAIN_CSV, os.path.join(OUT_DIR, "class_distribution.png"))

    # Map/build the three existing plots
    # We’ll rebuild train/val curves from each model dir containing metrics.csv, prefer cnn_small then resnet18
    for model_name in ["cnn_small", "resnet18"]:
        model_dir = os.path.join(MODELS_ROOT, model_name)
        if os.path.isdir(model_dir) and os.path.exists(os.path.join(model_dir, "metrics.csv")):
            fig_train_and_val_curves(
                model_dir,
                out_train_png=os.path.join(OUT_DIR, "train_curves.png"),
                out_val_png=os.path.join(OUT_DIR, "val_curves.png"),
            )
            break
    # Accuracy comparison (Fig. 6)
    fig_accuracy_comparison(MODELS_ROOT, os.path.join(OUT_DIR, "accuracy_comparison.png"))

    # B) Table 2 — if you have a variants file, compute; else drop a template
    variants_csv = os.path.join(MODELS_ROOT, "compare", "variants.csv")  # optional file you can create
    table2_ablation(variants_csv if os.path.exists(variants_csv) else [],
                    os.path.join(OUT_DIR, "table2_ablation.csv"))

    # D) helpful bullets to paste
    write_details_template(os.path.join(OUT_DIR, "details_template.md"))


if __name__ == "__main__":
    build_all_assets()
    print(f"\nDone. Paper assets are in: {OUT_DIR}\n"
          f"- Fig. 1  samples_per_class.png\n"
          f"- Fig. 2  pipeline_diagram.png\n"
          f"- Fig. 3  train_curves.png (rebuilt)\n"
          f"- Fig. 4  val_curves.png (rebuilt)\n"
          f"- Fig. 5  confusion_matrix_private.png (+ predictions_private.csv, predictions_private_probs.npz)\n"
          f"- Fig. 6  accuracy_comparison.png\n"
          f"- (opt)   class_distribution.png\n"
          f"- Table 1 table1_private_per_class_accuracy.csv\n"
          f"- Table 2 table2_ablation.csv\n"
          f"- Notes   details_template.md\n")
