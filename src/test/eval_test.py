# src/train/eval_test.py
import os, sys, time
import torch
from torch.utils.data import DataLoader

# --------- Path bootstrap so "src.*" works when run as a module or file ---------
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # <project>
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")  # <project>/src
for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

# --------- Imports from your project (with graceful fallbacks) ---------
try:
    from src.data.fer2013 import FER2013Class as FERDS  # if you have this name
except Exception:
    from src.data.fer2013 import FERCSV as FERDS  # our earlier class name

from src.data.fer2013 import DEFAULT_CLASSES
from src.models.factory import create_model
from src.utils.io import load_checkpoint

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None  # progress bar optional

# --------- Config (no argparse) ---------
# Prefer the full test set; change to "test_private.csv" if you want
CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "datasets", "test_private.csv"),
    os.path.join(PROJECT_ROOT, "datasets", "test_public.csv"),
]

CKPT_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "models", "best.pt"),
    os.path.join(PROJECT_ROOT, "checkpoints", "best.pt"),
]

BATCH_SIZE = 256
NUM_WORKERS = 0  # safe default on Windows; raise to 2–8 on Linux/mac if you like
USE_CLAHE = True
USE_CACHE = True  # if your dataset class supports it; ignored otherwise
MODEL_NAME = "cnn_small"


# --------- Device selection (CUDA -> DirectML -> CPU) ---------
def pick_device():
    # CUDA first
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            print(f"[eval] Using CUDA GPU: {name}")
        except Exception:
            print("[eval] Using CUDA GPU")
        return torch.device("cuda"), True, False
    # DirectML (AMD/Intel GPUs on Windows)
    try:
        import torch_directml  # type: ignore
        dml_device = torch_directml.device()
        print("[eval] Using DirectML device")
        return dml_device, False, True
    except Exception:
        pass
    print("[eval] Using CPU")
    return torch.device("cpu"), False, False


DEVICE, IS_CUDA, IS_DML = pick_device()
PIN_MEMORY = bool(IS_CUDA)  # pinning helps only for CUDA

# --------- Resolve files ---------
TEST_CSV = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
if TEST_CSV is None:
    raise FileNotFoundError(
        "Could not find test CSV. Expected one of: \n  - datasets/test_public.csv\n  - datasets/test_private.csv"
    )
CKPT = next((p for p in CKPT_CANDIDATES if os.path.exists(p)), None)
if CKPT is None:
    raise FileNotFoundError(
        "Could not find checkpoint. Expected one of: \n"
        "  - models/best.pt\n  - checkpoints/best.pt\n  - models/fer2013/cnn_small/best.pt"
    )


def main():
    # Dataset
    try:
        ds = FERDS(TEST_CSV, classes=DEFAULT_CLASSES, augment=False, use_clahe=USE_CLAHE, cache=USE_CACHE)
    except TypeError:
        # If your FER class doesn't support cache=
        ds = FERDS(TEST_CSV, classes=DEFAULT_CLASSES, augment=False, use_clahe=USE_CLAHE)

    n = len(ds)
    print(f"[eval] Test set: {TEST_CSV}  |  samples: {n:,}")

    dl = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    # Model
    state = load_checkpoint(CKPT)  # maps to CPU; we move to DEVICE below
    classes = state.get("classes", DEFAULT_CLASSES)
    model = create_model(MODEL_NAME, num_classes=len(classes))
    model.load_state_dict(state["model"])
    model.eval()
    model.to(DEVICE)

    # Optional CUDA inference tweaks
    if IS_CUDA:
        torch.backends.cudnn.benchmark = True

    # Metrics
    total_correct = 0
    total_seen = 0
    num_classes = len(classes)
    per_cls_correct = torch.zeros(num_classes, dtype=torch.long)
    per_cls_total = torch.zeros(num_classes, dtype=torch.long)

    start = time.perf_counter()
    iterator = dl
    if tqdm is not None:
        iterator = tqdm(dl, total=len(dl), desc="Evaluating", leave=False, dynamic_ncols=True)

    # Use inference_mode (faster than no_grad for eval)
    ctx_autocast = (
        torch.autocast(device_type="cuda", dtype=torch.float16) if IS_CUDA else
        torch.no_grad()
    )

    with torch.inference_mode():
        # nested autocast only for CUDA
        if IS_CUDA:
            cm = ctx_autocast
        else:
            cm = torch.no_grad()
        with cm:
            for xb, yb in iterator:
                xb = xb.to(DEVICE, non_blocking=IS_CUDA)  # yb kept on CPU for counting
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu()
                yb_cpu = yb if isinstance(yb, torch.Tensor) else torch.tensor(yb)

                # Global accuracy
                correct = (preds == yb_cpu).sum().item()
                total_correct += correct
                total_seen += yb_cpu.numel()

                # Per-class stats
                for c in range(num_classes):
                    mask = (yb_cpu == c)
                    per_cls_total[c] += int(mask.sum().item())
                    if per_cls_total[c] > 0:
                        per_cls_correct[c] += int((preds[mask] == c).sum().item())

                # Optional batchwise display
                if tqdm is not None:
                    iterator.set_postfix(acc=f"{(total_correct / max(1, total_seen)):.3f}")

    elapsed = time.perf_counter() - start
    overall_acc = total_correct / max(1, total_seen)

    print(f"[eval] Checkpoint: {CKPT}")
    print(f"[eval] Overall accuracy: {overall_acc:.3%}  |  samples: {total_seen:,}  |  time: {elapsed:.1f}s")

    # Per-class report
    print("[eval] Per-class accuracy:")
    for i, name in enumerate(classes):
        tot = int(per_cls_total[i].item())
        if tot == 0:
            pc_acc = 0.0
        else:
            pc_acc = per_cls_correct[i].item() / tot
        print(f"  - {name:<9} : {pc_acc:.3%}  (n={tot})")


if __name__ == "__main__":
    main()
