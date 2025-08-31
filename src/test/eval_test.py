# src/test/eval_test.py
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

# --- extra deps for media smoke tests
import cv2, numpy as np, torch.nn.functional as F

# --------- Config (no argparse) ---------
# Prefer the full test set; change to "test_private.csv" if you want
TEST_CSV = os.path.join(PROJECT_ROOT, "datasets", "test_private.csv")  # User can change this path

CKPT = os.path.join(PROJECT_ROOT, "models", "best.pt")  # User can change this path

BATCH_SIZE = 256
NUM_WORKERS = 0  # safe default on Windows; raise to 2–8 on Linux/mac if you like
USE_CLAHE = True
USE_CACHE = True  # if your dataset class supports it; ignored otherwise
MODEL_NAME = "cnn_small"

# Media (images + video) for quick smoke tests
MEDIA_DIR = os.path.join(HERE, "media")  # expected: src/test/media
CASCADE_PATH = os.path.join(PROJECT_ROOT, "assets", "haarcascade_frontalface_default.xml")


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
if TEST_CSV is None:
    raise FileNotFoundError(
        "Could not find test CSV. Expected one of: \n  - datasets/test_public.csv\n  - datasets/test_private.csv"
    )

if CKPT is None:
    raise FileNotFoundError(
        "Could not find checkpoint. Expected one of: \n"
        "  - models/best.pt\n  - checkpoints/best.pt\n  - models/fer2013/cnn_small/best.pt"
    )


# -------------------------
# Helpers for media smoke tests
# -------------------------
def _load_cascade():
    if os.path.exists(CASCADE_PATH):
        c = cv2.CascadeClassifier(CASCADE_PATH)
        if not c.empty():
            return c
    return None


def _detect_biggest_face(gray, cascade):
    if cascade is None:
        return None
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE)
    if len(faces) == 0:
        return None
    # biggest by area
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return (x, y, w, h)


def _preprocess_face_48(gray_or_bgr, use_clahe=True):
    if gray_or_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
    gray = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    if use_clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    arr = (gray.astype("float32") / 255.0)[None, None, :, :]  # [1,1,48,48]
    return torch.from_numpy(arr)


def _crop_or_center(gray_bgr, cascade):
    # Try to detect face; if none, center crop square
    if gray_bgr.ndim == 3:
        gray = cv2.cvtColor(gray_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_bgr
    box = _detect_biggest_face(gray, cascade)
    if box is None:
        h, w = gray.shape[:2]
        s = min(h, w)
        cy, cx = h // 2, w // 2
        y0, x0 = max(0, cy - s // 2), max(0, cx - s // 2)
        crop = gray[y0:y0 + s, x0:x0 + s]
        return crop
    x, y, w, h = box
    return gray[y:y + h, x:x + w]


def _run_image_smoke_tests(model, classes, device, limit=3):
    if not os.path.isdir(MEDIA_DIR):
        print(f"[eval][img] No media dir at {MEDIA_DIR}. Skipping image tests.")
        return
    imgs = [f for f in os.listdir(MEDIA_DIR) if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))]
    if not imgs:
        print(f"[eval][img] No images found in {MEDIA_DIR}. Skipping.")
        return
    imgs = imgs[:limit]

    cascade = _load_cascade()
    model.eval()
    print(f"[eval][img] Testing {len(imgs)} images from {MEDIA_DIR}")
    for name in imgs:
        path = os.path.join(MEDIA_DIR, name)
        im = cv2.imread(path)
        if im is None:
            print(f"  - {name}: unreadable")
            continue
        crop = _crop_or_center(im, cascade)
        xb = _preprocess_face_48(crop, use_clahe=True).to(device)
        with torch.inference_mode():
            logits = model(xb)
            prob = F.softmax(logits[0], dim=0).cpu().numpy()
        order = np.argsort(prob)[::-1]
        top1 = (classes[order[0]], float(prob[order[0]]))
        top3 = [(classes[i], float(prob[i])) for i in order[:3]]
        print(f"  - {name}: TOP1={top1[0]} ({top1[1]*100:.1f}%) | TOP3={[(c, f'{p*100:.1f}%') for c,p in top3]}")


def _run_video_majority_vote(model, classes, device, frame_stride=5, max_frames=600):
    if not os.path.isdir(MEDIA_DIR):
        print(f"[eval][video] No media dir at {MEDIA_DIR}. Skipping video test.")
        return
    vids = [f for f in os.listdir(MEDIA_DIR) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]
    if not vids:
        print(f"[eval][video] No videos found in {MEDIA_DIR}. Skipping.")
        return
    path = os.path.join(MEDIA_DIR, vids[0])
    cascade = _load_cascade()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[eval][video] Could not open {path}")
        return

    model.eval()
    counts = np.zeros(len(classes), dtype=np.int64)
    fidx = 0
    used = 0

    print(f"[eval][video] Scanning: {path} (stride={frame_stride})")
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok or used >= max_frames:
                break
            fidx += 1
            if fidx % frame_stride != 0:
                continue
            crop = _crop_or_center(frame, cascade)
            xb = _preprocess_face_48(crop, use_clahe=True).to(device)
            logits = model(xb)
            pred = int(logits.argmax(dim=1).item())
            counts[pred] += 1
            used += 1
    cap.release()

    if used == 0:
        print("[eval][video] No frames processed.")
        return

    majority = int(counts.argmax())
    majority_name = classes[majority]
    conf = counts[majority] / counts.sum()
    print(f"[eval][video] Majority emotion: {majority_name} ({conf*100:.1f}%) over {used} frames.")
    print("[eval][video] Distribution:", {classes[i]: int(counts[i]) for i in range(len(classes))})


# -------------------------
# Main test evaluation
# -------------------------

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

    # ---- NEW: quick media checks ----
    try:
        _run_image_smoke_tests(model, classes, DEVICE)
        _run_video_majority_vote(model, classes, DEVICE)
    except Exception as e:
        print(f"[eval][media] Skipped due to error: {e}")


if __name__ == "__main__":
    main()
