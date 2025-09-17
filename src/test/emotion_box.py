# demo_all_models.py — run all models, draw face box, show prediction per model (root-fix + auto-discover v3)
import os, sys, glob, cv2, numpy as np, torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Robust project root detection (works from src/test/**, scripts/**, etc.)
# -----------------------------------------------------------------------------
HERE = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()


def _find_project_root(start: str, max_up: int = 8) -> str:
    cur = os.path.abspath(start)
    for _ in range(max_up):
        has_models = os.path.isdir(os.path.join(cur, "models"))
        has_src = os.path.isdir(os.path.join(cur, "src"))
        if has_models or has_src:
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(start)


# Allow overrides via env vars (handy during debugging)
ENV_PROJECT_ROOT = os.environ.get("PROJECT_ROOT")
PROJECT_ROOT = os.path.abspath(ENV_PROJECT_ROOT) if ENV_PROJECT_ROOT else _find_project_root(HERE)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
MODELS_DIR = os.environ.get("MODELS_DIR") or os.path.join(PROJECT_ROOT, "..", "models")

# Ensure project + src are importable
for p in (PROJECT_ROOT, SRC_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"[root] HERE={HERE}")
print(f"[root] PROJECT_ROOT={PROJECT_ROOT}")
print(f"[root] MODELS_DIR={MODELS_DIR}")

# --------- Project imports (adjust if your paths differ) ---------
from src.models.factory import create_model
from src.utils.io import load_checkpoint

# Fallback classes; may be replaced by checkpoint['classes'] when present
DEFAULT_CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ImageNet stats for ResNet-like models
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# -----------------------------------------------------------------------------
# Face detection + preprocessing
# -----------------------------------------------------------------------------

def _build_haar():
    hp = getattr(cv2.data, "haarcascades", "")
    cascade = os.path.join(hp, "haarcascade_frontalface_default.xml")
    if not os.path.exists(cascade):
        raise FileNotFoundError("Could not find haarcascade_frontalface_default.xml.")
    return cv2.CascadeClassifier(cascade)


def _detect_largest_face(gray: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    faces = HAAR.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32))
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    # add ~12% padding around the box (clamped to image size)
    pad = int(0.12 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(gray.shape[1], x + w + pad)
    y1 = min(gray.shape[0], y + h + pad)
    return (x0, y0, x1 - x0, y1 - y0)


def _clahe_gray(gray):
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def _to_cnn48(gray_face, size: int) -> torch.Tensor:
    # (H,W) uint8 -> (1,1,H,W) float in [0,1]
    g = cv2.resize(gray_face, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(g).float().div(255.0).unsqueeze(0).unsqueeze(0)
    return t


def _to_resnet(gray_face, size: int) -> torch.Tensor:
    # (H,W) uint8 -> (1,3,H,W) normalized to ImageNet
    g = cv2.resize(gray_face, (size, size), interpolation=cv2.INTER_AREA).astype("float32") / 255.0
    g3 = np.repeat(g[None, :, :], 3, axis=0)  # (3,H,W)
    t = torch.from_numpy(g3).unsqueeze(0)  # (1,3,H,W)
    return (t - _IMAGENET_MEAN) / _IMAGENET_STD


def _prepare_face_once(path: str, max_height: int = 600):
    """Read + (optionally) resize image, apply CLAHE, detect largest face once.
    Returns (bgr, gray, box, face_gray) or (None, None, None, None) if no face.
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    h, w = bgr.shape[:2]
    if h > max_height:
        scale = max_height / h
        bgr = cv2.resize(bgr, (int(w * scale), max_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = _clahe_gray(gray)
    box = _detect_largest_face(gray)
    if box is None:
        return None, None, None, None
    x, y, bw, bh = box
    face_gray = gray[y:y + bh, x:x + bw].copy()
    return bgr, gray, box, face_gray


# -----------------------------------------------------------------------------
# Checkpoint discovery + introspection
# -----------------------------------------------------------------------------

def _infer_model_key_from_state_dict(sd: Dict[str, Any]) -> str:
    """Heuristic: detect resnet vs small cnn by param name patterns.
    Prefers 'layer1.' as a strong ResNet signal.
    """
    if not isinstance(sd, dict):
        return "emotion_cnn"
    names = list(sd.keys())
    if any(n.startswith("layer1.") for n in names):
        return "resnet18"
    return "emotion_cnn"


def _first_conv_in_channels(sd: Dict[str, Any]) -> Optional[int]:
    """Try to read the first conv weight and return its input channels (1 or 3)."""
    if not isinstance(sd, dict):
        return None
    for k in ("conv1.weight", "features.0.weight", "features.conv1.weight"):
        w = sd.get(k)
        if isinstance(w, torch.Tensor) and w.ndim == 4:
            return int(w.shape[1])
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and v.ndim == 4:
            return int(v.shape[1])
    return None


def _extract_in_size(state: Dict[str, Any]) -> Optional[int]:
    """Pull input size from common metadata fields if present."""
    for k in ("in_size", "input_size", "img_size", "image_size"):
        if k in state:
            val = state[k]
            if isinstance(val, (list, tuple)) and len(val) >= 1:
                return int(val[0])
            if isinstance(val, int):
                return int(val)
    for container in ("config", "hparams", "meta"):
        if container in state and isinstance(state[container], dict):
            sub = state[container]
            for k in ("in_size", "input_size", "img_size", "image_size"):
                if k in sub:
                    v = sub[k]
                    if isinstance(v, (list, tuple)) and len(v) >= 1:
                        return int(v[0])
                    if isinstance(v, int):
                        return int(v)
    return None


def _key_hint_from_dir(ckpt_path: str) -> Optional[str]:
    """Use parent directory names as hints (e.g., resnet18_25 -> resnet18, cnn_small_* -> emotion_cnn)."""
    d = os.path.basename(os.path.dirname(ckpt_path)).lower()
    if d.startswith("resnet"):
        return "resnet18"
    if d.startswith("cnn_small") or d.startswith("cnn"):
        return "emotion_cnn"
    return None


def _discover_model_specs() -> List[Dict[str, Any]]:
    """Find all checkpoints under MODELS_DIR/** and build specs automatically."""
    roots = [MODELS_DIR]
    # Also consider a sibling models/ next to HERE (in case user runs from nested folder)
    local_models = os.path.abspath(os.path.join(HERE, "models"))
    if os.path.isdir(local_models) and local_models not in roots:
        roots.append(local_models)

    patterns = ["*.pt", "*.pth", "*.ckpt"]
    ckpts = set()
    for root in roots:
        for ext in patterns:
            pat = os.path.join(root, "**", ext)
            for p in glob.glob(pat, recursive=True):
                if os.path.isfile(p):
                    ckpts.add(os.path.normpath(p))
    ckpts = sorted(ckpts)

    if not ckpts:
        tried = " | ".join(roots)
        raise FileNotFoundError(f"No checkpoints found under: {tried} (looking for *.pt, *.pth, *.ckpt)")

    specs: List[Dict[str, Any]] = []
    for ck in ckpts:
        try:
            state = load_checkpoint(ck)
        except Exception as e:
            print(f"[warn] Skipping {ck}: cannot load ({e})")
            continue

        classes = state.get("classes", DEFAULT_CLASSES)
        sd = state.get("state_dict") or state.get("model") or state

        # Decide factory key: directory hint -> metadata -> state_dict heuristic
        model_key = _key_hint_from_dir(ck) or state.get("model_key") or state.get("arch") or state.get(
            "backbone") or _infer_model_key_from_state_dict(sd)

        in_ch = _first_conv_in_channels(sd) or 1
        in_size = _extract_in_size(state) or (160 if (str(model_key).lower().find("resnet") >= 0 or in_ch == 3) else 48)

        name = f"{os.path.basename(os.path.dirname(ck))} | {os.path.basename(ck)}"
        specs.append(dict(
            name=name,
            key=(model_key or ("resnet18" if in_ch == 3 else "emotion_cnn")),
            ckpt=ck,
            in_size=int(in_size),
            in_ch=int(in_ch),
            classes=classes,
        ))
    print(f"[discover] Using {len(specs)} checkpoints:")
    for s in specs:
        print(
            f"  - {s['name']}  key={s['key']}  in_ch={s['in_ch']}  in_size={s['in_size']}  #classes={len(s['classes'])}")
    return specs


# -----------------------------------------------------------------------------
# Loading + prediction
# -----------------------------------------------------------------------------

def _load_models(device: str = "cpu"):
    models = []
    specs = _discover_model_specs()
    for spec in specs:
        state = load_checkpoint(spec["ckpt"])  # your util typically handles device internally
        classes = spec.get("classes") or state.get("classes", DEFAULT_CLASSES)
        key = str(spec["key"]).lower()
        try:
            m = create_model(key, num_classes=len(classes))
        except Exception:
            key = "resnet18" if spec.get("in_ch", 1) == 3 else "emotion_cnn"
            print(f"[warn] Unknown model key from ckpt; falling back to {key}")
            m = create_model(key, num_classes=len(classes))
        sd = state.get("state_dict") or state.get("model") or state
        m.load_state_dict(sd, strict=False)
        m.eval().to(device)
        spec["classes"] = classes
        models.append((spec, m, classes))
        print(
            f"[load] {spec['name']}: key={key}  in_ch={spec['in_ch']}  in_size={spec['in_size']}  ckpt={os.path.relpath(spec['ckpt'], PROJECT_ROOT)}")
    return models


@torch.no_grad()
def predict_on_face(models, gray_face: np.ndarray):
    """Predict on a single gray face ROI for every model.
    Returns list of dicts: {name, probs, label, conf}
    """
    outs = []
    for spec, m, classes in models:
        if int(spec.get("in_ch", 1)) == 3:
            x = _to_resnet(gray_face, spec["in_size"])
        else:
            x = _to_cnn48(gray_face, spec["in_size"])
        x = x.to(next(m.parameters()).device)
        logits = m(x)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs))
        outs.append(dict(
            name=spec["name"],
            probs=probs,
            label=classes[idx],
            conf=float(probs[idx]),
        ))
    return outs


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------

def annotate_per_model(base_bgr: np.ndarray, box: Tuple[int, int, int, int], results: List[Dict[str, Any]], max_cols=2):
    x, y, w, h = box
    panels = []
    for r in results:
        panel = base_bgr.copy()
        cv2.rectangle(panel, (x, y), (x + w, y + h), (0, 255, 0), 2)
        txt1 = f"{r['name']}"
        txt2 = f"{r['label']} ({r['conf'] * 100:.1f}%)"
        _put_text(panel, txt1, (10, 28), thick=2)
        _put_text(panel, txt2, (10, 58))
        panels.append(panel)

    if not panels:
        return base_bgr
    H, W = panels[0].shape[:2]
    cols = min(max_cols, len(panels))
    rows = int(np.ceil(len(panels) / cols))
    panels = [cv2.resize(p, (W, H)) for p in panels]
    grid_rows = []
    for r in range(rows):
        row_imgs = panels[r * cols:(r + 1) * cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.full_like(panels[0], 255))
        grid_rows.append(cv2.hconcat(row_imgs))
    grid = cv2.vconcat(grid_rows)
    return grid


def _put_text(img, text, org, thick=1, font_scale=0.8):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness=thick + 2,
                lineType=cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=thick,
                lineType=cv2.LINE_AA)


# -----------------------------------------------------------------------------
# Drivers
# -----------------------------------------------------------------------------

def run_on_image(path: str, models, device="cpu", out_path: Optional[str] = None):
    """Single pass: read/resize + detect once, reuse the same face crop for
    both annotated outputs and the bar-plot values. Returns (results, bgr, box, face_gray).
    """
    bgr, gray, box, face_gray = _prepare_face_once(path, max_height=600)
    if box is None:
        print(f"No face detected: {os.path.basename(path)}")
        return None, None, None, None

    results = predict_on_face(models, face_gray)

    # Save each model's annotated image separately
    x, y, w, h = box
    if out_path:
        base, ext = os.path.splitext(out_path)
    for r in results:
        panel = bgr.copy()
        cv2.rectangle(panel, (x, y), (x + w, y + h), (0, 255, 0), 4)  # Fatter box (thickness=4)
        txt1 = f"{r['name']}"
        txt2 = f"{r['label']} ({r['conf'] * 100:.1f}%)"
        _put_text(panel, txt1, (10, 38), thick=6, font_scale=1.2)  # Larger font, thicker
        _put_text(panel, txt2, (10, 78), thick=5, font_scale=1.1)
        if out_path:
            safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in r['name'])
            model_out_path = f"{base}_{safe_name}{ext}"
            os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
            cv2.imwrite(model_out_path, panel)
            print(f"Saved: {model_out_path}")

    return results, bgr, box, face_gray


def run_webcam(models, cam=0, device="cpu"):
    cap = cv2.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = _clahe_gray(gray)
            box = _detect_largest_face(gray)
            if box is None:
                cv2.imshow("Results (per model)", frame)
            else:
                x, y, w, h = box
                face_gray = gray[y:y + h, x:x + w].copy()
                results = predict_on_face(models, face_gray)
                grid = annotate_per_model(frame, box, results)
                cv2.imshow("Results (per model)", grid)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    image_dir = 'media'
    out_dir = os.path.join(PROJECT_ROOT, 'test/test_results')
    os.makedirs(out_dir, exist_ok=True)

    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    global HAAR
    HAAR = _build_haar()

    # Load all models ONCE
    models = _load_models(device=device)

    if image_paths:
        import matplotlib.pyplot as plt
        import numpy as np
        for img_path in image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            out_path = os.path.join(out_dir, f"{base}_tiled.png")
            print(f"[img] {img_path}")

            # --- Single-preprocess path: returns results used for BOTH outputs ---
            results, bgr, box, face_gray = run_on_image(img_path, models=models, device=device, out_path=out_path)
            if results is None:
                continue

            # Build bar-plot using the SAME results (no re-read / re-detect) -> identical numbers
            model_names = [r['name'].split(' | ')[0] for r in results]
            confs = [r['conf'] for r in results]
            preds = [r['label'] for r in results]

            plt.figure(figsize=(8, 8))
            bars = plt.bar(model_names, confs, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names))))
            plt.ylim(0, 1.05)
            plt.ylabel("Confidence")
            plt.title(f"Predictions for {base}", pad=18)
            plt.xticks(rotation=20, ha='right')
            for bar, pred, c in zip(bars, preds, confs):
                plt.text(bar.get_x() + bar.get_width() / 2, c + 0.02, f"{pred}\n{c:.3f}",
                         ha='center', va='bottom', fontsize=8)
            plt.tight_layout(rect=[0.04, 0.08, 0.98, 0.92])
            plt.savefig(os.path.join(out_dir, f"{base}_plot.png"))
    else:
        run_webcam(models=models, device=device)
