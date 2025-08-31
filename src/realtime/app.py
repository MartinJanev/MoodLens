import cv2, numpy as np, torch, torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import tkinter as tk
from PIL import Image, ImageTk

from ..realtime.HaarDetector import HaarFaceDetector
from ..utils.io import load_checkpoint
from ..models.factory import create_model

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _preprocess_face_bgr_to_resnet(gray_or_bgr, size: int = 160, use_clahe: bool = True) -> torch.Tensor:
    """
    Convert a face crop to a (1, 3, size, size) tensor normalized to ImageNet stats.
    Keeps CLAHE behavior consistent with your 48x48 pipeline.
    """
    if len(gray_or_bgr.shape) == 3:
        g = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        g = gray_or_bgr
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    if use_clahe:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    g = (g.astype("float32") / 255.0)  # (H, W) in [0,1]
    g3 = np.repeat(g[None, :, :], 3, axis=0)  # (3, H, W)
    t = torch.from_numpy(g3).unsqueeze(0)  # (1, 3, H, W) on CPU
    # normalize (CPU is fine; you can move to GPU after)
    t = (t - _IMAGENET_MEAN.to(t.device)) / _IMAGENET_STD.to(t.device)
    return t


def _preprocess_face_bgr_to_model(gray_or_bgr, use_clahe: bool = True) -> np.ndarray:
    """
    Preprocess a face image (BGR or grayscale) for model input.
    - Convert to grayscale if needed
    - Resize to 48x48
    - Optionally apply CLAHE
    - Scale to [0, 1] float32
    Returns a (48, 48) float32 numpy array.
    :param gray_or_bgr: Input image, either grayscale (H, W) or BGR (H, W, 3)
    :param use_clahe: Whether to apply CLAHE (default True)
    :return: Preprocessed grayscale image as (48, 48) float32 array
    """
    if len(gray_or_bgr.shape) == 3:
        g = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        g = gray_or_bgr
    g = cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA)
    if use_clahe:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    g = (g.astype("float32") / 255.0)
    return g


def softmax_to_label(probs: np.ndarray, classes: List[str]) -> Tuple[str, int]:
    """
    Convert softmax probabilities to a label string and index.
    :param probs: 1D numpy array of softmax probabilities
    :param classes: List of class names corresponding to probabilities
    :return: Tuple of (label string with confidence, class index)
    0 <= class index < len(classes)
    Example: "Happy 0.85", 3
    0.0 <= confidence <= 1.0
    """
    k = int(probs.argmax())
    return f"{classes[k]} {probs[k]:.2f}", k


_DEFAULT_COLOR_MAP = {
    "anger": (0, 0, 255),
    "disgust": (0, 128, 0),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (255, 255, 0),
    "neutral": (200, 200, 200),
}


def _build_color_map(classes: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """
    Build a color map for the given classes.
    Uses _DEFAULT_COLOR_MAP for known classes; defaults to green (0, 255,
    :param classes: List of class names
    :return: Dict mapping class name to BGR color tuple
    """
    return {c: _DEFAULT_COLOR_MAP.get(c.lower(), (0, 255, 0)) for c in classes}


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) of two bounding boxes.
    Boxes are (x, y, w, h).
    Returns IoU in [0.0, 1.0].
    :param a: First bounding box (x, y, w, h)
    :param b: Second bounding box (x, y, w, h)
    :return: IoU value
    """
    ax, ay, aw, ah = a;
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ---------------- Tkinter version ----------------
def _set_tk_icon(root: tk.Tk, icon_path: Optional[str]) -> None:
    """
    Cross-platform-ish icon setter. Keeps a reference to avoid GC.
    :param root: Tk root window
    :param icon_path: Path to icon file ('.ico' on Windows; PNG elsewhere)
    """
    if not icon_path:
        return
    import os
    ext = os.path.splitext(icon_path)[1].lower()
    try:
        if ext == ".ico" and os.name == "nt":
            root.iconbitmap(icon_path)  # Windows .ico
        else:
            img = tk.PhotoImage(file=icon_path)  # PNG works cross-platform
            root.iconphoto(True, img)
            root._icon_ref = img
    except Exception as e:
        print(f"[icon] Could not set icon: {e}")


def run_webcam(
        model_path: str,
        cascade_path: str,
        device: str = "cpu",
        use_clahe: bool = True,
        model_name: str = "cnn_small",  # pass "resnet18" or leave; we auto-detect if mismatched
        classes: Optional[List[str]] = None,
        camera_index: int = 0,
        detect_every_n: int = 2,
        ema_decay: float = 0.9,
        box_thickness: int = 3,
        window_title: str = "Emotion Vision",
        icon_path: Optional[str] = "assets/favicon.ico",
        keep_on_top: bool = True,
        resnet_input_size: int = 160,  # NEW: match your training size (160 default)
) -> None:
    """
    Tkinter UI wrapper around your real-time detector.
    """
    # Load checkpoint
    state = load_checkpoint(model_path)
    classes = classes or state.get("classes") or ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

    # --- Auto-detect checkpoint type and instantiate correct model ---
    def _looks_like_resnet(sd_keys) -> bool:
        # ResNet checkpoints typically have 'layer1.0.conv1.weight' and 'fc.weight'
        return any(k.startswith("layer1.0.conv1.weight") for k in sd_keys) or ("fc.weight" in sd_keys)

    sd_keys = state["model"].keys()
    is_resnet = "resnet" in (model_name or "").lower()

    model = create_model(model_name, num_classes=len(classes))
    try:
        model.load_state_dict(state["model"])
    except RuntimeError:
        # auto-switch if checkpoint & requested model mismatch
        if _looks_like_resnet(sd_keys) and not is_resnet:
            model_name = "resnet18"
            is_resnet = True
            model = create_model(model_name, num_classes=len(classes))
            model.load_state_dict(state["model"])
            print("[webcam] Auto-detected ResNet checkpoint — switched model to resnet18.")
        elif not _looks_like_resnet(sd_keys) and is_resnet:
            model_name = "cnn_small"
            is_resnet = False
            model = create_model(model_name, num_classes=len(classes))
            model.load_state_dict(state["model"])
            print("[webcam] Auto-detected small CNN checkpoint — switched model to cnn_small.")
        else:
            raise  # still mismatched → rethrow

    model.eval().to(device)

    color_map = _build_color_map(classes)
    detector = HaarFaceDetector(cascade_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    # --- Tk setup (unchanged) ---
    root = tk.Tk()
    root.title(window_title)
    if keep_on_top:
        try:
            root.attributes("-topmost", True)
        except:
            pass
    _set_tk_icon(root, icon_path)

    video_label = tk.Label(root)
    video_label.pack()

    # state carried between frames
    frame_i = 0
    cached_boxes: List[Tuple[int, int, int, int]] = []
    prev_boxes: List[Tuple[int, int, int, int]] = []
    prev_probs: List[np.ndarray] = []

    running = True

    def on_close():
        nonlocal running
        running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.bind("<Escape>", lambda e: on_close())
    root.bind("<q>", lambda e: on_close())

    def update_frame():
        nonlocal frame_i, cached_boxes, prev_boxes, prev_probs
        if not running:
            return

        ok, frame = cap.read()
        if not ok:
            on_close()
            return

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Re-run detector every Nth frame; cache between runs
        if frame_i % detect_every_n == 0:
            boxes = detector(gray_full)
            cached_boxes = list(boxes)
        else:
            boxes = cached_boxes

        # Step 1: current raw probabilities
        curr_probs: List[np.ndarray] = []
        for (x, y, w, h) in boxes:
            face = frame[y:y + h, x:x + w]

            if is_resnet:
                tens = _preprocess_face_bgr_to_resnet(face, size=resnet_input_size, use_clahe=use_clahe).to(device)
            else:
                g = _preprocess_face_bgr_to_model(face, use_clahe=use_clahe)
                tens = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,48,48)

            with torch.no_grad():
                probs = F.softmax(model(tens), dim=1)[0].detach().cpu().numpy()
            curr_probs.append(probs)

        # Step 2: EMA smoothing by best IoU match to previous frame (unchanged)
        smoothed_probs: List[np.ndarray] = []
        for i, b in enumerate(boxes):
            best_j, best = -1, 0.0
            for j, pb in enumerate(prev_boxes):
                iou = _iou(b, pb)
                if iou > best:
                    best, best_j = iou, j
            if best_j >= 0 and best >= 0.30:
                smoothed = ema_decay * prev_probs[best_j] + (1.0 - ema_decay) * curr_probs[i]
            else:
                smoothed = curr_probs[i]
            smoothed_probs.append(smoothed)

        # Step 3: draw overlays (unchanged)
        for (x, y, w, h), probs in zip(boxes, smoothed_probs):
            lab, k = softmax_to_label(probs, classes)
            color = color_map.get(classes[k], (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)
            cv2.putText(frame, lab, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Show frame (unchanged)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        frame_i += 1
        prev_boxes = list(boxes)
        prev_probs = [p.copy() for p in smoothed_probs]

        root.after(10, update_frame)

    update_frame()
    try:
        root.mainloop()
    finally:
        cap.release()
        cv2.destroyAllWindows()
