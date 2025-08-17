import cv2, numpy as np, torch, torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
import tkinter as tk
from PIL import Image, ImageTk

from ..realtime.face_class import HaarFaceDetector
from ..utils.io import load_checkpoint
from ..models.factory import create_model

# ---------------- existing helpers (unchanged) ----------------
def _preprocess_face_bgr_to_model(gray_or_bgr, use_clahe: bool = True) -> np.ndarray:
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
    return {c: _DEFAULT_COLOR_MAP.get(c.lower(), (0, 255, 0)) for c in classes}

def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
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
    """Cross-platform-ish icon setter. Keeps a reference to avoid GC."""
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
        model_name: str = "cnn_small",
        classes: Optional[List[str]] = None,
        camera_index: int = 0,
        detect_every_n: int = 2,
        ema_decay: float = 0.9,
        box_thickness: int = 3,
        window_title: str = "Emotion Vision",
        icon_path: Optional[str] = "assets/favicon.ico",     # <-- add your icon here ('.ico' on Windows; PNG elsewhere)
        keep_on_top: bool = True,
) -> None:
    """Tkinter UI wrapper around your real-time detector."""
    # Load model
    state = load_checkpoint(model_path)
    classes = classes or state.get("classes") or ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    model = create_model(model_name, num_classes=len(classes))
    model.load_state_dict(state["model"])
    model.eval().to(device)

    color_map = _build_color_map(classes)
    detector = HaarFaceDetector(cascade_path)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    # --- Tk setup
    root = tk.Tk()
    root.title(window_title)
    if keep_on_top:
        try: root.attributes("-topmost", True)
        except: pass
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
            face = frame[y:y+h, x:x+w]
            g = _preprocess_face_bgr_to_model(face, use_clahe=use_clahe)
            tens = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                probs = F.softmax(model(tens), dim=1)[0].detach().cpu().numpy()
            curr_probs.append(probs)

        # Step 2: EMA smoothing by best IoU match to previous frame
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

        # Step 3: draw overlays directly on BGR frame
        for (x, y, w, h), probs in zip(boxes, smoothed_probs):
            lab, k = softmax_to_label(probs, classes)
            color = color_map.get(classes[k], (0, 255, 0))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)
            cv2.putText(frame, lab, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Convert BGR -> RGB, then to Tk image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=im)
        # Keep a reference (Tk GC quirk)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        frame_i += 1
        prev_boxes = list(boxes)
        prev_probs = [p.copy() for p in smoothed_probs]

        # Schedule next frame
        root.after(10, update_frame)  # ~100 fps cap; adjust if needed

    update_frame()
    try:
        root.mainloop()
    finally:
        cap.release()
        cv2.destroyAllWindows()
