import cv2, numpy as np, torch, torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from ..realtime.face_class import HaarFaceDetector
from ..utils.io import load_checkpoint
from ..models.factory import create_model


def _preprocess_face_bgr_to_model(gray_or_bgr, use_clahe: bool = True) -> np.ndarray:
    """
    Accepts a BGR face crop or a grayscale array, returns float32 HxW in [0,1]
    resized to 48x48 with optional CLAHE.
    If input is BGR, it will be converted to grayscale first.
    If input is already grayscale, it will be resized directly.
    Uses OpenCV's cv2.INTER_AREA for resizing.
    Uses cv2.createCLAHE for CLAHE if use_clahe is True.
    Returns a 2D numpy array of shape (48, 48) with pixel values
    normalized to the range [0, 1].
    If use_clahe is True, applies CLAHE to the grayscale image.
    """
    if len(gray_or_bgr.shape) == 3:
        g = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        g = gray_or_bgr
    g = cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA)
    if use_clahe:
        g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    g = (g.astype("float32") / 255.0)
    return g  # HxW float32


def softmax_to_label(probs: np.ndarray, classes: List[str]) -> Tuple[str, int]:
    """
    Converts a softmax output (1D array of probabilities) to a label string
    and the index of the top class.
    :param probs: 1D numpy array of probabilities (e.g. from softmax)
    :param classes: List of class names corresponding to the probabilities
    :return: Tuple of (label string, index of top class)
    The label string is formatted as "class_name probability", e.g. "happy 0.85".
    The index is the position of the highest probability in the probs array.
    """
    k = int(probs.argmax())
    return f"{classes[k]} {probs[k]:.2f}", k  # return label and top-class index


# --- Colors (BGR) per emotion; falls back to green if a class name is unknown
_DEFAULT_COLOR_MAP = {
    "anger": (0, 0, 255),  # red
    "disgust": (0, 128, 0),  # dark green
    "fear": (128, 0, 128),  # purple
    "happy": (0, 255, 255),  # yellow
    "sad": (255, 0, 0),  # blue
    "surprise": (255, 255, 0),  # cyan
    "neutral": (200, 200, 200),  # gray
}


def _build_color_map(classes: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """
    Builds a color map for the given classes, mapping each class name to a BGR color.
    If a class name is not found in the default map, it defaults to green (0, 255, 0).
    :param classes: List of class names (case-insensitive)
    :return: Dictionary mapping class names to BGR tuples
    """
    return {c: _DEFAULT_COLOR_MAP.get(c.lower(), (0, 255, 0)) for c in classes}


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.
    Each box is defined by (x, y, width, height).
    Returns the IoU as a float in the range [0, 1].
    If the boxes do not overlap, returns 0.0.
    :param a: First bounding box (x, y, width, height)
    :param b: Second bounding box (x, y, width, height)
    :return: IoU as a float in [0, 1]
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


def run_webcam(
        model_path: str,
        cascade_path: str,
        device: str = "cpu",
        use_clahe: bool = True,
        model_name: str = "cnn_small",
        classes: Optional[List[str]] = None,
        camera_index: int = 0,
        detect_every_n: int = 2,
        ema_decay: float = 0.9,  # ↑ bigger = smoother/slower changes (0..1)
        box_thickness: int = 3  # slightly thicker box
) -> None:
    """
    Runs a real-time emotion detection webcam app using a pre-trained model.
    :param model_path: Path to the model checkpoint file.
    :param cascade_path: Path to the Haar cascade XML file for face detection.
    :param device: Device to run the model on ("cpu" or "cuda").
    :param use_clahe: Whether to apply CLAHE preprocessing to face crops.
    :param model_name: Name of the model architecture to create.
    :param classes: Optional list of emotion class names. If None, uses default classes.
    :param camera_index: Index of the webcam to use (default is 0).
    :param detect_every_n: How often to run the face detector (every N frames).
    :param ema_decay: Exponential moving average decay factor for smoothing probabilities.
                      Higher values mean smoother but slower changes (0..1).
    :param box_thickness: Thickness of the bounding box drawn around detected faces.
    :return: None. Displays a window with real-time emotion detection.

    This function captures video from the webcam, detects faces using Haar cascades,
    and applies the specified emotion detection model to each detected face.
    """
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

    frame_i = 0
    cached_boxes: List[Tuple[int, int, int, int]] = []

    # Keep smoothed probabilities matched to boxes across frames
    prev_boxes: List[Tuple[int, int, int, int]] = []
    prev_probs: List[np.ndarray] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Re-run detector every Nth frame for speed; reuse between
            if frame_i % detect_every_n == 0:
                boxes = detector(gray_full)
                cached_boxes = list(boxes)
            else:
                boxes = cached_boxes

            # Step 1: raw probs for current boxes
            curr_probs: List[np.ndarray] = []
            for (x, y, w, h) in boxes:
                face = frame[y:y + h, x:x + w]
                g = _preprocess_face_bgr_to_model(face, use_clahe=use_clahe)
                tens = torch.from_numpy(g).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,48,48)
                with torch.no_grad():
                    probs = F.softmax(model(tens), dim=1)[0].detach().cpu().numpy()
                curr_probs.append(probs)

            # Step 2: match to previous boxes & apply EMA smoothing
            smoothed_probs: List[np.ndarray] = []
            for i, b in enumerate(boxes):
                # find best previous match by IoU
                best_j, best = -1, 0.0
                for j, pb in enumerate(prev_boxes):
                    iou = _iou(b, pb)
                    if iou > best:
                        best, best_j = iou, j
                if best_j >= 0 and best >= 0.30:  # good enough match
                    smoothed = ema_decay * prev_probs[best_j] + (1.0 - ema_decay) * curr_probs[i]
                else:
                    smoothed = curr_probs[i]
                smoothed_probs.append(smoothed)

            # Step 3: draw with per-emotion colors and thicker boxes
            for (x, y, w, h), probs in zip(boxes, smoothed_probs):
                lab, k = softmax_to_label(probs, classes)
                color = color_map.get(classes[k], (0, 255, 0))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, box_thickness)
                cv2.putText(frame, lab, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            cv2.imshow("Emotion Vision", frame)
            cv2.setWindowProperty("Emotion Vision", cv2.WND_PROP_TOPMOST, 1)
            frame_i += 1

            # update state for next frame
            prev_boxes = list(boxes)
            prev_probs = [p.copy() for p in smoothed_probs]

            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
