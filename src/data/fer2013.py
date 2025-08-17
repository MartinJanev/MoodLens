import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional

DEFAULT_CLASSES = ["Anger","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def _apply_clahe_uint8(gray_uint8: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image.
    Input: gray_uint8 - 2D numpy array of uint8 grayscale image.
    Output: 2D numpy array of uint8 grayscale image after CLAHE.
    Uses OpenCV for CLAHE.
    Note: This function assumes the input is a single-channel grayscale image.
    If the input is not uint8, it will be converted to uint8 before applying CLAHE.
    The output will also be in uint8 format.
    If the input is not in the expected format, it may raise an error.
    :param gray_uint8: 2D numpy array of uint8 grayscale image.
    :return: 2D numpy array of uint8 grayscale image after CLAHE.
    """
    import cv2
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray_uint8)

class FER2013Class(Dataset):
    """
    Reads FER-style CSV with 'emotion','pixels' (48*48 space-separated).
    Preprocess: optional CLAHE + normalize to [0,1]. No transforms.py dependency.
    """
    def __init__(
        self,
        csv_path: str,
        classes: Optional[List[str]] = None,
        augment: bool = False,
        use_clahe: bool = True,
        cache: bool = False,  # set True if you want one-time RAM cache for speed
    ) -> None:
        self.df = pd.read_csv(csv_path)
        self.classes = classes or DEFAULT_CLASSES
        self.augment = augment
        self.use_clahe = use_clahe
        self.cache = cache

        self._xs = None
        self._ys = None
        if self.cache:
            self._build_cache()

    def __len__(self) -> int:
        return len(self.df)

    def _row_to_image(self, row_pixels: str) -> np.ndarray:
        """
        Convert a string of space-separated pixel values into a 48x48 numpy array.
        Expects 2304 float32 values (48*48).
        :param row_pixels: String of space-separated pixel values.
        :return: 2D numpy array of shape (48, 48) with float32 pixel values.
        """
        arr = np.fromstring(row_pixels, sep=" ", dtype=np.float32)
        if arr.size != 48*48:
            raise ValueError(f"Expected 2304 pixels; got {arr.size}")
        return arr.reshape(48,48)

    def _preprocess_gray(self, gray: np.ndarray) -> np.ndarray:
        """
        Preprocess the grayscale image:
        - If use_clahe is True, apply CLAHE.
        - Normalize to [0, 1] by dividing by 255.0.
        Expects gray to be a 2D numpy array of float32.
        Returns a 2D numpy array of float32 in [0, 1].
        :param gray: 2D numpy array of grayscale image.
        :return: Preprocessed grayscale image as a 2D numpy array in [0,
        """
        if self.use_clahe:
            gray = _apply_clahe_uint8(gray.astype("uint8")).astype("float32")
        return gray.astype("float32") / 255.0

    def _build_cache(self) -> None:
        """
        Build a cache of preprocessed images and labels in RAM.
        This speeds up __getitem__() significantly, especially with augmentations.
        Uses self._xs and self._ys to store the preprocessed data.
        Each image is a 2D numpy array of shape (48, 48) and
        each label is an integer in the range [0, len(classes)-1].
        The cache is built only once when the dataset is initialized with cache=True.
        The cache is not saved to disk; it exists only in memory.
        """
        xs, ys = [], []
        for i in range(len(self.df)):
            rec = self.df.iloc[i]
            y = int(rec["emotion"])
            gray = self._row_to_image(rec["pixels"])
            gray = self._preprocess_gray(gray)
            xs.append(gray)
            ys.append(y)
        self._xs = torch.from_numpy(np.stack(xs)).unsqueeze(1).contiguous()  # (N,1,48,48)
        self._ys = torch.tensor(ys, dtype=torch.long)

    def __getitem__(self, idx: int):
        """
        Get an item by index.
        If cache is enabled, returns preprocessed image and label from cache.
        If augment is True, applies random horizontal flip and rotation.
        Returns a tuple (x, y) where:
        - x is a tensor of shape (1, 48, 48) representing the
            preprocessed grayscale image.
        - y is a tensor of shape () representing the label (emotion index).
        If cache is disabled, reads the row from the DataFrame,
        preprocesses the image, and returns it along with the label.
        :param idx: Index of the item to retrieve.
        :return: A tuple (x, y) where x is the preprocessed image tensor
        """
        if self.cache and self._xs is not None:
            x = self._xs[idx]
            y = self._ys[idx]
            if self.augment:
                import numpy as np, cv2
                g = x[0].numpy()
                if np.random.rand() < 0.5:
                    g = np.fliplr(g).copy()
                deg = (np.random.rand() * 2 - 1) * 12.0
                M = cv2.getRotationMatrix2D((24, 24), deg, 1.0)
                g = cv2.warpAffine(g, M, (48,48), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                x = torch.from_numpy(g).unsqueeze(0)
            return x, y

        rec = self.df.iloc[idx]
        y = int(rec["emotion"])
        gray = self._row_to_image(rec["pixels"])
        gray = self._preprocess_gray(gray)
        x = torch.from_numpy(gray).unsqueeze(0)  # (1,48,48)
        y = torch.tensor(y, dtype=torch.long)
        return x, y
