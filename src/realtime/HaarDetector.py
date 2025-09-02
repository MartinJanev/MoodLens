import cv2
from typing import List, Tuple

class HaarFaceDetector:
    def __init__(self, cascade_path: str, scale_factor: float = 1.2, min_neighbors: int = 6, min_size: int = 24):
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise FileNotFoundError(f"Could not load Haar cascade at {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = (min_size, min_size)

    def __call__(self, gray_frame) -> List[Tuple[int,int,int,int]]:
        # returns list of (x,y,w,h)
        faces = self.detector.detectMultiScale(
            gray_frame,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces
