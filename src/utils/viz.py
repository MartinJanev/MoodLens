import cv2, numpy as np
from typing import Sequence

def draw_label(img, text: str, x: int, y: int):
    """
    Draw a label on the image at the specified position.
    :param img: The image on which to draw the label.
    :param text: The label text to draw.
    :param x: The x-coordinate for the label position.
    :param y: The y-coordinate for the label position.
    """
    cv2.putText(img, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

def draw_boxes_with_labels(img, boxes: Sequence[Sequence[int]], labels: Sequence[str]):
    """
    Draw bounding boxes and labels on the image.
    :param img: The image on which to draw the boxes and labels.
    :param boxes: A sequence of bounding boxes, each defined as (x, y, w, h).
    :param labels: A sequence of labels corresponding to each bounding box.
    :return: The image with drawn boxes and labels.
    """
    for (x,y,w,h), lab in zip(boxes, labels):
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        draw_label(img, lab, x, y)
    return img
