from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .preprocessing import load_grayscale_image


@dataclass
class IrisPrediction:
    center_x: int
    center_y: int
    radius: int
    confidence: float


class IrisSegmentationModel:
    """Classical iris localization model using Hough circles.

    This baseline is CPU-friendly for local Windows/PyCharm usage.
    """

    def __init__(self, min_radius: int = 25, max_radius: int = 120) -> None:
        self.min_radius = min_radius
        self.max_radius = max_radius

    def predict(self, image_path: Path) -> IrisPrediction:
        image = load_grayscale_image(image_path)
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=image.shape[0] // 4,
            param1=120,
            param2=22,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )

        if circles is None:
            h, w = image.shape
            return IrisPrediction(center_x=w // 2, center_y=h // 2, radius=min(h, w) // 5, confidence=0.1)

        circle = np.round(circles[0][0]).astype(int)
        cx, cy, r = int(circle[0]), int(circle[1]), int(circle[2])
        confidence = min(1.0, max(0.2, r / max(self.max_radius, 1)))
        return IrisPrediction(center_x=cx, center_y=cy, radius=r, confidence=float(confidence))

    @staticmethod
    def annotate(image_path: Path, prediction: IrisPrediction, output_path: Path) -> Path:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")

        cv2.circle(image, (prediction.center_x, prediction.center_y), prediction.radius, (0, 255, 0), 2)
        cv2.circle(image, (prediction.center_x, prediction.center_y), 2, (0, 0, 255), 3)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        return output_path
