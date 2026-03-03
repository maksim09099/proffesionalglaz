from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from .data_audit import IMAGE_EXTENSIONS


@dataclass
class SplitPaths:
    train: list[Path]
    val: list[Path]
    test: list[Path]


def load_grayscale_image(image_path: Path, size: tuple[int, int] = (256, 256)) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    image = cv2.equalizeHist(image)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def list_images(dataset_dir: Path) -> list[Path]:
    return sorted([p for p in dataset_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])


def split_dataset(image_paths: list[Path], test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> SplitPaths:
    train_val, test = train_test_split(image_paths, test_size=test_size, random_state=random_state)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=random_state)
    return SplitPaths(train=train, val=val, test=test)
