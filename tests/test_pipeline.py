from pathlib import Path

import cv2
import numpy as np

from src.iris_project.pipeline import run_prediction


def test_run_prediction_creates_annotation(tmp_path: Path) -> None:
    img = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(img, (100, 100), 45, 255, -1)
    image_path = tmp_path / "eye.jpg"
    cv2.imwrite(str(image_path), img)

    result = run_prediction(image_path, tmp_path / "artifacts")
    assert Path(result["annotated_image"]).exists()
    assert result["task"] == "iris_segmentation_or_localization"
