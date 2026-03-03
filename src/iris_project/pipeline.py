from __future__ import annotations

from pathlib import Path

from .model import IrisSegmentationModel


def run_prediction(image_path: Path, artifacts_dir: Path) -> dict:
    model = IrisSegmentationModel()
    pred = model.predict(image_path)
    annotated_path = artifacts_dir / f"{image_path.stem}_annotated.jpg"
    model.annotate(image_path, pred, annotated_path)
    return {
        "task": "iris_segmentation_or_localization",
        "center": {"x": pred.center_x, "y": pred.center_y},
        "radius": pred.radius,
        "confidence": pred.confidence,
        "annotated_image": str(annotated_path),
    }
