from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

try:
    import h5py
except Exception:  # optional
    h5py = None

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class MatSummary:
    path: Path
    keys: list[str]
    arrays: dict[str, tuple[int, ...]]


@dataclass
class DatasetAuditReport:
    image_count: int
    mat_count: int
    image_to_mat_matches: int
    missing_mat_for_image: list[str]
    missing_image_for_mat: list[str]
    mat_summaries: list[MatSummary]


def _load_mat(path: Path) -> dict[str, Any]:
    try:
        return scipy.io.loadmat(path)
    except NotImplementedError:
        if h5py is None:
            raise
        payload: dict[str, Any] = {}
        with h5py.File(path, "r") as f:  # type: ignore[arg-type]
            for key in f.keys():
                payload[key] = np.array(f[key])
        return payload


def summarize_mat(path: Path) -> MatSummary:
    payload = _load_mat(path)
    keys = [k for k in payload.keys() if not k.startswith("__")]
    arrays: dict[str, tuple[int, ...]] = {}
    for key in keys:
        value = payload[key]
        if isinstance(value, np.ndarray):
            arrays[key] = value.shape
    return MatSummary(path=path, keys=keys, arrays=arrays)


def infer_task(mat_summaries: list[MatSummary]) -> str:
    """Heuristic task inference from .mat keys and tensor ranks."""
    keyset = {key.lower() for summary in mat_summaries for key in summary.keys}
    if any(k in keyset for k in {"mask", "masks", "segmentation", "contour", "contours", "iris_mask"}):
        return "iris_segmentation_or_localization"
    if any(k in keyset for k in {"subject_id", "person_id", "class", "label", "identity"}):
        return "iris_biometric_identification"
    if any(len(shape) >= 2 for summary in mat_summaries for shape in summary.arrays.values()):
        return "iris_feature_extraction"
    return "unknown_requires_manual_review"


def audit_dataset(dataset_dir: Path) -> DatasetAuditReport:
    image_paths = sorted([p for p in dataset_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS])
    mat_paths = sorted(dataset_dir.rglob("*.mat"))

    image_stems = {p.stem for p in image_paths}
    mat_stems = {p.stem for p in mat_paths}

    matches = sorted(image_stems & mat_stems)
    missing_mat = sorted(image_stems - mat_stems)
    missing_image = sorted(stem for stem in mat_stems - image_stems if stem != "all_images")

    summaries = [summarize_mat(path) for path in mat_paths]

    return DatasetAuditReport(
        image_count=len(image_paths),
        mat_count=len(mat_paths),
        image_to_mat_matches=len(matches),
        missing_mat_for_image=missing_mat,
        missing_image_for_mat=missing_image,
        mat_summaries=summaries,
    )
