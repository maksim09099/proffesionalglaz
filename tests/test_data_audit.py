from pathlib import Path

import numpy as np
from scipy.io import savemat

from src.iris_project.data_audit import audit_dataset, infer_task


def test_audit_and_infer_task(tmp_path: Path) -> None:
    image = tmp_path / "foto01.jpg"
    image.write_bytes(b"fake")
    savemat(tmp_path / "foto01.mat", {"mask": np.zeros((10, 10), dtype=np.uint8)})
    savemat(tmp_path / "all_images.mat", {"image_ids": np.array([1])})

    report = audit_dataset(tmp_path)
    assert report.image_count == 1
    assert report.image_to_mat_matches == 1
    assert infer_task(report.mat_summaries) == "iris_segmentation_or_localization"
