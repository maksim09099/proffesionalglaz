from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from src.iris_project.paths import DEFAULT_OUTPUT_DIR
from src.iris_project.pipeline import run_prediction

app = FastAPI(title="Iris Analysis API", version="1.0.0")


class ModelInfo(BaseModel):
    task: str
    model_type: str
    notes: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info", response_model=ModelInfo)
def model_info() -> ModelInfo:
    return ModelInfo(
        task="iris_segmentation_or_localization",
        model_type="Classical CV (HoughCircles)",
        notes="Optimized for local CPU inference, suitable as baseline for eye/iris dataset.",
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = DEFAULT_OUTPUT_DIR / file.filename
    with tmp_path.open("wb") as buffer:
        buffer.write(await file.read())
    return run_prediction(tmp_path, DEFAULT_OUTPUT_DIR)
