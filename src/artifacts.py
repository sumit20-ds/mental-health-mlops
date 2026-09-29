"""Save / load the three files that define a deployable model (same layout locally, in Docker, on SageMaker)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.config import (DRIFT_CATEGORICAL, DRIFT_NUMERIC, FEATURE_COLS, META_FILE, MODEL_FILE, REFERENCE_FILE,
                        REGISTERED_MODEL_NAME)
from src.drift import build_reference


def save_artifacts(out_dir: Path, pipeline, meta: dict, reference_sample: pd.DataFrame) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, out_dir / MODEL_FILE, compress=3)
    (out_dir / META_FILE).write_text(json.dumps(meta, indent=2))
    reference_sample.to_csv(out_dir / REFERENCE_FILE, index=False)


def load_artifacts(model_dir: Path):
    model_dir = Path(model_dir)
    pipeline = joblib.load(model_dir / MODEL_FILE)
    meta = json.loads((model_dir / META_FILE).read_text())
    ref_path = model_dir / REFERENCE_FILE
    ref_sample = pd.read_csv(ref_path) if ref_path.exists() else None
    return pipeline, meta, ref_sample


def build_meta(algorithm: str, metrics: dict, candidates: list[dict], top_countries: list[str],
               X_ref: pd.DataFrame, ref_predictions) -> tuple[dict, pd.DataFrame]:
    """Everything serving + monitoring needs to know about a trained model. Reference = held-out data."""
    reference_sample = X_ref.copy()
    reference_sample["prediction"] = ref_predictions
    meta = {
        "model_name": REGISTERED_MODEL_NAME,
        "algorithm": algorithm,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "feature_columns": FEATURE_COLS,
        "top_countries": top_countries,
        "metrics": metrics,
        "candidates": candidates,
        "reference": build_reference(reference_sample, DRIFT_NUMERIC, DRIFT_CATEGORICAL),
    }
    return meta, reference_sample
