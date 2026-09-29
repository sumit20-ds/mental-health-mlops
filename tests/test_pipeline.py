import numpy as np

from src.artifacts import load_artifacts
from src.config import FEATURE_COLS


def test_exported_model_predicts_sensible_scores(model_dir, dataset):
    from src.features import prepare_features
    pipe, meta, ref = load_artifacts(model_dir)
    X = prepare_features(dataset.head(50), meta["top_countries"])
    preds = pipe.predict(X)
    assert preds.shape == (50,) and np.isfinite(preds).all()
    assert 0 < preds.mean() < 10
    assert meta["metrics"]["r2"] > 0.6, "baseline should comfortably beat 0.6 R2 on this dataset"


def test_meta_and_reference_are_complete(model_dir):
    _, meta, ref = load_artifacts(model_dir)
    assert set(FEATURE_COLS) <= set(ref.columns) and "prediction" in ref.columns
    assert "prediction" in meta["reference"]["numeric"]
