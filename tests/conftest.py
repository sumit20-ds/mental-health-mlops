import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.artifacts import build_meta, save_artifacts
from src.config import DATA_PATH, SEED, TARGET
from src.features import clean, fit_top_countries, prepare_features
from src.pipeline import build_pipeline, regression_metrics


@pytest.fixture(scope="session")
def dataset():
    return clean(pd.read_csv(DATA_PATH))


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, dataset):
    """Trains a tiny model once and exports it exactly like src/train.py does."""
    top = fit_top_countries(dataset)
    X, y = prepare_features(dataset, top), dataset[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    pipe = build_pipeline(LinearRegression()).fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    metrics = regression_metrics(y_te, preds)
    meta, ref = build_meta("linear_regression", metrics, [{"name": "linear_regression", **metrics}], top, X_te, preds)
    out = tmp_path_factory.mktemp("model")
    save_artifacts(out, pipe, meta, ref)
    return out
