import pandas as pd

from src.config import FEATURE_COLS
from src.features import clean, fit_top_countries, prepare_features


def test_clean_clips_negative_activity_and_drops_duplicates(dataset):
    assert (dataset["Physical_Activity_Hours"] >= 0).all()
    assert not dataset.duplicated().any()


def test_top_countries_and_grouping(dataset):
    top = fit_top_countries(dataset, 3)
    row = dataset.head(1).copy()
    row["Country"] = "Narnia"
    assert prepare_features(row, top)["Grouped_country"].iloc[0] == "Other"
    row["Country"] = top[0]
    assert prepare_features(row, top)["Grouped_country"].iloc[0] == top[0]


def test_prepare_features_returns_exact_columns(dataset):
    X = prepare_features(dataset.head(5), fit_top_countries(dataset))
    assert list(X.columns) == FEATURE_COLS


def test_negative_activity_is_clipped_at_serving_time(dataset):
    row = dataset.head(1).copy()
    row["Physical_Activity_Hours"] = -0.4
    assert prepare_features(row, ["Other"])["Physical_Activity_Hours"].iloc[0] == 0
    assert isinstance(clean(pd.DataFrame(dataset.head(3))), pd.DataFrame)
