"""Data cleaning + feature preparation shared by training AND serving (no train/serve skew)."""
from __future__ import annotations

import pandas as pd

from src.config import FEATURE_COLS, TOP_N_COUNTRIES


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Notebook section 5: drop duplicates, clip impossible negative activity hours to 0."""
    out = df.drop_duplicates().copy()
    out["Physical_Activity_Hours"] = out["Physical_Activity_Hours"].clip(lower=0)
    return out.reset_index(drop=True)


def fit_top_countries(df: pd.DataFrame, n: int = TOP_N_COUNTRIES) -> list[str]:
    """Notebook section 7: keep the n most frequent countries, bucket the rest as 'Other'."""
    return df["Country"].value_counts().index[:n].tolist()


def prepare_features(df: pd.DataFrame, top_countries: list[str]) -> pd.DataFrame:
    """Raw request/data frame -> exact feature frame the sklearn pipeline expects."""
    out = df.copy()
    out["Physical_Activity_Hours"] = out["Physical_Activity_Hours"].astype(float).clip(lower=0)
    out["Grouped_country"] = out["Country"].where(out["Country"].isin(top_countries), "Other")
    return out[FEATURE_COLS]
