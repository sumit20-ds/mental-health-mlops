"""sklearn pipelines and candidate models (ported from the notebook, plus imputers for production safety)."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import NOMINAL_COLS, NUMERIC_COLS, ORDINAL_COLS, SEED, SKEWED_COLS, STRESS_ORDER


def build_preprocessor() -> ColumnTransformer:
    skew = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    ordinal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(categories=[STRESS_ORDER])),
    ])
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("skewed", skew, SKEWED_COLS),
        ("numeric", numeric, NUMERIC_COLS),
        ("ordinal", ordinal, ORDINAL_COLS),
        ("nominal", nominal, NOMINAL_COLS),
    ])


def build_pipeline(regressor) -> Pipeline:
    return Pipeline([("preprocessor", build_preprocessor()), ("regressor", regressor)])


def get_candidates(seed: int = SEED, quick: bool = False) -> dict:
    """Same three experiments as the notebook: LR baseline, RF default, RF tuned."""
    tuned = RandomizedSearchCV(
        build_pipeline(RandomForestRegressor(random_state=seed, n_jobs=1)),
        param_distributions={
            "regressor__n_estimators": [100, 200, 300],
            "regressor__max_depth": [5, 10, 15],
            "regressor__min_samples_split": [2, 5, 10],
            "regressor__min_samples_leaf": [1, 2, 4],
        },
        n_iter=3 if quick else 15,
        cv=3 if quick else 5,
        scoring="r2",
        random_state=seed,
        n_jobs=-1,
    )
    return {
        "linear_regression": build_pipeline(LinearRegression()),
        "random_forest_default": build_pipeline(RandomForestRegressor(random_state=seed, n_jobs=-1)),
        "random_forest_tuned": tuned,
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
