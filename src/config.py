"""Single source of truth for paths, column groups and MLflow names."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = Path(os.getenv("DATA_PATH", ROOT / "data/raw/Student_Social_Media_And_Mental_Health_Impact.csv"))
EXPORT_DIR = Path(os.getenv("EXPORT_DIR", ROOT / "models"))

TARGET = "Mental_Health_Score"

# --- column groups (mirrors the notebook) -----------------------------------
SKEWED_COLS = ["Study_Hours"]
NUMERIC_COLS = ["Age", "Avg_Daily_Usage_Hours", "Daily_Unlocks", "Physical_Activity_Hours", "Sleep_Hours_Per_Night"]
ORDINAL_COLS = ["Stress_Level"]
STRESS_ORDER = ["Low", "Medium", "High", "Very High"]
NOMINAL_COLS = ["Gender", "Academic_Level", "Most_Used_Platform", "Purpose_Of_Use", "Grouped_country"]

FEATURE_COLS = SKEWED_COLS + NUMERIC_COLS + ORDINAL_COLS + NOMINAL_COLS
# What a client sends: raw `Country`; the server derives `Grouped_country`.
RAW_INPUT_COLS = [c if c != "Grouped_country" else "Country" for c in FEATURE_COLS]

DRIFT_NUMERIC = SKEWED_COLS + NUMERIC_COLS + ["prediction"]
DRIFT_CATEGORICAL = ORDINAL_COLS + NOMINAL_COLS

TOP_N_COUNTRIES = 10
TEST_SIZE = 0.2
SEED = 42

# --- MLflow ------------------------------------------------------------------
EXPERIMENT_NAME = "student-mental-health"
REGISTERED_MODEL_NAME = "student-mental-health-regressor"
CHAMPION_ALIAS = "champion"

# --- artifact file names (identical for local, Docker and SageMaker) ---------
MODEL_FILE = "model.joblib"
META_FILE = "model_meta.json"
REFERENCE_FILE = "reference_sample.csv"
