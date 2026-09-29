"""FastAPI model server.

Runs identically on a laptop, in docker-compose and inside a SageMaker endpoint:
  /predict, /predict/batch  -> application API (used by Streamlit)
  /ping, /invocations       -> SageMaker container contract (port 8080)
  /metrics                  -> Prometheus scrape target
  /monitoring/*             -> drift report + traffic simulator for demos
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import ValidationError

from src.artifacts import load_artifacts
from src.config import FEATURE_COLS, META_FILE, MODEL_FILE, RAW_INPUT_COLS
from src.drift import compute_drift
from src.features import prepare_features
from serving.schemas import (BatchRequest, BatchResponse, PredictionResponse, SimulationRequest,
                             StudentFeatures)

log = logging.getLogger("mh-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---- Prometheus metrics ------------------------------------------------------------------------
PREDICTIONS = Counter("mh_predictions_total", "Predictions served", ["endpoint"])
ERRORS = Counter("mh_errors_total", "Failed requests", ["endpoint"])
LATENCY = Histogram("mh_inference_latency_seconds", "Model inference latency",
                    buckets=(.005, .01, .025, .05, .1, .25, .5, 1, 2.5))
PRED_VALUE = Histogram("mh_prediction_value", "Distribution of predicted scores", buckets=tuple(range(1, 11)))
FEATURE_PSI = Gauge("mh_feature_psi", "Population Stability Index per feature", ["feature"])
DRIFT_LEVEL = Gauge("mh_drift_level", "0=stable 1=warning 2=drift 3=not enough data")

WINDOW = int(os.getenv("DRIFT_WINDOW", "2000"))
LOG_PATH = Path(os.getenv("PREDICTION_LOG_PATH", "logs/predictions.jsonl"))
LOG_TO_STDOUT = os.getenv("LOG_TO_STDOUT", "false").lower() == "true"
ENABLE_SIMULATION = os.getenv("ENABLE_SIMULATION", "true").lower() == "true"


def resolve_model_dir() -> Path:
    """MODEL_DIR env > SageMaker's /opt/ml/model > ./models."""
    for cand in (os.getenv("MODEL_DIR"), "/opt/ml/model", "models"):
        if cand and (Path(cand) / MODEL_FILE).exists() and (Path(cand) / META_FILE).exists():
            return Path(cand)
    return Path(os.getenv("MODEL_DIR", "models"))


class State:
    pipeline = None
    meta: dict = {}
    ref_sample: pd.DataFrame | None = None
    window: deque = deque(maxlen=WINDOW)  # recent served rows (features + prediction) for drift
    lock = threading.Lock()


def load_model() -> None:
    model_dir = resolve_model_dir()
    State.pipeline, State.meta, State.ref_sample = load_artifacts(model_dir)
    log.info("Loaded model from %s (%s)", model_dir, State.meta.get("algorithm"))
    if LOG_PATH.exists():  # warm the drift window after a restart
        with LOG_PATH.open() as fh:
            for line in list(fh)[-WINDOW:]:
                try:
                    State.window.append(json.loads(line)["row"])
                except (ValueError, KeyError):
                    continue


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_model()
    except Exception as exc:  # keep the process alive so /health can report the reason
        log.error("Model failed to load: %s", exc)
    yield


app = FastAPI(title="Student Mental-Health Score API", version="1.0.0", lifespan=lifespan,
              description="Predicts a student's Mental_Health_Score (regression) from social-media & lifestyle features.")


def model_version() -> str:
    reg = State.meta.get("registry", {})
    return f"{reg.get('name', State.meta.get('model_name', 'model'))}:v{reg.get('version', '?')}"


def band(score: float) -> str:
    return "Low" if score < 5 else "Moderate" if score < 6.5 else "Good" if score < 8 else "Excellent"


def require_model() -> None:
    if State.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `make train` or mount a model directory.")


def predict_rows(raw: pd.DataFrame, endpoint: str, source: str = "api", record: bool = True) -> tuple[np.ndarray, float]:
    """Single code path for every endpoint: features -> predict -> metrics -> prediction log."""
    require_model()
    start = time.perf_counter()
    try:
        X = prepare_features(raw[RAW_INPUT_COLS], State.meta["top_countries"])
        preds = np.clip(State.pipeline.predict(X), 0, 10)
    except Exception as exc:
        ERRORS.labels(endpoint).inc()
        raise HTTPException(status_code=422, detail=f"Prediction failed: {exc}") from exc
    elapsed = time.perf_counter() - start
    LATENCY.observe(elapsed)
    PREDICTIONS.labels(endpoint).inc(len(preds))
    for p in preds:
        PRED_VALUE.observe(float(p))
    if record:
        _record(X, preds, source)
    return preds, elapsed * 1000


def _record(X: pd.DataFrame, preds: np.ndarray, source: str) -> None:
    rows = X.assign(prediction=np.round(preds, 4)).to_dict(orient="records")
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with State.lock:
        State.window.extend(rows)
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with LOG_PATH.open("a") as fh:
                for r in rows:
                    fh.write(json.dumps({"ts": ts, "source": source, "row": r}) + "\n")
        except OSError as exc:
            log.warning("could not write prediction log: %s", exc)
    if LOG_TO_STDOUT:  # SageMaker -> CloudWatch Logs (query with Logs Insights)
        for r in rows:
            print(json.dumps({"event": "prediction", "ts": ts, "row": r}), flush=True)


def to_frame(items: list[StudentFeatures]) -> pd.DataFrame:
    return pd.DataFrame([i.model_dump() for i in items])


# ---- health -------------------------------------------------------------------------------------
@app.get("/health", tags=["ops"])
def health():
    if State.pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "model": model_version(), "algorithm": State.meta.get("algorithm")}


@app.get("/ping", tags=["sagemaker"])
def ping():
    """SageMaker health check: must return 200 when ready."""
    if State.pipeline is None:
        raise HTTPException(status_code=500, detail="model not loaded")
    return Response(status_code=200)


# ---- inference ----------------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(payload: StudentFeatures):
    preds, ms = predict_rows(to_frame([payload]), "predict")
    score = round(float(preds[0]), 2)
    return PredictionResponse(prediction=score, band=band(score), model_version=model_version(), latency_ms=round(ms, 2))


@app.post("/predict/batch", response_model=BatchResponse, tags=["inference"])
def predict_batch(payload: BatchRequest, log: bool = Query(default=True, description="False for what-if/analysis calls")):
    preds, ms = predict_rows(to_frame(payload.instances), "batch", record=log)
    return BatchResponse(predictions=[round(float(p), 3) for p in preds], model_version=model_version(),
                         latency_ms=round(ms, 2))


@app.post("/invocations", tags=["sagemaker"])
async def invocations(request: Request):
    """SageMaker contract. Accepts one record, a list of records, or {"instances": [...]}."""
    try:
        body = await request.json()
        records = body["instances"] if isinstance(body, dict) and "instances" in body else body
        records = [records] if isinstance(records, dict) else records
        items = [StudentFeatures.model_validate(r) for r in records]
    except (ValidationError, ValueError, KeyError, TypeError) as exc:
        ERRORS.labels("invocations").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    preds, _ = predict_rows(to_frame(items), "invocations")
    return {"predictions": [round(float(p), 3) for p in preds]}


# ---- model + monitoring -------------------------------------------------------------------------
@app.get("/model/info", tags=["ops"])
def model_info():
    require_model()
    ref_num = State.meta["reference"]["numeric"]
    return {
        "model_version": model_version(),
        "algorithm": State.meta.get("algorithm"),
        "trained_at": State.meta.get("trained_at"),
        "metrics": State.meta.get("metrics"),
        "candidates": State.meta.get("candidates", []),
        "registry": State.meta.get("registry"),
        "features": FEATURE_COLS,
        "reference_means": {k: round(v["mean"], 3) for k, v in ref_num.items()},
        "reference_ranges": {k: [v["min"], v["max"]] for k, v in ref_num.items()},
    }


def _current_window(n: int) -> pd.DataFrame:
    with State.lock:
        return pd.DataFrame(list(State.window)[-n:])


@app.get("/monitoring/drift", tags=["monitoring"])
def drift_report(window: int = Query(default=1000, ge=30, le=5000)):
    require_model()
    cur = _current_window(window)
    return compute_drift(State.meta["reference"], cur, State.ref_sample, with_ks=True) if len(cur) else \
        {"ready": False, "n_current": 0, "overall": "insufficient_data", "features": {},
         "message": "No predictions logged yet."}


@app.post("/monitoring/simulate", tags=["monitoring"])
def simulate(req: SimulationRequest):
    """Demo helper: replay held-out rows (normal) or perturbed rows (drift) through the real predict path."""
    if not ENABLE_SIMULATION:
        raise HTTPException(status_code=403, detail="simulation disabled")
    require_model()
    if State.ref_sample is None:
        raise HTTPException(status_code=409, detail="no reference sample available")
    rng = np.random.default_rng(int(time.time()))
    sample = State.ref_sample.sample(req.n, replace=True, random_state=int(rng.integers(10**9))).reset_index(drop=True)
    raw = sample.drop(columns=["prediction"]).rename(columns={"Grouped_country": "Country"})
    if req.drift:
        raw["Avg_Daily_Usage_Hours"] = (raw["Avg_Daily_Usage_Hours"] + 2.5).clip(upper=12)
        raw["Sleep_Hours_Per_Night"] = (raw["Sleep_Hours_Per_Night"] - 1.5).clip(lower=3)
        raw["Daily_Unlocks"] = raw["Daily_Unlocks"] + 70
        raw["Stress_Level"] = np.where(rng.random(req.n) < 0.7, "Very High", raw["Stress_Level"])
        raw["Most_Used_Platform"] = np.where(rng.random(req.n) < 0.6, "TikTok", raw["Most_Used_Platform"])
    preds, _ = predict_rows(raw, "simulation", source="drift_sim" if req.drift else "normal_sim")
    return {"simulated": req.n, "drift": req.drift, "mean_prediction": round(float(preds.mean()), 3)}


@app.delete("/monitoring/reset", tags=["monitoring"])
def reset_window():
    with State.lock:
        State.window.clear()
        LOG_PATH.unlink(missing_ok=True)
    return {"status": "cleared"}


@app.get("/metrics", tags=["monitoring"])
def metrics():
    """Prometheus endpoint. Drift gauges are refreshed on every scrape (cheap: PSI only, no KS)."""
    if State.pipeline is not None:
        report = compute_drift(State.meta["reference"], _current_window(WINDOW), with_ks=False)
        DRIFT_LEVEL.set({"stable": 0, "warning": 1, "drift": 2}.get(report["overall"], 3))
        for feat, item in report["features"].items():
            FEATURE_PSI.labels(feat).set(item["psi"])
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
