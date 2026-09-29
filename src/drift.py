"""Lightweight data/prediction drift: PSI (all columns) + KS test (numeric columns).

PSI rule of thumb: < 0.1 stable | 0.1-0.25 moderate shift | > 0.25 significant drift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PSI_WARN = 0.1
PSI_ALERT = 0.25
EPS = 1e-4


def _status(psi: float) -> str:
    return "drift" if psi >= PSI_ALERT else "warning" if psi >= PSI_WARN else "stable"


def _psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _bins(edges: list[float]) -> np.ndarray:
    return np.array([-np.inf, *edges, np.inf])


def build_reference(ref_df: pd.DataFrame, numeric: list[str], categorical: list[str], n_bins: int = 10) -> dict:
    """Summarise the reference (held-out) data once, at training time."""
    ref = {"n": int(len(ref_df)), "numeric": {}, "categorical": {}}
    for col in numeric:
        s = ref_df[col].astype(float)
        edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1))[1:-1]).tolist()
        counts, _ = np.histogram(s, bins=_bins(edges))
        ref["numeric"][col] = {
            "edges": edges,
            "props": (counts / counts.sum()).tolist(),
            "mean": float(s.mean()),
            "std": float(s.std()),
            "min": float(s.min()),
            "max": float(s.max()),
        }
    for col in categorical:
        props = ref_df[col].value_counts(normalize=True)
        ref["categorical"][col] = {"props": {str(k): float(v) for k, v in props.items()}}
    return ref


def compute_drift(reference: dict, current: pd.DataFrame, ref_sample: pd.DataFrame | None = None,
                  min_samples: int = 30, with_ks: bool = True) -> dict:
    n = int(len(current))
    if n < min_samples:
        return {"ready": False, "n_current": n, "min_samples": min_samples, "overall": "insufficient_data",
                "message": f"Need at least {min_samples} logged predictions (have {n}).", "features": {}}
    features: dict[str, dict] = {}
    for col, r in reference["numeric"].items():
        if col not in current:
            continue
        s = current[col].astype(float).dropna()
        counts, _ = np.histogram(s, bins=_bins(r["edges"]))
        psi = _psi(np.array(r["props"]), counts / max(counts.sum(), 1))
        item = {"type": "numeric", "psi": round(psi, 4), "status": _status(psi),
                "ref_mean": round(r["mean"], 3), "cur_mean": round(float(s.mean()), 3)}
        if with_ks and ref_sample is not None and col in ref_sample:
            item["ks_p_value"] = round(float(stats.ks_2samp(ref_sample[col].astype(float), s).pvalue), 4)
        features[col] = item
    for col, r in reference["categorical"].items():
        if col not in current:
            continue
        cur_props = current[col].astype(str).value_counts(normalize=True).to_dict()
        cats = sorted(set(r["props"]) | set(cur_props))
        psi = _psi(np.array([r["props"].get(c, 0.0) for c in cats]), np.array([cur_props.get(c, 0.0) for c in cats]))
        features[col] = {"type": "categorical", "psi": round(psi, 4), "status": _status(psi)}
    statuses = {f["status"] for f in features.values()}
    overall = "drift" if "drift" in statuses else "warning" if "warning" in statuses else "stable"
    return {"ready": True, "n_current": n, "overall": overall, "features": features,
            "thresholds": {"warning": PSI_WARN, "drift": PSI_ALERT}}
