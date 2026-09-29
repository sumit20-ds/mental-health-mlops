from src.artifacts import load_artifacts
from src.drift import compute_drift


def test_no_drift_on_reference_data(model_dir):
    _, meta, ref = load_artifacts(model_dir)
    report = compute_drift(meta["reference"], ref, ref)
    assert report["overall"] == "stable"
    assert max(f["psi"] for f in report["features"].values()) < 0.05


def test_shifted_data_is_flagged(model_dir):
    _, meta, ref = load_artifacts(model_dir)
    cur = ref.copy()
    cur["Avg_Daily_Usage_Hours"] += 3
    cur["Stress_Level"] = "Very High"
    report = compute_drift(meta["reference"], cur, ref)
    assert report["overall"] == "drift"
    assert report["features"]["Avg_Daily_Usage_Hours"]["status"] == "drift"
    assert report["features"]["Stress_Level"]["status"] == "drift"


def test_not_enough_data_is_reported(model_dir):
    _, meta, ref = load_artifacts(model_dir)
    assert compute_drift(meta["reference"], ref.head(5))["ready"] is False
