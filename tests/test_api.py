import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

VALID = {"Age": 21, "Gender": "Female", "Country": "India", "Academic_Level": "Undergraduate",
         "Most_Used_Platform": "Instagram", "Purpose_Of_Use": "Entertainment", "Avg_Daily_Usage_Hours": 4.5,
         "Daily_Unlocks": 150, "Study_Hours": 3.0, "Physical_Activity_Hours": 1.5,
         "Sleep_Hours_Per_Night": 7.0, "Stress_Level": "Medium"}


@pytest.fixture(scope="module")
def client(model_dir, tmp_path_factory):
    import os
    os.environ["MODEL_DIR"] = str(model_dir)
    os.environ["PREDICTION_LOG_PATH"] = str(tmp_path_factory.mktemp("logs") / "p.jsonl")
    from fastapi.testclient import TestClient

    from serving.app import app
    with TestClient(app) as c:  # runs lifespan -> loads the model
        yield c


def test_health_and_ping(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ping").status_code == 200


def test_predict(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["prediction"] <= 10 and body["band"] in {"Low", "Moderate", "Good", "Excellent"}


def test_higher_stress_and_less_sleep_lower_the_score(client):
    worse = {**VALID, "Stress_Level": "Very High", "Sleep_Hours_Per_Night": 4.0, "Avg_Daily_Usage_Hours": 8.0}
    assert client.post("/predict", json=worse).json()["prediction"] < client.post("/predict", json=VALID).json()["prediction"]


def test_validation_rejects_bad_input(client):
    assert client.post("/predict", json={**VALID, "Stress_Level": "Extreme"}).status_code == 422
    assert client.post("/predict", json={**VALID, "Sleep_Hours_Per_Night": 99}).status_code == 422


def test_batch_and_sagemaker_contract(client):
    assert len(client.post("/predict/batch", json={"instances": [VALID, VALID]}).json()["predictions"]) == 2
    assert len(client.post("/invocations", json=VALID).json()["predictions"]) == 1
    assert len(client.post("/invocations", json={"instances": [VALID] * 3}).json()["predictions"]) == 3
    assert client.post("/invocations", json={"nope": 1}).status_code == 400


def test_monitoring_flow(client):
    client.delete("/monitoring/reset")
    assert client.get("/monitoring/drift").json()["ready"] is False
    client.post("/monitoring/simulate", json={"n": 200, "drift": False})
    assert client.get("/monitoring/drift").json()["overall"] in {"stable", "warning"}
    client.post("/monitoring/simulate", json={"n": 400, "drift": True})
    assert client.get("/monitoring/drift").json()["overall"] in {"warning", "drift"}
    metrics = client.get("/metrics").text
    assert "mh_predictions_total" in metrics and "mh_feature_psi" in metrics
