"""One client for both deployment targets, chosen by PREDICT_BACKEND=fastapi|sagemaker."""
from __future__ import annotations

import json
import os
import time

import requests


def band(score: float) -> str:
    return "Low" if score < 5 else "Moderate" if score < 6.5 else "Good" if score < 8 else "Excellent"


class Backend:
    def __init__(self) -> None:
        self.mode = os.getenv("PREDICT_BACKEND", "fastapi").lower()
        self.api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
        self.endpoint = os.getenv("SAGEMAKER_ENDPOINT", "student-mental-health-endpoint")
        self.region = os.getenv("AWS_REGION", "us-east-1")
        self._runtime = None
    def _aws_kwargs(self) -> dict:
        try:
            import streamlit as st
            key, secret = st.secrets.get("AWS_ACCESS_KEY_ID"), st.secrets.get("AWS_SECRET_ACCESS_KEY")
            if key and secret:
                return {"aws_access_key_id": key, "aws_secret_access_key": secret}
        except Exception:
            pass
        return {}

    # ---- helpers -------------------------------------------------------------------------------
    def _get(self, path: str, **kw):
        r = requests.get(f"{self.api_url}{path}", timeout=15, **kw)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, body: dict, **kw):
        r = requests.post(f"{self.api_url}{path}", json=body, timeout=60, **kw)
        r.raise_for_status()
        return r.json()

    # def _invoke_sagemaker(self, records: list[dict]) -> list[float]:
    #     if self._runtime is None:
    #         import boto3
    #         self._runtime = boto3.client("sagemaker-runtime", region_name=self.region)
    #     resp = self._runtime.invoke_endpoint(EndpointName=self.endpoint, ContentType="application/json",
    #                                          Body=json.dumps({"instances": records}))
    #     return json.loads(resp["Body"].read())["predictions"]
    def _invoke_sagemaker(self, records: list[dict]) -> list[float]:
        if self._runtime is None:
            import boto3
            self._runtime = boto3.client("sagemaker-runtime", region_name=self.region, **self._aws_kwargs())
        resp = self._runtime.invoke_endpoint(EndpointName=self.endpoint, ContentType="application/json",
                                             Body=json.dumps({"instances": records}))
        return json.loads(resp["Body"].read())["predictions"]

    # ---- public API ----------------------------------------------------------------------------
    @property
    def supports_monitoring(self) -> bool:
        return self.mode == "fastapi"

    # def health(self) -> dict | None:
    #     try:
    #         if self.mode == "sagemaker":
    #             import boto3
    #             status = boto3.client("sagemaker", region_name=self.region).describe_endpoint(
    #                 EndpointName=self.endpoint)["EndpointStatus"]
    #             return {"status": "ok" if status == "InService" else status, "model": f"sagemaker:{self.endpoint}"}
    #         return self._get("/health")
    #     except Exception:
    #         return None
    def health(self) -> dict | None:
        try:
            if self.mode == "sagemaker":
                import boto3
                status = boto3.client("sagemaker", region_name=self.region, **self._aws_kwargs()).describe_endpoint(
                    EndpointName=self.endpoint)["EndpointStatus"]
                return {"status": "ok" if status == "InService" else status, "model": f"sagemaker:{self.endpoint}"}
            return self._get("/health")
        except Exception:
            return None

    def predict(self, payload: dict) -> dict:
        if self.mode == "sagemaker":
            t = time.perf_counter()
            score = round(float(self._invoke_sagemaker([payload])[0]), 2)
            return {"prediction": score, "band": band(score), "model_version": f"sagemaker:{self.endpoint}",
                    "latency_ms": round((time.perf_counter() - t) * 1000, 1)}
        return self._post("/predict", payload)

    def predict_batch(self, payloads: list[dict], log: bool = False) -> list[float]:
        """log=False (default): what-if analysis, excluded from the drift window."""
        if self.mode == "sagemaker":
            return [float(p) for p in self._invoke_sagemaker(payloads)]
        return self._post("/predict/batch", {"instances": payloads}, params={"log": str(log).lower()})["predictions"]

    def info(self) -> dict | None:
        try:
            return self._get("/model/info") if self.mode == "fastapi" else None
        except Exception:
            return None

    def drift(self, window: int = 1000) -> dict | None:
        try:
            return self._get("/monitoring/drift", params={"window": window})
        except Exception:
            return None

    def simulate(self, n: int, drift: bool) -> dict:
        return self._post("/monitoring/simulate", {"n": n, "drift": drift})

    def reset(self) -> None:
        requests.delete(f"{self.api_url}/monitoring/reset", timeout=15)
