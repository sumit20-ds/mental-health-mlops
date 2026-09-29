"""Pydantic v2 request/response contracts for the API (also used by SageMaker /invocations)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Platform = Literal["Facebook", "LinkedIn", "Instagram", "Snapchat", "Twitter", "YouTube",
                   "TikTok", "LINE", "KakaoTalk", "VKontakte", "WhatsApp", "WeChat"]


class StudentFeatures(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "Age": 21, "Gender": "Female", "Country": "India", "Academic_Level": "Undergraduate",
        "Most_Used_Platform": "Instagram", "Purpose_Of_Use": "Entertainment",
        "Avg_Daily_Usage_Hours": 4.5, "Daily_Unlocks": 150, "Study_Hours": 3.0,
        "Physical_Activity_Hours": 1.5, "Sleep_Hours_Per_Night": 7.0, "Stress_Level": "Medium"}})

    Age: int = Field(ge=13, le=60)
    Gender: Literal["Male", "Female"]
    Country: str = Field(min_length=1, max_length=60)
    Academic_Level: Literal["High School", "Undergraduate", "Graduate"]
    Most_Used_Platform: Platform
    Purpose_Of_Use: Literal["Networking", "Education", "Entertainment", "News"]
    Avg_Daily_Usage_Hours: float = Field(ge=0, le=24)
    Daily_Unlocks: int = Field(ge=0, le=1000)
    Study_Hours: float = Field(ge=0, le=24)
    Physical_Activity_Hours: float = Field(ge=0, le=24)
    Sleep_Hours_Per_Night: float = Field(ge=0, le=24)
    Stress_Level: Literal["Low", "Medium", "High", "Very High"]


class PredictionResponse(BaseModel):
    prediction: float
    band: str
    model_version: str
    latency_ms: float


class BatchRequest(BaseModel):
    instances: list[StudentFeatures] = Field(min_length=1, max_length=500)


class BatchResponse(BaseModel):
    predictions: list[float]
    model_version: str
    latency_ms: float


class SimulationRequest(BaseModel):
    n: int = Field(default=200, ge=1, le=1000)
    drift: bool = False
