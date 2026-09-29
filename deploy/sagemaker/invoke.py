"""Smoke test used by CD (and by you): invoke the live endpoint and sanity-check the answer."""
from __future__ import annotations

import argparse
import json
import sys

import boto3

SAMPLE = {"Age": 21, "Gender": "Female", "Country": "India", "Academic_Level": "Undergraduate",
          "Most_Used_Platform": "Instagram", "Purpose_Of_Use": "Entertainment", "Avg_Daily_Usage_Hours": 4.5,
          "Daily_Unlocks": 150, "Study_Hours": 3.0, "Physical_Activity_Hours": 1.5,
          "Sleep_Hours_Per_Night": 7.0, "Stress_Level": "Medium"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--region", default=None)
    a = p.parse_args()
    rt = boto3.client("sagemaker-runtime", region_name=a.region)
    resp = rt.invoke_endpoint(EndpointName=a.endpoint_name, ContentType="application/json", Body=json.dumps(SAMPLE))
    pred = json.loads(resp["Body"].read())["predictions"][0]
    print(f"prediction = {pred}")
    if not 0 <= pred <= 10:
        print("prediction out of range", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
