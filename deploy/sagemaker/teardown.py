"""Delete the endpoint + every endpoint-config/model created by deploy.py (stops all SageMaker charges)."""
from __future__ import annotations

import argparse

import boto3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--region", default=None)
    a = p.parse_args()
    sm = boto3.client("sagemaker", region_name=a.region)
    try:
        sm.delete_endpoint(EndpointName=a.endpoint_name)
        print("deleted endpoint", a.endpoint_name)
    except sm.exceptions.ClientError as exc:
        print("endpoint:", exc)
    for cfg in sm.list_endpoint_configs(NameContains=a.endpoint_name, MaxResults=100)["EndpointConfigs"]:
        sm.delete_endpoint_config(EndpointConfigName=cfg["EndpointConfigName"])
        print("deleted config", cfg["EndpointConfigName"])
    for m in sm.list_models(NameContains=a.endpoint_name, MaxResults=100)["Models"]:
        sm.delete_model(ModelName=m["ModelName"])
        print("deleted model", m["ModelName"])


if __name__ == "__main__":
    main()
