"""Create or update a SageMaker endpoint from (ECR image + model.tar.gz in S3). Idempotent: safe to re-run in CI.

Serverless (default): pay-per-request, ideal for a portfolio project. Real-time: `--mode realtime` also enables
Data Capture (request/response payloads -> S3) that SageMaker Model Monitor can analyse.
"""
from __future__ import annotations

import argparse
import sys
import time

import boto3
from botocore.exceptions import ClientError


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--endpoint-name", required=True)
    p.add_argument("--image-uri", required=True)
    p.add_argument("--model-data", required=True, help="s3://bucket/key/model.tar.gz")
    p.add_argument("--role-arn", required=True)
    p.add_argument("--region", default=None)
    p.add_argument("--mode", choices=["serverless", "realtime"], default="serverless")
    p.add_argument("--memory-mb", type=int, default=2048)
    p.add_argument("--max-concurrency", type=int, default=5)
    p.add_argument("--instance-type", default="ml.t2.medium")
    p.add_argument("--capture-bucket", default=None, help="realtime only: enable Data Capture to this bucket")
    a = p.parse_args()

    sm = boto3.client("sagemaker", region_name=a.region)
    ts = time.strftime("%Y%m%d%H%M%S")
    model_name = f"{a.endpoint_name}-{ts}"

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=a.role_arn,
        PrimaryContainer={"Image": a.image_uri, "ModelDataUrl": a.model_data,
                          "Environment": {"LOG_TO_STDOUT": "true", "ENABLE_SIMULATION": "false", "MODEL_DIR": "/opt/ml/model"}},
    )
    variant = {"VariantName": "AllTraffic", "ModelName": model_name}
    config_kwargs = {}
    if a.mode == "serverless":
        variant["ServerlessConfig"] = {"MemorySizeInMB": a.memory_mb, "MaxConcurrency": a.max_concurrency}
    else:
        variant.update({"InstanceType": a.instance_type, "InitialInstanceCount": 1})
        if a.capture_bucket:
            config_kwargs["DataCaptureConfig"] = {
                "EnableCapture": True, "InitialSamplingPercentage": 100,
                "DestinationS3Uri": f"s3://{a.capture_bucket}/data-capture",
                "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
                "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]}}
    sm.create_endpoint_config(EndpointConfigName=model_name, ProductionVariants=[variant], **config_kwargs)

    try:
        sm.describe_endpoint(EndpointName=a.endpoint_name)
        sm.update_endpoint(EndpointName=a.endpoint_name, EndpointConfigName=model_name)  # zero-downtime swap
        print(f"Updating endpoint {a.endpoint_name} -> {model_name}")
    except ClientError as exc:
        if "Could not find endpoint" not in str(exc):
            raise
        sm.create_endpoint(EndpointName=a.endpoint_name, EndpointConfigName=model_name)
        print(f"Creating endpoint {a.endpoint_name}")

    sm.get_waiter("endpoint_in_service").wait(EndpointName=a.endpoint_name, WaiterConfig={"Delay": 20, "MaxAttempts": 90})
    status = sm.describe_endpoint(EndpointName=a.endpoint_name)["EndpointStatus"]
    print(f"Endpoint status: {status}")
    return 0 if status == "InService" else 1


if __name__ == "__main__":
    sys.exit(main())
