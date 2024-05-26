import boto3
from typing import Optional


def get_model_ids(client: Optional = None):
    if not client:
        client = boto3.client("bedrock")
    models = client.list_foundation_models()
    model_ids = [model["modelId"] for model in models["modelSummaries"]]
    return model_ids
