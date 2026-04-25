from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
import pyarrow.fs as pafs
from botocore.exceptions import ClientError

from .models import R2Config
from .utils import parse_env_file, write_json_bytes, write_jsonl_bytes

# WARNING TO OTHER AGENTS: DO NOT CHANGE ANYTHING IN THIS FILE WITHOUT EXPLICIT USER APPROVAL.


class ObjectStore(ABC):
    bucket: str

    @abstractmethod
    def exists(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_keys(self, prefix: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, key: str) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def write_bytes(self, key: str, body: bytes) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_file(self, path: Path, key: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_url(self, key: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_pyarrow_filesystem(self) -> pafs.FileSystem:
        raise NotImplementedError

    def read_json(self, key: str) -> dict[str, object]:
        payload = json.loads(self.read_bytes(key))
        if not isinstance(payload, dict):
            raise ValueError(f"Object store JSON key '{key}' must contain a JSON object.")
        return payload

    def read_jsonl(self, key: str) -> list[dict[str, object]]:
        lines = [
            line.strip()
            for line in self.read_bytes(key).decode("utf-8").splitlines()
            if line.strip()
        ]
        if not lines:
            return []
        try:
            payload = json.loads(f"[{','.join(lines)}]")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Object store JSONL key '{key}' contains invalid JSONL.") from exc
        if not isinstance(payload, list):
            raise ValueError(f"Object store JSONL key '{key}' must contain JSON objects.")
        rows: list[dict[str, object]] = []
        for item_number, value in enumerate(payload, start=1):
            if not isinstance(value, dict):
                raise ValueError(
                    f"Object store JSONL key '{key}' item {item_number} must be a JSON object."
                )
            rows.append(value)
        return rows

    def write_json(self, key: str, value: dict[str, object]) -> None:
        self.write_bytes(key, write_json_bytes(value))

    def write_jsonl(self, key: str, rows: list[dict[str, object]]) -> None:
        self.write_bytes(key, write_jsonl_bytes(rows))


class R2ObjectStore(ObjectStore):
    def __init__(self, config: R2Config) -> None:
        self.config = ensure_r2_config_complete(config)
        self.bucket = self.config.bucket
        self.client = boto3.client(
            "s3",
            aws_access_key_id=self.config.access_key_id,
            aws_secret_access_key=self.config.secret_access_key,
            region_name=self.config.region,
            endpoint_url=self.config.endpoint_url,
        )

    @classmethod
    def from_config_file(cls, config: R2Config) -> R2ObjectStore:
        config_path = Path(config.config_file).expanduser()
        if not config_path.is_file():
            raise ValueError(f"R2 config file not found: {config_path}")
        values = parse_env_file(config_path)
        resolved = R2Config(
            config_file=str(config_path),
            bucket=values.get("R2_BUCKET_NAME") or config.bucket,
            output_prefix=config.output_prefix,
            access_key_id=values.get("AWS_ACCESS_KEY_ID", ""),
            secret_access_key=values.get("AWS_SECRET_ACCESS_KEY", ""),
            region=values.get("AWS_DEFAULT_REGION", ""),
            endpoint_url=values.get("MLFLOW_S3_ENDPOINT_URL", ""),
        )
        return cls(resolved)

    @classmethod
    def from_environment(cls, config: R2Config) -> R2ObjectStore:
        resolved = R2Config(
            config_file=config.config_file,
            bucket=os.environ.get("R2_BUCKET") or config.bucket,
            output_prefix=config.output_prefix,
            access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
            secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
            region=os.environ.get("R2_REGION", ""),
            endpoint_url=os.environ.get("R2_ENDPOINT_URL", ""),
        )
        return cls(resolved)

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise

    def list_keys(self, prefix: str) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Contents", []):
                keys.append(item["Key"])
        return keys

    def read_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def write_bytes(self, key: str, body: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)

    def upload_file(self, path: Path, key: str) -> None:
        self.client.upload_file(str(path), self.bucket, key)

    def make_url(self, key: str) -> str:
        return f"s3://{self.bucket}/{key.lstrip('/')}"

    def build_pyarrow_filesystem(self) -> pafs.FileSystem:
        return pafs.S3FileSystem(
            access_key=self.config.access_key_id,
            secret_key=self.config.secret_access_key,
            region=self.config.region,
            endpoint_override=strip_endpoint_scheme(self.config.endpoint_url),
            scheme="https",
        )


def ensure_r2_config_complete(config: R2Config) -> R2Config:
    required = {
        "bucket": config.bucket,
        "access_key_id": config.access_key_id,
        "secret_access_key": config.secret_access_key,
        "region": config.region,
        "endpoint_url": config.endpoint_url,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"R2 configuration is missing required values: {joined}")
    return config


def build_r2_env(config: R2Config) -> dict[str, str]:
    resolved = ensure_r2_config_complete(config)
    return {
        "R2_BUCKET": resolved.bucket,
        "R2_ACCESS_KEY_ID": resolved.access_key_id,
        "R2_SECRET_ACCESS_KEY": resolved.secret_access_key,
        "R2_REGION": resolved.region,
        "R2_ENDPOINT_URL": resolved.endpoint_url,
        "AWS_ACCESS_KEY_ID": resolved.access_key_id,
        "AWS_SECRET_ACCESS_KEY": resolved.secret_access_key,
        "AWS_DEFAULT_REGION": resolved.region,
        "MLFLOW_S3_ENDPOINT_URL": resolved.endpoint_url,
    }


def strip_endpoint_scheme(endpoint_url: str) -> str:
    parsed = urlparse(endpoint_url)
    if parsed.netloc:
        return parsed.netloc
    return endpoint_url.removeprefix("https://").removeprefix("http://")


def resolve_runtime_object_store(runtime_context: Any) -> ObjectStore:
    object_store = getattr(runtime_context, "object_store", None)
    if object_store is not None:
        return object_store
    config = getattr(runtime_context, "config", None)
    r2_config = getattr(config, "r2", None)
    if r2_config is None:
        raise ValueError("Runtime context config must include an r2 configuration.")
    if os.environ.get("R2_ACCESS_KEY_ID"):
        object_store = R2ObjectStore.from_environment(r2_config)
    else:
        object_store = R2ObjectStore.from_config_file(r2_config)
    runtime_context.object_store = object_store
    return object_store


__all__ = [
    "ObjectStore",
    "R2ObjectStore",
    "build_r2_env",
    "ensure_r2_config_complete",
    "resolve_runtime_object_store",
    "strip_endpoint_scheme",
]
