"""Object storage client — abstracts S3/Azure Blob/local filesystem."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol


class ObjectStore(Protocol):
    """Protocol for artifact storage backends."""

    async def put(self, key: str, data: bytes) -> str: ...
    async def get(self, key: str) -> bytes: ...
    async def exists(self, key: str) -> bool: ...


class LocalObjectStore:
    """Local filesystem-based artifact store (dev / testing)."""

    def __init__(self, base_path: str = "./artifacts"):
        self.base = Path(base_path)
        self.base.mkdir(parents=True, exist_ok=True)

    async def put(self, key: str, data: bytes) -> str:
        path = self.base / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return f"file://{path.resolve()}"

    async def get(self, key: str) -> bytes:
        path = self.base / key
        return path.read_bytes()

    async def exists(self, key: str) -> bool:
        return (self.base / key).exists()


class S3ObjectStore:
    """S3-compatible artifact store (production)."""

    def __init__(self, bucket: str, region: str = "us-east-1"):
        import aioboto3

        self.bucket = bucket
        self.region = region
        self._session = aioboto3.Session()

    async def put(self, key: str, data: bytes) -> str:
        async with self._session.client("s3", region_name=self.region) as s3:
            await s3.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    async def get(self, key: str) -> bytes:
        async with self._session.client("s3", region_name=self.region) as s3:
            response = await s3.get_object(Bucket=self.bucket, Key=key)
            return await response["Body"].read()

    async def exists(self, key: str) -> bool:
        async with self._session.client("s3", region_name=self.region) as s3:
            try:
                await s3.head_object(Bucket=self.bucket, Key=key)
                return True
            except Exception:
                return False


def create_object_store(store_type: str = "local", **kwargs) -> ObjectStore:
    """Factory for object store backends."""
    if store_type == "s3":
        return S3ObjectStore(
            bucket=kwargs.get("bucket", "evolve-artifacts"),
            region=kwargs.get("region", "us-east-1"),
        )
    return LocalObjectStore(base_path=kwargs.get("base_path", "./artifacts"))
