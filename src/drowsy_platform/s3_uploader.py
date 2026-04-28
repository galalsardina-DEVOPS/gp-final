from __future__ import annotations

from pathlib import Path


class S3Uploader:
    def __init__(self, bucket: str, prefix: str) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = None

    def upload_file(self, path: Path) -> str:
        client = self._get_client()
        key = f"{self.prefix}/{path.name}" if self.prefix else path.name
        client.upload_file(str(path), self.bucket, key)
        return f"s3://{self.bucket}/{key}"

    def _get_client(self):
        if self._client is None:
            import boto3

            self._client = boto3.client("s3")
        return self._client

