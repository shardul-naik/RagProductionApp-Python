import os
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlparse

import boto3


def _storage_location() -> tuple[object, str, str]:
    bucket_url = os.environ["STORAGE_BUCKET_URL"].rstrip("/")
    parsed = urlparse(bucket_url)

    if parsed.scheme == "s3":
        if not parsed.netloc:
            raise ValueError("STORAGE_BUCKET_URL must include an S3 bucket name")
        endpoint_url = ""
        bucket = parsed.netloc
        prefix = parsed.path.strip("/")
    elif parsed.scheme in {"http", "https"} and parsed.netloc:
        path_parts = [part for part in parsed.path.split("/") if part]
        if not path_parts:
            raise ValueError(
                "STORAGE_BUCKET_URL must include the bucket in its URL path"
            )
        endpoint_url = f"{parsed.scheme}://{parsed.netloc}"
        bucket = path_parts[0]
        prefix = "/".join(path_parts[1:])
    else:
        raise ValueError(
            "STORAGE_BUCKET_URL must be an s3:// URL or an HTTPS bucket URL"
        )

    client = boto3.client(
        "s3",
        endpoint_url=endpoint_url or None,
        aws_access_key_id=os.environ["STORAGE_ACCESS_KEY"],
        aws_secret_access_key=os.environ["STORAGE_SECRET_KEY"],
    )
    return client, bucket, prefix


def upload_pdf(file_name: str, file_bytes: bytes) -> str:
    client, bucket, prefix = _storage_location()
    object_name = Path(file_name).name
    object_key = f"{uuid.uuid4()}/{object_name}"
    if prefix:
        object_key = f"{prefix}/{object_key}"

    client.put_object(
        Bucket=bucket,
        Key=object_key,
        Body=file_bytes,
        ContentType="application/pdf",
    )
    return object_key


def download_pdf(object_key: str) -> str:
    client, bucket, _ = _storage_location()
    temporary_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    temporary_file.close()
    try:
        client.download_file(bucket, object_key, temporary_file.name)
    except Exception:
        Path(temporary_file.name).unlink(missing_ok=True)
        raise
    return temporary_file.name
