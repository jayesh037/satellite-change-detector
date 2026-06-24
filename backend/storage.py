"""
Backblaze B2 object storage helpers.

B2 is S3-compatible cloud storage — free 10GB, no card needed.
We use boto3 (AWS S3 SDK) because B2 implements the S3 API.

What we store:
- Inference results (GeoJSON, GeoTIFF) → results/{task_id}/
- COG tiles (TCI, NDVI, NDWI, NDBI)  → tiles/
- ML model checkpoint                  → models/

Pre-signed URLs: since bucket is Private, we generate temporary
URLs (valid 1hr) so frontend can fetch files without exposing credentials.
"""
import os
import boto3
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BUCKET = os.environ.get("B2_BUCKET_NAME", "satchange-outputs")
ENDPOINT = os.environ.get("B2_ENDPOINT", "")


def get_b2_client():
    """
    Create boto3 S3 client pointed at Backblaze B2.
    endpoint_url tells boto3 to use B2 instead of AWS S3.
    """
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        aws_access_key_id=os.environ["B2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["B2_SECRET_ACCESS_KEY"],
        region_name="auto",
    )


def upload_file(local_path: str, b2_key: str) -> str:
    """
    Upload local file to B2.
    b2_key = path inside bucket e.g. 'results/task-123/mask.geojson'
    Returns the b2_key (use get_signed_url to get accessible URL).
    """
    client = get_b2_client()
    local_path = str(local_path)
    logger.info(f"Uploading {local_path} → B2:{b2_key}")
    client.upload_file(local_path, BUCKET, b2_key)
    logger.info(f"Upload complete: {b2_key}")
    return b2_key


def upload_bytes(data: bytes, b2_key: str, content_type: str = "application/octet-stream") -> str:
    """Upload raw bytes to B2. Returns b2_key."""
    client = get_b2_client()
    client.put_object(
        Bucket=BUCKET,
        Key=b2_key,
        Body=data,
        ContentType=content_type
    )
    return b2_key


def download_file(b2_key: str, local_path: str) -> None:
    """Download file from B2 to local disk."""
    client = get_b2_client()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    client.download_file(BUCKET, b2_key, local_path)
    logger.info(f"Downloaded B2:{b2_key} → {local_path}")


def get_signed_url(b2_key: str, expires_in: int = 3600) -> str:
    """
    Generate a pre-signed URL for a private file.
    
    Pre-signed URL = temporary URL valid for `expires_in` seconds (default 1hr).
    Anyone with this URL can download the file during that window.
    Used so frontend can fetch results without exposing API credentials.
    
    This is the industry standard pattern (used by AWS, Dropbox, etc).
    """
    client = get_b2_client()
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": b2_key},
        ExpiresIn=expires_in,
    )
    return url


def delete_file(b2_key: str) -> None:
    """Delete file from B2."""
    client = get_b2_client()
    client.delete_object(Bucket=BUCKET, Key=b2_key)


def file_exists(b2_key: str) -> bool:
    """Check if file exists in B2 using list_objects (head_object gives 403 on B2)."""
    try:
        client = get_b2_client()
        response = client.list_objects_v2(Bucket=BUCKET, Prefix=b2_key, MaxKeys=1)
        return response.get("KeyCount", 0) > 0
    except Exception:
        return False


def list_files(prefix: str) -> list[str]:
    """List all files under a prefix in B2."""
    client = get_b2_client()
    response = client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]
