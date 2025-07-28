"""
Common utilities for CQC Rating Predictor ML System.

This module provides shared utilities used across the project including:
- Date/time helpers
- Data validation functions
- Logging configuration
- GCP helper functions
- Common constants
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import re
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import secretmanager


# ==================== Constants ====================

# CQC API Constants
CQC_API_BASE_URL = "https://api.cqc.org.uk/public/v1"
CQC_API_PROVIDERS_ENDPOINT = f"{CQC_API_BASE_URL}/providers"
CQC_API_LOCATIONS_ENDPOINT = f"{CQC_API_BASE_URL}/locations"

# Rating Values
CQC_RATINGS = ["Outstanding", "Good", "Requires improvement", "Inadequate", "Not yet rated"]
RATING_SCORES = {
    "Outstanding": 4,
    "Good": 3,
    "Requires improvement": 2,
    "Inadequate": 1,
    "Not yet rated": 0
}

# GCP Resource Naming
GCS_RAW_DATA_PREFIX = "raw-data"
GCS_PROCESSED_DATA_PREFIX = "processed-data"
GCS_MODEL_ARTIFACTS_PREFIX = "model-artifacts"

# BigQuery Dataset Names
BQ_RAW_DATASET = "cqc_raw"
BQ_PROCESSED_DATASET = "cqc_processed"
BQ_ML_DATASET = "cqc_ml"

# Vertex AI Constants
VERTEX_AI_PIPELINE_ROOT = "gs://{bucket}/vertex-ai/pipelines"
VERTEX_AI_MODEL_REGISTRY = "cqc-rating-predictor"


# ==================== Logging Configuration ====================

def setup_logging(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration with structured logging for GCP.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler with formatting
    handler = logging.StreamHandler()
    handler.setLevel(level)
    
    # Create formatter - using structured format for GCP
    formatter = logging.Formatter(
        '{"timestamp": "%(asctime)s", "severity": "%(levelname)s", '
        '"logger": "%(name)s", "message": "%(message)s", '
        '"function": "%(funcName)s", "line": %(lineno)d}'
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# ==================== Date/Time Helpers ====================

def get_utc_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def parse_iso_timestamp(timestamp_str: str) -> datetime:
    """
    Parse ISO format timestamp string to datetime object.
    
    Args:
        timestamp_str: ISO format timestamp string
        
    Returns:
        datetime object
    """
    # Handle different ISO formats
    if timestamp_str.endswith('Z'):
        timestamp_str = timestamp_str[:-1] + '+00:00'
    return datetime.fromisoformat(timestamp_str)


def format_date_for_filename(date: datetime = None) -> str:
    """
    Format date for use in filenames (YYYYMMDD format).
    
    Args:
        date: datetime object (defaults to current date)
        
    Returns:
        Formatted date string
    """
    if date is None:
        date = datetime.now(timezone.utc)
    return date.strftime("%Y%m%d")


def format_datetime_for_filename(dt: datetime = None) -> str:
    """
    Format datetime for use in filenames (YYYYMMDD_HHMMSS format).
    
    Args:
        dt: datetime object (defaults to current datetime)
        
    Returns:
        Formatted datetime string
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y%m%d_%H%M%S")


# ==================== Data Validation Functions ====================

def validate_cqc_provider_id(provider_id: str) -> bool:
    """
    Validate CQC provider ID format.
    
    Args:
        provider_id: Provider ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    # CQC provider IDs are typically 1-6 characters, alphanumeric
    pattern = r'^[A-Z0-9]{1,6}$'
    return bool(re.match(pattern, str(provider_id).upper()))


def validate_cqc_location_id(location_id: str) -> bool:
    """
    Validate CQC location ID format.
    
    Args:
        location_id: Location ID to validate
        
    Returns:
        True if valid, False otherwise
    """
    # CQC location IDs are typically longer alphanumeric strings
    pattern = r'^[A-Z0-9\-]{1,20}$'
    return bool(re.match(pattern, str(location_id).upper()))


def validate_postcode(postcode: str) -> bool:
    """
    Validate UK postcode format.
    
    Args:
        postcode: Postcode to validate
        
    Returns:
        True if valid, False otherwise
    """
    # UK postcode regex pattern
    pattern = r'^[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}$'
    return bool(re.match(pattern, str(postcode).upper().strip()))


def validate_rating(rating: str) -> bool:
    """
    Validate CQC rating value.
    
    Args:
        rating: Rating to validate
        
    Returns:
        True if valid, False otherwise
    """
    return rating in CQC_RATINGS


def clean_text_field(text: Optional[str]) -> str:
    """
    Clean and normalize text fields.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = " ".join(str(text).split())
    
    # Remove control characters
    text = "".join(char for char in text if ord(char) >= 32 or char == '\n')
    
    return text.strip()


# ==================== GCP Helper Functions ====================

def get_gcs_client(project_id: Optional[str] = None) -> storage.Client:
    """
    Get Google Cloud Storage client.
    
    Args:
        project_id: GCP project ID (optional)
        
    Returns:
        Storage client instance
    """
    return storage.Client(project=project_id)


def get_bigquery_client(project_id: Optional[str] = None) -> bigquery.Client:
    """
    Get BigQuery client.
    
    Args:
        project_id: GCP project ID (optional)
        
    Returns:
        BigQuery client instance
    """
    return bigquery.Client(project=project_id)


def get_secret(secret_id: str, project_id: str, version: str = "latest") -> str:
    """
    Retrieve secret from Google Secret Manager.
    
    Args:
        secret_id: Secret identifier
        project_id: GCP project ID
        version: Secret version (default: "latest")
        
    Returns:
        Secret value as string
    """
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def upload_to_gcs(
    bucket_name: str,
    blob_name: str,
    data: Any,
    content_type: str = "application/json"
) -> str:
    """
    Upload data to Google Cloud Storage.
    
    Args:
        bucket_name: GCS bucket name
        blob_name: Object name in bucket
        data: Data to upload (dict, string, or bytes)
        content_type: MIME type of the data
        
    Returns:
        GCS URI of uploaded object
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Convert data to bytes if needed
    if isinstance(data, dict):
        data = json.dumps(data, indent=2)
    if isinstance(data, str):
        data = data.encode('utf-8')
    
    blob.upload_from_string(data, content_type=content_type)
    
    return f"gs://{bucket_name}/{blob_name}"


def download_from_gcs(bucket_name: str, blob_name: str) -> bytes:
    """
    Download data from Google Cloud Storage.
    
    Args:
        bucket_name: GCS bucket name
        blob_name: Object name in bucket
        
    Returns:
        Downloaded data as bytes
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    return blob.download_as_bytes()


def list_gcs_objects(
    bucket_name: str,
    prefix: Optional[str] = None,
    delimiter: Optional[str] = None
) -> List[str]:
    """
    List objects in GCS bucket.
    
    Args:
        bucket_name: GCS bucket name
        prefix: Filter results to objects with this prefix
        delimiter: Group results by delimiter
        
    Returns:
        List of object names
    """
    client = get_gcs_client()
    blobs = client.list_blobs(bucket_name, prefix=prefix, delimiter=delimiter)
    
    return [blob.name for blob in blobs]


def parse_gcs_uri(gcs_uri: str) -> tuple[str, str]:
    """
    Parse GCS URI into bucket and object name.
    
    Args:
        gcs_uri: GCS URI (gs://bucket/object)
        
    Returns:
        Tuple of (bucket_name, object_name)
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    
    path = gcs_uri[5:]  # Remove 'gs://'
    parts = path.split("/", 1)
    
    if len(parts) != 2:
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    
    return parts[0], parts[1]


def build_gcs_uri(bucket_name: str, object_name: str) -> str:
    """
    Build GCS URI from bucket and object name.
    
    Args:
        bucket_name: GCS bucket name
        object_name: Object name
        
    Returns:
        GCS URI
    """
    return f"gs://{bucket_name}/{object_name}"


# ==================== Error Handling ====================

class CQCDataError(Exception):
    """Base exception for CQC data processing errors."""
    pass


class ValidationError(CQCDataError):
    """Exception raised for data validation errors."""
    pass


class GCPError(CQCDataError):
    """Exception raised for GCP-related errors."""
    pass


# ==================== Utility Functions ====================

def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """
    Split a list into batches.
    
    Args:
        items: List to batch
        batch_size: Size of each batch
        
    Returns:
        List of batches
    """
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for flattened keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    Safely get nested dictionary value.
    
    Args:
        d: Dictionary
        path: Dot-separated path (e.g., "location.address.postcode")
        default: Default value if path not found
        
    Returns:
        Value at path or default
    """
    keys = path.split('.')
    value = d
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value