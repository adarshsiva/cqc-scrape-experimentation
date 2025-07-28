# Google Cloud Storage (GCS) Documentation

## Overview
Google Cloud Storage is a RESTful online file storage web service for storing and accessing data on Google Cloud Platform infrastructure. It provides object storage with global edge-caching, integrated site hosting capabilities, and a robust API.

## Installation
```bash
pip install google-cloud-storage

# With OpenTelemetry tracing support
pip install google-cloud-storage[tracing]
```

## Basic Setup

### Client Initialization
```python
from google.cloud import storage

# Instantiate a client
storage_client = storage.Client()
```

## Bucket Operations

### Creating a Bucket
```python
from google.cloud import storage

storage_client = storage.Client()
bucket_name = "my-new-bucket"

# Create the new bucket
bucket = storage_client.create_bucket(bucket_name)
print(f"Bucket {bucket.name} created.")
```

### Creating Bucket with Specific Configuration
```python
# With storage class and location
python storage_create_bucket_class_location.py <BUCKET_NAME>

# With dual-region configuration
python storage_create_bucket_dual_region.py <BUCKET_NAME> <LOCATION> <REGION_1> <REGION_2>

# With turbo replication
python storage_create_bucket_turbo_replication.py <BUCKET_NAME>
```

### Bucket Management
```python
# List all buckets
buckets = storage_client.list_buckets()

# Get specific bucket
bucket = storage_client.get_bucket(bucket_name)

# Delete a bucket
python storage_delete_bucket.py <BUCKET_NAME>
```

### Bucket Metadata and Configuration
```python
# Get bucket metadata
python storage_get_bucket_metadata.py <BUCKET_NAME>

# Set bucket labels
python storage_add_bucket_label.py <BUCKET_NAME>

# Remove bucket labels
python storage_remove_bucket_label.py <BUCKET_NAME>
```

## File Operations

### Uploading Files
```python
# Upload a file
python storage_upload_file.py <BUCKET_NAME> <SOURCE_FILE_NAME> <DESTINATION_BLOB_NAME>

# Upload with encryption
python storage_upload_encrypted_file.py <BUCKET_NAME> <SOURCE_FILE_NAME> <DESTINATION_BLOB_NAME> <BASE64_ENCRYPTION_KEY>

# Upload from memory
python storage_upload_from_memory.py <BUCKET_NAME> <CONTENTS> <DESTINATION_BLOB_NAME>

# Upload with KMS key
python storage_upload_with_kms_key.py <BUCKET_NAME> <SOURCE_FILE_NAME> <DESTINATION_BLOB_NAME> <KMS_KEY_NAME>
```

### Downloading Files
```python
# Download a file
python storage_download_file.py <BUCKET_NAME> <SOURCE_BLOB_NAME> <DESTINATION_FILE_NAME>

# Download public file
python storage_download_public_file.py <BUCKET_NAME> <SOURCE_BLOB_NAME> <DESTINATION_FILE_NAME>

# Download into memory
python storage_download_into_memory.py <BUCKET_NAME> <BLOB_NAME>

# Download with requester pays
python storage_download_file_requester_pays.py <BUCKET_NAME> <PROJECT_ID> <SOURCE_BLOB_NAME> <DESTINATION_FILE_NAME>

# Download byte range
python storage_download_byte_range.py <BUCKET_NAME> <SOURCE_BLOB_NAME> <START_BYTE> <END_BYTE> <DESTINATION_FILE_NAME>
```

### File Management
```python
# List files
python storage_list_files.py <BUCKET_NAME>

# List files with prefix
python storage_list_files_with_prefix.py <BUCKET_NAME> <PREFIX>

# Copy file
python storage_copy_file.py <BUCKET_NAME> <BLOB_NAME> <DESTINATION_BUCKET_NAME> <DESTINATION_BLOB_NAME>

# Move file
python storage_move_file.py <BUCKET_NAME> <BLOB_NAME> <DESTINATION_BUCKET_NAME> <DESTINATION_BLOB_NAME>

# Rename file
python storage_rename_file.py <BUCKET_NAME> <BLOB_NAME> <NEW_NAME>

# Delete file
python storage_delete_file.py <BUCKET_NAME> <BLOB_NAME>
```

### File Metadata
```python
# Set metadata
python storage_set_metadata.py <BUCKET_NAME> <BLOB_NAME>

# Change storage class
python storage_change_file_storage_class.py <BUCKET_NAME> <BLOB_NAME>
```

## Security Features

### Access Control

#### Uniform Bucket-Level Access
```python
# Enable uniform bucket-level access
python storage_enable_uniform_bucket_level_access.py <BUCKET_NAME>

# Disable uniform bucket-level access
python storage_disable_uniform_bucket_level_access.py <BUCKET_NAME>
```

#### Public Access Prevention
```python
# Enforce public access prevention
python storage_set_public_access_prevention_enforced.py <BUCKET_NAME>

# Set to inherited
python storage_set_public_access_prevention_inherited.py <BUCKET_NAME>
```

#### IAM Policies
```python
# Add IAM member
python storage_add_bucket_iam_member.py <BUCKET_NAME> <ROLE> <MEMBER>

# Remove IAM member
python storage_remove_bucket_iam_member.py <BUCKET_NAME> <ROLE> <MEMBER>

# View IAM members
python storage_view_bucket_iam_members.py <BUCKET_NAME>

# Make bucket public
python storage_set_bucket_public_iam.py <BUCKET_NAME>
```

#### ACLs (Access Control Lists)
```python
# Working with ACLs
client = storage.Client()
bucket = client.get_bucket(bucket_name)
acl = bucket.acl

# Grant permissions
acl.user("me@example.org").grant_read()
acl.all_authenticated().grant_write()

# Save ACL changes
acl.save()

# Print ACLs
python storage_print_bucket_acl.py <BUCKET_NAME>
python storage_print_file_acl.py <BUCKET_NAME> <BLOB_NAME>
```

### Encryption

#### Customer-Managed Encryption Keys (CMEK)
```python
# Set default KMS key for bucket
python storage_set_bucket_default_kms_key.py <BUCKET_NAME> <KMS_KEY_NAME>

# Delete default KMS key
python storage_bucket_delete_default_kms_key.py <BUCKET_NAME>

# Get object KMS key
python storage_object_get_kms_key.py <BUCKET_NAME> <BLOB_NAME>
```

#### Customer-Supplied Encryption Keys (CSEK)
```python
# Generate encryption key
python storage_generate_encryption_key.py

# Rotate encryption key
python storage_rotate_encryption_key.py <BUCKET_NAME> <BLOB_NAME> <BASE64_ENCRYPTION_KEY> <BASE64_NEW_ENCRYPTION_KEY>

# Convert CSEK to CMEK
python storage_object_csek_to_cmek.py <BUCKET_NAME> <BLOB_NAME> <ENCRYPTION_KEY> <KMS_KEY_NAME>
```

## Advanced Features

### Lifecycle Management
```python
# Enable lifecycle management
python storage_enable_bucket_lifecycle_management.py <BUCKET_NAME>

# Disable lifecycle management
python storage_disable_bucket_lifecycle_management.py <BUCKET_NAME>
```

### Versioning
```python
# Enable versioning
python storage_enable_versioning.py <BUCKET_NAME>

# Disable versioning
python storage_disable_versioning.py <BUCKET_NAME>

# List archived generations
python storage_list_file_archived_generations.py <BUCKET_NAME>
```

### Retention Policies
```python
# Set retention policy
python storage_set_retention_policy.py <BUCKET_NAME> <RETENTION_PERIOD>

# Lock retention policy
python storage_lock_retention_policy.py <BUCKET_NAME>

# Remove retention policy
python storage_remove_retention_policy.py <BUCKET_NAME>
```

### Event-Based and Temporary Holds
```python
# Enable default event-based hold
python storage_enable_default_event_based_hold.py <BUCKET_NAME>

# Set event-based hold on object
python storage_set_event_based_hold.py <BUCKET_NAME> <BLOB_NAME>

# Set temporary hold
python storage_set_temporary_hold.py <BUCKET_NAME> <BLOB_NAME>

# Release holds
python storage_release_event_based_hold.py <BUCKET_NAME> <BLOB_NAME>
python storage_release_temporary_hold.py <BUCKET_NAME> <BLOB_NAME>
```

### Signed URLs
```python
# Generate signed URL v4
python storage_generate_signed_url_v4.py <BUCKET_NAME> <BLOB_NAME>

# Generate upload signed URL v4
python storage_generate_upload_signed_url_v4.py <BUCKET_NAME> <BLOB_NAME>

# Generate signed post policy v4
python storage_generate_signed_post_policy_v4.py <BUCKET_NAME> <BLOB_NAME>
```

### CORS Configuration
```python
# Configure CORS
python storage_cors_configuration.py <BUCKET_NAME>

# Remove CORS configuration
python storage_remove_cors_configuration.py <BUCKET_NAME>
```

### Pub/Sub Notifications
```python
# Create bucket notifications
python storage_create_bucket_notifications.py <BUCKET_NAME> <TOPIC_NAME>

# Print notification details
python storage_print_pubsub_bucket_notification.py <BUCKET_NAME> <NOTIFICATION_ID>

# Delete notification
python storage_delete_bucket_notification.py <BUCKET_NAME> <NOTIFICATION_ID>
```

## Performance and Reliability

### Retry Configuration
```python
from google.api_core import exceptions
from google.api_core.retry import Retry

# Define custom retry predicate
_MY_RETRIABLE_TYPES = [
   exceptions.TooManyRequests,  # 429
   exceptions.InternalServerError,  # 500
   exceptions.BadGateway,  # 502
   exceptions.ServiceUnavailable,  # 503
]

def is_retryable(exc):
    return isinstance(exc, _MY_RETRIABLE_TYPES)

my_retry_policy = Retry(predicate=is_retryable)
bucket = client.get_bucket(BUCKET_NAME, retry=my_retry_policy)
```

### Timeouts
```python
# Set timeout (applies to both connect and read)
bucket = client.get_bucket(BUCKET_NAME, timeout=300.0)  # five minutes
```

### Recovery Point Objective (RPO)
```python
# Set RPO to Async Turbo
python storage_set_rpo_async_turbo.py <BUCKET_NAME>

# Set RPO to Default
python storage_set_rpo_default.py <BUCKET_NAME>
```

## Monitoring with OpenTelemetry

### Setup Tracing
```bash
# Install dependencies
pip install google-cloud-storage[tracing]
pip install opentelemetry-exporter-gcp-trace opentelemetry-propagator-gcp
pip install opentelemetry-instrumentation-requests

# Enable tracing
export ENABLE_GCS_PYTHON_CLIENT_OTEL_TRACES=True
```

### Configure Tracer
```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
trace.set_tracer_provider(tracer_provider)

# Optional: instrument requests library
from opentelemetry.instrumentation.requests import RequestsInstrumentor
RequestsInstrumentor().instrument(tracer_provider=tracer_provider)
```

## HMAC Keys for Service Accounts
```python
# Create HMAC key
python storage_create_hmac_key.py <PROJECT_ID> <SERVICE_ACCOUNT_EMAIL>

# List HMAC keys
python storage_list_hmac_keys.py <PROJECT_ID>

# Activate/Deactivate HMAC key
python storage_activate_hmac_key.py <ACCESS_ID> <PROJECT_ID>
python storage_deactivate_hmac_key.py <ACCESS_ID> <PROJECT_ID>

# Delete HMAC key
python storage_delete_hmac_key.py <ACCESS_ID> <PROJECT_ID>
```

## Best Practices for CQC Project

1. **Use Batch Operations**: For multiple operations, use batch requests to improve performance
2. **Enable Versioning**: Keep versioning enabled for critical data buckets
3. **Set Lifecycle Policies**: Configure automatic deletion of old data to manage costs
4. **Use Uniform Bucket-Level Access**: Simplify access management with uniform policies
5. **Implement Retry Logic**: Use custom retry policies for resilient operations
6. **Monitor with Cloud Trace**: Enable OpenTelemetry tracing for performance monitoring
7. **Secure with CMEK**: Use Cloud KMS for encryption key management
8. **Set Appropriate Timeouts**: Configure timeouts based on expected file sizes
9. **Use Signed URLs**: Generate signed URLs for temporary access to private objects
10. **Enable Public Access Prevention**: Prevent accidental public exposure of sensitive data