# Google Cloud Scheduler Documentation

## Overview

Google Cloud Scheduler is a fully managed enterprise-grade cron job scheduler that allows you to schedule virtually any job, including batch, big data jobs, cloud infrastructure operations, and more. This documentation covers the Python SDK usage, configuration, and best practices relevant to the CQC ML project.

## Table of Contents

- [Installation](#installation)
- [SDK Configuration](#sdk-configuration)
- [Job Management](#job-management)
- [Job Types](#job-types)
- [Scheduling Configuration](#scheduling-configuration)
- [Integration with Other Services](#integration-with-other-services)
- [Logging Configuration](#logging-configuration)
- [Best Practices](#best-practices)

## Installation

### Mac/Linux
```bash
python3 -m venv <your-env>
source <your-env>/bin/activate
pip install google-cloud-scheduler
```

### Windows
```bash
py -m venv <your-env>
.\<your-env>\Scripts\activate
pip install google-cloud-scheduler
```

## SDK Configuration

### Import the Library
```python
from google.cloud import scheduler
from google.cloud import scheduler_v1
```

### Create a Client
```python
# Create the Cloud Scheduler client
client = scheduler.CloudSchedulerClient()

# Or with specific version
client = scheduler_v1.CloudSchedulerClient()
```

### Access Library Version
```python
import google.cloud.scheduler

# Access the client library version
version = google.cloud.scheduler.__version__
print(f"Google Cloud Scheduler Client Version: {version}")
```

## Job Management

### Creating Jobs

#### HTTP Target Job
```python
from google.cloud import scheduler_v1

# Initialize client
client = scheduler_v1.CloudSchedulerClient()

# Define parent location
parent = client.common_location_path(project_id, location_id)

# Create HTTP job
job = scheduler_v1.Job(
    name=f"{parent}/jobs/http-job",
    description="HTTP job example",
    schedule="0 */4 * * *",  # Every 4 hours
    http_target=scheduler_v1.HttpTarget(
        uri="https://example.com/endpoint",
        http_method=scheduler_v1.HttpMethod.POST,
        headers={"Content-Type": "application/json"},
        body=b'{"key": "value"}',
    ),
    time_zone="America/New_York",
)

# Create the job
response = client.create_job(
    parent=parent,
    job=job,
)
```

#### Pub/Sub Target Job
```python
# Create Pub/Sub job
job = scheduler_v1.Job(
    name=f"{parent}/jobs/pubsub-job",
    description="Pub/Sub job example",
    schedule="0 2 * * *",  # Daily at 2 AM
    pubsub_target=scheduler_v1.PubsubTarget(
        topic_name=f"projects/{project_id}/topics/{topic_name}",
        data=b"Message data",
        attributes={"key": "value"},
    ),
    time_zone="UTC",
)

response = client.create_job(
    parent=parent,
    job=job,
)
```

#### App Engine Target Job
```python
# Create App Engine job
job = scheduler_v1.Job(
    name=f"{parent}/jobs/appengine-job",
    description="App Engine job example",
    schedule="0 0 * * 0",  # Weekly on Sunday
    app_engine_http_target=scheduler_v1.AppEngineHttpTarget(
        relative_uri="/tasks/weekly",
        http_method=scheduler_v1.HttpMethod.GET,
        app_engine_routing=scheduler_v1.AppEngineRouting(
            service="worker",
            version="v1",
        ),
    ),
    time_zone="Europe/London",
)

response = client.create_job(
    parent=parent,
    job=job,
)
```

### Listing Jobs

```python
# List all jobs in a location
parent = client.common_location_path(project_id, location_id)

# Using pager for efficient iteration
pager = client.list_jobs(parent=parent)

for job in pager:
    print(f"Job: {job.name}")
    print(f"Schedule: {job.schedule}")
    print(f"State: {job.state}")
```

### Getting a Job

```python
# Get a specific job
job_name = client.job_path(project_id, location_id, job_id)
job = client.get_job(name=job_name)

print(f"Job details: {job}")
```

### Updating a Job

```python
# Update job schedule
job.schedule = "0 */2 * * *"  # Every 2 hours

# Update the job
update_mask = {"paths": ["schedule"]}
updated_job = client.update_job(
    job=job,
    update_mask=update_mask,
)
```

### Pausing and Resuming Jobs

```python
# Pause a job
job_name = client.job_path(project_id, location_id, job_id)
paused_job = client.pause_job(name=job_name)

# Resume a job
resumed_job = client.resume_job(name=job_name)
```

### Running a Job Manually

```python
# Force run a job immediately
job_name = client.job_path(project_id, location_id, job_id)
response = client.run_job(name=job_name)
```

### Deleting a Job

```python
# Delete a job
job_name = client.job_path(project_id, location_id, job_id)
client.delete_job(name=job_name)
```

## Job Types

### HTTP Target Jobs with Authentication

#### OIDC Token Authentication
```python
job = scheduler_v1.Job(
    name=f"{parent}/jobs/oidc-job",
    schedule="0 */1 * * *",
    http_target=scheduler_v1.HttpTarget(
        uri="https://example.com/secure-endpoint",
        http_method=scheduler_v1.HttpMethod.POST,
        oidc_token=scheduler_v1.OidcToken(
            service_account_email="service-account@project.iam.gserviceaccount.com",
            audience="https://example.com",
        ),
    ),
)
```

#### OAuth Token Authentication
```python
job = scheduler_v1.Job(
    name=f"{parent}/jobs/oauth-job",
    schedule="0 */1 * * *",
    http_target=scheduler_v1.HttpTarget(
        uri="https://example.com/api/endpoint",
        http_method=scheduler_v1.HttpMethod.POST,
        oauth_token=scheduler_v1.OAuthToken(
            service_account_email="service-account@project.iam.gserviceaccount.com",
            scope="https://www.googleapis.com/auth/cloud-platform",
        ),
    ),
)
```

## Scheduling Configuration

### Cron Expressions
Cloud Scheduler uses standard cron format:
- `* * * * *` - Minute, Hour, Day of month, Month, Day of week
- Examples:
  - `0 10 * * *` - Daily at 10:00 AM
  - `0 */6 * * *` - Every 6 hours
  - `0 9 * * 1` - Every Monday at 9:00 AM
  - `0 0 1 * *` - First day of every month

### Time Zones
Always specify time zones to avoid confusion:
```python
job.time_zone = "America/New_York"  # Eastern Time
job.time_zone = "Europe/London"     # UK Time
job.time_zone = "UTC"               # Coordinated Universal Time
```

### Retry Configuration
```python
job = scheduler_v1.Job(
    name=f"{parent}/jobs/retry-job",
    schedule="0 */1 * * *",
    http_target=scheduler_v1.HttpTarget(
        uri="https://example.com/endpoint",
    ),
    retry_config=scheduler_v1.RetryConfig(
        retry_count=3,
        max_retry_duration={"seconds": 3600},  # 1 hour
        min_backoff_duration={"seconds": 5},
        max_backoff_duration={"seconds": 3600},
        max_doublings=5,
    ),
)
```

### Deadline Configuration
```python
job = scheduler_v1.Job(
    name=f"{parent}/jobs/deadline-job",
    schedule="0 */1 * * *",
    http_target=scheduler_v1.HttpTarget(
        uri="https://example.com/endpoint",
    ),
    attempt_deadline={"seconds": 1800},  # 30 minutes
)
```

## Integration with Other Services

### Cloud Functions Triggering
```python
# Trigger a Cloud Function via HTTP
job = scheduler_v1.Job(
    name=f"{parent}/jobs/function-trigger",
    schedule="0 0 * * *",  # Daily at midnight
    http_target=scheduler_v1.HttpTarget(
        uri=f"https://region-{project_id}.cloudfunctions.net/function-name",
        http_method=scheduler_v1.HttpMethod.POST,
        oidc_token=scheduler_v1.OidcToken(
            service_account_email=f"service-account@{project_id}.iam.gserviceaccount.com",
            audience=f"https://region-{project_id}.cloudfunctions.net/function-name",
        ),
    ),
)
```

### Cloud Run Triggering
```python
# Trigger a Cloud Run service
job = scheduler_v1.Job(
    name=f"{parent}/jobs/cloudrun-trigger",
    schedule="0 */4 * * *",  # Every 4 hours
    http_target=scheduler_v1.HttpTarget(
        uri=f"https://service-name-{project_hash}-uc.a.run.app/endpoint",
        http_method=scheduler_v1.HttpMethod.POST,
        headers={"Content-Type": "application/json"},
        body=b'{"task": "process"}',
        oidc_token=scheduler_v1.OidcToken(
            service_account_email=f"service-account@{project_id}.iam.gserviceaccount.com",
        ),
    ),
)
```

### Dataflow Job Triggering
```python
# Trigger Dataflow job via Cloud Functions or direct API
job = scheduler_v1.Job(
    name=f"{parent}/jobs/dataflow-trigger",
    schedule="0 3 * * *",  # Daily at 3 AM
    http_target=scheduler_v1.HttpTarget(
        uri="https://dataflow.googleapis.com/v1b3/projects/{project}/templates:launch",
        http_method=scheduler_v1.HttpMethod.POST,
        headers={"Content-Type": "application/json"},
        body=b'''{
            "jobName": "scheduled-job",
            "parameters": {
                "inputFile": "gs://bucket/input.txt",
                "outputFile": "gs://bucket/output.txt"
            },
            "environment": {
                "tempLocation": "gs://bucket/temp"
            }
        }''',
        oauth_token=scheduler_v1.OAuthToken(
            service_account_email=f"service-account@{project_id}.iam.gserviceaccount.com",
            scope="https://www.googleapis.com/auth/cloud-platform",
        ),
    ),
)
```

### BigQuery Scheduled Queries
While BigQuery has its own scheduling, Cloud Scheduler can trigger complex workflows:
```python
# Trigger BigQuery job via Cloud Function
job = scheduler_v1.Job(
    name=f"{parent}/jobs/bigquery-etl",
    schedule="0 1 * * *",  # Daily at 1 AM
    pubsub_target=scheduler_v1.PubsubTarget(
        topic_name=f"projects/{project_id}/topics/bigquery-etl-trigger",
        data=b'''{
            "query": "SELECT * FROM dataset.table WHERE date = CURRENT_DATE()",
            "destination": "dataset.processed_table"
        }''',
    ),
)
```

## Terraform Configuration Examples

### Cloud Function with Scheduler Trigger
```hcl
locals {
  project = "my-project-name"
}

resource "google_service_account" "account" {
  account_id   = "gcf-sa"
  display_name = "Test Service Account"
}

resource "google_storage_bucket" "bucket" {
  name                        = "${local.project}-gcf-source"
  location                    = "US"
  uniform_bucket_level_access = true
}
 
resource "google_storage_bucket_object" "object" {
  name   = "function-source.zip"
  bucket = google_storage_bucket.bucket.name
  source = "function-source.zip"
}

resource "google_cloudfunctions2_function" "function" {
  name        = "gcf-function"
  location    = "us-central1"
  description = "a new function"
 
  build_config {
    runtime     = "nodejs20"
    entry_point = "helloHttp"
    source {
      storage_source {
        bucket = google_storage_bucket.bucket.name
        object = google_storage_bucket_object.object.name
      }
    }
  }
 
  service_config {
    min_instance_count    = 1
    available_memory      = "256M"
    timeout_seconds       = 60
    service_account_email = google_service_account.account.email
  }
}

resource "google_cloud_scheduler_job" "invoke_cloud_function" {
  name        = "invoke-gcf-function"
  description = "Schedule the HTTPS trigger for cloud function"
  schedule    = "0 0 * * *" # every day at midnight
  project     = google_cloudfunctions2_function.function.project
  region      = google_cloudfunctions2_function.function.location

  http_target {
    uri         = google_cloudfunctions2_function.function.service_config[0].uri
    http_method = "POST"
    oidc_token {
      audience              = "${google_cloudfunctions2_function.function.service_config[0].uri}/"
      service_account_email = google_service_account.account.email
    }
  }
}
```

## Logging Configuration

### Enable Default Logging for All Google Modules
```bash
export GOOGLE_SDK_PYTHON_LOGGING_SCOPE=google
```

### Enable Default Logging for Cloud Scheduler
```bash
export GOOGLE_SDK_PYTHON_LOGGING_SCOPE=google.cloud.library_v1
```

### Programmatic Logging Configuration
```python
import logging

# Configure logging for all Google modules
base_logger = logging.getLogger("google")
base_logger.addHandler(logging.StreamHandler())
base_logger.setLevel(logging.DEBUG)

# Configure logging for Cloud Scheduler specifically
base_logger = logging.getLogger("google.cloud.library_v1")
base_logger.addHandler(logging.StreamHandler())
base_logger.setLevel(logging.DEBUG)
```

## Error Handling and Monitoring

### Job State Monitoring
```python
# Check job state
job = client.get_job(name=job_name)

if job.state == scheduler_v1.Job.State.ENABLED:
    print("Job is enabled and running on schedule")
elif job.state == scheduler_v1.Job.State.PAUSED:
    print("Job is paused")
elif job.state == scheduler_v1.Job.State.DISABLED:
    print("Job is disabled")
```

### Last Attempt Status
```python
# Check last attempt
if job.status:
    print(f"Last attempt time: {job.status.last_attempt_time}")
    print(f"Last attempt status: {job.status.code}")
```

## Best Practices

### 1. Job Design
- Keep job payloads small and focused
- Use idempotent operations
- Implement proper error handling in target services
- Set appropriate timeouts and retry configurations

### 2. Scheduling Strategy
- Spread job execution times to avoid resource contention
- Use appropriate time zones for business logic
- Consider using jitter for jobs that might create thundering herd problems

### 3. Security
- Use service accounts with minimal required permissions
- Implement OIDC tokens for HTTP targets
- Encrypt sensitive data in job payloads
- Regularly rotate service account keys

### 4. Monitoring and Alerting
- Set up Cloud Monitoring alerts for job failures
- Monitor job execution duration
- Track retry rates and adjust configurations
- Use Cloud Logging for debugging

### 5. Cost Optimization
- Delete unused jobs
- Optimize job frequency based on actual needs
- Use appropriate retry configurations to avoid excessive attempts
- Monitor API usage and costs

### 6. Development Workflow
- Use separate projects for dev/staging/production
- Test cron expressions thoroughly
- Implement proper CI/CD for job configurations
- Document job purposes and dependencies

### 7. Integration Best Practices
- Use Pub/Sub for decoupling when possible
- Implement circuit breakers in target services
- Use Cloud Tasks for more complex queueing needs
- Consider using Cloud Workflows for multi-step processes

## Common Use Cases for CQC ML Project

### 1. Data Pipeline Scheduling
```python
# Schedule daily data ingestion
job = scheduler_v1.Job(
    name=f"{parent}/jobs/daily-data-ingestion",
    schedule="0 2 * * *",  # 2 AM daily
    pubsub_target=scheduler_v1.PubsubTarget(
        topic_name=f"projects/{project_id}/topics/data-ingestion",
        data=b'{"source": "cqc-api", "date": "today"}',
    ),
)
```

### 2. Model Training Triggers
```python
# Schedule weekly model retraining
job = scheduler_v1.Job(
    name=f"{parent}/jobs/weekly-model-training",
    schedule="0 0 * * 0",  # Sunday midnight
    http_target=scheduler_v1.HttpTarget(
        uri="https://vertex-ai-training-trigger.run.app/train",
        http_method=scheduler_v1.HttpMethod.POST,
        body=b'{"model": "cqc-classifier", "version": "auto"}',
        oidc_token=scheduler_v1.OidcToken(
            service_account_email="ml-training@project.iam.gserviceaccount.com",
        ),
    ),
)
```

### 3. Report Generation
```python
# Schedule monthly reports
job = scheduler_v1.Job(
    name=f"{parent}/jobs/monthly-reports",
    schedule="0 9 1 * *",  # First day of month at 9 AM
    http_target=scheduler_v1.HttpTarget(
        uri="https://report-generator.run.app/generate",
        http_method=scheduler_v1.HttpMethod.POST,
        body=b'{"type": "monthly", "recipients": ["team@example.com"]}',
    ),
)
```

### 4. Data Quality Checks
```python
# Schedule hourly data quality checks
job = scheduler_v1.Job(
    name=f"{parent}/jobs/data-quality-check",
    schedule="0 * * * *",  # Every hour
    pubsub_target=scheduler_v1.PubsubTarget(
        topic_name=f"projects/{project_id}/topics/data-quality",
        attributes={
            "check_type": "completeness",
            "severity": "warning"
        },
    ),
)
```

### 5. Backup Operations
```python
# Schedule daily backups
job = scheduler_v1.Job(
    name=f"{parent}/jobs/daily-backup",
    schedule="0 3 * * *",  # 3 AM daily
    http_target=scheduler_v1.HttpTarget(
        uri="https://backup-service.run.app/backup",
        http_method=scheduler_v1.HttpMethod.POST,
        body=b'{"datasets": ["cqc-ml-data", "cqc-ml-models"]}',
    ),
)
```

## Dependency Versions

The Cloud Scheduler client library requires specific versions of dependencies:
- Python 3.7+
- `google-api-core` >= 1.34.0, >= 2.11.0
- `proto-plus` >= 1.22.0
- `protobuf` >= 3.20.2, < 5.0.0

## Additional Resources

- [Cloud Scheduler Documentation](https://cloud.google.com/scheduler/docs)
- [Cloud Scheduler Python Client Library](https://github.com/googleapis/google-cloud-python/tree/main/packages/google-cloud-scheduler)
- [Cron Expression Reference](https://cloud.google.com/scheduler/docs/configuring/cron-job-schedules)
- [Cloud Scheduler Quotas and Limits](https://cloud.google.com/scheduler/quotas)