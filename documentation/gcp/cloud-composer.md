# Google Cloud Composer Documentation

## Overview

Google Cloud Composer is a fully managed workflow orchestration service built on Apache Airflow. This documentation covers the Python SDK usage, configuration, and best practices relevant to the CQC ML project.

## Table of Contents

- [Installation](#installation)
- [API Reference](#api-reference)
- [Environment Management](#environment-management)
- [User Workloads Secrets](#user-workloads-secrets)
- [Maintenance and Updates](#maintenance-and-updates)
- [Logging Configuration](#logging-configuration)
- [Best Practices](#best-practices)

## Installation

### Mac/Linux
```bash
python3 -m venv <your-env>
source <your-env>/bin/activate
pip install google-cloud-orchestration-airflow
```

### Windows
```bash
py -m venv <your-env>
.\<your-env>\Scripts\activate
pip install google-cloud-orchestration-airflow
```

## API Reference

### Core Services

#### Environments Service
- **Module**: `google.cloud.orchestration.airflow.service_v1.services.environments`
- **Description**: Main service for managing Google Cloud Composer environments
- **Key Methods**:
  - `CheckUpgrade()`: Check for available environment upgrades

#### ImageVersions Service
- **Module**: `google.cloud.orchestration.airflow.service_v1.services.image_versions`
- **Methods**:
  - `ListImageVersions(request: ListImageVersionsRequest) -> ListImageVersionsResponse`
    - Lists available image versions for Google Cloud Composer
    - Parameters:
      - `parent` (str, required): The parent project and location
      - `page_size` (int, optional): Maximum number of items to return
      - `page_token` (str, optional): Page token from previous call

### API Changes and Updates

#### Version 1.15.0 - New Features
- Added `CheckUpgrade` method to `Environments` service
- Added `CheckUpgradeRequest` message type
- Added `satisfies_pzi` field to `Environment` message (indicates PZI compliance)
- Added `airflow_metadata_retention_config` field to `DataRetentionConfig`
- Added `AirflowMetadataRetentionPolicyConfig` message

#### Python 3.13 Support
The Google Cloud Orchestration Airflow client library now supports Python 3.13.

### Data Types and Messages

#### Environment Configuration
- **EnvironmentConfig**:
  - `node_config`: Node configuration settings
  - `software_config`: Software configuration including Airflow version
  - `private_environment_config`: Private environment settings
  - `maintenance_window`: Maintenance window configuration

#### Workloads Configuration
- **WorkloadsConfig**: Resource configurations for Airflow components
  - `scheduler`: Configuration for Airflow schedulers
  - `triggerer`: Configuration for Airflow triggerer
  - `web_server`: Configuration for Airflow web server
  - `worker`: Configuration for Airflow workers

#### Scheduler Configuration
```yaml
scheduler:
  cpu: (Optional) Number of CPUs for a single Airflow scheduler
  memory_gb: (Optional) Amount of memory (GB) for a single Airflow scheduler
  storage_gb: (Optional) Amount of storage (GB) for a single Airflow scheduler
  count: (Optional) Number of schedulers
```

#### Data Retention Configuration
- **DataRetentionConfig**:
  - `airflow_metadata_retention_config`: Airflow metadata retention policies
- **TaskLogsRetentionConfig**:
  - `storage_mode`: Storage mode for task logs

## Environment Management

### Creating an Environment
```python
from google.cloud import orchestration_airflow

# Initialize client
client = orchestration_airflow.EnvironmentsClient()

# Create environment configuration
environment = {
    "name": "example-environment",
    "config": {
        "software_config": {
            "image_version": "composer-3-airflow-2"
        }
    }
}

# Create the environment
operation = client.create_environment(
    parent="projects/PROJECT_ID/locations/LOCATION",
    environment=environment
)
```

### Checking for Upgrades
```python
# Check if upgrades are available for an environment
response = client.check_upgrade(
    environment="projects/PROJECT_ID/locations/LOCATION/environments/ENVIRONMENT_NAME"
)
```

### Getting Maintenance Schedule
The API supports retrieving the maintenance schedule configured for a specific AlloyDB cluster (integrated with Composer environments).

## User Workloads Secrets

### Creating a User Workloads Secret
```python
import base64
from google.cloud import orchestration_airflow

# Create a secret with base64 encoded data
secret_data = {
    "username": base64.b64encode("username".encode()).decode(),
    "password": base64.b64encode("password".encode()).decode(),
}

# Resource creation would be done through Terraform or gcloud CLI
```

### Accessing User Workloads Secrets
User workloads secrets are managed through Kubernetes secrets within the Composer environment. The `data` field contains the secret values, and comments have been updated in the API to clarify usage.

## Maintenance and Updates

### Scheduled Maintenance
Cloud Composer supports scheduling maintenance windows for environments:
- Configure maintenance schedules for controlled updates
- Retrieve current maintenance schedule information

### Image Versions
List available Composer image versions:
```python
# List available image versions
pager = image_versions_client.list_image_versions(
    parent="projects/PROJECT_ID/locations/LOCATION"
)

for image_version in pager:
    print(f"Version: {image_version.image_version_id}")
```

## Logging Configuration

### Enable Logging for All Google Modules
```bash
export GOOGLE_SDK_PYTHON_LOGGING_SCOPE=google
```

### Enable Logging for Specific Module
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

# Configure logging for specific module
base_logger = logging.getLogger("google.cloud.library_v1")
base_logger.addHandler(logging.StreamHandler())
base_logger.setLevel(logging.DEBUG)
```

## Integration with Other Services

### Cloud Workflows Integration
Cloud Composer can integrate with Cloud Workflows for complex orchestration scenarios:
- Workflow state management includes `UNAVAILABLE` state
- Support for `call_log_level`, `state_error`, and `user_env_vars` fields
- Execution tokens for immediate job execution

### Dataproc Integration
- Support for Flink and Trino job types in workflow templates
- Enhanced job management with unreachable output fields

### Document Processing
- Pipeline Service integration for Document Warehouse API v1
- Support for complex data processing workflows

## Best Practices

### 1. Environment Configuration
- Use appropriate Composer image versions for your Airflow requirements
- Configure resource allocation based on workload requirements
- Enable PZI compliance when required (`satisfies_pzi` field)

### 2. Security
- Use User Workloads Secrets for sensitive data
- Implement proper IAM policies for environment access
- Enable private environment configuration when needed

### 3. Maintenance
- Schedule maintenance windows during low-traffic periods
- Regularly check for available upgrades
- Monitor environment health and performance

### 4. Resource Management
- Configure appropriate CPU, memory, and storage for each component
- Use auto-scaling where available
- Monitor resource utilization

### 5. Data Retention
- Configure appropriate retention policies for Airflow metadata
- Manage task log retention based on compliance requirements

### 6. Error Handling
- Implement proper error handling in DAGs
- Monitor workflow execution status
- Set up alerts for failed workflows

### 7. Performance Optimization
- Optimize scheduler configuration for your workload
- Configure appropriate worker resources
- Use connection pooling for database connections

## Terraform Configuration Examples

### Basic Environment with User Workloads Secret
```hcl
resource "google_composer_environment" "example" {
    name = "example-environment"
    config{
        software_config {
            image_version = "composer-3-airflow-2"
        }
    }
}

resource "google_composer_user_workloads_secret" "example" {
    environment = google_composer_environment.example.name
    name = "example-secret"
    data = {
        username: base64encode("username"),
        password: base64encode("password"),
    }
}

data "google_composer_user_workloads_secret" "example" {
    environment = google_composer_environment.example.name
    name = resource.google_composer_user_workloads_secret.example.name
}

output "debug" {
    value = data.google_composer_user_workloads_secret.example
}
```

## Additional Resources

- [Cloud Composer Documentation](https://cloud.google.com/composer/docs)
- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Cloud Composer Python Client Library](https://github.com/googleapis/google-cloud-python/tree/main/packages/google-cloud-orchestration-airflow)
- [Cloud Workflows Documentation](https://cloud.google.com/workflows/docs)