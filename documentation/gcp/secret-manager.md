# Google Cloud Secret Manager Documentation

## Overview

Google Cloud Secret Manager is a secure and convenient service for storing API keys, passwords, certificates, and other sensitive data. This documentation covers the Python SDK usage, configuration, and best practices relevant to the CQC ML project.

## Table of Contents

- [Installation](#installation)
- [SDK Configuration](#sdk-configuration)
- [Secret Management](#secret-management)
- [Access Control](#access-control)
- [Integration with Other Services](#integration-with-other-services)
- [Regional Secrets](#regional-secrets)
- [Logging Configuration](#logging-configuration)
- [Best Practices](#best-practices)

## Installation

### Mac/Linux
```bash
python3 -m venv <your-env>
source <your-env>/bin/activate
pip install google-cloud-secret-manager
```

### Windows
```bash
py -m venv <your-env>
.\<your-env>\Scripts\activate
pip install google-cloud-secret-manager
```

## SDK Configuration

### Import the Library
```python
from google.cloud import secretmanager
```

### Create a Client
```python
# Create the Secret Manager client
client = secretmanager.SecretManagerServiceClient()
```

## Secret Management

### Creating Secrets

#### Basic Secret
```python
# Define parent project
parent = f"projects/{project_id}"

# Create the secret
response = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "my-secret",
        "secret": {
            "replication": {
                "automatic": {}
            }
        },
    }
)
```

#### Secret with Labels and Annotations
```python
response = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "my-secret",
        "secret": {
            "labels": {
                "env": "production",
                "team": "ml-team"
            },
            "annotations": {
                "created_by": "ml-pipeline",
                "purpose": "api-authentication"
            },
            "replication": {
                "automatic": {}
            }
        },
    }
)
```

### Adding Secret Versions

```python
# Add a secret version
parent = client.secret_path(project_id, "my-secret")

# Convert the secret payload to bytes
payload = "my-secret-value".encode("UTF-8")

# Add the secret version
response = client.add_secret_version(
    request={
        "parent": parent,
        "payload": {"data": payload},
    }
)
```

### Accessing Secrets

#### Access Latest Version
```python
# Build the resource name of the secret version
name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"

# Access the secret version
response = client.access_secret_version(request={"name": name})

# Get the secret data
secret_value = response.payload.data.decode("UTF-8")
```

#### Access Specific Version
```python
# Access a specific version
name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
response = client.access_secret_version(request={"name": name})
```

### Listing Secrets

```python
# List all secrets in a project
parent = f"projects/{project_id}"

for secret in client.list_secrets(request={"parent": parent}):
    print(f"Secret: {secret.name}")
```

### Deleting Secrets

```python
# Delete a secret
name = client.secret_path(project_id, secret_id)
client.delete_secret(request={"name": name})
```

## Access Control

### IAM Permissions for Service Accounts

#### Grant Secret Accessor Role
```python
from google.cloud import secretmanager_v1
from google.iam.v1 import iam_policy_pb2, policy_pb2

# Get the current IAM policy
name = client.secret_path(project_id, secret_id)
policy = client.get_iam_policy(request={"resource": name})

# Add a new member
policy.bindings.add(
    role="roles/secretmanager.secretAccessor",
    members=[f"serviceAccount:{service_account_email}"]
)

# Update the IAM policy
client.set_iam_policy(
    request={
        "resource": name,
        "policy": policy,
    }
)
```

## Integration with Other Services

### Google Kubernetes Engine (GKE)
The Secret Manager CSI (Container Storage Interface) component can be enabled/disabled on GKE clusters via API:
```python
# This functionality is typically managed through the GKE API
# Enable/disable Secret Manager CSI component on GKE clusters
```

### Cloud Functions Integration
Secrets can be mounted as environment variables or volumes in Cloud Functions:

#### Environment Variables
```python
# When deploying a Cloud Function, reference secrets as environment variables
# This is typically done through deployment configuration, not direct API calls
```

#### Secret Volumes
```python
# Mount secrets as files in Cloud Functions
# Configuration done during function deployment
```

### Cloud Run Integration
Similar to Cloud Functions, Cloud Run services can access secrets through environment variables or mounted volumes.

### Vertex AI Integration
Secrets can be used to store API keys and credentials for ML pipelines and model deployments.

## Regional Secrets

### Creating Regional Secrets

#### Basic Regional Secret
```python
# Regional secrets are created in a specific location
parent = f"projects/{project_id}/locations/{location}"

response = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "regional-secret",
        "secret": {
            "replication": {
                "user_managed": {
                    "replicas": [
                        {"location": "us-central1"},
                        {"location": "us-east1"}
                    ]
                }
            }
        },
    }
)
```

#### Regional Secret with Rotation
```python
# Create a secret with automatic rotation
response = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "rotating-secret",
        "secret": {
            "rotation": {
                "rotation_period": {"seconds": 2592000},  # 30 days
                "next_rotation_time": {"seconds": next_rotation_timestamp}
            },
            "topics": [
                {"name": f"projects/{project_id}/topics/{topic_name}"}
            ]
        },
    }
)
```

#### Regional Secret with TTL
```python
# Create a secret with time-to-live
response = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "ttl-secret",
        "secret": {
            "ttl": {"seconds": 86400},  # 24 hours
            "replication": {
                "user_managed": {
                    "replicas": [{"location": "us-central1"}]
                }
            }
        },
    }
)
```

#### Regional Secret with Expiration
```python
# Create a secret that expires at a specific time
response = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "expiring-secret",
        "secret": {
            "expire_time": {"seconds": expiration_timestamp},
            "replication": {
                "user_managed": {
                    "replicas": [{"location": "us-central1"}]
                }
            }
        },
    }
)
```

## Logging Configuration

### Enable Default Logging for All Google Modules
```bash
export GOOGLE_SDK_PYTHON_LOGGING_SCOPE=google
```

### Enable Default Logging for Secret Manager
```bash
export GOOGLE_SDK_PYTHON_LOGGING_SCOPE=google.cloud.secretmanager
```

### Programmatic Logging Configuration
```python
import logging

# Configure logging for all Google modules
base_logger = logging.getLogger("google")
base_logger.addHandler(logging.StreamHandler())
base_logger.setLevel(logging.DEBUG)

# Configure logging for Secret Manager specifically
base_logger = logging.getLogger("google.cloud.secretmanager")
base_logger.addHandler(logging.StreamHandler())
base_logger.setLevel(logging.DEBUG)
```

## Terraform Configuration Examples

### Basic Secret with Annotations
```hcl
resource "google_secret_manager_secret" "secret-with-annotations" {
  secret_id = "secret"

  labels = {
    label = "my-label"
  }

  annotations = {
    key1 = "someval"
    key2 = "someval2"
    key3 = "someval3"
    key4 = "someval4"
    key5 = "someval5"
  }

  replication {
    auto {}
  }
}
```

### Secret with Version
```hcl
resource "google_secret_manager_secret" "secret" {
  secret_id = "secret-1"
  replication {
    auto {}
  }
}

resource "google_secret_manager_secret_version" "secret-version-data" {
  secret = google_secret_manager_secret.secret.name
  secret_data = "secret-data"
}
```

### Regional Secret with Rotation
```hcl
data "google_project" "project" {}

resource "google_pubsub_topic" "topic" {
  name = "tf-topic"
}

resource "google_pubsub_topic_iam_member" "secrets_manager_access" {
  topic  = google_pubsub_topic.topic.name
  role   = "roles/pubsub.publisher"
  member = "serviceAccount:service-${data.google_project.project.number}@gcp-sa-secretmanager.iam.gserviceaccount.com"
}

resource "google_secret_manager_regional_secret" "regional-secret-with-rotation" {
  secret_id = "tf-reg-secret"
  location = "us-central1"

  topics {
    name = google_pubsub_topic.topic.id
  }

  rotation {
    rotation_period = "3600s"
    next_rotation_time = "2045-11-30T00:00:00Z"
  }

  depends_on = [
    google_pubsub_topic_iam_member.secrets_manager_access,
  ]
}
```

## Best Practices

### 1. Access Control
- Use the principle of least privilege when granting access
- Create service accounts with specific roles for accessing secrets
- Regularly audit secret access logs

### 2. Secret Rotation
- Implement automatic rotation for long-lived secrets
- Use Pub/Sub notifications to trigger rotation workflows
- Test rotation procedures regularly

### 3. Version Management
- Avoid using "latest" version in production
- Pin to specific versions for reproducibility
- Implement gradual rollout for secret updates

### 4. Encryption
- Secrets are encrypted at rest by default
- Consider using customer-managed encryption keys (CMEK) for additional control
- Encrypt sensitive data before storing as secrets

### 5. Organization
- Use consistent naming conventions
- Apply labels and annotations for better organization
- Group related secrets by project or environment

### 6. Cost Optimization
- Delete unused secret versions
- Use appropriate replication settings
- Monitor secret access patterns

### 7. Integration Best Practices
- Cache secret values appropriately to reduce API calls
- Implement retry logic with exponential backoff
- Handle secret access failures gracefully

### 8. Development Workflow
- Use different projects for development and production secrets
- Implement secret templates for consistency
- Document secret purposes and rotation schedules

## Common Use Cases for CQC ML Project

### 1. API Keys Storage
```python
# Store third-party API keys
api_key_secret = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "third-party-api-key",
        "secret": {
            "labels": {"service": "external-api", "env": "prod"},
            "replication": {"automatic": {}}
        },
    }
)
```

### 2. Database Credentials
```python
# Store database connection strings
db_secret = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "db-connection-string",
        "secret": {
            "labels": {"database": "bigquery", "env": "prod"},
            "replication": {"automatic": {}}
        },
    }
)
```

### 3. Service Account Keys
```python
# Store service account keys for cross-project access
sa_key_secret = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "service-account-key",
        "secret": {
            "labels": {"purpose": "cross-project-access"},
            "replication": {"automatic": {}}
        },
    }
)
```

### 4. ML Model Credentials
```python
# Store credentials for model registries or artifact stores
model_creds = client.create_secret(
    request={
        "parent": parent,
        "secret_id": "model-registry-credentials",
        "secret": {
            "labels": {"component": "ml-pipeline"},
            "replication": {"automatic": {}}
        },
    }
)
```

## Additional Resources

- [Secret Manager Documentation](https://cloud.google.com/secret-manager/docs)
- [Secret Manager Python Client Library](https://github.com/googleapis/google-cloud-python/tree/main/packages/google-cloud-secret-manager)
- [Secret Manager Best Practices](https://cloud.google.com/secret-manager/docs/best-practices)
- [IAM Documentation for Secret Manager](https://cloud.google.com/secret-manager/docs/access-control)