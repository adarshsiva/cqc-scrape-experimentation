# Terraform Google Cloud Provider Documentation

## Overview

The Terraform Google Cloud Provider enables infrastructure as code management for Google Cloud Platform resources. This documentation covers provider configuration and resource usage relevant to the CQC ML project.

## Table of Contents

- [Provider Configuration](#provider-configuration)
- [Vertex AI Resources](#vertex-ai-resources)
- [Cloud Functions Resources](#cloud-functions-resources)
- [Cloud Run Resources](#cloud-run-resources)
- [BigQuery Resources](#bigquery-resources)
- [Cloud Storage Resources](#cloud-storage-resources)
- [Secret Manager Resources](#secret-manager-resources)
- [Cloud Composer Resources](#cloud-composer-resources)
- [Cloud Scheduler Resources](#cloud-scheduler-resources)
- [Best Practices](#best-practices)

## Provider Configuration

### Basic Provider Setup
```hcl
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = "my-project-id"
  region  = "us-central1"
}

provider "google-beta" {
  project = "my-project-id"
  region  = "us-central1"
}
```

## Vertex AI Resources

### Feature Online Store

#### Basic Feature Online Store with Bigtable
```hcl
resource "google_vertex_ai_feature_online_store" "featureonlinestore" {
  name     = "example_feature_view"
  labels = {
    foo = "bar"
  }
  region = "us-central1"
  bigtable {
    auto_scaling {
      min_node_count         = 1
      max_node_count         = 2
      cpu_utilization_target = 80
    }
  }
}
```

#### Feature Online Store with Embedding Management (Beta)
```hcl
resource "google_vertex_ai_feature_online_store" "featureonlinestore" {
  provider = google-beta
  name     = "example_feature_online_store_beta_bigtable"
  labels = {
    foo = "bar"
  }
  region = "us-central1"
  bigtable {
    auto_scaling {
      min_node_count         = 1
      max_node_count         = 2
      cpu_utilization_target = 80
    }
  }
  embedding_management {
    enabled = true
  }
  force_destroy = true
}
```

### Feature Online Store Feature View

#### Basic Feature View with BigQuery Source
```hcl
resource "google_bigquery_dataset" "tf-test-dataset" {
  dataset_id    = "example_feature_view"
  friendly_name = "test"
  description   = "This is a test description"
  location      = "US"
}

resource "google_bigquery_table" "tf-test-table" {
  deletion_protection = false
  dataset_id = google_bigquery_dataset.tf-test-dataset.dataset_id
  table_id   = "example_feature_view"
  schema     = <<EOF
  [
  {
    "name": "entity_id",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": "Test default entity_id"
  },
  {
    "name": "test_entity_column",
    "mode": "NULLABLE",
    "type": "STRING",
    "description": "test secondary entity column"
  },
  {
    "name": "feature_timestamp",
    "mode": "NULLABLE",
    "type": "TIMESTAMP",
    "description": "Default timestamp value"
  }
]
EOF
}

resource "google_vertex_ai_feature_online_store_featureview" "featureview" {
  name                 = "example_feature_view"
  region               = "us-central1"
  feature_online_store = google_vertex_ai_feature_online_store.featureonlinestore.name
  sync_config {
    cron = "0 0 * * *"
  }
  big_query_source {
    uri               = "bq://${google_bigquery_table.tf-test-table.project}.${google_bigquery_table.tf-test-table.dataset_id}.${google_bigquery_table.tf-test-table.table_id}"
    entity_id_columns = ["test_entity_column"]
  }
}
```

#### Feature View with Vector Search
```hcl
resource "google_vertex_ai_feature_online_store_featureview" "featureview_vector_search" {
  provider             = google-beta
  name                 = "example_feature_view_vector_search"
  region               = "us-central1"
  feature_online_store = google_vertex_ai_feature_online_store.featureonlinestore.name
  sync_config {
    cron = "0 0 * * *"
  }
  big_query_source {
    uri               = "bq://${google_bigquery_table.tf-test-table.project}.${google_bigquery_table.tf-test-table.dataset_id}.${google_bigquery_table.tf-test-table.table_id}"
    entity_id_columns = ["test_entity_column"]
  }
  vector_search_config {
    embedding_column      = "embedding"
    filter_columns        = ["country"]
    crowding_column       = "test_crowding_column"
    distance_measure_type = "DOT_PRODUCT_DISTANCE"
    tree_ah_config {
      leaf_node_embedding_count = "1000"
    }
    embedding_dimension = "2"
  }
}
```

### Feature Group
```hcl
resource "google_vertex_ai_feature_group" "feature_group" {
  name = "example_feature_group"
  description = "A sample feature group"
  region = "us-central1"
  labels = {
      label-one = "value-one"
  }
  big_query {
    big_query_source {
        # The source table must have a column named 'feature_timestamp' of type TIMESTAMP.
        input_uri = "bq://${google_bigquery_table.sample_table.project}.${google_bigquery_table.sample_table.dataset_id}.${google_bigquery_table.sample_table.table_id}"
    }
    entity_id_columns = ["feature_id"]
  }
}
```

### Vertex AI Endpoint
```hcl
resource "google_vertex_ai_endpoint" "endpoint" {
  name         = "endpoint-name"
  display_name = "sample-endpoint"
  description  = "A sample vertex endpoint"
  location     = "us-central1"
  region       = "us-central1"
  labels       = {
    label-one = "value-one"
  }
  network      = "projects/${data.google_project.project.number}/global/networks/${google_compute_network.vertex_network.name}"
  encryption_spec {
    kms_key_name = "kms-name"
  }
  predict_request_response_logging_config {
    bigquery_destination {
      output_uri = "bq://${data.google_project.project.project_id}.${google_bigquery_dataset.bq_dataset.dataset_id}.request_response_logging"
    }
    enabled       = true
    sampling_rate = 0.1
  }
  depends_on   = [
    google_service_networking_connection.vertex_vpc_connection
  ]
}
```

### Vertex AI Index

#### Batch Update Index
```hcl
resource "google_vertex_ai_index" "index" {
  labels = {
    foo = "bar"
  }
  region   = "us-central1"
  display_name = "test-index"
  description = "index for test"
  metadata {
    contents_delta_uri = "gs://${google_storage_bucket.bucket.name}/contents"
    config {
      dimensions = 2
      approximate_neighbors_count = 150
      shard_size = "SHARD_SIZE_SMALL"
      distance_measure_type = "DOT_PRODUCT_DISTANCE"
      algorithm_config {
        tree_ah_config {
          leaf_node_embedding_count = 500
          leaf_nodes_to_search_percent = 7
        }
      }
    }
  }
  index_update_method = "BATCH_UPDATE"
}
```

#### Streaming Update Index
```hcl
resource "google_vertex_ai_index" "index" {
  labels = {
    foo = "bar"
  }
  region   = "us-central1"
  display_name = "test-index"
  description = "index for test"
  metadata {
    contents_delta_uri = "gs://${google_storage_bucket.bucket.name}/contents"
    config {
      dimensions = 2
      shard_size = "SHARD_SIZE_LARGE"
      distance_measure_type = "COSINE_DISTANCE"
      feature_norm_type = "UNIT_L2_NORM"
      algorithm_config {
        brute_force_config {}
      }
    }
  }
  index_update_method = "STREAM_UPDATE"
}
```

## Cloud Functions Resources

### Cloud Functions (2nd Gen) with Secret Environment Variables
```hcl
locals {
  project = "my-project-name"
}

resource "google_storage_bucket" "bucket" {
  name     = "${local.project}-gcf-source"
  location = "US"
  uniform_bucket_level_access = true
}
 
resource "google_storage_bucket_object" "object" {
  name   = "function-source.zip"
  bucket = google_storage_bucket.bucket.name
  source = "function-source.zip"
}
 
resource "google_cloudfunctions2_function" "function" {
  name = "function-secret"
  location = "us-central1"
  description = "a new function"
 
  build_config {
    runtime = "nodejs20"
    entry_point = "helloHttp"
    source {
      storage_source {
        bucket = google_storage_bucket.bucket.name
        object = google_storage_bucket_object.object.name
      }
    }
  }
 
  service_config {
    max_instance_count  = 1
    available_memory    = "256M"
    timeout_seconds     = 60

    secret_environment_variables {
      key        = "TEST"
      project_id = local.project
      secret     = google_secret_manager_secret.secret.secret_id
      version    = "latest"
    }
  }
  depends_on = [google_secret_manager_secret_version.secret]
}

resource "google_secret_manager_secret" "secret" {
  secret_id = "secret"

  replication {
    user_managed {
      replicas {
        location = "us-central1"
      }
    }
  }  
}

resource "google_secret_manager_secret_version" "secret" {
  secret = google_secret_manager_secret.secret.name
  secret_data = "secret"
  enabled = true
}
```

### Cloud Functions with Secret Volumes
```hcl
resource "google_cloudfunctions2_function" "function" {
  name = "function-secret"
  location = "us-central1"
  description = "a new function"
 
  build_config {
    runtime = "nodejs20"
    entry_point = "helloHttp"
    source {
      storage_source {
        bucket = google_storage_bucket.bucket.name
        object = google_storage_bucket_object.object.name
      }
    }
  }
 
  service_config {
    max_instance_count  = 1
    available_memory    = "256M"
    timeout_seconds     = 60

    secret_volumes {
      mount_path = "/etc/secrets"
      project_id = local.project
      secret     = google_secret_manager_secret.secret.secret_id
    }
  }
  depends_on = [google_secret_manager_secret_version.secret]
}
```

## Cloud Run Resources

### Cloud Run v2 Service with Cloud SQL and Secrets
```hcl
resource "google_cloud_run_v2_service" "default" {
  name     = "cloudrun-service"
  location = "us-central1"
  deletion_protection = false
  ingress = "INGRESS_TRAFFIC_ALL"
  
  template {
    scaling {
      max_instance_count = 2
    }
  
    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.instance.connection_name]
      }
    }

    containers {
      image = "us-docker.pkg.dev/cloudrun/container/hello"

      env {
        name = "FOO"
        value = "bar"
      }
      env {
        name = "SECRET_ENV_VAR"
        value_source {
          secret_key_ref {
            secret = google_secret_manager_secret.secret.secret_id
            version = "1"
          }
        }
      }
      volume_mounts {
        name = "cloudsql"
        mount_path = "/cloudsql"
      }
    }
  }

  traffic {
    type = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
  depends_on = [google_secret_manager_secret_version.secret-version-data]
}
```

### Cloud Run v2 Job
```hcl
resource "google_cloud_run_v2_job" "default" {
  name     = "cloudrun-job"
  location = "us-central1"
  deletion_protection = false
  template {
    template{
      volumes {
        name = "cloudsql"
        cloud_sql_instance {
          instances = [google_sql_database_instance.instance.connection_name]
        }
      }

      containers {
        image = "us-docker.pkg.dev/cloudrun/container/job"

        env {
          name = "FOO"
          value = "bar"
        }
        env {
          name = "latestdclsecret"
          value_source {
            secret_key_ref {
              secret = google_secret_manager_secret.secret.secret_id
              version = "1"
            }
          }
        }
        volume_mounts {
          name = "cloudsql"
          mount_path = "/cloudsql"
        }
      }
    }
  }
}
```

## BigQuery Resources

### Dataset
```hcl
resource "google_bigquery_dataset" "dataset" {
  dataset_id                  = "example_dataset"
  friendly_name               = "test"
  description                 = "This is a test description"
  location                    = "US"
  default_table_expiration_ms = 3600000

  labels = {
    env = "default"
  }
}
```

### Table with Schema
```hcl
resource "google_bigquery_table" "table" {
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  table_id   = "example_table"
  deletion_protection = false

  time_partitioning {
    type = "DAY"
  }

  labels = {
    env = "default"
  }

  schema = <<EOF
[
  {
    "name": "permalink",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "The Permalink"
  },
  {
    "name": "state",
    "type": "STRING",
    "mode": "NULLABLE",
    "description": "State where the head office is located"
  }
]
EOF
}
```

### BigQuery Connection for External Data
```hcl
resource "google_bigquery_connection" "connection" {
   connection_id = "tf-test-connection"
   location      = "us-central1"
   friendly_name = "tf-test-connection"
   description   = "a bigquery connection for tf test"
   cloud_resource {}
}
```

### Dataset IAM
```hcl
resource "google_bigquery_dataset_iam_member" "viewer" {
  project = data.google_project.project.project_id
  dataset_id = google_bigquery_dataset.dataset.dataset_id
  role       = "roles/bigquery.dataViewer"
  member  = "serviceAccount:service-${google_project.project.number}@gcp-sa-aiplatform.iam.gserviceaccount.com"
}
```

## Cloud Storage Resources

### Storage Bucket
```hcl
resource "google_storage_bucket" "bucket" {
  name     = "vertex-ai-ml-bucket"
  location = "US"
  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }
  
  versioning {
    enabled = true
  }
}
```

### Storage Object
```hcl
resource "google_storage_bucket_object" "data" {
  name   = "ml-data/training-data.json"
  bucket = google_storage_bucket.bucket.name
  source = "local-training-data.json"
}
```

## Secret Manager Resources

### Secret with Automatic Replication
```hcl
resource "google_secret_manager_secret" "secret" {
  secret_id = "api-key"

  labels = {
    environment = "production"
  }

  replication {
    auto {}
  }
}
```

### Secret Version
```hcl
resource "google_secret_manager_secret_version" "secret_version" {
  secret = google_secret_manager_secret.secret.name
  secret_data = "my-secret-data"
}
```

### Secret IAM
```hcl
resource "google_secret_manager_secret_iam_member" "secret_accessor" {
  secret_id = google_secret_manager_secret.secret.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}
```

### Regional Secret with Rotation
```hcl
resource "google_secret_manager_regional_secret" "regional_secret" {
  secret_id = "regional-secret"
  location = "us-central1"

  rotation {
    rotation_period = "3600s"
    next_rotation_time = "2045-11-30T00:00:00Z"
  }
}
```

## Cloud Composer Resources

### Composer Environment
```hcl
resource "google_composer_environment" "environment" {
  name = "example-environment"
  region = "us-central1"
  
  config {
    node_count = 3
    
    node_config {
      zone         = "us-central1-a"
      machine_type = "n1-standard-1"
    }
    
    software_config {
      image_version = "composer-2-airflow-2"
      
      pypi_packages = {
        numpy = ""
        pandas = ">=1.0.0"
      }
      
      env_variables = {
        ENVIRONMENT = "production"
      }
    }
    
    workloads_config {
      scheduler {
        cpu        = 0.5
        memory_gb  = 1.875
        storage_gb = 1
        count      = 1
      }
      
      web_server {
        cpu        = 0.5
        memory_gb  = 1.875
        storage_gb = 1
      }
      
      worker {
        cpu        = 0.5
        memory_gb  = 1.875
        storage_gb = 1
        min_count  = 1
        max_count  = 3
      }
    }
  }
}
```

### User Workloads Secret
```hcl
resource "google_composer_user_workloads_secret" "secret" {
  environment = google_composer_environment.environment.name
  name = "airflow-secret"
  
  data = {
    username: base64encode("admin"),
    password: base64encode("secure-password"),
  }
}
```

## Cloud Scheduler Resources

### HTTP Target with OIDC
```hcl
resource "google_cloud_scheduler_job" "job" {
  name        = "test-job"
  description = "test http job"
  schedule    = "*/8 * * * *"
  region      = "us-central1"

  http_target {
    http_method = "POST"
    uri         = "https://example.com/ping"
    
    body = base64encode("{\"foo\":\"bar\"}")
    
    headers = {
      "Content-Type" = "application/json"
    }
    
    oidc_token {
      service_account_email = google_service_account.service_account.email
      audience              = "https://example.com"
    }
  }
  
  retry_config {
    retry_count = 3
    min_backoff_duration = "5s"
    max_backoff_duration = "3600s"
    max_doublings = 5
  }
}
```

### Pub/Sub Target
```hcl
resource "google_cloud_scheduler_job" "job" {
  name        = "test-job"
  description = "test job"
  schedule    = "*/2 * * * *"
  region      = "us-central1"

  pubsub_target {
    topic_name = google_pubsub_topic.topic.id
    data       = base64encode("test message")
    
    attributes = {
      foo = "bar"
    }
  }
}
```

## IAM Resources

### Service Account
```hcl
resource "google_service_account" "service_account" {
  account_id   = "ml-pipeline-sa"
  display_name = "ML Pipeline Service Account"
}
```

### Service Account IAM
```hcl
resource "google_project_iam_member" "sa_user" {
  project = var.project_id
  role    = "roles/iam.serviceAccountUser"
  member  = "serviceAccount:${google_service_account.service_account.email}"
}
```

### Workload Identity
```hcl
resource "google_service_account_iam_binding" "workload_identity" {
  service_account_id = google_service_account.service_account.name
  role               = "roles/iam.workloadIdentityUser"
  
  members = [
    "serviceAccount:${var.project_id}.svc.id.goog[default/ml-pipeline]"
  ]
}
```

## Best Practices

### 1. Resource Organization
- Use consistent naming conventions
- Apply appropriate labels and tags
- Group related resources in modules
- Use data sources for existing resources

### 2. State Management
- Use remote state backends (GCS)
- Enable state locking
- Implement state file encryption
- Regular state backups

### 3. Security
- Use service accounts with minimal permissions
- Implement resource-level IAM bindings
- Enable audit logging
- Use customer-managed encryption keys (CMEK)

### 4. Cost Optimization
- Use lifecycle rules for storage
- Implement auto-scaling where appropriate
- Set resource quotas
- Use preemptible instances for non-critical workloads

### 5. Dependency Management
- Use explicit dependencies with `depends_on`
- Order resource creation properly
- Handle circular dependencies
- Use `terraform graph` to visualize dependencies

### 6. Variable Management
- Use input variables for environment-specific values
- Implement variable validation
- Use sensitive variables for secrets
- Document variable purposes

### 7. Module Design
- Create reusable modules
- Version your modules
- Document module interfaces
- Test modules independently

### 8. CI/CD Integration
- Automate terraform plan/apply
- Implement approval workflows
- Use workspaces for environments
- Automated testing with terratest

## Example Module Structure

```
terraform/
├── modules/
│   ├── vertex-ai/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── README.md
│   ├── data-pipeline/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── README.md
│   └── ml-infrastructure/
│       ├── main.tf
│       ├── variables.tf
│       ├── outputs.tf
│       └── README.md
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   └── prod/
│       ├── main.tf
│       ├── variables.tf
│       └── terraform.tfvars
└── README.md
```

## Additional Resources

- [Terraform Google Provider Documentation](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Google Cloud Terraform Examples](https://github.com/GoogleCloudPlatform/terraform-google-examples)
- [Terraform Best Practices](https://www.terraform.io/docs/cloud/guides/recommended-practices/index.html)
- [Google Cloud Architecture Framework](https://cloud.google.com/architecture/framework)