terraform {
  required_version = ">= 1.0"
  
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
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "cloudfunctions.googleapis.com",
    "cloudbuild.googleapis.com",
    "cloudscheduler.googleapis.com",
    "storage.googleapis.com",
    "bigquery.googleapis.com",
    "dataflow.googleapis.com",
    "composer.googleapis.com",
    "aiplatform.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com"
  ])
  
  service            = each.value
  disable_on_destroy = false
}

# Service Account for Cloud Functions
resource "google_service_account" "cf_service_account" {
  account_id   = "cqc-cf-service-account"
  display_name = "CQC Cloud Functions Service Account"
  description  = "Service account for CQC Cloud Functions"
}

# Service Account for Dataflow
resource "google_service_account" "dataflow_service_account" {
  account_id   = "cqc-dataflow-service-account"
  display_name = "CQC Dataflow Service Account"
  description  = "Service account for CQC Dataflow pipelines"
}

# Service Account for Vertex AI
resource "google_service_account" "vertex_service_account" {
  account_id   = "cqc-vertex-service-account"
  display_name = "CQC Vertex AI Service Account"
  description  = "Service account for CQC Vertex AI operations"
}

# Storage Buckets
resource "google_storage_bucket" "raw_data" {
  name          = "${var.project_id}-cqc-raw-data"
  location      = var.region
  force_destroy = false
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
  
  versioning {
    enabled = true
  }
}

resource "google_storage_bucket" "temp_dataflow" {
  name          = "${var.project_id}-cqc-dataflow-temp"
  location      = var.region
  force_destroy = true
  
  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }
}

resource "google_storage_bucket" "ml_artifacts" {
  name          = "${var.project_id}-cqc-ml-artifacts"
  location      = var.region
  force_destroy = false
  
  versioning {
    enabled = true
  }
}

# BigQuery Dataset
resource "google_bigquery_dataset" "cqc_dataset" {
  dataset_id                  = var.bq_dataset_id
  friendly_name               = "CQC Data"
  description                 = "Dataset for CQC ratings data and predictions"
  location                    = var.region
  default_table_expiration_ms = null
  
  access {
    role          = "OWNER"
    user_by_email = google_service_account.cf_service_account.email
  }
  
  access {
    role          = "OWNER"
    user_by_email = google_service_account.dataflow_service_account.email
  }
  
  access {
    role          = "OWNER"
    user_by_email = google_service_account.vertex_service_account.email
  }
}

# BigQuery Tables
resource "google_bigquery_table" "locations" {
  dataset_id = google_bigquery_dataset.cqc_dataset.dataset_id
  table_id   = "locations"
  
  schema = file("${path.module}/../config/bigquery_schema.json")
  
  time_partitioning {
    type  = "DAY"
    field = "ingestion_timestamp"
  }
  
  clustering = ["region", "overall_rating"]
}

resource "google_bigquery_table" "providers" {
  dataset_id = google_bigquery_dataset.cqc_dataset.dataset_id
  table_id   = "providers"
  
  schema = file("${path.module}/../config/bigquery_schema.json")
  
  time_partitioning {
    type  = "DAY"
    field = "ingestion_timestamp"
  }
}

resource "google_bigquery_table" "predictions" {
  dataset_id = google_bigquery_dataset.cqc_dataset.dataset_id
  table_id   = "predictions"
  
  schema = file("${path.module}/../config/bigquery_schema.json")
  
  time_partitioning {
    type  = "DAY"
    field = "prediction_timestamp"
  }
}

# Secret Manager Secrets
resource "google_secret_manager_secret" "cqc_subscription_key" {
  secret_id = "cqc-subscription-key"
  
  replication {
    automatic = true
  }
}

resource "google_secret_manager_secret" "cqc_partner_code" {
  secret_id = "cqc-partner-code"
  
  replication {
    automatic = true
  }
}

# Cloud Scheduler Job
resource "google_cloud_scheduler_job" "cqc_ingestion" {
  name        = "cqc-weekly-ingestion"
  description = "Weekly CQC data ingestion"
  schedule    = "0 2 * * 1"  # Every Monday at 2 AM
  time_zone   = "Europe/London"
  
  http_target {
    uri         = google_cloudfunctions_function.ingestion.https_trigger_url
    http_method = "POST"
    
    oidc_token {
      service_account_email = google_service_account.cf_service_account.email
    }
  }
  
  depends_on = [google_cloudfunctions_function.ingestion]
}

# Cloud Functions
resource "google_cloudfunctions_function" "ingestion" {
  name        = "cqc-data-ingestion"
  description = "Ingests data from CQC API"
  runtime     = "python311"
  
  available_memory_mb   = 512
  source_archive_bucket = google_storage_bucket.cf_source.name
  source_archive_object = google_storage_bucket_object.ingestion_source.name
  entry_point          = "ingest_cqc_data"
  
  service_account_email = google_service_account.cf_service_account.email
  
  environment_variables = {
    GCP_PROJECT = var.project_id
    GCS_BUCKET  = google_storage_bucket.raw_data.name
  }
  
  trigger_http = true
}

resource "google_cloudfunctions_function" "prediction" {
  name        = "cqc-rating-prediction"
  description = "Serves CQC rating predictions"
  runtime     = "python311"
  
  available_memory_mb   = 1024
  source_archive_bucket = google_storage_bucket.cf_source.name
  source_archive_object = google_storage_bucket_object.prediction_source.name
  entry_point          = "predict_cqc_rating"
  
  service_account_email = google_service_account.cf_service_account.email
  
  environment_variables = {
    GCP_PROJECT        = var.project_id
    VERTEX_ENDPOINT_ID = var.vertex_endpoint_id
  }
  
  trigger_http = true
}

# Source bucket for Cloud Functions
resource "google_storage_bucket" "cf_source" {
  name          = "${var.project_id}-cqc-cf-source"
  location      = var.region
  force_destroy = true
}

# Upload source code (in practice, this would be done by CI/CD)
resource "google_storage_bucket_object" "ingestion_source" {
  name   = "ingestion-${data.archive_file.ingestion.output_md5}.zip"
  bucket = google_storage_bucket.cf_source.name
  source = data.archive_file.ingestion.output_path
}

resource "google_storage_bucket_object" "prediction_source" {
  name   = "prediction-${data.archive_file.prediction.output_md5}.zip"
  bucket = google_storage_bucket.cf_source.name
  source = data.archive_file.prediction.output_path
}

data "archive_file" "ingestion" {
  type        = "zip"
  source_dir  = "${path.module}/../src/ingestion"
  output_path = "${path.module}/.terraform/tmp/ingestion.zip"
}

data "archive_file" "prediction" {
  type        = "zip"
  source_dir  = "${path.module}/../src/prediction"
  output_path = "${path.module}/.terraform/tmp/prediction.zip"
}

# IAM Bindings
resource "google_project_iam_member" "cf_secret_accessor" {
  project = var.project_id
  role    = "roles/secretmanager.secretAccessor"
  member  = "serviceAccount:${google_service_account.cf_service_account.email}"
}

resource "google_project_iam_member" "cf_storage_admin" {
  project = var.project_id
  role    = "roles/storage.admin"
  member  = "serviceAccount:${google_service_account.cf_service_account.email}"
}

resource "google_project_iam_member" "dataflow_worker" {
  project = var.project_id
  role    = "roles/dataflow.worker"
  member  = "serviceAccount:${google_service_account.dataflow_service_account.email}"
}

resource "google_project_iam_member" "dataflow_bigquery" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.dataflow_service_account.email}"
}

resource "google_project_iam_member" "vertex_ai_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.vertex_service_account.email}"
}

# Outputs
output "ingestion_function_url" {
  value = google_cloudfunctions_function.ingestion.https_trigger_url
}

output "prediction_function_url" {
  value = google_cloudfunctions_function.prediction.https_trigger_url
}

output "bigquery_dataset_id" {
  value = google_bigquery_dataset.cqc_dataset.dataset_id
}