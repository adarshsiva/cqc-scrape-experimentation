output "project_id" {
  description = "GCP Project ID"
  value       = var.project_id
}

output "region" {
  description = "GCP region"
  value       = var.region
}

output "raw_data_bucket" {
  description = "GCS bucket for raw CQC data"
  value       = google_storage_bucket.raw_data.name
}

output "dataflow_temp_bucket" {
  description = "GCS bucket for Dataflow temporary files"
  value       = google_storage_bucket.temp_dataflow.name
}

output "ml_artifacts_bucket" {
  description = "GCS bucket for ML artifacts"
  value       = google_storage_bucket.ml_artifacts.name
}

output "bigquery_dataset" {
  description = "BigQuery dataset for CQC data"
  value       = google_bigquery_dataset.cqc_dataset.dataset_id
}

output "service_accounts" {
  description = "Service account emails"
  value = {
    cloud_functions = google_service_account.cf_service_account.email
    dataflow        = google_service_account.dataflow_service_account.email
    vertex_ai       = google_service_account.vertex_service_account.email
  }
}

output "cloud_functions" {
  description = "Cloud Function URLs"
  value = {
    ingestion  = google_cloudfunctions_function.ingestion.https_trigger_url
    prediction = google_cloudfunctions_function.prediction.https_trigger_url
  }
}

output "scheduler_job" {
  description = "Cloud Scheduler job name"
  value       = google_cloud_scheduler_job.cqc_ingestion.name
}

output "secret_ids" {
  description = "Secret Manager secret IDs"
  value = {
    subscription_key = google_secret_manager_secret.cqc_subscription_key.secret_id
    partner_code     = google_secret_manager_secret.cqc_partner_code.secret_id
  }
}