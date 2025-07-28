variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "europe-west2"  # London
}

variable "bq_dataset_id" {
  description = "BigQuery dataset ID"
  type        = string
  default     = "cqc_data"
}

variable "vertex_endpoint_id" {
  description = "Vertex AI endpoint ID for serving predictions"
  type        = string
  default     = ""  # Will be set after model deployment
}

variable "enable_apis" {
  description = "Whether to enable required GCP APIs"
  type        = bool
  default     = true
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "labels" {
  description = "Labels to apply to resources"
  type        = map(string)
  default = {
    project     = "cqc-predictor"
    managed_by  = "terraform"
  }
}