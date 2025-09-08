#!/bin/bash

# CQC ML Pipeline Monitoring Setup Script
# Sets up comprehensive monitoring for all pipeline components

set -e

PROJECT_ID=${GCP_PROJECT:-"machine-learning-exp-467008"}
REGION="europe-west2"

echo "=========================================="
echo "SETTING UP MONITORING FOR CQC ML PIPELINE"
echo "=========================================="

# Step 1: Create monitoring workspace
echo "Creating Cloud Monitoring workspace..."
gcloud alpha monitoring workspaces create \
    --project=$PROJECT_ID \
    --display-name="CQC ML Pipeline Monitoring" || echo "Workspace may already exist"

# Step 2: Create alert policies
echo "Creating alert policies..."

# Alert for Cloud Run job failures
cat > /tmp/cloudrun-alert.json <<EOF
{
  "displayName": "CQC Data Fetcher Job Failure",
  "conditions": [{
    "displayName": "Job execution failed",
    "conditionThreshold": {
      "filter": "resource.type=\"cloud_run_job\" AND metric.type=\"run.googleapis.com/job/completed_execution_count\" AND metric.label.result=\"failed\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 0,
      "duration": "60s",
      "aggregations": [{
        "alignmentPeriod": "60s",
        "perSeriesAligner": "ALIGN_RATE"
      }]
    }
  }],
  "notificationChannels": [],
  "alertStrategy": {
    "autoClose": "1800s"
  }
}
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/cloudrun-alert.json

# Alert for BigQuery quota issues
cat > /tmp/bigquery-alert.json <<EOF
{
  "displayName": "BigQuery Quota Exceeded",
  "conditions": [{
    "displayName": "Query quota exceeded",
    "conditionThreshold": {
      "filter": "resource.type=\"bigquery_project\" AND metric.type=\"bigquery.googleapis.com/quota/exceeded\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 0,
      "duration": "60s"
    }
  }],
  "notificationChannels": [],
  "alertStrategy": {
    "autoClose": "3600s"
  }
}
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/bigquery-alert.json

# Alert for Vertex AI training failures
cat > /tmp/vertex-alert.json <<EOF
{
  "displayName": "Vertex AI Training Job Failure",
  "conditions": [{
    "displayName": "Training job failed",
    "conditionThreshold": {
      "filter": "resource.type=\"ml_job\" AND metric.type=\"ml.googleapis.com/training/job/error_count\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 0,
      "duration": "60s"
    }
  }],
  "notificationChannels": [],
  "alertStrategy": {
    "autoClose": "7200s"
  }
}
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/vertex-alert.json

# Alert for Cloud Functions errors
cat > /tmp/functions-alert.json <<EOF
{
  "displayName": "Prediction API High Error Rate",
  "conditions": [{
    "displayName": "Error rate > 5%",
    "conditionThreshold": {
      "filter": "resource.type=\"cloud_function\" AND metric.type=\"cloudfunctions.googleapis.com/function/execution_count\" AND metric.label.status!=\"ok\"",
      "comparison": "COMPARISON_GT",
      "thresholdValue": 0.05,
      "duration": "300s",
      "aggregations": [{
        "alignmentPeriod": "60s",
        "perSeriesAligner": "ALIGN_RATE"
      }]
    }
  }],
  "notificationChannels": [],
  "alertStrategy": {
    "autoClose": "1800s"
  }
}
EOF

gcloud alpha monitoring policies create --policy-from-file=/tmp/functions-alert.json

# Step 3: Create custom dashboard
echo "Creating monitoring dashboard..."

cat > /tmp/dashboard.json <<EOF
{
  "displayName": "CQC ML Pipeline Dashboard",
  "mosaicLayout": {
    "columns": 12,
    "tiles": [
      {
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Cloud Run Job Executions",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"cloud_run_job\" AND metric.type=\"run.googleapis.com/job/completed_execution_count\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_RATE"
                  }
                }
              }
            }]
          }
        }
      },
      {
        "xPos": 6,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "BigQuery Slot Usage",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"bigquery_project\" AND metric.type=\"bigquery.googleapis.com/slots/total_allocated\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_MEAN"
                  }
                }
              }
            }]
          }
        }
      },
      {
        "yPos": 4,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Prediction API Latency",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"cloud_function\" AND metric.type=\"cloudfunctions.googleapis.com/function/execution_times\"",
                  "aggregation": {
                    "alignmentPeriod": "60s",
                    "perSeriesAligner": "ALIGN_DELTA"
                  }
                }
              }
            }]
          }
        }
      },
      {
        "xPos": 6,
        "yPos": 4,
        "width": 6,
        "height": 4,
        "widget": {
          "title": "Vertex AI Training Duration",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"ml_job\" AND metric.type=\"ml.googleapis.com/training/job/duration\"",
                  "aggregation": {
                    "alignmentPeriod": "3600s",
                    "perSeriesAligner": "ALIGN_MAX"
                  }
                }
              }
            }]
          }
        }
      },
      {
        "yPos": 8,
        "width": 12,
        "height": 4,
        "widget": {
          "title": "Storage Usage",
          "xyChart": {
            "dataSets": [{
              "timeSeriesQuery": {
                "timeSeriesFilter": {
                  "filter": "resource.type=\"gcs_bucket\" AND metric.type=\"storage.googleapis.com/storage/total_bytes\"",
                  "aggregation": {
                    "alignmentPeriod": "3600s",
                    "perSeriesAligner": "ALIGN_MAX"
                  }
                }
              }
            }]
          }
        }
      }
    ]
  }
}
EOF

gcloud monitoring dashboards create --config-from-file=/tmp/dashboard.json

# Step 4: Set up logging exports to BigQuery
echo "Setting up log exports to BigQuery..."

# Create dataset for logs
bq mk -d \
    --location=$REGION \
    --description="CQC ML Pipeline Logs" \
    ${PROJECT_ID}:cqc_logs || echo "Dataset may already exist"

# Export Cloud Run logs
gcloud logging sinks create cqc-cloudrun-sink \
    bigquery.googleapis.com/projects/${PROJECT_ID}/datasets/cqc_logs \
    --log-filter='resource.type="cloud_run_job"' || echo "Sink may already exist"

# Export Cloud Functions logs
gcloud logging sinks create cqc-functions-sink \
    bigquery.googleapis.com/projects/${PROJECT_ID}/datasets/cqc_logs \
    --log-filter='resource.type="cloud_function"' || echo "Sink may already exist"

# Export Vertex AI logs
gcloud logging sinks create cqc-vertex-sink \
    bigquery.googleapis.com/projects/${PROJECT_ID}/datasets/cqc_logs \
    --log-filter='resource.type="ml_job"' || echo "Sink may already exist"

# Step 5: Create uptime checks
echo "Creating uptime checks..."

# Check for prediction API
gcloud monitoring uptime-checks create https predict-api-check \
    --display-name="Prediction API Health Check" \
    --uri="https://europe-west2-${PROJECT_ID}.cloudfunctions.net/health-check" \
    --check-interval=5m || echo "Uptime check may already exist"

# Step 6: Set up cost monitoring
echo "Setting up cost monitoring..."

# Create budget alert
gcloud billing budgets create \
    --billing-account=$(gcloud beta billing projects describe $PROJECT_ID --format="value(billingAccountName)") \
    --display-name="CQC ML Pipeline Budget" \
    --budget-amount=1000 \
    --threshold-rule=percent=50 \
    --threshold-rule=percent=90 \
    --threshold-rule=percent=100 || echo "Budget may already exist"

# Step 7: Create log-based metrics
echo "Creating log-based metrics..."

# Metric for successful predictions
gcloud logging metrics create cqc_prediction_success \
    --description="Successful CQC rating predictions" \
    --log-metric-type=counter \
    --log-filter='resource.type="cloud_function"
    AND jsonPayload.success=true
    AND jsonPayload.function="predict_rating"' || echo "Metric may already exist"

# Metric for data ingestion volume
gcloud logging metrics create cqc_data_ingested \
    --description="Number of CQC locations ingested" \
    --log-metric-type=counter \
    --value-extractor='EXTRACT(jsonPayload.locations_fetched)' \
    --log-filter='resource.type="cloud_run_job"
    AND jsonPayload.job="cqc-complete-fetcher"' || echo "Metric may already exist"

# Step 8: Display monitoring summary
echo ""
echo "=========================================="
echo "MONITORING SETUP COMPLETE"
echo "=========================================="
echo ""
echo "✅ Alert policies created:"
echo "   - Cloud Run job failures"
echo "   - BigQuery quota exceeded"
echo "   - Vertex AI training failures"
echo "   - Prediction API error rate"
echo ""
echo "✅ Monitoring dashboard created"
echo "✅ Log exports to BigQuery configured"
echo "✅ Uptime checks configured"
echo "✅ Budget alerts configured"
echo "✅ Custom metrics created"
echo ""
echo "Access your monitoring dashboard at:"
echo "https://console.cloud.google.com/monitoring/dashboards"
echo ""
echo "View alerts at:"
echo "https://console.cloud.google.com/monitoring/alerting"
echo ""
echo "Check logs at:"
echo "https://console.cloud.google.com/logs"