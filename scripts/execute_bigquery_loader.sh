#!/bin/bash
# Execute the BigQuery loader job on Cloud Run

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
JOB_NAME="cqc-bigquery-loader"

echo "Executing CQC BigQuery loader job..."
echo ""

# Execute the job
gcloud run jobs execute ${JOB_NAME} \
  --region ${REGION} \
  --project ${PROJECT_ID} \
  --wait

# Check the status
if [ $? -eq 0 ]; then
    echo ""
    echo "Job executed successfully!"
    
    # Show recent executions
    echo ""
    echo "Recent job executions:"
    gcloud run jobs executions list \
      --job ${JOB_NAME} \
      --region ${REGION} \
      --project ${PROJECT_ID} \
      --limit 5 \
      --format="table(name,status,completionTime)"
    
    # Get the latest execution name
    LATEST_EXECUTION=$(gcloud run jobs executions list \
      --job ${JOB_NAME} \
      --region ${REGION} \
      --project ${PROJECT_ID} \
      --limit 1 \
      --format="value(name)")
    
    echo ""
    echo "To view logs for the latest execution:"
    echo "gcloud run jobs executions logs ${LATEST_EXECUTION} --region ${REGION} --project ${PROJECT_ID}"
else
    echo ""
    echo "Job execution failed!"
    echo "Check logs for details."
fi