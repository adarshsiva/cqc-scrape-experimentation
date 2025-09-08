#!/bin/bash
# Run Comprehensive CQC Data Extraction
# This script demonstrates the usage patterns from plan.md Phase 1.1

set -e

PROJECT_ID=${GCP_PROJECT:-"machine-learning-exp-467008"}
REGION=${GCP_REGION:-"europe-west2"}

echo "==================================================================="
echo "Comprehensive CQC Data Extraction - As per plan.md Phase 1.1"
echo "==================================================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Example 1: Full comprehensive extraction (from plan.md)
echo "Example 1: Full Comprehensive Extraction"
echo "Command from plan.md Phase 1.1:"
echo ""
cat << 'EOF'
gcloud run jobs execute cqc-comprehensive-extractor \
  --region europe-west2 \
  --update-env-vars="
    ENDPOINTS=locations,providers,inspection_areas,reports,assessment_groups,
    MAX_LOCATIONS=50000,
    INCLUDE_HISTORICAL=true,
    FETCH_REPORTS=true,
    RATE_LIMIT=1800,
    PARALLEL_WORKERS=10" \
  --task-timeout=21600 --wait
EOF
echo ""
read -p "Execute full extraction? (y/N): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gcloud run jobs execute cqc-comprehensive-extractor \
      --region=$REGION \
      --project=$PROJECT_ID \
      --update-env-vars="ENDPOINTS=locations,providers,inspection_areas,reports,assessment_groups,MAX_LOCATIONS=50000,INCLUDE_HISTORICAL=true,FETCH_REPORTS=true,RATE_LIMIT=1800,PARALLEL_WORKERS=10" \
      --task-timeout=21600 \
      --wait
fi

echo ""
echo "-------------------------------------------------------------------"

# Example 2: Quick test run
echo "Example 2: Quick Test Run (1000 locations)"
echo ""
read -p "Execute test run? (y/N): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gcloud run jobs execute cqc-comprehensive-extractor \
      --region=$REGION \
      --project=$PROJECT_ID \
      --update-env-vars="ENDPOINTS=locations,providers,inspection_areas,MAX_LOCATIONS=1000,INCLUDE_HISTORICAL=false,FETCH_REPORTS=false,RATE_LIMIT=1800,PARALLEL_WORKERS=5" \
      --task-timeout=3600 \
      --wait
fi

echo ""
echo "-------------------------------------------------------------------"

# Example 3: Historical data only
echo "Example 3: Historical Inspection Data Focus"
echo ""
read -p "Execute historical data extraction? (y/N): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gcloud run jobs execute cqc-comprehensive-extractor \
      --region=$REGION \
      --project=$PROJECT_ID \
      --update-env-vars="ENDPOINTS=locations,inspection_areas,assessment_groups,MAX_LOCATIONS=10000,INCLUDE_HISTORICAL=true,FETCH_REPORTS=false,RATE_LIMIT=1800,PARALLEL_WORKERS=8" \
      --task-timeout=7200 \
      --wait
fi

echo ""
echo "-------------------------------------------------------------------"

# Example 4: Reports focus
echo "Example 4: Detailed Reports Extraction"
echo ""
read -p "Execute reports extraction? (y/N): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    gcloud run jobs execute cqc-comprehensive-extractor \
      --region=$REGION \
      --project=$PROJECT_ID \
      --update-env-vars="ENDPOINTS=locations,reports,MAX_LOCATIONS=5000,INCLUDE_HISTORICAL=false,FETCH_REPORTS=true,RATE_LIMIT=1000,PARALLEL_WORKERS=3" \
      --task-timeout=10800 \
      --wait
fi

echo ""
echo "==================================================================="
echo "Monitoring and Management Commands"
echo "==================================================================="
echo ""

echo "List recent executions:"
echo "gcloud run jobs executions list --job=cqc-comprehensive-extractor --region=$REGION --project=$PROJECT_ID"
echo ""

echo "View execution logs:"
echo "gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=cqc-comprehensive-extractor\" --project=$PROJECT_ID --limit=50"
echo ""

echo "Check Cloud Storage buckets:"
echo "gsutil ls gs://$PROJECT_ID-cqc-raw-data/comprehensive/"
echo "gsutil ls gs://$PROJECT_ID-cqc-processed/metadata/"
echo ""

echo "Query BigQuery results:"
echo "bq query --use_legacy_sql=false 'SELECT COUNT(*) as total_locations FROM \`$PROJECT_ID.cqc_data.ml_training_features_comprehensive\`'"
echo ""

echo "==================================================================="
echo "Next Steps (from plan.md)"
echo "==================================================================="
echo ""
echo "Phase 1.2: Feature Engineering"
echo "- Review extracted data in BigQuery"
echo "- Validate feature completeness"
echo "- Prepare for ML model training"
echo ""
echo "Phase 2: Dashboard Feature Extraction Service"
echo "- Build dashboard feature extraction service"
echo "- Create feature alignment and transformation logic" 
echo "- Validate feature compatibility between sources"
echo ""
echo "Phase 3: Unified ML Pipeline"
echo "- Train models on comprehensive CQC dataset"
echo "- Deploy models with dashboard feature support"
echo "- Create prediction API endpoints"
echo ""

echo "For detailed guidance, see: src/ml/README_comprehensive_extractor.md"