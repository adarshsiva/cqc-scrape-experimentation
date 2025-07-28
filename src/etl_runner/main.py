import os
import json
import logging
from datetime import datetime
import functions_framework
from google.cloud import dataflow_v1beta3
from google.api_core import retry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
REGION = os.environ.get('GCP_REGION', 'europe-west2')
TEMPLATE_GCS_PATH = os.environ.get('DATAFLOW_TEMPLATE_PATH')

@functions_framework.http
def run_etl_pipeline(request):
    """Cloud Function to trigger Dataflow ETL pipeline."""
    try:
        # Parse request
        request_json = request.get_json() if request.method == 'POST' else {}
        data_type = request_json.get('data_type', 'locations')  # locations or providers
        
        # Create Dataflow client
        client = dataflow_v1beta3.JobsV1Beta3Client()
        
        # Job configuration
        job_name = f"cqc-etl-{data_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Template parameters
        template_parameters = {
            "input_pattern": f"gs://{PROJECT_ID}-cqc-raw-data/raw/{data_type}/*.json",
            "output_dataset": "cqc_data",
            "output_table": data_type,
            "data_type": data_type
        }
        
        # Environment configuration
        environment = dataflow_v1beta3.RuntimeEnvironment(
            temp_location=f"gs://{PROJECT_ID}-cqc-dataflow-temp/tmp",
            machine_type="n1-standard-2",
            max_workers=10,
            service_account_email=f"cqc-df-service-account@{PROJECT_ID}.iam.gserviceaccount.com"
        )
        
        # Create job request
        template_job_name = f"projects/{PROJECT_ID}/locations/{REGION}/jobs/{job_name}"
        
        if TEMPLATE_GCS_PATH:
            # Use pre-built template
            request = dataflow_v1beta3.LaunchTemplateRequest(
                project_id=PROJECT_ID,
                gcs_path=TEMPLATE_GCS_PATH,
                launch_parameters=dataflow_v1beta3.LaunchTemplateParameters(
                    job_name=job_name,
                    parameters=template_parameters,
                    environment=environment
                )
            )
            
            # Launch template
            response = client.launch_template(request=request, retry=retry.Retry())
        else:
            # Use flex template
            flex_template_spec = {
                "launch_parameter": {
                    "jobName": job_name,
                    "parameters": template_parameters,
                    "environment": {
                        "tempLocation": f"gs://{PROJECT_ID}-cqc-dataflow-temp/tmp",
                        "machineType": "n1-standard-2",
                        "maxWorkers": 10,
                        "serviceAccountEmail": f"cqc-df-service-account@{PROJECT_ID}.iam.gserviceaccount.com"
                    },
                    "containerSpecGcsPath": f"gs://{PROJECT_ID}-cqc-dataflow-templates/etl-pipeline/template.json"
                }
            }
            
            request = dataflow_v1beta3.LaunchFlexTemplateRequest(
                project_id=PROJECT_ID,
                location=REGION,
                launch_parameter=flex_template_spec["launch_parameter"]
            )
            
            response = client.launch_flex_template(request=request)
        
        logger.info(f"Launched Dataflow job: {response.job.name}")
        
        return {
            'status': 'success',
            'job_name': response.job.name,
            'job_id': response.job.id,
            'data_type': data_type,
            'timestamp': datetime.utcnow().isoformat()
        }, 200
        
    except Exception as e:
        logger.error(f"Failed to launch Dataflow job: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, 500

@functions_framework.http
def etl_status(request):
    """Check status of Dataflow jobs."""
    try:
        client = dataflow_v1beta3.JobsV1Beta3Client()
        
        # List recent jobs
        parent = f"projects/{PROJECT_ID}/locations/{REGION}"
        jobs = client.list_jobs(parent=parent, page_size=10)
        
        job_statuses = []
        for job in jobs:
            if job.name.startswith('cqc-etl-'):
                job_statuses.append({
                    'name': job.name,
                    'id': job.id,
                    'state': job.current_state.name,
                    'create_time': job.create_time.isoformat() if job.create_time else None,
                    'type': job.type.name
                })
        
        return {
            'status': 'success',
            'jobs': job_statuses,
            'timestamp': datetime.utcnow().isoformat()
        }, 200
        
    except Exception as e:
        logger.error(f"Failed to get job status: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, 500