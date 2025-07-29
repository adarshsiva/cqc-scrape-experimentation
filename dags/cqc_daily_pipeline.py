"""
CQC Daily Pipeline DAG

This DAG orchestrates the daily CQC data pipeline:
1. Fetches data from CQC API
2. Updates BigQuery tables
3. Runs weekly model retraining
4. Performs risk assessments for all locations
5. Sends alerts for high-risk locations
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.functions import CloudFunctionInvokeFunctionOperator
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCheckOperator,
    BigQueryExecuteQueryOperator,
    BigQueryTableExistenceOperator
)
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import (
    CreateCustomTrainingJobOperator
)
from airflow.providers.google.cloud.operators.vertex_ai.batch_prediction_job import (
    CreateBatchPredictionJobOperator
)
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowSkipException
import logging

# Configuration
PROJECT_ID = "{{ var.value.gcp_project_id }}"
REGION = "{{ var.value.gcp_region }}"
DATASET_ID = "{{ var.value.bq_dataset_id }}"
BUCKET_NAME = "{{ var.value.gcs_bucket_name }}"
CLOUD_FUNCTION_NAME = "{{ var.value.cf_ingestion_function }}"
VERTEX_AI_ENDPOINT = "{{ var.value.vertex_ai_endpoint }}"
ALERT_EMAIL = "{{ var.value.alert_email }}"

# Default DAG arguments
default_args = {
    'owner': 'cqc-pipeline',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': [ALERT_EMAIL],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}

# Create DAG
dag = DAG(
    'cqc_daily_pipeline',
    default_args=default_args,
    description='Daily CQC data pipeline with weekly model retraining',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False,
    max_active_runs=1,
    tags=['cqc', 'etl', 'ml', 'production'],
)

# Task 1: Trigger CQC API data ingestion
fetch_cqc_data = CloudFunctionInvokeFunctionOperator(
    task_id='fetch_cqc_data',
    function_id=CLOUD_FUNCTION_NAME,
    location=REGION,
    project_id=PROJECT_ID,
    input_data={
        'fetch_providers': True,
        'fetch_locations': True,
        'incremental': True
    },
    dag=dag,
)

# Task 2: Check if new data was fetched
check_new_data = BigQueryCheckOperator(
    task_id='check_new_data',
    sql=f"""
    SELECT COUNT(*) as new_records
    FROM `{PROJECT_ID}.{DATASET_ID}.providers_staging`
    WHERE DATE(ingestion_timestamp) = CURRENT_DATE()
    """,
    use_legacy_sql=False,
    dag=dag,
)

# Task 3: Load data from GCS to BigQuery staging tables
load_providers_to_bq = GCSToBigQueryOperator(
    task_id='load_providers_to_bq',
    bucket=BUCKET_NAME,
    source_objects=[f'raw/providers/{datetime.now().strftime("%Y/%m/%d")}/*.json'],
    destination_project_dataset_table=f'{PROJECT_ID}.{DATASET_ID}.providers_staging',
    source_format='NEWLINE_DELIMITED_JSON',
    write_disposition='WRITE_APPEND',
    autodetect=True,
    max_bad_records=100,
    dag=dag,
)

load_locations_to_bq = GCSToBigQueryOperator(
    task_id='load_locations_to_bq',
    bucket=BUCKET_NAME,
    source_objects=[f'raw/locations/{datetime.now().strftime("%Y/%m/%d")}/*.json'],
    destination_project_dataset_table=f'{PROJECT_ID}.{DATASET_ID}.locations_staging',
    source_format='NEWLINE_DELIMITED_JSON',
    write_disposition='WRITE_APPEND',
    autodetect=True,
    max_bad_records=100,
    dag=dag,
)

# Task 4: Run data quality checks
data_quality_checks = BigQueryCheckOperator(
    task_id='data_quality_checks',
    sql=f"""
    WITH quality_metrics AS (
        SELECT 
            'providers' as table_name,
            COUNT(*) as record_count,
            COUNT(DISTINCT provider_id) as unique_count,
            SUM(CASE WHEN provider_id IS NULL THEN 1 ELSE 0 END) as null_ids
        FROM `{PROJECT_ID}.{DATASET_ID}.providers_staging`
        WHERE DATE(ingestion_timestamp) = CURRENT_DATE()
        
        UNION ALL
        
        SELECT 
            'locations' as table_name,
            COUNT(*) as record_count,
            COUNT(DISTINCT location_id) as unique_count,
            SUM(CASE WHEN location_id IS NULL THEN 1 ELSE 0 END) as null_ids
        FROM `{PROJECT_ID}.{DATASET_ID}.locations_staging`
        WHERE DATE(ingestion_timestamp) = CURRENT_DATE()
    )
    SELECT 
        SUM(record_count) > 0 as has_records,
        SUM(null_ids) = 0 as no_null_ids
    FROM quality_metrics
    """,
    use_legacy_sql=False,
    dag=dag,
)

# Task 5: Merge staging data into production tables
merge_providers = BigQueryExecuteQueryOperator(
    task_id='merge_providers_to_production',
    sql=f"""
    MERGE `{PROJECT_ID}.{DATASET_ID}.providers` T
    USING (
        SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.providers_staging`
        WHERE DATE(ingestion_timestamp) = CURRENT_DATE()
        QUALIFY ROW_NUMBER() OVER (PARTITION BY provider_id ORDER BY ingestion_timestamp DESC) = 1
    ) S
    ON T.provider_id = S.provider_id
    WHEN MATCHED THEN
        UPDATE SET
            T.name = S.name,
            T.also_known_as = S.also_known_as,
            T.address = S.address,
            T.phone_number = S.phone_number,
            T.website = S.website,
            T.provider_type = S.provider_type,
            T.ownership_type = S.ownership_type,
            T.last_updated = S.ingestion_timestamp
    WHEN NOT MATCHED THEN
        INSERT (provider_id, name, also_known_as, address, phone_number, website, 
                provider_type, ownership_type, last_updated)
        VALUES (S.provider_id, S.name, S.also_known_as, S.address, S.phone_number, 
                S.website, S.provider_type, S.ownership_type, S.ingestion_timestamp)
    """,
    use_legacy_sql=False,
    dag=dag,
)

merge_locations = BigQueryExecuteQueryOperator(
    task_id='merge_locations_to_production',
    sql=f"""
    MERGE `{PROJECT_ID}.{DATASET_ID}.locations` T
    USING (
        SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.locations_staging`
        WHERE DATE(ingestion_timestamp) = CURRENT_DATE()
        QUALIFY ROW_NUMBER() OVER (PARTITION BY location_id ORDER BY ingestion_timestamp DESC) = 1
    ) S
    ON T.location_id = S.location_id
    WHEN MATCHED THEN
        UPDATE SET
            T.name = S.name,
            T.provider_id = S.provider_id,
            T.type = S.type,
            T.regulated_activities = S.regulated_activities,
            T.service_types = S.service_types,
            T.specialisms = S.specialisms,
            T.current_ratings = S.current_ratings,
            T.reports = S.reports,
            T.last_inspection_date = S.last_inspection_date,
            T.last_updated = S.ingestion_timestamp
    WHEN NOT MATCHED THEN
        INSERT (location_id, name, provider_id, type, regulated_activities, 
                service_types, specialisms, current_ratings, reports, 
                last_inspection_date, last_updated)
        VALUES (S.location_id, S.name, S.provider_id, S.type, S.regulated_activities,
                S.service_types, S.specialisms, S.current_ratings, S.reports,
                S.last_inspection_date, S.ingestion_timestamp)
    """,
    use_legacy_sql=False,
    dag=dag,
)

# Task 6: Create/update feature engineering tables
update_features = BigQueryExecuteQueryOperator(
    task_id='update_feature_tables',
    sql=f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.ml_features` AS
    WITH location_features AS (
        SELECT 
            l.location_id,
            l.provider_id,
            l.type as location_type,
            
            -- Rating features
            CAST(JSON_EXTRACT_SCALAR(l.current_ratings, '$.overall.rating') AS STRING) as current_overall_rating,
            DATE_DIFF(CURRENT_DATE(), l.last_inspection_date, DAY) as days_since_inspection,
            
            -- Activity features
            ARRAY_LENGTH(JSON_EXTRACT_ARRAY(l.regulated_activities)) as num_regulated_activities,
            ARRAY_LENGTH(JSON_EXTRACT_ARRAY(l.service_types)) as num_service_types,
            ARRAY_LENGTH(JSON_EXTRACT_ARRAY(l.specialisms)) as num_specialisms,
            
            -- Provider features
            p.provider_type,
            p.ownership_type,
            
            -- Historical features (to be expanded)
            COUNT(*) OVER (PARTITION BY l.provider_id) as provider_location_count
            
        FROM `{PROJECT_ID}.{DATASET_ID}.locations` l
        LEFT JOIN `{PROJECT_ID}.{DATASET_ID}.providers` p
        ON l.provider_id = p.provider_id
    )
    SELECT * FROM location_features
    """,
    use_legacy_sql=False,
    dag=dag,
)

# Task 7: Check if it's time for weekly model retraining
def check_retraining_schedule(**context):
    """Check if today is the scheduled day for model retraining"""
    # Retrain every Sunday
    if datetime.now().weekday() == 6:
        return 'trigger_model_training'
    else:
        return 'skip_model_training'

check_retraining = BranchPythonOperator(
    task_id='check_retraining_schedule',
    python_callable=check_retraining_schedule,
    dag=dag,
)

# Task 8a: Trigger model training (weekly)
trigger_model_training = CreateCustomTrainingJobOperator(
    task_id='trigger_model_training',
    project_id=PROJECT_ID,
    location=REGION,
    display_name=f'cqc-model-training-{datetime.now().strftime("%Y%m%d")}',
    worker_pool_specs=[{
        'machine_spec': {
            'machine_type': 'n1-standard-8',
            'accelerator_type': 'NVIDIA_TESLA_T4',
            'accelerator_count': 1,
        },
        'replica_count': 1,
        'python_package_spec': {
            'executor_image_uri': f'gcr.io/{PROJECT_ID}/cqc-training:latest',
            'package_uris': [f'gs://{BUCKET_NAME}/training/packages/trainer-0.1.tar.gz'],
            'python_module': 'trainer.task',
            'args': [
                '--project-id', PROJECT_ID,
                '--dataset-id', DATASET_ID,
                '--model-dir', f'gs://{BUCKET_NAME}/models/',
                '--experiment-name', 'weekly-training',
            ],
        },
    }],
    base_output_directory=f'gs://{BUCKET_NAME}/training-outputs/',
    dag=dag,
)

# Task 8b: Skip model training
skip_model_training = DummyOperator(
    task_id='skip_model_training',
    dag=dag,
)

# Task 9: Join after conditional training
join_after_training = DummyOperator(
    task_id='join_after_training',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

# Task 10: Run batch predictions for risk assessment
run_risk_assessment = CreateBatchPredictionJobOperator(
    task_id='run_risk_assessment',
    project_id=PROJECT_ID,
    location=REGION,
    job_display_name=f'cqc-risk-assessment-{datetime.now().strftime("%Y%m%d")}',
    model_name=f'projects/{PROJECT_ID}/locations/{REGION}/models/cqc-risk-model',
    instances_format='bigquery',
    bigquery_source_input_uri=f'bq://{PROJECT_ID}.{DATASET_ID}.ml_features',
    predictions_format='bigquery',
    bigquery_destination_output_uri=f'bq://{PROJECT_ID}.{DATASET_ID}.risk_predictions',
    dag=dag,
)

# Task 11: Identify high-risk locations
identify_high_risk = BigQueryExecuteQueryOperator(
    task_id='identify_high_risk_locations',
    sql=f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.high_risk_alerts` AS
    WITH risk_scores AS (
        SELECT 
            p.location_id,
            l.name as location_name,
            l.provider_id,
            pr.name as provider_name,
            p.predicted_risk_score,
            p.risk_factors,
            CASE 
                WHEN p.predicted_risk_score >= 0.8 THEN 'CRITICAL'
                WHEN p.predicted_risk_score >= 0.6 THEN 'HIGH'
                WHEN p.predicted_risk_score >= 0.4 THEN 'MEDIUM'
                ELSE 'LOW'
            END as risk_level,
            CURRENT_DATETIME() as assessment_timestamp
        FROM `{PROJECT_ID}.{DATASET_ID}.risk_predictions` p
        JOIN `{PROJECT_ID}.{DATASET_ID}.locations` l
        ON p.location_id = l.location_id
        JOIN `{PROJECT_ID}.{DATASET_ID}.providers` pr
        ON l.provider_id = pr.provider_id
        WHERE DATE(p.prediction_timestamp) = CURRENT_DATE()
    )
    SELECT * FROM risk_scores
    WHERE risk_level IN ('CRITICAL', 'HIGH')
    ORDER BY predicted_risk_score DESC
    """,
    use_legacy_sql=False,
    dag=dag,
)

# Task 12: Check if there are high-risk alerts
def check_alerts(**context):
    """Check if there are any high-risk locations to alert about"""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
    SELECT COUNT(*) as alert_count
    FROM `{PROJECT_ID}.{DATASET_ID}.high_risk_alerts`
    WHERE DATE(assessment_timestamp) = CURRENT_DATE()
    """
    
    result = client.query(query).result()
    alert_count = list(result)[0].alert_count
    
    if alert_count > 0:
        return 'send_alert_notifications'
    else:
        return 'no_alerts_needed'

check_for_alerts = BranchPythonOperator(
    task_id='check_for_alerts',
    python_callable=check_alerts,
    dag=dag,
)

# Task 13a: Send alert notifications
def prepare_alert_email(**context):
    """Prepare email content for high-risk alerts"""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=PROJECT_ID)
    query = f"""
    SELECT 
        location_name,
        provider_name,
        risk_level,
        ROUND(predicted_risk_score, 3) as risk_score,
        risk_factors
    FROM `{PROJECT_ID}.{DATASET_ID}.high_risk_alerts`
    WHERE DATE(assessment_timestamp) = CURRENT_DATE()
    ORDER BY predicted_risk_score DESC
    LIMIT 20
    """
    
    results = client.query(query).result()
    
    email_content = """
    <h2>CQC High Risk Location Alerts</h2>
    <p>The following locations have been identified as high-risk based on today's assessment:</p>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr>
            <th>Location</th>
            <th>Provider</th>
            <th>Risk Level</th>
            <th>Risk Score</th>
            <th>Key Risk Factors</th>
        </tr>
    """
    
    for row in results:
        email_content += f"""
        <tr>
            <td>{row.location_name}</td>
            <td>{row.provider_name}</td>
            <td><strong>{row.risk_level}</strong></td>
            <td>{row.risk_score}</td>
            <td>{row.risk_factors}</td>
        </tr>
        """
    
    email_content += """
    </table>
    <p>Please review these locations for immediate attention.</p>
    <p>Generated by CQC Daily Pipeline</p>
    """
    
    return email_content

prepare_alerts = PythonOperator(
    task_id='prepare_alert_email',
    python_callable=prepare_alert_email,
    dag=dag,
)

send_alert_notifications = EmailOperator(
    task_id='send_alert_notifications',
    to=[ALERT_EMAIL],
    subject=f'CQC High Risk Alerts - {datetime.now().strftime("%Y-%m-%d")}',
    html_content="{{ task_instance.xcom_pull(task_ids='prepare_alert_email') }}",
    dag=dag,
)

# Task 13b: No alerts needed
no_alerts_needed = DummyOperator(
    task_id='no_alerts_needed',
    dag=dag,
)

# Task 14: Pipeline completion
pipeline_complete = DummyOperator(
    task_id='pipeline_complete',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

# Define task dependencies
fetch_cqc_data >> check_new_data >> [load_providers_to_bq, load_locations_to_bq]

[load_providers_to_bq, load_locations_to_bq] >> data_quality_checks

data_quality_checks >> [merge_providers, merge_locations]

[merge_providers, merge_locations] >> update_features

update_features >> check_retraining

check_retraining >> [trigger_model_training, skip_model_training]

[trigger_model_training, skip_model_training] >> join_after_training

join_after_training >> run_risk_assessment

run_risk_assessment >> identify_high_risk

identify_high_risk >> check_for_alerts

check_for_alerts >> [prepare_alerts, no_alerts_needed]

prepare_alerts >> send_alert_notifications

[send_alert_notifications, no_alerts_needed] >> pipeline_complete