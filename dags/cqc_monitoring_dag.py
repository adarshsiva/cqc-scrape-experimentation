"""
CQC System Monitoring DAG

This DAG monitors the health of the CQC pipeline system:
1. Checks data freshness in BigQuery
2. Monitors Cloud Function execution status
3. Validates model endpoint health
4. Monitors storage usage and costs
5. Checks for data quality issues
6. Sends alerts for any system anomalies
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import (
    BigQueryCheckOperator,
    BigQueryValueCheckOperator,
    BigQueryExecuteQueryOperator
)
from airflow.providers.google.cloud.operators.functions import CloudFunctionInvokeFunctionOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.providers.google.cloud.sensors.gcs import GCSObjectsWithPrefixExistenceSensor
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowException
import logging
from typing import Dict, List, Any

# Configuration
PROJECT_ID = "{{ var.value.gcp_project_id }}"
REGION = "{{ var.value.gcp_region }}"
DATASET_ID = "{{ var.value.bq_dataset_id }}"
BUCKET_NAME = "{{ var.value.gcs_bucket_name }}"
CLOUD_FUNCTION_NAME = "{{ var.value.cf_ingestion_function }}"
VERTEX_AI_ENDPOINT = "{{ var.value.vertex_ai_endpoint }}"
ALERT_EMAIL = "{{ var.value.alert_email }}"
MONITORING_DATASET = "{{ var.value.monitoring_dataset_id }}"

# Monitoring thresholds
DATA_FRESHNESS_HOURS = 26  # Alert if data is older than 26 hours
MIN_DAILY_RECORDS = 100  # Minimum expected records per day
MAX_ERROR_RATE = 0.05  # 5% error rate threshold
MAX_STORAGE_GB = 1000  # Alert if storage exceeds 1TB
MAX_DAILY_COST = 100  # Alert if daily cost exceeds $100

# Default DAG arguments
default_args = {
    'owner': 'cqc-monitoring',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': [ALERT_EMAIL],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'cqc_monitoring_dag',
    default_args=default_args,
    description='Monitor CQC pipeline system health',
    schedule_interval='0 */4 * * *',  # Run every 4 hours
    catchup=False,
    max_active_runs=1,
    tags=['cqc', 'monitoring', 'operations'],
)

# Task 1: Check data freshness
check_data_freshness = BigQueryValueCheckOperator(
    task_id='check_data_freshness',
    sql=f"""
    WITH freshness_check AS (
        SELECT 
            'providers' as table_name,
            MAX(last_updated) as latest_update,
            DATETIME_DIFF(CURRENT_DATETIME(), MAX(last_updated), HOUR) as hours_since_update
        FROM `{PROJECT_ID}.{DATASET_ID}.providers`
        
        UNION ALL
        
        SELECT 
            'locations' as table_name,
            MAX(last_updated) as latest_update,
            DATETIME_DIFF(CURRENT_DATETIME(), MAX(last_updated), HOUR) as hours_since_update
        FROM `{PROJECT_ID}.{DATASET_ID}.locations`
    )
    SELECT MAX(hours_since_update) as max_hours_since_update
    FROM freshness_check
    """,
    pass_value={DATA_FRESHNESS_HOURS},
    tolerance=0,
    use_legacy_sql=False,
    dag=dag,
)

# Task 2: Check daily record counts
check_record_counts = BigQueryCheckOperator(
    task_id='check_daily_record_counts',
    sql=f"""
    WITH daily_counts AS (
        SELECT 
            DATE(ingestion_timestamp) as ingestion_date,
            COUNT(*) as record_count
        FROM `{PROJECT_ID}.{DATASET_ID}.providers_staging`
        WHERE DATE(ingestion_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
        GROUP BY ingestion_date
        
        UNION ALL
        
        SELECT 
            DATE(ingestion_timestamp) as ingestion_date,
            COUNT(*) as record_count
        FROM `{PROJECT_ID}.{DATASET_ID}.locations_staging`
        WHERE DATE(ingestion_timestamp) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
        GROUP BY ingestion_date
    )
    SELECT 
        MIN(record_count) >= {MIN_DAILY_RECORDS} as meets_minimum
    FROM daily_counts
    WHERE ingestion_date = CURRENT_DATE()
    """,
    use_legacy_sql=False,
    dag=dag,
)

# Task 3: Check Cloud Function execution logs
def check_function_health(**context):
    """Check Cloud Function execution health from logs"""
    from google.cloud import logging
    from datetime import datetime, timedelta
    
    client = logging.Client(project=PROJECT_ID)
    
    # Query logs for the last 24 hours
    yesterday = datetime.now() - timedelta(days=1)
    filter_str = f"""
    resource.type="cloud_function"
    resource.labels.function_name="{CLOUD_FUNCTION_NAME}"
    timestamp >= "{yesterday.isoformat()}"
    """
    
    entries = list(client.list_entries(filter_=filter_str))
    
    total_executions = 0
    errors = 0
    
    for entry in entries:
        total_executions += 1
        if entry.severity in ['ERROR', 'CRITICAL']:
            errors += 1
    
    error_rate = errors / total_executions if total_executions > 0 else 0
    
    context['task_instance'].xcom_push(key='function_error_rate', value=error_rate)
    context['task_instance'].xcom_push(key='function_executions', value=total_executions)
    
    if error_rate > MAX_ERROR_RATE:
        raise AirflowException(f"Cloud Function error rate ({error_rate:.2%}) exceeds threshold ({MAX_ERROR_RATE:.2%})")
    
    return {
        'total_executions': total_executions,
        'errors': errors,
        'error_rate': error_rate
    }

check_function_logs = PythonOperator(
    task_id='check_function_health',
    python_callable=check_function_health,
    dag=dag,
)

# Task 4: Check model endpoint health
def check_model_endpoint(**context):
    """Test model endpoint with a sample prediction"""
    from google.cloud import aiplatform
    import json
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Sample test instance
    test_instance = {
        "location_type": "Care home with nursing",
        "days_since_inspection": 180,
        "num_regulated_activities": 3,
        "num_service_types": 2,
        "provider_type": "Individual",
        "ownership_type": "Private"
    }
    
    try:
        endpoint = aiplatform.Endpoint(VERTEX_AI_ENDPOINT)
        
        # Test prediction
        start_time = datetime.now()
        predictions = endpoint.predict(instances=[test_instance])
        latency = (datetime.now() - start_time).total_seconds()
        
        context['task_instance'].xcom_push(key='endpoint_latency', value=latency)
        context['task_instance'].xcom_push(key='endpoint_status', value='healthy')
        
        logging.info(f"Model endpoint healthy. Latency: {latency:.3f}s")
        
        return {
            'status': 'healthy',
            'latency': latency,
            'prediction': predictions.predictions[0] if predictions.predictions else None
        }
        
    except Exception as e:
        context['task_instance'].xcom_push(key='endpoint_status', value='unhealthy')
        raise AirflowException(f"Model endpoint unhealthy: {str(e)}")

test_model_endpoint = PythonOperator(
    task_id='check_model_endpoint',
    python_callable=check_model_endpoint,
    dag=dag,
)

# Task 5: Monitor storage usage
def check_storage_usage(**context):
    """Monitor GCS bucket storage usage"""
    from google.cloud import storage
    
    client = storage.Client(project=PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    total_size = 0
    file_count = 0
    
    # Calculate total storage usage
    for blob in bucket.list_blobs():
        total_size += blob.size
        file_count += 1
    
    total_size_gb = total_size / (1024**3)  # Convert to GB
    
    context['task_instance'].xcom_push(key='storage_size_gb', value=total_size_gb)
    context['task_instance'].xcom_push(key='file_count', value=file_count)
    
    if total_size_gb > MAX_STORAGE_GB:
        raise AirflowException(f"Storage usage ({total_size_gb:.2f} GB) exceeds threshold ({MAX_STORAGE_GB} GB)")
    
    return {
        'total_size_gb': total_size_gb,
        'file_count': file_count
    }

monitor_storage = PythonOperator(
    task_id='check_storage_usage',
    python_callable=check_storage_usage,
    dag=dag,
)

# Task 6: Check data quality metrics
check_data_quality = BigQueryExecuteQueryOperator(
    task_id='check_data_quality_metrics',
    sql=f"""
    CREATE OR REPLACE TABLE `{PROJECT_ID}.{MONITORING_DATASET}.data_quality_metrics` AS
    WITH quality_metrics AS (
        -- Provider data quality
        SELECT 
            'providers' as table_name,
            COUNT(*) as total_records,
            COUNT(DISTINCT provider_id) as unique_records,
            SUM(CASE WHEN provider_id IS NULL THEN 1 ELSE 0 END) as null_ids,
            SUM(CASE WHEN name IS NULL OR name = '' THEN 1 ELSE 0 END) as missing_names,
            SUM(CASE WHEN address IS NULL THEN 1 ELSE 0 END) as missing_addresses,
            CURRENT_DATETIME() as check_timestamp
        FROM `{PROJECT_ID}.{DATASET_ID}.providers`
        
        UNION ALL
        
        -- Location data quality
        SELECT 
            'locations' as table_name,
            COUNT(*) as total_records,
            COUNT(DISTINCT location_id) as unique_records,
            SUM(CASE WHEN location_id IS NULL THEN 1 ELSE 0 END) as null_ids,
            SUM(CASE WHEN name IS NULL OR name = '' THEN 1 ELSE 0 END) as missing_names,
            SUM(CASE WHEN current_ratings IS NULL THEN 1 ELSE 0 END) as missing_ratings,
            CURRENT_DATETIME() as check_timestamp
        FROM `{PROJECT_ID}.{DATASET_ID}.locations`
    )
    SELECT 
        *,
        ROUND(100.0 * null_ids / NULLIF(total_records, 0), 2) as null_id_rate,
        ROUND(100.0 * missing_names / NULLIF(total_records, 0), 2) as missing_name_rate
    FROM quality_metrics
    """,
    use_legacy_sql=False,
    dag=dag,
)

# Task 7: Check for duplicate records
check_duplicates = BigQueryCheckOperator(
    task_id='check_duplicate_records',
    sql=f"""
    WITH duplicate_check AS (
        SELECT 
            'providers' as table_name,
            COUNT(*) - COUNT(DISTINCT provider_id) as duplicate_count
        FROM `{PROJECT_ID}.{DATASET_ID}.providers`
        
        UNION ALL
        
        SELECT 
            'locations' as table_name,
            COUNT(*) - COUNT(DISTINCT location_id) as duplicate_count
        FROM `{PROJECT_ID}.{DATASET_ID}.locations`
    )
    SELECT SUM(duplicate_count) = 0 as no_duplicates
    FROM duplicate_check
    """,
    use_legacy_sql=False,
    dag=dag,
)

# Task 8: Monitor BigQuery costs
def check_bigquery_costs(**context):
    """Monitor BigQuery usage and estimated costs"""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=PROJECT_ID)
    
    # Query to get today's query costs
    query = f"""
    SELECT 
        SUM(total_bytes_processed) / POW(10, 12) as tb_processed,
        COUNT(*) as query_count,
        -- Estimate cost at $5 per TB processed
        ROUND(SUM(total_bytes_processed) / POW(10, 12) * 5, 2) as estimated_cost_usd
    FROM `{PROJECT_ID}.region-{REGION}.INFORMATION_SCHEMA.JOBS_BY_PROJECT`
    WHERE DATE(creation_time) = CURRENT_DATE()
        AND statement_type = 'SELECT'
    """
    
    result = list(client.query(query).result())[0]
    
    context['task_instance'].xcom_push(key='bq_tb_processed', value=result.tb_processed or 0)
    context['task_instance'].xcom_push(key='bq_query_count', value=result.query_count or 0)
    context['task_instance'].xcom_push(key='bq_estimated_cost', value=result.estimated_cost_usd or 0)
    
    if result.estimated_cost_usd and result.estimated_cost_usd > MAX_DAILY_COST:
        logging.warning(f"BigQuery daily cost (${result.estimated_cost_usd}) exceeds threshold (${MAX_DAILY_COST})")
    
    return {
        'tb_processed': result.tb_processed or 0,
        'query_count': result.query_count or 0,
        'estimated_cost_usd': result.estimated_cost_usd or 0
    }

monitor_costs = PythonOperator(
    task_id='check_bigquery_costs',
    python_callable=check_bigquery_costs,
    dag=dag,
)

# Task 9: Compile monitoring report
def compile_monitoring_report(**context):
    """Compile all monitoring metrics into a report"""
    ti = context['task_instance']
    
    # Gather all metrics from XCom
    metrics = {
        'function_error_rate': ti.xcom_pull(task_ids='check_function_health', key='function_error_rate'),
        'function_executions': ti.xcom_pull(task_ids='check_function_health', key='function_executions'),
        'endpoint_status': ti.xcom_pull(task_ids='check_model_endpoint', key='endpoint_status'),
        'endpoint_latency': ti.xcom_pull(task_ids='check_model_endpoint', key='endpoint_latency'),
        'storage_size_gb': ti.xcom_pull(task_ids='check_storage_usage', key='storage_size_gb'),
        'file_count': ti.xcom_pull(task_ids='check_storage_usage', key='file_count'),
        'bq_tb_processed': ti.xcom_pull(task_ids='check_bigquery_costs', key='bq_tb_processed'),
        'bq_query_count': ti.xcom_pull(task_ids='check_bigquery_costs', key='bq_query_count'),
        'bq_estimated_cost': ti.xcom_pull(task_ids='check_bigquery_costs', key='bq_estimated_cost'),
    }
    
    # Create HTML report
    report_html = f"""
    <h2>CQC Pipeline Monitoring Report</h2>
    <p>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <h3>System Health Summary</h3>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr><th>Component</th><th>Status</th><th>Details</th></tr>
        <tr>
            <td>Data Freshness</td>
            <td style="color: green;">✓ Passed</td>
            <td>Data is up to date</td>
        </tr>
        <tr>
            <td>Cloud Function</td>
            <td style="color: {'green' if metrics['function_error_rate'] < MAX_ERROR_RATE else 'red'};">
                {'✓ Healthy' if metrics['function_error_rate'] < MAX_ERROR_RATE else '✗ Issues Detected'}
            </td>
            <td>{metrics['function_executions']} executions, {metrics['function_error_rate']:.1%} error rate</td>
        </tr>
        <tr>
            <td>Model Endpoint</td>
            <td style="color: {'green' if metrics['endpoint_status'] == 'healthy' else 'red'};">
                {'✓ Healthy' if metrics['endpoint_status'] == 'healthy' else '✗ Unhealthy'}
            </td>
            <td>Latency: {metrics['endpoint_latency']:.3f}s</td>
        </tr>
        <tr>
            <td>Storage Usage</td>
            <td style="color: {'green' if metrics['storage_size_gb'] < MAX_STORAGE_GB else 'red'};">
                {'✓ Normal' if metrics['storage_size_gb'] < MAX_STORAGE_GB else '✗ High Usage'}
            </td>
            <td>{metrics['storage_size_gb']:.2f} GB ({metrics['file_count']:,} files)</td>
        </tr>
    </table>
    
    <h3>Cost Summary</h3>
    <table border="1" cellpadding="5" cellspacing="0">
        <tr><th>Service</th><th>Usage</th><th>Estimated Cost (Today)</th></tr>
        <tr>
            <td>BigQuery</td>
            <td>{metrics['bq_tb_processed']:.3f} TB processed ({metrics['bq_query_count']} queries)</td>
            <td>${metrics['bq_estimated_cost']:.2f}</td>
        </tr>
    </table>
    
    <h3>Data Quality</h3>
    <p>✓ No duplicate records detected</p>
    <p>✓ Daily record counts meet minimum threshold</p>
    
    <p><em>This is an automated monitoring report from the CQC Pipeline Monitoring system.</em></p>
    """
    
    return report_html

compile_report = PythonOperator(
    task_id='compile_monitoring_report',
    python_callable=compile_monitoring_report,
    dag=dag,
)

# Task 10: Decide if alerts are needed
def check_if_alerts_needed(**context):
    """Check if any monitoring alerts should be sent"""
    ti = context['task_instance']
    
    # Check various thresholds
    issues = []
    
    function_error_rate = ti.xcom_pull(task_ids='check_function_health', key='function_error_rate')
    if function_error_rate > MAX_ERROR_RATE:
        issues.append(f"Cloud Function error rate: {function_error_rate:.1%}")
    
    endpoint_status = ti.xcom_pull(task_ids='check_model_endpoint', key='endpoint_status')
    if endpoint_status != 'healthy':
        issues.append("Model endpoint is unhealthy")
    
    storage_size_gb = ti.xcom_pull(task_ids='check_storage_usage', key='storage_size_gb')
    if storage_size_gb > MAX_STORAGE_GB:
        issues.append(f"Storage usage high: {storage_size_gb:.2f} GB")
    
    bq_cost = ti.xcom_pull(task_ids='check_bigquery_costs', key='bq_estimated_cost')
    if bq_cost > MAX_DAILY_COST:
        issues.append(f"BigQuery daily cost high: ${bq_cost:.2f}")
    
    context['task_instance'].xcom_push(key='monitoring_issues', value=issues)
    
    if issues:
        return 'send_alert_email'
    else:
        return 'no_alerts_needed'

check_alerts = BranchPythonOperator(
    task_id='check_if_alerts_needed',
    python_callable=check_if_alerts_needed,
    dag=dag,
)

# Task 11a: Send alert email
send_alert_email = EmailOperator(
    task_id='send_alert_email',
    to=[ALERT_EMAIL],
    subject=f'CQC Pipeline Monitoring Alert - {datetime.now().strftime("%Y-%m-%d")}',
    html_content="{{ task_instance.xcom_pull(task_ids='compile_monitoring_report') }}",
    dag=dag,
)

# Task 11b: No alerts needed
no_alerts_needed = DummyOperator(
    task_id='no_alerts_needed',
    dag=dag,
)

# Task 12: Update monitoring dashboard
update_dashboard = BigQueryExecuteQueryOperator(
    task_id='update_monitoring_dashboard',
    sql=f"""
    INSERT INTO `{PROJECT_ID}.{MONITORING_DATASET}.system_health_log`
    (timestamp, function_error_rate, endpoint_status, endpoint_latency, 
     storage_gb, file_count, bq_tb_processed, bq_estimated_cost, alerts_sent)
    VALUES (
        CURRENT_DATETIME(),
        {{{{ ti.xcom_pull(task_ids='check_function_health', key='function_error_rate') }}}},
        '{{{{ ti.xcom_pull(task_ids='check_model_endpoint', key='endpoint_status') }}}}',
        {{{{ ti.xcom_pull(task_ids='check_model_endpoint', key='endpoint_latency') }}}},
        {{{{ ti.xcom_pull(task_ids='check_storage_usage', key='storage_size_gb') }}}},
        {{{{ ti.xcom_pull(task_ids='check_storage_usage', key='file_count') }}}},
        {{{{ ti.xcom_pull(task_ids='check_bigquery_costs', key='bq_tb_processed') }}}},
        {{{{ ti.xcom_pull(task_ids='check_bigquery_costs', key='bq_estimated_cost') }}}},
        ARRAY_LENGTH({{{{ ti.xcom_pull(task_ids='check_if_alerts_needed', key='monitoring_issues') }}}}) > 0
    )
    """,
    use_legacy_sql=False,
    dag=dag,
)

# Task 13: Monitoring complete
monitoring_complete = DummyOperator(
    task_id='monitoring_complete',
    trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    dag=dag,
)

# Define task dependencies
# Parallel health checks
[check_data_freshness, check_record_counts, check_function_logs, 
 test_model_endpoint, monitor_storage, check_data_quality, 
 check_duplicates, monitor_costs] >> compile_report

compile_report >> check_alerts

check_alerts >> [send_alert_email, no_alerts_needed]

[send_alert_email, no_alerts_needed] >> update_dashboard

update_dashboard >> monitoring_complete