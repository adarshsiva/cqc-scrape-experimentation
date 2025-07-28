# Google BigQuery Documentation

## Overview
Google BigQuery is a fully-managed, serverless enterprise data warehouse that enables scalable analysis over petabytes of data. It supports ANSI SQL queries and provides built-in machine learning capabilities, making it ideal for data analytics and business intelligence.

## Installation
```bash
# Basic installation
pip install google-cloud-bigquery

# With pandas support
pip install google-cloud-bigquery[pandas]

# With pandas and PyArrow for better performance
pip install google-cloud-bigquery[pandas,pyarrow]

# With OpenTelemetry tracing
pip install google-cloud-bigquery[opentelemetry] opentelemetry-exporter-gcp-trace
```

## Basic Setup

### Client Initialization
```python
from google.cloud import bigquery

# Create client with default project
client = bigquery.Client()

# Create client with explicit project
client = bigquery.Client(project='YOUR_PROJECT_ID')
```

## Dataset Operations

### Creating Datasets
```python
# Create a dataset
dataset_id = f"{client.project}.your_dataset_id"
dataset = bigquery.Dataset(dataset_id)
dataset.location = "US"
dataset = client.create_dataset(dataset, timeout=30)
print(f"Created dataset {dataset.project}.{dataset.dataset_id}")
```

### Listing Datasets
```python
# List all datasets
datasets = list(client.list_datasets())
for dataset in datasets:
    print(f"Dataset ID: {dataset.dataset_id}")

# List datasets with filter
datasets = client.list_datasets(filter="labels.department:engineering")
```

### Managing Dataset Properties
```python
# Get dataset
dataset = client.get_dataset(dataset_id)

# Update dataset description
dataset.description = "This dataset contains CQC ratings data"
dataset = client.update_dataset(dataset, ["description"])

# Add labels to dataset
dataset.labels = {"department": "analytics", "project": "cqc"}
dataset = client.update_dataset(dataset, ["labels"])
```

### Dataset Access Control
```python
# Update dataset access
from google.cloud.bigquery import AccessEntry

# Grant access to a user
entry = AccessEntry(
    role="READER",
    entity_type="userByEmail",
    entity_id="user@example.com",
)
dataset.access_entries.append(entry)
dataset = client.update_dataset(dataset, ["access_entries"])
```

## Table Operations

### Creating Tables
```python
# Define schema
schema = [
    bigquery.SchemaField("location_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("provider_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("rating", "STRING", mode="NULLABLE"),
    bigquery.SchemaField("inspection_date", "DATE", mode="NULLABLE"),
    bigquery.SchemaField("score", "FLOAT64", mode="NULLABLE"),
]

# Create table
table_id = f"{client.project}.{dataset_id}.cqc_ratings"
table = bigquery.Table(table_id, schema=schema)
table = client.create_table(table)
print(f"Created table {table.table_id}")
```

### Loading Data

#### Load from Cloud Storage
```python
# Load CSV from GCS
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.CSV,
    skip_leading_rows=1,
    autodetect=True,  # Auto-detect schema
)

uri = "gs://your-bucket/cqc-data.csv"
load_job = client.load_table_from_uri(
    uri, table_id, job_config=job_config
)
load_job.result()  # Wait for job to complete
```

#### Load JSON with auto-detection
```python
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
    autodetect=True,
)

uri = "gs://your-bucket/cqc-data.json"
load_job = client.load_table_from_uri(uri, table_id, job_config=job_config)
```

#### Load from Pandas DataFrame
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    'location_id': ['L001', 'L002'],
    'provider_id': ['P001', 'P001'],
    'rating': ['Good', 'Outstanding'],
    'inspection_date': ['2024-01-15', '2024-01-20'],
    'score': [85.5, 92.0]
})

# Load to BigQuery
job_config = bigquery.LoadJobConfig(
    schema=schema,
    write_disposition="WRITE_APPEND",
)

job = client.load_table_from_dataframe(
    df, table_id, job_config=job_config
)
job.result()
```

### Streaming Inserts
```python
# Insert rows via streaming
rows_to_insert = [
    {"location_id": "L003", "provider_id": "P002", "rating": "Good"},
    {"location_id": "L004", "provider_id": "P002", "rating": "Requires improvement"},
]

errors = client.insert_rows_json(table_id, rows_to_insert)
if errors:
    print(f"Failed to insert rows: {errors}")
```

### Table Management
```python
# Get table
table = client.get_table(table_id)
print(f"Table has {table.num_rows} rows")

# Update table
table.description = "CQC ratings data"
table = client.update_table(table, ["description"])

# Copy table
source_table_id = table_id
destination_table_id = f"{dataset_id}.cqc_ratings_backup"
job = client.copy_table(source_table_id, destination_table_id)
job.result()

# Delete table
client.delete_table(table_id, not_found_ok=True)
```

## Querying Data

### Basic Queries
```python
# Simple query
query = """
    SELECT location_id, rating, inspection_date
    FROM `project.dataset.cqc_ratings`
    WHERE rating = 'Outstanding'
    LIMIT 1000
"""

query_job = client.query(query)
results = query_job.result()

for row in results:
    print(f"{row.location_id}: {row.rating} on {row.inspection_date}")
```

### Parameterized Queries
```python
# Query with parameters
query = """
    SELECT location_id, rating, score
    FROM `project.dataset.cqc_ratings`
    WHERE rating = @rating
    AND score > @min_score
"""

job_config = bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ScalarQueryParameter("rating", "STRING", "Good"),
        bigquery.ScalarQueryParameter("min_score", "FLOAT64", 80.0),
    ]
)

query_job = client.query(query, job_config=job_config)
results = query_job.result()
```

### Query to DataFrame
```python
# Query results to pandas DataFrame
query = """
    SELECT rating, COUNT(*) as count
    FROM `project.dataset.cqc_ratings`
    GROUP BY rating
"""

df = client.query(query).to_dataframe()
print(df)
```

### DDL Queries
```python
# Create table using DDL
query = """
    CREATE OR REPLACE TABLE `project.dataset.cqc_summary`
    AS
    SELECT 
        provider_id,
        COUNT(*) as location_count,
        AVG(score) as avg_score
    FROM `project.dataset.cqc_ratings`
    GROUP BY provider_id
"""

query_job = client.query(query)
query_job.result()
```

## Advanced Features

### Partitioned Tables
```python
# Create partitioned table
schema = [
    bigquery.SchemaField("location_id", "STRING"),
    bigquery.SchemaField("rating", "STRING"),
    bigquery.SchemaField("inspection_date", "DATE"),
]

table = bigquery.Table(table_id, schema=schema)
table.time_partitioning = bigquery.TimePartitioning(
    type_=bigquery.TimePartitioningType.DAY,
    field="inspection_date",
)
table = client.create_table(table)
```

### Clustered Tables
```python
# Create clustered table
table = bigquery.Table(table_id, schema=schema)
table.clustering_fields = ["provider_id", "rating"]
table = client.create_table(table)
```

### Table Snapshots and Clones
```python
# Create table snapshot
snapshot_id = f"{table_id}_snapshot_{datetime.now().strftime('%Y%m%d')}"
client.copy_table(
    table_id,
    snapshot_id,
    job_config=bigquery.CopyJobConfig(
        operation_type=bigquery.OperationType.SNAPSHOT
    )
).result()

# Create table clone
clone_id = f"{table_id}_clone"
client.copy_table(
    table_id,
    clone_id,
    job_config=bigquery.CopyJobConfig(
        operation_type=bigquery.OperationType.CLONE
    )
).result()
```

### Views
```python
# Create view
view_id = f"{dataset_id}.outstanding_providers"
view = bigquery.Table(view_id)
view.view_query = """
    SELECT DISTINCT provider_id
    FROM `project.dataset.cqc_ratings`
    WHERE rating = 'Outstanding'
"""
view = client.create_table(view)
```

### External Tables
```python
# Create external table from GCS
table_id = f"{dataset_id}.external_cqc_data"
table = bigquery.Table(table_id)

external_config = bigquery.ExternalConfig("CSV")
external_config.source_uris = ["gs://bucket/cqc-data/*.csv"]
external_config.autodetect = True

table.external_data_configuration = external_config
table = client.create_table(table)
```

## BigQuery Storage Write API
```python
# High-performance writes
from google.cloud import bigquery_storage_v1

# Use storage write API for better performance
job_config = bigquery.LoadJobConfig(
    use_avro_logical_types=True,
    write_disposition="WRITE_APPEND",
)

# The client will automatically use Storage Write API when available
```

## Export Data
```python
# Extract to Cloud Storage
destination_uri = "gs://your-bucket/exports/cqc-data-*.csv"
extract_job = client.extract_table(
    table_id,
    destination_uri,
    location="US",
)
extract_job.result()
```

## Monitoring and Management

### Job Management
```python
# List recent jobs
for job in client.list_jobs(max_results=10):
    print(f"Job {job.job_id} ({job.job_type}): {job.state}")

# Get specific job
job = client.get_job(job_id)
print(f"Job {job.job_id} started at {job.started}")
```

### Table Metadata
```python
# Get table schema
table = client.get_table(table_id)
for field in table.schema:
    print(f"{field.name}: {field.field_type} ({field.mode})")

# Table properties
print(f"Created: {table.created}")
print(f"Modified: {table.modified}")
print(f"Rows: {table.num_rows}")
print(f"Size: {table.num_bytes} bytes")
```

## OpenTelemetry Tracing
```python
# Enable tracing
import os
os.environ["ENABLE_GCS_PYTHON_CLIENT_OTEL_TRACES"] = "True"

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(CloudTraceSpanExporter()))
trace.set_tracer_provider(tracer_provider)
```

## DB-API Interface
```python
from google.cloud.bigquery import dbapi

# Create connection
conn = dbapi.connect()
cursor = conn.cursor()

# Execute query with parameters
cursor.execute(
    "SELECT * FROM `project.dataset.table` WHERE rating = %s",
    ("Outstanding",)
)

# Fetch results
for row in cursor.fetchall():
    print(row)

cursor.close()
conn.close()
```

## Best Practices for CQC Project

### Schema Design
```python
# Comprehensive CQC table schema
cqc_schema = [
    bigquery.SchemaField("location_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("provider_id", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("location_name", "STRING"),
    bigquery.SchemaField("provider_name", "STRING"),
    bigquery.SchemaField("service_type", "STRING", mode="REPEATED"),
    bigquery.SchemaField("registration_date", "DATE"),
    bigquery.SchemaField("current_rating", "STRING"),
    bigquery.SchemaField("rating_date", "DATE"),
    bigquery.SchemaField("region", "STRING"),
    bigquery.SchemaField("postal_code", "STRING"),
    bigquery.SchemaField("regulated_activities", "STRING", mode="REPEATED"),
    bigquery.SchemaField("inspection_categories", "RECORD", mode="REPEATED", fields=[
        bigquery.SchemaField("category", "STRING"),
        bigquery.SchemaField("rating", "STRING"),
    ]),
    bigquery.SchemaField("last_updated", "TIMESTAMP"),
]
```

### Performance Optimization
1. **Use partitioning** on inspection_date for time-based queries
2. **Use clustering** on provider_id and rating for efficient filtering
3. **Enable query caching** for repeated queries
4. **Use BigQuery Storage API** for large data reads
5. **Batch inserts** instead of individual row inserts
6. **Use materialized views** for frequently accessed aggregations
7. **Set appropriate table expiration** for temporary tables
8. **Use INFORMATION_SCHEMA** for metadata queries
9. **Monitor slot usage** and query performance
10. **Use BI Engine** for interactive dashboards

### Cost Management
```python
# Set dataset default table expiration
dataset.default_table_expiration_ms = 7 * 24 * 60 * 60 * 1000  # 7 days
dataset = client.update_dataset(dataset, ["default_table_expiration_ms"])

# Query with maximum bytes billed
job_config = bigquery.QueryJobConfig(
    maximum_bytes_billed=10_000_000_000  # 10 GB limit
)

# Dry run to estimate costs
job_config = bigquery.QueryJobConfig(dry_run=True)
query_job = client.query(query, job_config=job_config)
print(f"Query will process {query_job.total_bytes_processed} bytes")
```