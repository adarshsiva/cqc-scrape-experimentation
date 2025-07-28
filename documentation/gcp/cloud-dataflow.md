# Google Cloud Dataflow Documentation

## Overview
Google Cloud Dataflow is a fully-managed stream and batch processing service based on Apache Beam. It enables you to execute data processing pipelines at scale with automatic resource management, autoscaling, and fault tolerance.

## Installation
```bash
# Install Apache Beam with GCP support
pip install apache-beam[gcp]

# For additional features like ML
pip install apache-beam[gcp,ml]
```

## Basic Pipeline Structure

### Simple Pipeline Example
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

# Create pipeline
with beam.Pipeline(options=PipelineOptions()) as pipeline:
    (pipeline
     | 'Create Data' >> beam.Create([1, 2, 3, 4, 5])
     | 'Square Numbers' >> beam.Map(lambda x: x * x)
     | 'Print Results' >> beam.Map(print)
    )
```

### Word Count Example
```python
import apache_beam as beam
from apache_beam.io import ReadFromText, WriteToText
from apache_beam.options.pipeline_options import PipelineOptions

def run():
    options = PipelineOptions()
    
    with beam.Pipeline(options=options) as p:
        (p
         | 'Read' >> ReadFromText('gs://dataflow-samples/shakespeare/kinglear.txt')
         | 'Split' >> beam.FlatMap(lambda line: line.split())
         | 'Count' >> beam.combiners.Count.PerElement()
         | 'Format' >> beam.Map(lambda word_count: f'{word_count[0]}: {word_count[1]}')
         | 'Write' >> WriteToText('gs://YOUR_BUCKET/output')
        )
```

## Running on Dataflow

### Basic Dataflow Execution
```bash
python -m apache_beam.examples.wordcount \
    --region DATAFLOW_REGION \
    --input gs://dataflow-samples/shakespeare/kinglear.txt \
    --output gs://STORAGE_BUCKET/results/outputs \
    --runner DataflowRunner \
    --project PROJECT_ID \
    --temp_location gs://STORAGE_BUCKET/tmp/
```

### Configuring Pipeline Options
```python
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, WorkerOptions, SetupOptions

# Create pipeline options
options = PipelineOptions()

# Configure Google Cloud options
options.view_as(GoogleCloudOptions).project = 'YOUR_PROJECT_ID'
options.view_as(GoogleCloudOptions).region = 'us-central1'
options.view_as(GoogleCloudOptions).staging_location = 'gs://YOUR_BUCKET/staging'
options.view_as(GoogleCloudOptions).temp_location = 'gs://YOUR_BUCKET/temp'

# Configure worker options
options.view_as(WorkerOptions).num_workers = 5
options.view_as(WorkerOptions).max_num_workers = 10
options.view_as(WorkerOptions).autoscaling_algorithm = 'THROUGHPUT_BASED'
options.view_as(WorkerOptions).machine_type = 'n1-standard-4'

# Save main session for pickling
options.view_as(SetupOptions).save_main_session = True
```

## Streaming Pipelines

### Streaming Configuration
```python
from apache_beam.options.pipeline_options import StandardOptions

options = PipelineOptions()
options.view_as(StandardOptions).streaming = True
options.view_as(StandardOptions).runner = 'DataflowRunner'
```

### Streaming Word Count with Pub/Sub
```bash
python -m apache_beam.examples.streaming_wordcount \
  --runner DataflowRunner \
  --project YOUR_GCP_PROJECT \
  --region YOUR_GCP_REGION \
  --temp_location gs://YOUR_GCS_BUCKET/tmp/ \
  --input_topic "projects/YOUR_PUBSUB_PROJECT_NAME/topics/YOUR_INPUT_TOPIC" \
  --output_topic "projects/YOUR_PUBSUB_PROJECT_NAME/topics/YOUR_OUTPUT_TOPIC" \
  --streaming
```

## I/O Connectors

### Reading from Various Sources
```python
# Read from text file
lines = p | 'ReadText' >> beam.io.ReadFromText('gs://bucket/input.txt')

# Read from BigQuery
query = 'SELECT * FROM `project.dataset.table`'
rows = p | 'ReadBQ' >> beam.io.ReadFromBigQuery(query=query, use_standard_sql=True)

# Read from Pub/Sub
messages = p | 'ReadPubSub' >> beam.io.ReadFromPubSub(topic='projects/PROJECT/topics/TOPIC')
```

### Writing to Various Sinks
```python
# Write to text file
output | 'WriteText' >> beam.io.WriteToText('gs://bucket/output')

# Write to BigQuery
output | 'WriteBQ' >> beam.io.WriteToBigQuery(
    table='project:dataset.table',
    schema='name:STRING,value:INTEGER',
    write_disposition=beam.io.BigQueryDisposition.WRITE_TRUNCATE
)

# Write to Pub/Sub
output | 'WritePubSub' >> beam.io.WriteToPubSub(topic='projects/PROJECT/topics/TOPIC')
```

## Common Transforms

### Core Transforms
```python
# Map - Apply function to each element
squared = numbers | 'Square' >> beam.Map(lambda x: x * x)

# FlatMap - Apply function and flatten results
words = lines | 'ExtractWords' >> beam.FlatMap(lambda line: line.split())

# Filter - Keep elements matching condition
evens = numbers | 'FilterEven' >> beam.Filter(lambda x: x % 2 == 0)

# GroupByKey - Group values by key
grouped = kvs | 'Group' >> beam.GroupByKey()

# CombinePerKey - Combine values per key
summed = kvs | 'Sum' >> beam.CombinePerKey(sum)
```

### Windowing
```python
from apache_beam import window

# Fixed windows
windowed = (
    data 
    | 'Window' >> beam.WindowInto(window.FixedWindows(60))  # 60-second windows
    | 'Count' >> beam.CombineGlobally(beam.combiners.CountCombineFn()).without_defaults()
)

# Sliding windows
sliding = (
    data
    | 'SlidingWindow' >> beam.WindowInto(
        window.SlidingWindows(size=60, period=30)  # 60s windows every 30s
    )
)

# Session windows
sessions = (
    data
    | 'SessionWindow' >> beam.WindowInto(
        window.Sessions(gap_size=10)  # 10-second gap
    )
)
```

## Advanced Features

### Side Inputs
```python
# Create side input
average = (
    data 
    | 'ComputeAverage' >> beam.CombineGlobally(beam.combiners.MeanCombineFn())
    | 'AsSingleton' >> beam.pvalue.AsSingleton()
)

# Use side input
normalized = (
    data 
    | 'Normalize' >> beam.Map(
        lambda x, avg: x / avg, 
        avg=average
    )
)
```

### DoFn with State and Timers
```python
class StatefulDoFn(beam.DoFn):
    STATE_SPEC = beam.transforms.userstate.ReadModifyWriteStateSpec(
        'count', beam.coders.VarIntCoder()
    )
    TIMER_SPEC = beam.transforms.userstate.TimerSpec(
        'timer', beam.transforms.userstate.TimeDomain.WATERMARK
    )
    
    def process(self, element, count=beam.DoFn.StateParam(STATE_SPEC),
                timer=beam.DoFn.TimerParam(TIMER_SPEC)):
        current_count = count.read() or 0
        count.write(current_count + 1)
        
        # Set timer
        timer.set(beam.utils.timestamp.Timestamp.now() + 60)
        
        yield element, current_count
    
    @beam.DoFn.timer_method(TIMER_SPEC)
    def on_timer(self, count=beam.DoFn.StateParam(STATE_SPEC)):
        yield f'Timer fired, count: {count.read()}'
```

### Multi-Language Pipelines (Cross-Language)
```python
# Using external transforms
from apache_beam.transforms.external import ExternalTransform
from apache_beam.transforms.external import ImplicitSchemaPayloadBuilder

java_output = (
    input
    | 'JavaTransform' >> beam.ExternalTransform(
        'beam:transform:org.example:transform:v1',
        ImplicitSchemaPayloadBuilder({'parameter': 'value'}),
        "localhost:12345"  # Expansion service address
    )
)
```

## Dataflow-Specific Features

### Autoscaling
```python
# Configure autoscaling
options.view_as(WorkerOptions).autoscaling_algorithm = 'THROUGHPUT_BASED'
options.view_as(WorkerOptions).max_num_workers = 100

# For streaming
options.view_as(StandardOptions).streaming = True
# Streaming autoscaling is automatic based on backlog and CPU
```

### Dataflow Prime
```bash
# Enable Dataflow Prime features
python pipeline.py \
    --runner DataflowRunner \
    --project PROJECT_ID \
    --region REGION \
    --experiments=use_runner_v2 \
    --dataflow_service_options=enable_prime
```

### Custom Machine Types
```python
# Use custom machine configuration
options.view_as(WorkerOptions).machine_type = 'custom-4-16384'  # 4 vCPUs, 16GB RAM
options.view_as(WorkerOptions).disk_size_gb = 100
options.view_as(WorkerOptions).disk_type = 'pd-ssd'
```

## Environment Setup

### Using Custom Docker Images
```bash
# Build custom container
docker build -t gcr.io/PROJECT_ID/beam-custom:latest .

# Run with custom container
python pipeline.py \
    --runner DataflowRunner \
    --sdk_container_image=gcr.io/PROJECT_ID/beam-custom:latest \
    --experiment=use_runner_v2
```

### Dependencies Management
```python
# Using requirements.txt
options.view_as(SetupOptions).requirements_file = 'requirements.txt'

# Using setup.py
options.view_as(SetupOptions).setup_file = './setup.py'

# Extra packages
options.view_as(SetupOptions).extra_packages = ['./dist/mypackage-1.0.tar.gz']
```

## Monitoring and Debugging

### Enable Profiling
```python
options.view_as(ProfilingOptions).profile_cpu = True
options.view_as(ProfilingOptions).profile_memory = True
```

### Dataflow Job Submission Script
```bash
#!/bin/bash
export GCP_PROJECT=<project>
export GCS_BUCKET=<bucket>
export TEMP_LOCATION=gs://$GCS_BUCKET/tmp
export GCP_REGION=<region>
export JOB_NAME="dataflow-job-`date +%Y%m%d-%H%M%S`"
export NUM_WORKERS="1"

python pipeline.py \
  --runner DataflowRunner \
  --project $GCP_PROJECT \
  --region $GCP_REGION \
  --temp_location $TEMP_LOCATION \
  --staging_location gs://$GCS_BUCKET/staging \
  --job_name $JOB_NAME \
  --num_workers $NUM_WORKERS \
  --max_num_workers 10 \
  --experiments=use_runner_v2
```

## Best Practices for CQC Project

### ETL Pipeline Template
```python
class CQCDataflowPipeline:
    def __init__(self, project_id, region, bucket):
        self.options = PipelineOptions()
        self.options.view_as(GoogleCloudOptions).project = project_id
        self.options.view_as(GoogleCloudOptions).region = region
        self.options.view_as(GoogleCloudOptions).temp_location = f'gs://{bucket}/temp'
        self.options.view_as(GoogleCloudOptions).staging_location = f'gs://{bucket}/staging'
        self.options.view_as(SetupOptions).save_main_session = True
        
    def run(self):
        with beam.Pipeline(options=self.options) as p:
            # Read from Cloud Storage
            raw_data = (
                p 
                | 'ReadJSON' >> beam.io.ReadFromText('gs://bucket/cqc-data/*.json')
                | 'ParseJSON' >> beam.Map(json.loads)
            )
            
            # Transform data
            transformed = (
                raw_data
                | 'ExtractFields' >> beam.Map(self.extract_cqc_fields)
                | 'ValidateData' >> beam.Filter(self.validate_record)
                | 'EnrichData' >> beam.Map(self.enrich_with_metadata)
            )
            
            # Write to BigQuery
            transformed | 'WriteToBQ' >> beam.io.WriteToBigQuery(
                table='project:dataset.cqc_ratings',
                schema=self.get_bq_schema(),
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
    
    @staticmethod
    def extract_cqc_fields(record):
        return {
            'location_id': record.get('locationId'),
            'provider_id': record.get('providerId'),
            'rating': record.get('currentRatings', {}).get('overall', {}).get('rating'),
            'inspection_date': record.get('lastInspection', {}).get('date'),
            'service_types': record.get('regulatedActivities', [])
        }
```

### Key Recommendations
1. **Use Dataflow Prime** for better autoscaling and performance
2. **Enable streaming engine** for streaming pipelines to reduce costs
3. **Use regional endpoints** to reduce latency
4. **Implement proper error handling** with dead letter queues
5. **Monitor pipeline metrics** in Cloud Monitoring
6. **Use Flex Templates** for parameterized pipeline deployment
7. **Leverage BigQuery Storage Write API** for better performance
8. **Implement incremental processing** to avoid reprocessing data
9. **Use Cloud Composer** for orchestrating multiple Dataflow jobs
10. **Enable VPC Service Controls** for enhanced security