# Google Cloud Vertex AI Documentation

## Overview

Vertex AI is Google Cloud's unified platform for building, deploying, and scaling ML models. This documentation covers the Python SDK usage, best practices, and code examples relevant to the CQC ML project.

## Table of Contents

- [Installation](#installation)
- [SDK Initialization](#sdk-initialization)
- [Training Jobs](#training-jobs)
- [Model Deployment](#model-deployment)
- [Batch Prediction](#batch-prediction)
- [Pipelines](#pipelines)
- [Feature Store](#feature-store)
- [Model Evaluation](#model-evaluation)
- [Custom Containers](#custom-containers)
- [Best Practices](#best-practices)

## Installation

### Basic Installation
```bash
pip3 install --upgrade --user "google-cloud-aiplatform>=1.38"
```

### Installation with Prediction Support (Mac/Linux)
```bash
pip install virtualenv
virtualenv <your-env>
source <your-env>/bin/activate
<your-env>/bin/pip install "google-cloud-aiplatform[prediction]">=1.16.0"
```

### Installation with Prediction Support (Windows)
```bash
pip install virtualenv
virtualenv <your-env>
<your-env>\Scripts\activate
<your-env>\Scripts\pip.exe install "google-cloud-aiplatform[prediction]">=1.16.0"
```

## SDK Initialization

### Basic Initialization
```python
from google.cloud import aiplatform

aiplatform.init(
    # your Google Cloud Project ID or number
    # environment default used if not set
    project='my-project',

    # the Vertex AI region you will use
    # defaults to us-central1
    location='us-central1',

    # Google Cloud Storage bucket in same region as location
    # used to stage artifacts
    staging_bucket='gs://my_staging_bucket',

    # custom google.auth.credentials.Credentials
    # environment default credentials used if not set
    credentials=my_credentials,

    # customer managed encryption key resource name
    # will be applied to all Vertex AI resources if set
    encryption_spec_key_name=my_encryption_key_name,

    # the name of the experiment to use to track
    # logged metrics and parameters
    experiment='my-experiment',

    # description of the experiment above
    experiment_description='my experiment description'
)
```

### Alternative Initialization (vertexai module)
```python
import vertexai

vertexai.init(project='my-project', location='us-central1')
```

### Client Instantiation
```python
import vertexai
from vertexai import types

# Instantiate GenAI client from Vertex SDK
client = vertexai.Client(project='my-project', location='us-central1')
```

## Training Jobs

### Custom Training Job
```python
import google.cloud.aiplatform as aiplatform

job = aiplatform.CustomTrainingJob(
    display_name="my-training-job",
    script_path="training_script.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-2:latest",
    requirements=["gcsfs==0.7.1"],
    model_serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-2:latest",
)

model = job.run(
    my_dataset,
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type='NVIDIA_TESLA_K80',
    accelerator_count=1
)
```

### AutoML Tabular Training
```python
dataset = aiplatform.TabularDataset('projects/my-project/location/us-central1/datasets/{DATASET_ID}')

job = aiplatform.AutoMLTabularTrainingJob(
    display_name="train-automl",
    optimization_prediction_type="regression",
    optimization_objective="minimize-rmse",
)

model = job.run(
    dataset=dataset,
    target_column="target_column_name",
    training_fraction_split=0.6,
    validation_fraction_split=0.2,
    test_fraction_split=0.2,
    budget_milli_node_hours=1000,
    model_display_name="my-automl-model",
    disable_early_stopping=False,
)
```

### Custom Training Script Data Contract
```python
import os

# Accessing data URIs and format
data_format = os.environ['AIP_DATA_FORMAT']
training_data_uri = os.environ['AIP_TRAINING_DATA_URI']
validation_data_uri = os.environ['AIP_VALIDATION_DATA_URI']
test_data_uri = os.environ['AIP_TEST_DATA_URI']

# Writing model artifacts
model_output_dir = os.environ['AIP_MODEL_DIR']
```

## Model Deployment

### Upload Model
```python
model = aiplatform.Model.upload(
    display_name='my-model',
    artifact_uri="gs://python/to/my/model/dir",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-2:latest",
)
```

### Deploy Model to Endpoint
```python
# Create endpoint
endpoint = aiplatform.Endpoint.create(display_name='my-endpoint')

# Deploy model
endpoint = model.deploy(
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=5,
    accelerator_type='NVIDIA_TESLA_K80',
    accelerator_count=1
)
```

### Deploy Model with Advanced Container Configuration
```python
endpoint = model.deploy(
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/custom-container:latest",
    container_command=["python3"],
    container_args=["serve.py"],
    container_ports=[8888],
    container_env_vars={"ENV": "prod"},
    container_predict_route="/predict",
    container_health_route="/health",
    serving_container_shared_memory_size_mb=512,
    serving_container_grpc_ports=[9000],
    serving_container_startup_probe_exec=["/bin/check-start.sh"],
    serving_container_health_probe_exec=["/bin/health-check.sh"]
)
```

### Get Predictions
```python
predictions = endpoint.predict(instances=[[6.7, 3.1, 4.7, 1.5], [4.6, 3.1, 1.5, 0.2]])
```

### Undeploy Models
```python
endpoint.undeploy_all()
```

### Delete Endpoint
```python
endpoint.delete()
```

## Batch Prediction

### Synchronous Batch Prediction
```python
batch_prediction_job = model.batch_predict(
    job_display_name='my-batch-prediction-job',
    instances_format='csv',
    machine_type='n1-standard-4',
    gcs_source=['gs://path/to/my/file.csv'],
    gcs_destination_prefix='gs://path/to/my/batch_prediction/results/',
    service_account='my-sa@my-project.iam.gserviceaccount.com'
)
```

### Asynchronous Batch Prediction
```python
batch_prediction_job = model.batch_predict(..., sync=False)

# Wait for resource to be created
batch_prediction_job.wait_for_resource_creation()

# Get the state
print(f"Job state: {batch_prediction_job.state}")

# Block until job is complete
batch_prediction_job.wait()
```

## Pipelines

### Create and Run Pipeline Synchronously
```python
from google.cloud.aiplatform import PipelineJob

# Instantiate PipelineJob object
pl = PipelineJob(
    display_name="My first pipeline",

    # Whether or not to enable caching
    # True = always cache pipeline step result
    # False = never cache pipeline step result
    # None = defer to cache option for each pipeline component in the pipeline definition
    enable_caching=False,

    # Local or GCS path to a compiled pipeline definition
    template_path="pipeline.json",

    # Dictionary containing input parameters for your pipeline
    parameter_values=parameter_values,

    # GCS path to act as the pipeline root
    pipeline_root=pipeline_root,
)

# Execute pipeline in Vertex AI and monitor until completion
pl.run(
    # Email address of service account to use for the pipeline run
    # You must have iam.serviceAccounts.actAs permission on the service account to use it
    service_account=service_account,

    # Whether this function call should be synchronous (wait for pipeline run to finish before terminating)
    # or asynchronous (return immediately)
    sync=True
)
```

### Submit Pipeline Asynchronously
```python
# Submit the Pipeline to Vertex AI
pl.submit(
    service_account=service_account,
)
```

## Feature Store

Feature Store functionality is covered in the Terraform section with HCL examples. For Python SDK usage, refer to the official Vertex AI documentation.

## Model Evaluation

### List Model Evaluations
```python
model = aiplatform.Model('projects/my-project/locations/us-central1/models/{MODEL_ID}')
evaluations = model.list_model_evaluations()
```

### Get Specific Model Evaluation
```python
# Get the first evaluation
evaluation = model.get_model_evaluation()

# Get a specific evaluation by ID
evaluation = model.get_model_evaluation(evaluation_id='{EVALUATION_ID}')

eval_metrics = evaluation.metrics
```

### Create Model Evaluation Reference
```python
# Using full resource name
evaluation_by_name = aiplatform.ModelEvaluation(
    evaluation_name='projects/my-project/locations/us-central1/models/{MODEL_ID}/evaluations/{EVALUATION_ID}'
)

# Using model and evaluation IDs
evaluation_by_ids = aiplatform.ModelEvaluation(
    evaluation_name='{EVALUATION_ID}',
    model_id='{MODEL_ID}'
)
```

## Custom Containers

### Create Local Custom Prediction Routine Model
```python
from google.cloud.aiplatform.prediction import LocalModel

# {import your predictor and handler}

local_model = LocalModel.create_cpr_model(
    "{PATH_TO_THE_SOURCE_DIR}",
    f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE}",
    predictor={{PREDICTOR_CLASS}},
    handler={{HANDLER_CLASS}},
    requirements_path="{PATH_TO_REQUIREMENTS_TXT}",
)
```

### Get Serving Container Specification
```python
local_model.get_serving_container_spec()
```

### Push Custom Container Image
```python
local_model.push_image()
```

### Upload Custom Container Model
```python
model = aiplatform.Model.upload(
    local_model=local_model,
    display_name="{MODEL_DISPLAY_NAME}",
    artifact_uri="{GCS_PATH_TO_MODEL_ARTIFACTS}",
)
```

### Test Container Locally
```python
with local_model.deploy_to_local_endpoint(
    artifact_uri="{GCS_PATH_TO_MODEL_ARTIFACTS}",
    credential_path="{PATH_TO_CREDENTIALS}",
) as local_endpoint:
    predict_response = local_endpoint.predict(
        request_file="{PATH_TO_INPUT_FILE}",
        headers={{ANY_NEEDED_HEADERS}},
    )
    
    health_check_response = local_endpoint.run_health_check()
```

### Print Local Container Logs
```python
local_endpoint.print_container_logs(show_all=True)
```

## Generative AI Models

### Import Required Modules
```python
from vertexai.generative_models import GenerativeModel, Image, Content, Part, Tool, FunctionDeclaration, GenerationConfig
```

### Automatic Function Calling with Gemini
```python
# First, create functions that the model can use to answer your questions.
def get_current_weather(location: str, unit: str = "centigrade"):
    """Gets weather in the specified location.

    Args:
        location: The location for which to get the weather.
        unit: Optional. Temperature unit. Can be Centigrade or Fahrenheit. Defaults to Centigrade.
    """
    return dict(
        location=location,
        unit=unit,
        weather="Super nice, but maybe a bit hot.",
    )

# Infer function schema
get_current_weather_func = FunctionDeclaration.from_func(get_current_weather)
# Tool is a collection of related functions
weather_tool = Tool(
    function_declarations=[get_current_weather_func],
)

# Use tools in chat:
model = GenerativeModel(
    "gemini-pro",
    # You can specify tools when creating a model to avoid having to send them with every request.
    tools=[weather_tool],
)

# Activate automatic function calling:
afc_responder = AutomaticFunctionCallingResponder(
    # Optional:
    max_automatic_function_calls=5,
)
chat = model.start_chat(responder=afc_responder)
# Send a message to the model. The model will respond with a function call.
# The SDK will automatically call the requested function and respond to the model.
# The model will use the function call response to answer the original question.
print(chat.send_message("What is the weather like in Boston?"))
```

## Model Evaluation with EvalTask

### Run Gen AI Model Inference for Evaluation
```python
import pandas as pd

prompts_df = pd.DataFrame({
    "prompt": [
        "What is the capital of France?",
        "Write a haiku about a cat.",
        "Write a Python function to calculate the factorial of a number.",
        "Translate 'How are you?' to French.",
    ],

    "reference": [
        "Paris",
        "Sunbeam on the floor,\nA furry puddle sleeping,\nTwitching tail tells tales.",
        "def factorial(n):\n    if n < 0:\n        return 'Factorial does not exist for negative numbers'\n    elif n == 0:\n        return 1\n    else:\n        fact = 1\n        i = 1\n        while i <= n:\n            fact *= i\n            i += 1\n        return fact",
        "Comment ça va ?",
    ]
})

inference_results = client.evals.run_inference(
    model="gemini-2.5-flash-preview-05-20",
    src=prompts_df
)
```

### Evaluate Gen AI Model Responses
```python
eval_result = client.evals.evaluate(
    dataset=inference_results,
    metrics=[
        types.Metric(name='exact_match'),
        types.Metric(name='rouge_l_sum'),
        types.PrebuiltMetric.TEXT_QUALITY,
    ]
)
```

### Pairwise Metric Evaluation with Model Inference
```python
import pandas as pd
from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples, PairwiseMetric
from vertexai.generative_models import GenerativeModel

baseline_model = GenerativeModel("gemini-1.0-pro")
candidate_model = GenerativeModel("gemini-1.5-pro")

pairwise_groundedness = PairwiseMetric(
    metric_prompt_template=MetricPromptTemplateExamples.get_prompt_template(
        "pairwise_groundedness"
    ),
    baseline_model=baseline_model,
)
eval_dataset = pd.DataFrame({
    "prompt"  : [...],
})
result = EvalTask(
    dataset=eval_dataset,
    metrics=[pairwise_groundedness],
    experiment="my-pairwise-experiment",
).evaluate(
    model=candidate_model,
    experiment_run_name="gemini-pairwise-eval-run",
)
```

### BYOR (Bring Your Own Response) Evaluation
```python
import pandas as pd
from vertexai.evaluation import EvalTask, MetricPromptTemplateExamples

eval_dataset = pd.DataFrame({
        "prompt"  : [...],
        "reference": [...],
        "response" : [...],
        "baseline_model_response": [...],
})
eval_task = EvalTask(
    dataset=eval_dataset,
    metrics=[
            "bleu",
            "rouge_l_sum",
            MetricPromptTemplateExamples.Pointwise.FLUENCY,
            MetricPromptTemplateExamples.Pairwise.SAFETY
    ],
    experiment="my-experiment",
)
eval_result = eval_task.evaluate(experiment_run_name="eval-experiment-run")
```

## Cloud Profiler Integration

### Initialize Cloud Profiler for TensorFlow
```python
from google.cloud.aiplatform.training_utils import cloud_profiler

# ... your training code ...
cloud_profiler.init()
```

## Best Practices

### 1. Resource Management
- Always use absolute paths when specifying file locations
- Clean up resources (endpoints, models) when no longer needed
- Use `force_destroy` cautiously in production environments

### 2. Error Handling
- Implement proper error handling for asynchronous operations
- Check job states before proceeding with dependent operations
- Use timeouts appropriately for long-running operations

### 3. Cost Optimization
- Use appropriate machine types for your workload
- Configure auto-scaling for endpoints based on traffic patterns
- Clean up batch prediction results after processing

### 4. Security
- Use service accounts with minimal required permissions
- Encrypt sensitive data using customer-managed encryption keys (CMEK)
- Implement proper IAM policies for resources

### 5. Monitoring and Logging
- Enable request/response logging for endpoints
- Monitor model performance metrics regularly
- Set up alerts for failed jobs or degraded model performance

### 6. Data Management
- Use appropriate data formats for your use case (CSV, JSONL, TFRecord)
- Implement data validation before training
- Version your datasets and models

### 7. Pipeline Best Practices
- Enable caching for expensive pipeline steps
- Use parameter values for flexible pipeline configuration
- Monitor pipeline execution costs

### 8. Development Workflow
- Test models locally before deploying to production
- Use staging environments for validation
- Implement CI/CD pipelines for model deployment

## Additional Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Vertex AI Python Client Library](https://github.com/googleapis/python-aiplatform)
- [Vertex AI Samples](https://github.com/GoogleCloudPlatform/vertex-ai-samples)