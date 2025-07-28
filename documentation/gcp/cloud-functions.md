# Google Cloud Functions Documentation

## Overview
Google Cloud Functions can be implemented using the Functions Framework for Python, which provides a lightweight framework for writing portable Python functions that can be deployed to various serverless environments.

## Installation
```bash
pip install functions-framework
```

## Basic HTTP Function

### Simple HTTP Function
```python
import functions_framework

@functions_framework.http
def hello(request):
    return "Hello world!"
```

### HTTP Function with Flask Integration
```python
import flask
import functions_framework

@functions_framework.http
def hello(request: flask.Request) -> flask.typing.ResponseReturnValue:
    return "Hello world!"
```

## CloudEvent Functions
```python
import functions_framework
from cloudevents.http.event import CloudEvent

@functions_framework.cloud_event
def hello_cloud_event(cloud_event: CloudEvent) -> None:
   print(f"Received event with ID: {cloud_event['id']} and data {cloud_event.data}")
```

## Event-Style Functions (Pub/Sub)
```python
def hello(event, context):
    print(event)
    print(context)
    print("Received", context.event_id)
```

## Error Handling
```python
import functions_framework

@functions_framework.errorhandler(ZeroDivisionError)
def handle_zero_division(e):
    return "I'm a teapot", 418

def function(request):
    1 / 0
    return "Success", 200
```

## Local Development

### Running Functions Locally
```bash
# HTTP function
functions-framework --target=hello --debug

# CloudEvent function
functions-framework --target=hello_cloud_event

# Event function with specific port
functions-framework --target=hello --signature-type=event --debug --port=8080
```

### Testing with cURL
```bash
# HTTP function
curl localhost:8080

# CloudEvent function
curl -X POST localhost:8080 \
   -H "Content-Type: application/cloudevents+json" \
   -d '{
	"specversion" : "1.0",
	"type" : "example.com.cloud.event",
	"source" : "https://example.com/cloudevents/pull",
	"subject" : "123",
	"id" : "A234-1234-1234",
	"time" : "2018-04-05T17:31:00Z",
	"data" : "hello world"
}'
```

## Containerization

### Dockerfile Configuration
```dockerfile
# Use the official Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . .

# Install production dependencies.
RUN pip install functions-framework
RUN pip install -r requirements.txt

# Run the web service on container startup.
CMD exec functions-framework --target=hello
```

### Building and Running Containers
```bash
# Build container
docker build -t helloworld .

# Run locally
docker run --rm -p 8080:8080 -e PORT=8080 helloworld

# Build with Cloud Buildpacks
pack build \
    --builder gcr.io/buildpacks/builder:v1 \
    --env GOOGLE_FUNCTION_SIGNATURE_TYPE=http \
    --env GOOGLE_FUNCTION_TARGET=hello \
    my-first-function
```

## Deployment to Cloud Run

### Configure Docker Authentication
```bash
gcloud auth configure-docker
```

### Build, Push, and Deploy
```bash
docker build -t gcr.io/[PROJECT-ID]/helloworld .
docker push gcr.io/[PROJECT-ID]/helloworld
gcloud run deploy helloworld --image gcr.io/[PROJECT-ID]/helloworld --region us-central1
```

## Requirements File
```text
functions-framework==3.*
# Optional dependencies
cloudevents>=1.2.0
requests
```

## Testing with Pub/Sub Emulator
```bash
export PUBSUB_PROJECT_ID=my-project
gcloud beta emulators pubsub start \
    --project=$PUBSUB_PROJECT_ID \
    --host-port=localhost:8085

# Set up topic and subscription
$(gcloud beta emulators pubsub env-init)
python publisher.py $PUBSUB_PROJECT_ID create $TOPIC_ID
python subscriber.py $PUBSUB_PROJECT_ID create-push $TOPIC_ID $PUSH_SUBSCRIPTION_ID http://localhost:8080
python publisher.py $PUBSUB_PROJECT_ID publish $TOPIC_ID
```

## Best Practices for CQC Project

1. **Function Structure**: Use the decorator pattern for cleaner code
2. **Error Handling**: Implement proper error handlers for different exception types
3. **Local Testing**: Always test functions locally before deployment
4. **Containerization**: Use Docker for consistent deployment environments
5. **Dependencies**: Pin versions in requirements.txt for reproducibility
6. **Environment Variables**: Use Secret Manager for sensitive data like API keys