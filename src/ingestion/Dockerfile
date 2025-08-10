FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy all Python files
COPY cqc_fetcher_cloud.py main.py app.py ./

# For Cloud Run Job (batch processing)
ENV PYTHONUNBUFFERED=1
ENV GCP_PROJECT=machine-learning-exp-467008

# Default to running the fetcher, but can be overridden
CMD ["python", "cqc_fetcher_cloud.py"]