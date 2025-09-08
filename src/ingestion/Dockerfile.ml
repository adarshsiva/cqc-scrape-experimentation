FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    google-cloud-storage \
    google-cloud-bigquery \
    google-cloud-secret-manager \
    requests \
    numpy \
    pandas

# Copy the ML data extractor
COPY cqc_ml_data_extractor.py .

# Set environment
ENV PYTHONUNBUFFERED=1

# Run the extractor
CMD ["python", "cqc_ml_data_extractor.py"]