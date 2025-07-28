#!/usr/bin/env python3
"""
Deploy and run the CQC ETL pipeline on Google Cloud Dataflow.

This script will:
1. Deploy the ETL pipeline to process sample data from GCS
2. Write processed data to BigQuery
3. Monitor pipeline execution
"""

import argparse
import subprocess
import sys
import os
from datetime import datetime


def run_dataflow_pipeline(project_id: str, data_type: str, input_pattern: str):
    """Run the Dataflow ETL pipeline for the specified data type."""
    
    # Set up pipeline parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_name = f"cqc-etl-{data_type}-{timestamp}"
    temp_location = f"gs://{project_id}-cqc-ml-pipeline/temp"
    staging_location = f"gs://{project_id}-cqc-ml-pipeline/staging"
    
    # Build the command
    cmd = [
        "python3", "dataflow_pipeline.py",
        f"--project-id={project_id}",
        "--dataset-id=cqc_data",
        f"--temp-location={temp_location}",
        f"--input-path={input_pattern}",
        f"--data-type={data_type}",
        "--runner=DataflowRunner",
        f"--job_name={job_name}",
        f"--staging_location={staging_location}",
        "--region=us-central1",
        "--setup_file=./setup.py"
    ]
    
    print(f"\nRunning Dataflow pipeline for {data_type}...")
    print(f"Job name: {job_name}")
    print(f"Input: {input_pattern}")
    print("\nCommand: " + " ".join(cmd))
    
    # Run the command
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"\nPipeline submitted successfully!")
        print(result.stdout)
        return job_name
    except subprocess.CalledProcessError as e:
        print(f"\nError running pipeline: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        raise


def create_setup_file():
    """Create setup.py for Dataflow to package the transforms module."""
    setup_content = '''import setuptools

setuptools.setup(
    name='cqc_etl',
    version='1.0.0',
    install_requires=[],
    packages=setuptools.find_packages(),
    py_modules=['transforms']
)
'''
    with open('setup.py', 'w') as f:
        f.write(setup_content)
    print("Created setup.py for Dataflow packaging")


def main():
    parser = argparse.ArgumentParser(description='Deploy CQC ETL Pipeline to Dataflow')
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    args = parser.parse_args()
    
    project_id = args.project_id
    bucket_name = f"{project_id}-cqc-raw-data"
    
    # Create setup.py for packaging
    create_setup_file()
    
    # Define input patterns for the sample data
    locations_pattern = f"gs://{bucket_name}/raw/locations/20250728_191714_locations_sample.json"
    providers_pattern = f"gs://{bucket_name}/raw/providers/20250728_191714_providers_sample.json"
    
    try:
        # Run locations ETL pipeline
        locations_job = run_dataflow_pipeline(project_id, "locations", locations_pattern)
        
        # Run providers ETL pipeline
        providers_job = run_dataflow_pipeline(project_id, "providers", providers_pattern)
        
        print("\n" + "="*50)
        print("ETL Pipelines Deployed Successfully!")
        print("="*50)
        print(f"\nLocations job: {locations_job}")
        print(f"Providers job: {providers_job}")
        print(f"\nMonitor progress in the Dataflow console:")
        print(f"https://console.cloud.google.com/dataflow/jobs?project={project_id}")
        print("\nThe pipelines will:")
        print("1. Read sample data from GCS")
        print("2. Parse and transform the data")
        print("3. Extract features and calculate derived metrics")
        print("4. Write processed data to BigQuery tables:")
        print(f"   - {project_id}.cqc_data.locations")
        print(f"   - {project_id}.cqc_data.providers")
        
    except Exception as e:
        print(f"\nError deploying pipelines: {e}")
        sys.exit(1)
    finally:
        # Clean up setup.py
        if os.path.exists('setup.py'):
            os.remove('setup.py')
            print("\nCleaned up setup.py")


if __name__ == "__main__":
    main()