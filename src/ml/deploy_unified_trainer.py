#!/usr/bin/env python3
"""
Deployment script for Unified CQC Model Trainer.
Creates and submits Vertex AI custom training job for the unified ensemble model.
"""

import os
import argparse
from datetime import datetime
from google.cloud import aiplatform


def deploy_unified_training_job(project_id: str, 
                               region: str = 'europe-west2',
                               machine_type: str = 'n1-highmem-4',
                               replica_count: int = 1,
                               tune_hyperparameters: bool = False,
                               deploy_model: bool = True) -> str:
    """
    Deploy unified CQC model trainer as Vertex AI custom training job.
    
    Args:
        project_id: GCP project ID
        region: GCP region for training
        machine_type: Compute engine machine type
        replica_count: Number of training replicas
        tune_hyperparameters: Whether to perform hyperparameter tuning
        deploy_model: Whether to deploy trained model to endpoint
        
    Returns:
        Training job resource name
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Training container image
    training_image = "gcr.io/cloud-aiplatform/training/scikit-learn-cpu.1-0:latest"
    
    # Job display name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    job_display_name = f"cqc-unified-training-{timestamp}"
    
    # Environment variables
    env_vars = {
        'GCP_PROJECT': project_id,
        'GCP_REGION': region,
        'TUNE_HYPERPARAMETERS': str(tune_hyperparameters).lower(),
        'DEPLOY_MODEL': str(deploy_model).lower(),
        'SAVE_ARTIFACTS': 'true'
    }
    
    # Create custom training job
    job = aiplatform.CustomJob(
        display_name=job_display_name,
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": machine_type,
                },
                "replica_count": replica_count,
                "container_spec": {
                    "image_uri": training_image,
                    "command": ["python", "-m", "src.ml.unified_model_trainer"],
                    "env": [{"name": k, "value": v} for k, v in env_vars.items()]
                },
            }
        ],
        base_output_dir=f"gs://{project_id}-cqc-models/training-outputs/{timestamp}/",
        labels={
            "model_type": "unified_ensemble",
            "use_case": "cqc_rating_prediction",
            "training_date": timestamp
        }
    )
    
    print(f"Submitting unified CQC training job: {job_display_name}")
    print(f"Project: {project_id}")
    print(f"Region: {region}")
    print(f"Machine type: {machine_type}")
    print(f"Hyperparameter tuning: {tune_hyperparameters}")
    print(f"Deploy model: {deploy_model}")
    
    # Submit job
    job.run(sync=False)  # Submit asynchronously
    
    print(f"Training job submitted: {job.resource_name}")
    print(f"Monitor progress in GCP Console: https://console.cloud.google.com/ai/training/jobs")
    
    return job.resource_name


def main():
    parser = argparse.ArgumentParser(description="Deploy Unified CQC Model Training Job")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--region", default="europe-west2", help="GCP region")
    parser.add_argument("--machine-type", default="n1-highmem-4", help="Machine type")
    parser.add_argument("--tune-hyperparameters", action="store_true", 
                       help="Enable hyperparameter tuning")
    parser.add_argument("--no-deploy", action="store_true", 
                       help="Skip model deployment to endpoint")
    
    args = parser.parse_args()
    
    job_name = deploy_unified_training_job(
        project_id=args.project_id,
        region=args.region,
        machine_type=args.machine_type,
        tune_hyperparameters=args.tune_hyperparameters,
        deploy_model=not args.no_deploy
    )
    
    print(f"\nTraining job deployed successfully: {job_name}")
    print("\nTo monitor the job:")
    print(f"gcloud ai custom-jobs describe {job_name.split('/')[-1]} --region={args.region}")


if __name__ == "__main__":
    main()