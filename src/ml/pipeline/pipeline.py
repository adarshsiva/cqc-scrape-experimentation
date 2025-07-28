"""
Main Vertex AI pipeline for CQC ML system.

This module orchestrates all pipeline components to:
- Load and prepare data from BigQuery
- Preprocess features
- Train multiple models (XGBoost, LightGBM, AutoML)
- Evaluate and compare models
- Deploy the best performing model
- Handle model versioning
"""

from kfp import dsl
from kfp.dsl import pipeline
from typing import Optional, Dict
import google.cloud.aiplatform as aiplatform
from datetime import datetime

# Import pipeline components
from components import (
    data_preparation_component,
    feature_preprocessing_component,
    train_xgboost_component,
    train_lightgbm_component,
    train_automl_component,
    model_evaluation_component,
    model_deployment_component
)


@pipeline(
    name="cqc-ml-training-pipeline",
    description="End-to-end ML pipeline for CQC rating prediction",
    pipeline_root="gs://{{project-id}}-cqc-ml-pipeline",
)
def cqc_ml_pipeline(
    project_id: str,
    location: str = "us-central1",
    dataset_id: str = "cqc_data",
    table_id: str = "cqc_ratings",
    query: Optional[str] = None,
    train_split: float = 0.7,
    validation_split: float = 0.15,
    test_split: float = 0.15,
    target_column: str = "rating",
    xgboost_hyperparameters: Dict[str, float] = {},
    lightgbm_hyperparameters: Dict[str, float] = {},
    automl_budget_hours: float = 1.0,
    enable_automl: bool = False,
    endpoint_display_name: str = "cqc-rating-predictor",
    machine_type: str = "n1-standard-4",
    min_replicas: int = 1,
    max_replicas: int = 3,
    experiment_name: str = "cqc-ml-experiment"
):
    """CQC ML Training Pipeline.
    
    Args:
        project_id: GCP project ID
        location: GCP region for Vertex AI
        dataset_id: BigQuery dataset containing CQC data
        table_id: BigQuery table name
        query: Optional custom query for data selection
        train_split: Fraction of data for training
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        target_column: Name of target column in data
        xgboost_hyperparameters: Hyperparameters for XGBoost
        lightgbm_hyperparameters: Hyperparameters for LightGBM
        automl_budget_hours: Training budget for AutoML in hours
        enable_automl: Whether to train AutoML model
        endpoint_display_name: Display name for model endpoint
        machine_type: Machine type for model serving
        min_replicas: Minimum number of serving replicas
        max_replicas: Maximum number of serving replicas
        experiment_name: Name for tracking experiment
    """
    
    # Generate unique identifiers for this pipeline run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    
    # Step 1: Data Preparation
    data_prep_task = data_preparation_component(
        project_id=project_id,
        dataset_id=dataset_id,
        table_id=table_id,
        query=query,
        train_split=train_split,
        validation_split=validation_split,
        test_split=test_split
    )
    data_prep_task.set_display_name("Data Preparation")
    
    # Step 2: Feature Preprocessing
    feature_prep_task = feature_preprocessing_component(
        train_data=data_prep_task.outputs["train_data"],
        validation_data=data_prep_task.outputs["validation_data"],
        test_data=data_prep_task.outputs["test_data"],
        target_column=target_column
    )
    feature_prep_task.set_display_name("Feature Engineering")
    
    # Step 3a: Train XGBoost Model
    xgboost_task = train_xgboost_component(
        train_data=feature_prep_task.outputs["processed_train"],
        validation_data=feature_prep_task.outputs["processed_validation"],
        hyperparameters=xgboost_hyperparameters
    )
    xgboost_task.set_display_name("Train XGBoost")
    
    # Step 3b: Train LightGBM Model
    lightgbm_task = train_lightgbm_component(
        train_data=feature_prep_task.outputs["processed_train"],
        validation_data=feature_prep_task.outputs["processed_validation"],
        hyperparameters=lightgbm_hyperparameters
    )
    lightgbm_task.set_display_name("Train LightGBM")
    
    # Step 3c: Train AutoML Model (conditional)
    with dsl.Condition(enable_automl == True, name="AutoML Training"):
        automl_task = train_automl_component(
            project_id=project_id,
            location=location,
            dataset_display_name=f"cqc_automl_dataset_{run_id}",
            train_data=feature_prep_task.outputs["processed_train"],
            target_column="target",
            budget_hours=automl_budget_hours,
            model_display_name=f"cqc_automl_model_{run_id}"
        )
        automl_task.set_display_name("Train AutoML")
    
    # Step 4: Model Evaluation and Comparison
    # Evaluate XGBoost
    xgboost_eval_task = model_evaluation_component(
        model=xgboost_task.outputs["model"],
        test_data=feature_prep_task.outputs["processed_test"]
    )
    xgboost_eval_task.set_display_name("Evaluate XGBoost")
    
    # Evaluate LightGBM
    lightgbm_eval_task = model_evaluation_component(
        model=lightgbm_task.outputs["model"],
        test_data=feature_prep_task.outputs["processed_test"]
    )
    lightgbm_eval_task.set_display_name("Evaluate LightGBM")
    
    # Step 5: Model Selection (using dsl.Condition for conditional deployment)
    # Compare models and deploy the best one
    # For simplicity, we'll deploy based on individual model performance
    
    # Deploy XGBoost if it meets criteria
    with dsl.Condition(
        xgboost_eval_task.outputs["deploy_decision"] == "deploy",
        name="Deploy XGBoost"
    ):
        xgboost_deploy_task = model_deployment_component(
            project_id=project_id,
            location=location,
            model=xgboost_task.outputs["model"],
            deploy_decision=xgboost_eval_task.outputs["deploy_decision"],
            endpoint_display_name=f"{endpoint_display_name}_xgboost",
            machine_type=machine_type,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            deployed_model_display_name=f"cqc_xgboost_{run_id}"
        )
        xgboost_deploy_task.set_display_name("Deploy XGBoost Model")
    
    # Deploy LightGBM if it meets criteria
    with dsl.Condition(
        lightgbm_eval_task.outputs["deploy_decision"] == "deploy",
        name="Deploy LightGBM"
    ):
        lightgbm_deploy_task = model_deployment_component(
            project_id=project_id,
            location=location,
            model=lightgbm_task.outputs["model"],
            deploy_decision=lightgbm_eval_task.outputs["deploy_decision"],
            endpoint_display_name=f"{endpoint_display_name}_lightgbm",
            machine_type=machine_type,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            deployed_model_display_name=f"cqc_lightgbm_{run_id}"
        )
        lightgbm_deploy_task.set_display_name("Deploy LightGBM Model")
    
    # Configure pipeline-level settings
    dsl.get_pipeline_conf().set_timeout(7200)  # 2 hours timeout
    

def compile_pipeline(output_path: str = "cqc_ml_pipeline.json"):
    """Compile the pipeline to JSON format.
    
    Args:
        output_path: Path to save compiled pipeline
    """
    from kfp import compiler
    
    compiler.Compiler().compile(
        pipeline_func=cqc_ml_pipeline,
        package_path=output_path
    )
    print(f"Pipeline compiled and saved to: {output_path}")


def run_pipeline(
    project_id: str,
    location: str = "us-central1",
    pipeline_root: str = None,
    service_account: str = None,
    **pipeline_parameters
):
    """Run the pipeline on Vertex AI.
    
    Args:
        project_id: GCP project ID
        location: GCP region
        pipeline_root: GCS path for pipeline artifacts
        service_account: Service account for pipeline execution
        **pipeline_parameters: Parameters to pass to pipeline
    """
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Set pipeline root if not provided
    if not pipeline_root:
        pipeline_root = f"gs://{project_id}-cqc-ml-pipeline/pipeline_runs"
    
    # Compile pipeline
    pipeline_path = "cqc_ml_pipeline.json"
    compile_pipeline(pipeline_path)
    
    # Create pipeline job
    job = aiplatform.PipelineJob(
        display_name="cqc-ml-training-pipeline",
        template_path=pipeline_path,
        pipeline_root=pipeline_root,
        parameter_values={
            "project_id": project_id,
            "location": location,
            **pipeline_parameters
        }
    )
    
    # Submit pipeline
    job.submit(service_account=service_account)
    
    print(f"Pipeline submitted. Job name: {job.display_name}")
    print(f"View in console: {job._dashboard_uri()}")
    
    return job


def create_pipeline_schedule(
    project_id: str,
    location: str = "us-central1",
    pipeline_root: str = None,
    service_account: str = None,
    schedule: str = "0 2 * * 0",  # Weekly on Sunday at 2 AM
    **pipeline_parameters
):
    """Create a scheduled pipeline run.
    
    Args:
        project_id: GCP project ID
        location: GCP region
        pipeline_root: GCS path for pipeline artifacts
        service_account: Service account for pipeline execution
        schedule: Cron schedule expression
        **pipeline_parameters: Parameters to pass to pipeline
    """
    from google.cloud import aiplatform_v1
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Compile pipeline
    pipeline_path = "cqc_ml_pipeline.json"
    compile_pipeline(pipeline_path)
    
    # Upload compiled pipeline to GCS
    import os
    from google.cloud import storage
    
    bucket_name = f"{project_id}-cqc-ml-pipeline"
    blob_name = f"pipelines/cqc_ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(pipeline_path)
    
    pipeline_gcs_uri = f"gs://{bucket_name}/{blob_name}"
    
    # Create schedule
    schedule_client = aiplatform_v1.ScheduleServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )
    
    parent = f"projects/{project_id}/locations/{location}"
    
    schedule_config = {
        "display_name": "cqc-ml-weekly-training",
        "cron": schedule,
        "create_pipeline_job_request": {
            "parent": parent,
            "pipeline_job": {
                "display_name": "cqc-ml-scheduled-training",
                "template_uri": pipeline_gcs_uri,
                "pipeline_root": pipeline_root or f"gs://{project_id}-cqc-ml-pipeline/scheduled_runs",
                "parameter_values": {
                    "project_id": project_id,
                    "location": location,
                    **pipeline_parameters
                },
                "service_account": service_account
            }
        }
    }
    
    schedule = schedule_client.create_schedule(
        parent=parent,
        schedule=schedule_config
    )
    
    print(f"Schedule created: {schedule.name}")
    print(f"Cron expression: {schedule.cron}")
    
    return schedule


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="CQC ML Pipeline Runner")
    parser.add_argument("--project-id", required=True, help="GCP Project ID")
    parser.add_argument("--location", default="us-central1", help="GCP Region")
    parser.add_argument("--action", choices=["compile", "run", "schedule"], 
                       default="compile", help="Action to perform")
    parser.add_argument("--service-account", help="Service account for pipeline execution")
    parser.add_argument("--enable-automl", action="store_true", 
                       help="Enable AutoML training")
    
    args = parser.parse_args()
    
    if args.action == "compile":
        compile_pipeline()
    
    elif args.action == "run":
        job = run_pipeline(
            project_id=args.project_id,
            location=args.location,
            service_account=args.service_account,
            enable_automl=args.enable_automl
        )
        print(f"Pipeline job created: {job.resource_name}")
    
    elif args.action == "schedule":
        schedule = create_pipeline_schedule(
            project_id=args.project_id,
            location=args.location,
            service_account=args.service_account,
            enable_automl=args.enable_automl
        )
        print(f"Pipeline schedule created: {schedule.name}")