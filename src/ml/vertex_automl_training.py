"""
Vertex AI AutoML Tables Training Script for CQC Rating Predictions.

This script creates and trains an AutoML Tables model for predicting CQC ratings
using comprehensive dashboard features from BigQuery.
"""

import argparse
import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import yaml

from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud.aiplatform import gapic as aip
from google.api_core import exceptions
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VertexAutoMLTrainer:
    """Vertex AI AutoML Tables trainer for CQC rating predictions."""
    
    def __init__(self, project_id: str, region: str, config_path: str = None):
        """Initialize the AutoML trainer.
        
        Args:
            project_id: Google Cloud project ID
            region: GCP region for training
            config_path: Path to configuration YAML file
        """
        self.project_id = project_id
        self.region = region
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Initialize clients
        self.bq_client = bigquery.Client(project=project_id)
        
        logger.info(f"Initialized AutoML trainer for project {project_id} in region {region}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def validate_bigquery_dataset(self, table_id: str) -> bool:
        """Validate that the BigQuery dataset exists and has required columns.
        
        Args:
            table_id: BigQuery table ID in format 'dataset.table'
            
        Returns:
            True if dataset is valid
        """
        try:
            # Get table reference
            table_ref = self.bq_client.get_table(table_id)
            
            # Check if table exists
            logger.info(f"Found BigQuery table: {table_id}")
            logger.info(f"Table has {table_ref.num_rows} rows and {len(table_ref.schema)} columns")
            
            # Check for required columns
            required_columns = ['overall_rating']
            schema_columns = [field.name for field in table_ref.schema]
            
            missing_columns = [col for col in required_columns if col not in schema_columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            # Log column names for verification
            logger.info(f"Available columns: {schema_columns[:10]}...")  # Show first 10
            
            return True
            
        except exceptions.NotFound:
            logger.error(f"Table {table_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            return False
    
    def create_automl_dataset(self, 
                             display_name: str,
                             bq_table_uri: str) -> aiplatform.TabularDataset:
        """Create AutoML dataset from BigQuery table.
        
        Args:
            display_name: Display name for the dataset
            bq_table_uri: BigQuery table URI
            
        Returns:
            Created TabularDataset
        """
        logger.info(f"Creating AutoML dataset: {display_name}")
        
        try:
            # Create dataset
            dataset = aiplatform.TabularDataset.create(
                display_name=display_name,
                bq_source=bq_table_uri,
                sync=True
            )
            
            logger.info(f"Created dataset with resource name: {dataset.resource_name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            raise
    
    def get_dashboard_features(self) -> List[str]:
        """Get list of dashboard features to include in training.
        
        Returns:
            List of feature column names
        """
        # Dashboard features based on CQC comprehensive analysis
        dashboard_features = [
            # Basic information
            'provider_id',
            'location_id',
            'provider_type',
            'service_type',
            'location_status',
            
            # Geographic features  
            'region',
            'local_authority',
            'constituency',
            'postcode_district',
            
            # Temporal features
            'registration_age_years',
            'days_since_last_inspection',
            'months_since_last_inspection',
            'inspection_month',
            'inspection_quarter',
            'inspection_day_of_week',
            
            # Service features
            'num_regulated_activities',
            'num_inspection_areas',
            'is_residential_service',
            'is_community_service', 
            'is_hospital_service',
            'is_primary_service',
            
            # Historical performance features
            'regional_good_rating_rate',
            'provider_historical_rating_avg',
            'location_previous_rating',
            
            # Inspection complexity features
            'inspection_areas_complexity_score',
            'regulated_activities_complexity_score',
            
            # Text-based features (TF-IDF)
            'inspection_areas_tfidf_0', 'inspection_areas_tfidf_1', 'inspection_areas_tfidf_2',
            'regulated_activities_tfidf_0', 'regulated_activities_tfidf_1', 'regulated_activities_tfidf_2',
            
            # Aggregated regional features
            'region_avg_inspection_frequency',
            'region_provider_count',
            'region_location_count',
            
            # Provider-level features
            'provider_location_count',
            'provider_service_type_diversity',
            'provider_avg_rating',
            
            # Risk indicators
            'high_risk_service_flag',
            'inspection_overdue_flag',
            'new_registration_flag',
            'location_type_risk_score',
            
            # Seasonal features
            'is_winter_inspection',
            'is_summer_inspection', 
            'is_holiday_period_inspection'
        ]
        
        logger.info(f"Using {len(dashboard_features)} dashboard features")
        return dashboard_features
    
    def create_automl_training_job(self,
                                  dataset: aiplatform.TabularDataset,
                                  display_name: str,
                                  target_column: str = 'overall_rating',
                                  optimization_objective: str = 'maximize-precision-at-recall',
                                  budget_milli_node_hours: int = 10000) -> aiplatform.AutoMLTabularTrainingJob:
        """Create and configure AutoML training job.
        
        Args:
            dataset: TabularDataset to train on
            display_name: Display name for training job
            target_column: Target column name
            optimization_objective: AutoML optimization objective
            budget_milli_node_hours: Training budget in milli node hours
            
        Returns:
            Configured AutoMLTabularTrainingJob
        """
        logger.info(f"Creating AutoML training job: {display_name}")
        
        # Get dashboard features
        feature_columns = self.get_dashboard_features()
        
        # Configure column transformations
        column_specs = {}
        for feature in feature_columns:
            # Determine column type based on feature name patterns
            if any(keyword in feature.lower() for keyword in ['flag', 'is_', 'num_']):
                column_specs[feature] = 'numeric'
            elif any(keyword in feature.lower() for keyword in ['type', 'status', 'region', 'authority']):
                column_specs[feature] = 'categorical'
            else:
                column_specs[feature] = 'auto'  # Let AutoML decide
        
        # Target column is categorical
        column_specs[target_column] = 'categorical'
        
        # Create training job
        training_job = aiplatform.AutoMLTabularTrainingJob(
            display_name=display_name,
            optimization_objective=optimization_objective,
            column_specs=column_specs,
            optimization_objective_recall_value=0.8,  # Precision at 80% recall
        )
        
        logger.info(f"Configured training job with {len(feature_columns)} features")
        logger.info(f"Optimization objective: {optimization_objective}")
        logger.info(f"Budget: {budget_milli_node_hours} milli node hours")
        
        return training_job
    
    def train_model(self,
                   training_job: aiplatform.AutoMLTabularTrainingJob,
                   dataset: aiplatform.TabularDataset,
                   target_column: str = 'overall_rating',
                   budget_milli_node_hours: int = 10000,
                   model_display_name: str = None) -> aiplatform.Model:
        """Train the AutoML model.
        
        Args:
            training_job: Configured training job
            dataset: Dataset to train on
            target_column: Target column name
            budget_milli_node_hours: Training budget
            model_display_name: Display name for the trained model
            
        Returns:
            Trained Model
        """
        logger.info("Starting AutoML model training...")
        
        if not model_display_name:
            model_display_name = f"cqc-automl-model-{int(time.time())}"
        
        try:
            # Start training
            model = training_job.run(
                dataset=dataset,
                target_column=target_column,
                budget_milli_node_hours=budget_milli_node_hours,
                model_display_name=model_display_name,
                sync=True  # Wait for training to complete
            )
            
            logger.info(f"Training completed successfully!")
            logger.info(f"Model resource name: {model.resource_name}")
            
            return model
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
    
    def export_feature_importance(self, 
                                 model: aiplatform.Model,
                                 output_path: str = None) -> Dict:
        """Export feature importance from trained model.
        
        Args:
            model: Trained AutoML model
            output_path: Path to save feature importance JSON
            
        Returns:
            Feature importance dictionary
        """
        logger.info("Exporting feature importance...")
        
        try:
            # Get model evaluation
            evaluations = model.list_model_evaluations()
            
            if not evaluations:
                logger.warning("No model evaluations found")
                return {}
            
            evaluation = evaluations[0]
            
            # Extract feature importance from evaluation metrics
            metrics = evaluation.to_dict()
            
            # Feature attributions are typically in the evaluation metadata
            feature_importance = {}
            
            if 'metadata' in metrics and 'featureAttributions' in metrics['metadata']:
                attributions = metrics['metadata']['featureAttributions']
                
                for attribution in attributions:
                    feature_name = attribution.get('featureName', 'unknown')
                    importance = attribution.get('attribution', 0.0)
                    feature_importance[feature_name] = importance
                
                # Sort by importance
                feature_importance = dict(
                    sorted(feature_importance.items(), 
                          key=lambda x: x[1], reverse=True)
                )
                
                logger.info(f"Extracted importance for {len(feature_importance)} features")
                
                # Save to file if path provided
                if output_path:
                    import json
                    with open(output_path, 'w') as f:
                        json.dump(feature_importance, f, indent=2)
                    logger.info(f"Saved feature importance to {output_path}")
            
            return feature_importance
            
        except Exception as e:
            logger.error(f"Failed to export feature importance: {e}")
            return {}
    
    def deploy_model_to_endpoint(self,
                                model: aiplatform.Model,
                                endpoint_display_name: str = None,
                                machine_type: str = "n1-standard-2",
                                min_replica_count: int = 1,
                                max_replica_count: int = 3) -> aiplatform.Endpoint:
        """Deploy model to Vertex AI endpoint.
        
        Args:
            model: Trained model to deploy
            endpoint_display_name: Display name for endpoint
            machine_type: Machine type for serving
            min_replica_count: Minimum number of replicas
            max_replica_count: Maximum number of replicas
            
        Returns:
            Deployed Endpoint
        """
        if not endpoint_display_name:
            endpoint_display_name = f"cqc-automl-endpoint-{int(time.time())}"
        
        logger.info(f"Deploying model to endpoint: {endpoint_display_name}")
        
        try:
            # Create endpoint
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_display_name,
                sync=True
            )
            
            logger.info(f"Created endpoint: {endpoint.resource_name}")
            
            # Deploy model to endpoint
            deployed_model = endpoint.deploy(
                model=model,
                machine_type=machine_type,
                min_replica_count=min_replica_count,
                max_replica_count=max_replica_count,
                sync=True
            )
            
            logger.info(f"Model deployed successfully to endpoint")
            logger.info(f"Endpoint URL: {endpoint.resource_name}")
            
            return endpoint
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise
    
    def run_full_pipeline(self,
                         bq_table_id: str,
                         dataset_display_name: str = None,
                         training_job_display_name: str = None,
                         budget_milli_node_hours: int = 10000,
                         deploy_to_endpoint: bool = True) -> Tuple[aiplatform.Model, Optional[aiplatform.Endpoint]]:
        """Run the complete AutoML training pipeline.
        
        Args:
            bq_table_id: BigQuery table ID (format: dataset.table)
            dataset_display_name: Display name for dataset
            training_job_display_name: Display name for training job
            budget_milli_node_hours: Training budget
            deploy_to_endpoint: Whether to deploy to endpoint
            
        Returns:
            Tuple of (trained model, endpoint if deployed)
        """
        logger.info("Starting full AutoML pipeline...")
        
        # Default names with timestamp
        timestamp = int(time.time())
        if not dataset_display_name:
            dataset_display_name = f"cqc-automl-dataset-{timestamp}"
        if not training_job_display_name:
            training_job_display_name = f"cqc-automl-training-{timestamp}"
        
        try:
            # Step 1: Validate BigQuery dataset
            logger.info("Step 1: Validating BigQuery dataset...")
            if not self.validate_bigquery_dataset(bq_table_id):
                raise ValueError(f"Invalid BigQuery dataset: {bq_table_id}")
            
            # Step 2: Create AutoML dataset
            logger.info("Step 2: Creating AutoML dataset...")
            bq_uri = f"bq://{self.project_id}.{bq_table_id}"
            dataset = self.create_automl_dataset(dataset_display_name, bq_uri)
            
            # Step 3: Create training job
            logger.info("Step 3: Creating training job...")
            training_job = self.create_automl_training_job(
                dataset=dataset,
                display_name=training_job_display_name,
                budget_milli_node_hours=budget_milli_node_hours
            )
            
            # Step 4: Train model
            logger.info("Step 4: Training model...")
            model = self.train_model(
                training_job=training_job,
                dataset=dataset,
                budget_milli_node_hours=budget_milli_node_hours
            )
            
            # Step 5: Export feature importance
            logger.info("Step 5: Exporting feature importance...")
            feature_importance = self.export_feature_importance(
                model, f"feature_importance_{timestamp}.json"
            )
            
            # Step 6: Deploy to endpoint (optional)
            endpoint = None
            if deploy_to_endpoint:
                logger.info("Step 6: Deploying to endpoint...")
                endpoint = self.deploy_model_to_endpoint(model)
            
            logger.info("AutoML pipeline completed successfully!")
            
            # Print summary
            logger.info(f"Dataset: {dataset.resource_name}")
            logger.info(f"Model: {model.resource_name}")
            if endpoint:
                logger.info(f"Endpoint: {endpoint.resource_name}")
            
            return model, endpoint
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Main function to run AutoML training."""
    parser = argparse.ArgumentParser(description='Train CQC AutoML model')
    parser.add_argument('--project-id', required=True, help='Google Cloud project ID')
    parser.add_argument('--region', default='europe-west2', help='GCP region')
    parser.add_argument('--bq-table', required=True, help='BigQuery table ID (dataset.table)')
    parser.add_argument('--config', help='Path to configuration YAML file')
    parser.add_argument('--budget', type=int, default=10000, 
                       help='Training budget in milli node hours')
    parser.add_argument('--no-deploy', action='store_true', 
                       help='Skip endpoint deployment')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = VertexAutoMLTrainer(
        project_id=args.project_id,
        region=args.region,
        config_path=args.config
    )
    
    # Run pipeline
    try:
        model, endpoint = trainer.run_full_pipeline(
            bq_table_id=args.bq_table,
            budget_milli_node_hours=args.budget,
            deploy_to_endpoint=not args.no_deploy
        )
        
        print(f"\n✅ AutoML training completed successfully!")
        print(f"Model: {model.resource_name}")
        if endpoint:
            print(f"Endpoint: {endpoint.resource_name}")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()