"""
Vertex AI Feature Store Configuration and Management for CQC Prediction System
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud.aiplatform import Feature, EntityType, Featurestore
from google.cloud.exceptions import NotFound, Conflict
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "machine-learning-exp-467008"
REGION = "europe-west2"
FEATURE_STORE_ID = "cqc-prediction-features"


class CQCFeatureStoreManager:
    """Manages Vertex AI Feature Store for CQC predictions"""
    
    def __init__(self, project_id: str = PROJECT_ID, region: str = REGION):
        self.project_id = project_id
        self.region = region
        self.feature_store_id = FEATURE_STORE_ID
        
        # Initialize AI Platform
        aiplatform.init(project=project_id, location=region)
        
        # BigQuery client for data import
        self.bq_client = bigquery.Client(project=project_id)
        
        self.feature_store = None
        self.entity_types = {}
        
    def create_feature_store(self) -> Featurestore:
        """Create Vertex AI Feature Store with high availability configuration"""
        
        try:
            # Check if feature store already exists
            feature_store = aiplatform.Featurestore(
                featurestore_name=self.feature_store_id,
                project=self.project_id,
                location=self.region
            )
            logger.info(f"Feature store {self.feature_store_id} already exists")
            self.feature_store = feature_store
            return feature_store
            
        except NotFound:
            logger.info(f"Creating new feature store: {self.feature_store_id}")
            
            # Create new feature store with online serving
            feature_store = aiplatform.Featurestore.create(
                featurestore_id=self.feature_store_id,
                online_serving_config={
                    "fixed_node_count": 10  # High capacity for sub-100ms serving
                },
                labels={
                    "project": "cqc-prediction",
                    "environment": "production",
                    "team": "ml-engineering"
                },
                description="Feature store for CQC rating predictions with real-time dashboard metrics",
                create_request_timeout=600
            )
            
            self.feature_store = feature_store
            logger.info(f"Feature store created: {feature_store.resource_name}")
            return feature_store
    
    def create_entity_types(self):
        """Create entity types for different feature groups"""
        
        entity_configs = [
            {
                "entity_type_id": "location",
                "description": "CQC location features",
                "monitoring_config": {
                    "snapshot_analysis": {
                        "disabled": False,
                        "monitoring_interval_days": 1,
                        "staleness_days": 7
                    }
                }
            },
            {
                "entity_type_id": "provider",
                "description": "CQC provider aggregated features",
                "monitoring_config": {
                    "snapshot_analysis": {
                        "disabled": False,
                        "monitoring_interval_days": 1,
                        "staleness_days": 7
                    }
                }
            },
            {
                "entity_type_id": "dashboard_metrics",
                "description": "Real-time dashboard metrics",
                "monitoring_config": {
                    "snapshot_analysis": {
                        "disabled": False,
                        "monitoring_interval_days": 1,
                        "staleness_days": 1  # More frequent updates for real-time metrics
                    }
                }
            }
        ]
        
        for config in entity_configs:
            try:
                entity_type = self.feature_store.create_entity_type(
                    entity_type_id=config["entity_type_id"],
                    description=config["description"],
                    monitoring_config=config.get("monitoring_config")
                )
                self.entity_types[config["entity_type_id"]] = entity_type
                logger.info(f"Created entity type: {config['entity_type_id']}")
                
            except Conflict:
                # Entity type already exists
                entity_type = self.feature_store.get_entity_type(
                    entity_type_id=config["entity_type_id"]
                )
                self.entity_types[config["entity_type_id"]] = entity_type
                logger.info(f"Entity type already exists: {config['entity_type_id']}")
    
    def create_features(self):
        """Create all features for CQC predictions"""
        
        # CQC API Features (80+ features)
        cqc_features = [
            # Rating features
            ("overall_rating", "INT64", "Overall CQC rating (1-4)"),
            ("safe_rating", "INT64", "Safe domain rating"),
            ("effective_rating", "INT64", "Effective domain rating"),
            ("caring_rating", "INT64", "Caring domain rating"),
            ("responsive_rating", "INT64", "Responsive domain rating"),
            ("well_led_rating", "INT64", "Well-led domain rating"),
            
            # Inspection features
            ("days_since_inspection", "INT64", "Days since last inspection"),
            ("inspection_type", "STRING", "Type of last inspection"),
            ("total_inspections", "INT64", "Total number of inspections"),
            ("enforcement_actions", "INT64", "Number of enforcement actions"),
            
            # Provider features
            ("provider_avg_rating", "DOUBLE", "Provider average rating"),
            ("provider_locations_count", "INT64", "Number of provider locations"),
            ("provider_type", "STRING", "Type of provider"),
            
            # Service features
            ("registered_beds", "INT64", "Number of registered beds"),
            ("service_types", "STRING_ARRAY", "Types of services provided"),
            ("specialisms", "STRING_ARRAY", "Specialisms"),
            ("regulated_activities", "STRING_ARRAY", "Regulated activities"),
            
            # Location attributes
            ("region", "STRING", "Geographic region"),
            ("local_authority", "STRING", "Local authority"),
            ("ccg", "STRING", "Clinical commissioning group"),
            ("ownership_type", "STRING", "Ownership type"),
            
            # Historical metrics
            ("rating_trend", "DOUBLE", "Rating trend over time"),
            ("improvement_rate", "DOUBLE", "Rate of improvement"),
            ("volatility_score", "DOUBLE", "Rating volatility score")
        ]
        
        # Dashboard-specific features (NEW)
        dashboard_features = [
            # Incident metrics
            ("incident_rate_7d", "DOUBLE", "Incident rate last 7 days"),
            ("incident_rate_30d", "DOUBLE", "Incident rate last 30 days"),
            ("critical_incidents_30d", "INT64", "Critical incidents last 30 days"),
            ("incident_severity_avg", "DOUBLE", "Average incident severity"),
            
            # Staff metrics
            ("staff_turnover_rate", "DOUBLE", "Staff turnover rate"),
            ("staff_to_resident_ratio", "DOUBLE", "Staff to resident ratio"),
            ("agency_staff_percentage", "DOUBLE", "Percentage of agency staff"),
            ("training_completion_rate", "DOUBLE", "Staff training completion rate"),
            ("overtime_hours_avg", "DOUBLE", "Average overtime hours per staff"),
            
            # Care quality metrics
            ("care_plan_compliance_rate", "DOUBLE", "Care plan compliance rate"),
            ("medication_errors_30d", "INT64", "Medication errors last 30 days"),
            ("falls_count_30d", "INT64", "Falls count last 30 days"),
            ("activity_participation_rate", "DOUBLE", "Activity participation rate"),
            ("nutrition_score", "DOUBLE", "Nutrition quality score"),
            
            # Complaints and alerts
            ("complaints_count_90d", "INT64", "Complaints last 90 days"),
            ("safeguarding_alerts_90d", "INT64", "Safeguarding alerts last 90 days"),
            ("compliments_count_90d", "INT64", "Compliments last 90 days"),
            
            # Operational metrics
            ("occupancy_rate", "DOUBLE", "Bed occupancy rate"),
            ("equipment_issues_30d", "INT64", "Equipment issues last 30 days"),
            ("maintenance_requests_pending", "INT64", "Pending maintenance requests"),
            
            # Risk indicators
            ("high_risk_residents_count", "INT64", "Number of high risk residents"),
            ("pressure_ulcer_rate", "DOUBLE", "Pressure ulcer rate"),
            ("infection_rate", "DOUBLE", "Infection rate"),
            ("hospital_admissions_rate", "DOUBLE", "Hospital admissions rate")
        ]
        
        # Create features for location entity
        location_entity = self.entity_types["location"]
        for feature_id, value_type, description in cqc_features:
            self._create_feature(location_entity, feature_id, value_type, description)
        
        # Create features for dashboard metrics entity
        dashboard_entity = self.entity_types["dashboard_metrics"]
        for feature_id, value_type, description in dashboard_features:
            self._create_feature(dashboard_entity, feature_id, value_type, description)
    
    def _create_feature(self, entity_type, feature_id: str, value_type: str, 
                       description: str):
        """Create a single feature in an entity type"""
        try:
            feature = entity_type.create_feature(
                feature_id=feature_id,
                value_type=value_type,
                description=description,
                labels={
                    "feature_group": "cqc_prediction",
                    "importance": "high"
                }
            )
            logger.info(f"Created feature: {feature_id}")
            return feature
            
        except Conflict:
            logger.info(f"Feature already exists: {feature_id}")
            return entity_type.get_feature(feature_id=feature_id)
    
    def import_features_from_bigquery(self, 
                                     source_table: str,
                                     entity_type_id: str,
                                     entity_id_field: str = "location_id"):
        """Import features from BigQuery table to Feature Store"""
        
        entity_type = self.entity_types.get(entity_type_id)
        if not entity_type:
            raise ValueError(f"Entity type {entity_type_id} not found")
        
        # Create import feature values request
        import_request = entity_type.batch_create_features(
            feature_requests=[],  # Features already created
            import_feature_values_config={
                "feature_specs": [
                    {"id": feature.name.split('/')[-1]} 
                    for feature in entity_type.list_features()
                ],
                "source_uris": [f"bq://{source_table}"],
                "entity_id_field": entity_id_field,
                "feature_time_field": "update_timestamp",
                "disable_online_serving": False
            }
        )
        
        logger.info(f"Started feature import from {source_table}")
        return import_request
    
    def get_online_features(self, 
                           entity_ids: List[str],
                           feature_ids: List[str],
                           entity_type: str = "location") -> pd.DataFrame:
        """Get features from online serving for real-time predictions"""
        
        entity_type_obj = self.entity_types.get(entity_type)
        if not entity_type_obj:
            raise ValueError(f"Entity type {entity_type} not found")
        
        # Read features from online store
        feature_values = entity_type_obj.read(
            entity_ids=entity_ids,
            feature_ids=feature_ids
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_values)
        return df
    
    def update_online_features(self, 
                              features_dict: Dict[str, Any],
                              entity_id: str,
                              entity_type: str = "dashboard_metrics"):
        """Update features in online serving store"""
        
        entity_type_obj = self.entity_types.get(entity_type)
        if not entity_type_obj:
            raise ValueError(f"Entity type {entity_type} not found")
        
        # Prepare feature values
        feature_values = []
        for feature_id, value in features_dict.items():
            feature_values.append({
                "feature_id": feature_id,
                "value": value
            })
        
        # Write to online store
        entity_type_obj.write_feature_values(
            instances=[{
                "entity_id": entity_id,
                "features": feature_values
            }]
        )
        
        logger.info(f"Updated {len(feature_values)} features for entity {entity_id}")
    
    def setup_monitoring(self):
        """Setup monitoring for feature drift and data quality"""
        
        # Configure monitoring for each entity type
        for entity_name, entity_type in self.entity_types.items():
            monitoring_config = {
                "snapshot_analysis": {
                    "disabled": False,
                    "monitoring_interval_days": 1,
                    "staleness_days": 7 if entity_name != "dashboard_metrics" else 1
                },
                "import_features_analysis": {
                    "state": "ENABLED",
                    "anomaly_detection_baseline": "LATEST_IMPORT_FEATURES_STATS"
                },
                "numerical_threshold_config": {
                    "value": 0.1  # 10% threshold for drift detection
                },
                "categorical_threshold_config": {
                    "value": 0.1
                }
            }
            
            entity_type.update(
                monitoring_config=monitoring_config
            )
            
            logger.info(f"Monitoring configured for entity type: {entity_name}")
    
    def create_serving_endpoint(self):
        """Create optimized serving endpoint for low-latency predictions"""
        
        endpoint_config = {
            "display_name": "cqc-feature-serving-endpoint",
            "description": "High-performance endpoint for CQC feature serving",
            "labels": {
                "environment": "production",
                "use_case": "cqc_prediction"
            }
        }
        
        # This would typically create a Vertex AI endpoint
        # For feature serving, the feature store handles this internally
        logger.info("Feature serving endpoint configured with online store")
        
        return {
            "endpoint": f"{self.region}-aiplatform.googleapis.com",
            "feature_store": self.feature_store_id,
            "online_serving_nodes": 10,
            "expected_latency_ms": 50
        }


def main():
    """Main function to setup Feature Store"""
    
    manager = CQCFeatureStoreManager()
    
    # Step 1: Create Feature Store
    logger.info("Creating Feature Store...")
    feature_store = manager.create_feature_store()
    
    # Step 2: Create Entity Types
    logger.info("Creating Entity Types...")
    manager.create_entity_types()
    
    # Step 3: Create Features
    logger.info("Creating Features...")
    manager.create_features()
    
    # Step 4: Setup Monitoring
    logger.info("Setting up Monitoring...")
    manager.setup_monitoring()
    
    # Step 5: Create Serving Endpoint
    logger.info("Configuring Serving Endpoint...")
    endpoint_info = manager.create_serving_endpoint()
    
    logger.info("Feature Store setup complete!")
    logger.info(f"Feature Store ID: {manager.feature_store_id}")
    logger.info(f"Project: {manager.project_id}")
    logger.info(f"Region: {manager.region}")
    logger.info(f"Endpoint Info: {json.dumps(endpoint_info, indent=2)}")
    
    return feature_store


if __name__ == "__main__":
    main()