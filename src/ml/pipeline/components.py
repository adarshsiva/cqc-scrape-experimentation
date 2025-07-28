"""
Vertex AI pipeline components for CQC ML system.

This module contains reusable pipeline components for:
- Data preparation
- Feature preprocessing
- Model training (XGBoost, LightGBM, AutoML)
- Model evaluation
- Model deployment
"""

from typing import NamedTuple, Optional, Dict, List
from kfp import dsl
from kfp.dsl import (
    component, 
    Input, 
    Output, 
    Dataset, 
    Model, 
    Metrics,
    Artifact,
    OutputPath
)
import os


@component(
    base_image="python:3.9",
    packages_to_install=[
        "google-cloud-bigquery==3.11.4",
        "pandas==2.0.3",
        "pandas-gbq==0.19.2",
        "scikit-learn==1.3.0",
        "numpy==1.24.3"
    ]
)
def data_preparation_component(
    project_id: str,
    dataset_id: str,
    table_id: str,
    query: Optional[str],
    train_split: float,
    validation_split: float,
    test_split: float,
    train_data: Output[Dataset],
    validation_data: Output[Dataset],
    test_data: Output[Dataset],
    data_stats: Output[Metrics]
) -> None:
    """Prepare data from BigQuery for training.
    
    Args:
        project_id: GCP project ID
        dataset_id: BigQuery dataset ID
        table_id: BigQuery table ID
        query: Optional custom query to filter/transform data
        train_split: Fraction of data for training
        validation_split: Fraction of data for validation
        test_split: Fraction of data for testing
        train_data: Output training dataset
        validation_data: Output validation dataset
        test_data: Output test dataset
        data_stats: Output data statistics
    """
    from google.cloud import bigquery
    import pandas as pd
    import numpy as np
    import json
    
    # Initialize BigQuery client
    client = bigquery.Client(project=project_id)
    
    # Construct query
    if query:
        sql_query = query
    else:
        sql_query = f"""
        SELECT 
            locationId,
            providerId,
            name,
            type,
            serviceType,
            region,
            localAuthority,
            postalCode,
            currentRatings.overall.rating as rating,
            lastInspection.date as lastInspectionDate,
            registrationDate,
            regulatedActivities,
            inspectionAreas,
            numberOfBeds,
            locationStatus
        FROM `{project_id}.{dataset_id}.{table_id}`
        WHERE currentRatings.overall.rating IS NOT NULL
        """
    
    # Load data from BigQuery
    df = client.query(sql_query).to_dataframe()
    
    # Log data statistics
    data_stats.log_metric("total_records", len(df))
    data_stats.log_metric("features", len(df.columns) - 1)  # Exclude target
    
    # Class distribution
    if 'rating' in df.columns:
        class_dist = df['rating'].value_counts().to_dict()
        for rating, count in class_dist.items():
            data_stats.log_metric(f"class_{rating}", count)
    
    # Shuffle data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split indices
    n = len(df)
    train_idx = int(n * train_split)
    val_idx = train_idx + int(n * validation_split)
    
    # Split data
    train_df = df[:train_idx]
    val_df = df[train_idx:val_idx]
    test_df = df[val_idx:]
    
    # Save datasets
    train_df.to_csv(train_data.path, index=False)
    val_df.to_csv(validation_data.path, index=False)
    test_df.to_csv(test_data.path, index=False)
    
    # Log split statistics
    data_stats.log_metric("train_records", len(train_df))
    data_stats.log_metric("validation_records", len(val_df))
    data_stats.log_metric("test_records", len(test_df))


@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "numpy==1.24.3",
        "joblib==1.3.1"
    ]
)
def feature_preprocessing_component(
    train_data: Input[Dataset],
    validation_data: Input[Dataset],
    test_data: Input[Dataset],
    target_column: str,
    processed_train: Output[Dataset],
    processed_validation: Output[Dataset],
    processed_test: Output[Dataset],
    feature_preprocessor: Output[Artifact],
    feature_stats: Output[Metrics]
) -> None:
    """Preprocess features for ML training.
    
    Args:
        train_data: Input training dataset
        validation_data: Input validation dataset
        test_data: Input test dataset
        target_column: Name of target column
        processed_train: Output processed training dataset
        processed_validation: Output processed validation dataset
        processed_test: Output processed test dataset
        feature_preprocessor: Output fitted preprocessor artifact
        feature_stats: Output feature statistics
    """
    import pandas as pd
    import joblib
    import sys
    import os
    
    # Add parent directory to path to import features module
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Import feature engineering module
    # Note: In production, this would be packaged with the component
    from features import CQCFeatureEngineer
    
    # Load datasets
    train_df = pd.read_csv(train_data.path)
    val_df = pd.read_csv(validation_data.path)
    test_df = pd.read_csv(test_data.path)
    
    # Initialize feature engineer
    engineer = CQCFeatureEngineer()
    
    # Process training data
    X_train, y_train = engineer.create_feature_pipeline(
        train_df, target_column=target_column, is_training=True
    )
    
    # Process validation data
    X_val, y_val = engineer.create_feature_pipeline(
        val_df, target_column=target_column, is_training=False
    )
    
    # Process test data
    X_test, y_test = engineer.create_feature_pipeline(
        test_df, target_column=target_column, is_training=False
    )
    
    # Save processed datasets
    train_processed = pd.concat([X_train, y_train.rename('target')], axis=1)
    val_processed = pd.concat([X_val, y_val.rename('target')], axis=1)
    test_processed = pd.concat([X_test, y_test.rename('target')], axis=1)
    
    train_processed.to_csv(processed_train.path, index=False)
    val_processed.to_csv(processed_validation.path, index=False)
    test_processed.to_csv(processed_test.path, index=False)
    
    # Save preprocessor
    joblib.dump(engineer, feature_preprocessor.path)
    
    # Log feature statistics
    feature_stats.log_metric("num_features", X_train.shape[1])
    feature_stats.log_metric("train_samples", X_train.shape[0])
    feature_stats.log_metric("validation_samples", X_val.shape[0])
    feature_stats.log_metric("test_samples", X_test.shape[0])


@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "xgboost==1.7.6",
        "numpy==1.24.3",
        "joblib==1.3.1"
    ]
)
def train_xgboost_component(
    train_data: Input[Dataset],
    validation_data: Input[Dataset],
    hyperparameters: Dict[str, float],
    model: Output[Model],
    metrics: Output[Metrics]
) -> None:
    """Train XGBoost model.
    
    Args:
        train_data: Processed training dataset
        validation_data: Processed validation dataset
        hyperparameters: XGBoost hyperparameters
        model: Output trained model
        metrics: Output model metrics
    """
    import pandas as pd
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    import joblib
    import json
    
    # Load data
    train_df = pd.read_csv(train_data.path)
    val_df = pd.read_csv(validation_data.path)
    
    # Separate features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_val = val_df.drop('target', axis=1)
    y_val = val_df['target']
    
    # Set default hyperparameters
    default_params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'multi:softprob',
        'num_class': len(y_train.unique()),
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
    
    # Update with provided hyperparameters
    params = {**default_params, **hyperparameters}
    
    # Train model
    xgb_model = xgb.XGBClassifier(**params)
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Evaluate model
    y_pred = xgb_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average='weighted'
    )
    
    # Log metrics
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall", recall)
    metrics.log_metric("f1_score", f1)
    metrics.log_metric("best_iteration", xgb_model.best_iteration)
    
    # Save model
    joblib.dump(xgb_model, model.path)
    
    # Save model metadata
    model.metadata["framework"] = "xgboost"
    model.metadata["feature_names"] = X_train.columns.tolist()
    model.metadata["hyperparameters"] = json.dumps(params)


@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "lightgbm==3.3.5",
        "numpy==1.24.3",
        "joblib==1.3.1"
    ]
)
def train_lightgbm_component(
    train_data: Input[Dataset],
    validation_data: Input[Dataset],
    hyperparameters: Dict[str, float],
    model: Output[Model],
    metrics: Output[Metrics]
) -> None:
    """Train LightGBM model.
    
    Args:
        train_data: Processed training dataset
        validation_data: Processed validation dataset
        hyperparameters: LightGBM hyperparameters
        model: Output trained model
        metrics: Output model metrics
    """
    import pandas as pd
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import joblib
    import json
    
    # Load data
    train_df = pd.read_csv(train_data.path)
    val_df = pd.read_csv(validation_data.path)
    
    # Separate features and target
    X_train = train_df.drop('target', axis=1)
    y_train = train_df['target']
    X_val = val_df.drop('target', axis=1)
    y_val = val_df['target']
    
    # Set default hyperparameters
    default_params = {
        'objective': 'multiclass',
        'num_class': len(y_train.unique()),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # Update with provided hyperparameters
    params = {**default_params, **hyperparameters}
    
    # Create LightGBM datasets
    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)
    
    # Train model
    lgb_model = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
    )
    
    # Evaluate model
    y_pred = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    y_pred = y_pred.argmax(axis=1)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average='weighted'
    )
    
    # Log metrics
    metrics.log_metric("accuracy", accuracy)
    metrics.log_metric("precision", precision)
    metrics.log_metric("recall", recall)
    metrics.log_metric("f1_score", f1)
    metrics.log_metric("best_iteration", lgb_model.best_iteration)
    
    # Save model
    joblib.dump(lgb_model, model.path)
    
    # Save model metadata
    model.metadata["framework"] = "lightgbm"
    model.metadata["feature_names"] = X_train.columns.tolist()
    model.metadata["hyperparameters"] = json.dumps(params)


@component(
    base_image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
    packages_to_install=[
        "google-cloud-aiplatform==1.38.1",
        "pandas==2.0.3"
    ]
)
def train_automl_component(
    project_id: str,
    location: str,
    dataset_display_name: str,
    train_data: Input[Dataset],
    target_column: str,
    budget_hours: float,
    model_display_name: str,
    model: Output[Model],
    metrics: Output[Metrics]
) -> None:
    """Train AutoML Tabular model.
    
    Args:
        project_id: GCP project ID
        location: GCP region
        dataset_display_name: Display name for AutoML dataset
        train_data: Training dataset
        target_column: Target column name
        budget_hours: Training budget in hours
        model_display_name: Display name for trained model
        model: Output model reference
        metrics: Output metrics
    """
    from google.cloud import aiplatform
    import pandas as pd
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Create AutoML dataset
    dataset = aiplatform.TabularDataset.create(
        display_name=dataset_display_name,
        gcs_source=train_data.uri
    )
    
    # Configure and run AutoML training job
    job = aiplatform.AutoMLTabularTrainingJob(
        display_name=f"automl-training-{model_display_name}",
        optimization_prediction_type="classification"
    )
    
    automl_model = job.run(
        dataset=dataset,
        target_column=target_column,
        budget_milli_node_hours=int(budget_hours * 1000),
        model_display_name=model_display_name,
        disable_early_stopping=False
    )
    
    # Get model evaluation
    model_evaluations = automl_model.list_model_evaluations()
    eval_metrics = model_evaluations[0].metrics
    
    # Log metrics
    metrics.log_metric("accuracy", eval_metrics.get('accuracy', 0))
    metrics.log_metric("precision", eval_metrics.get('precision', 0))
    metrics.log_metric("recall", eval_metrics.get('recall', 0))
    metrics.log_metric("f1_score", eval_metrics.get('f1Score', 0))
    
    # Save model reference
    model.uri = automl_model.resource_name
    model.metadata["framework"] = "automl"
    model.metadata["model_resource_name"] = automl_model.resource_name


@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==2.0.3",
        "scikit-learn==1.3.0",
        "numpy==1.24.3",
        "joblib==1.3.1",
        "matplotlib==3.7.2",
        "seaborn==0.12.2"
    ]
)
def model_evaluation_component(
    model: Input[Model],
    test_data: Input[Dataset],
    evaluation_report: Output[Artifact],
    metrics: Output[Metrics],
    deploy_decision: OutputPath(str)
) -> None:
    """Evaluate model performance on test set.
    
    Args:
        model: Trained model
        test_data: Test dataset
        evaluation_report: Output evaluation report
        metrics: Output detailed metrics
        deploy_decision: Output deployment decision
    """
    import pandas as pd
    import joblib
    import json
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support,
        confusion_matrix, classification_report
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Load model and data
    model_obj = joblib.load(model.path)
    test_df = pd.read_csv(test_data.path)
    
    # Separate features and target
    X_test = test_df.drop('target', axis=1)
    y_test = test_df['target']
    
    # Make predictions
    if hasattr(model_obj, 'predict_proba'):
        y_pred_proba = model_obj.predict_proba(X_test)
        y_pred = y_pred_proba.argmax(axis=1)
    else:
        y_pred = model_obj.predict(X_test)
        y_pred_proba = None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )
    precision_avg, recall_avg, f1_avg, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted'
    )
    
    # Log metrics
    metrics.log_metric("test_accuracy", accuracy)
    metrics.log_metric("test_precision_weighted", precision_avg)
    metrics.log_metric("test_recall_weighted", recall_avg)
    metrics.log_metric("test_f1_weighted", f1_avg)
    
    # Per-class metrics
    classes = ['Outstanding', 'Good', 'Requires_improvement', 'Inadequate']
    for i, cls in enumerate(classes[:len(precision)]):
        metrics.log_metric(f"precision_{cls}", float(precision[i]))
        metrics.log_metric(f"recall_{cls}", float(recall[i]))
        metrics.log_metric(f"f1_{cls}", float(f1[i]))
        metrics.log_metric(f"support_{cls}", int(support[i]))
    
    # Create evaluation report
    report = {
        'model_framework': model.metadata.get('framework', 'unknown'),
        'test_accuracy': accuracy,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
    }
    
    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{evaluation_report.path}_confusion_matrix.png')
    plt.close()
    
    # Save evaluation report
    with open(evaluation_report.path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Make deployment decision
    # Deploy if accuracy > 0.7 and all class recalls > 0.5
    deploy = accuracy > 0.7 and all(r > 0.5 for r in recall)
    
    with open(deploy_decision, 'w') as f:
        f.write('deploy' if deploy else 'no_deploy')


@component(
    base_image="gcr.io/google.com/cloudsdktool/cloud-sdk:latest",
    packages_to_install=[
        "google-cloud-aiplatform==1.38.1"
    ]
)
def model_deployment_component(
    project_id: str,
    location: str,
    model: Input[Model],
    deploy_decision: InputPath(str),
    endpoint_display_name: str,
    machine_type: str,
    min_replicas: int,
    max_replicas: int,
    deployed_model_display_name: str,
    endpoint: Output[Artifact]
) -> None:
    """Deploy model to Vertex AI Endpoint.
    
    Args:
        project_id: GCP project ID
        location: GCP region
        model: Model to deploy
        deploy_decision: Deployment decision from evaluation
        endpoint_display_name: Display name for endpoint
        machine_type: Machine type for serving
        min_replicas: Minimum number of replicas
        max_replicas: Maximum number of replicas
        deployed_model_display_name: Display name for deployed model
        endpoint: Output endpoint information
    """
    from google.cloud import aiplatform
    import json
    
    # Read deployment decision
    with open(deploy_decision, 'r') as f:
        decision = f.read().strip()
    
    if decision != 'deploy':
        print("Model did not meet deployment criteria. Skipping deployment.")
        endpoint.metadata["deployed"] = "false"
        endpoint.metadata["reason"] = "Did not meet performance criteria"
        return
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Upload model to Vertex AI
    if model.metadata.get('framework') == 'automl':
        # AutoML model is already in Vertex AI
        vertex_model = aiplatform.Model(model.uri)
    else:
        # Upload custom model
        vertex_model = aiplatform.Model.upload(
            display_name=deployed_model_display_name,
            artifact_uri=model.uri,
            serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
        )
    
    # Create or get endpoint
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_display_name}"'
    )
    
    if endpoints:
        vertex_endpoint = endpoints[0]
    else:
        vertex_endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name
        )
    
    # Deploy model to endpoint
    vertex_model.deploy(
        endpoint=vertex_endpoint,
        deployed_model_display_name=deployed_model_display_name,
        machine_type=machine_type,
        min_replica_count=min_replicas,
        max_replica_count=max_replicas,
        traffic_percentage=100
    )
    
    # Save endpoint information
    endpoint.metadata["deployed"] = "true"
    endpoint.metadata["endpoint_resource_name"] = vertex_endpoint.resource_name
    endpoint.metadata["model_resource_name"] = vertex_model.resource_name
    
    with open(endpoint.path, 'w') as f:
        json.dump({
            "endpoint_name": vertex_endpoint.resource_name,
            "model_name": vertex_model.resource_name,
            "deployed": True
        }, f)