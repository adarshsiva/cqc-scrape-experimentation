"""
CQC Streaming Feature Pipeline Package

This package contains the Apache Beam streaming pipeline for real-time CQC feature ingestion.
The pipeline processes dashboard events from Pub/Sub and writes features to BigQuery and Vertex AI Feature Store.

Main Components:
- streaming_feature_pipeline: Main pipeline implementation
- Event parsing and validation
- Feature transformation and aggregation
- BigQuery and Feature Store writers
- Error handling and monitoring

Usage:
    python -m src.dataflow.streaming_feature_pipeline --project-id=PROJECT_ID

Author: Claude Code
Version: 1.0.0
"""

__version__ = '1.0.0'
__author__ = 'Claude Code'
__email__ = 'noreply@anthropic.com'

# Import main pipeline components for easy access
try:
    from .streaming_feature_pipeline import (
        StreamingFeaturePipeline,
        DashboardEventParser,
        DashboardMetricsTransformer,
        FeatureAggregator,
        VertexAIFeatureStoreWriter
    )
except ImportError:
    # Handle case where dependencies aren't installed
    pass

# Package metadata
__all__ = [
    'StreamingFeaturePipeline',
    'DashboardEventParser', 
    'DashboardMetricsTransformer',
    'FeatureAggregator',
    'VertexAIFeatureStoreWriter'
]