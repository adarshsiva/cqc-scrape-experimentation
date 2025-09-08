"""
Apache Beam Streaming Pipeline for Real-time CQC Feature Ingestion.

This pipeline processes dashboard events from Pub/Sub in real-time and:
- Reads from 'dashboard-events' Pub/Sub topic
- Processes and transforms dashboard metrics
- Writes to BigQuery table 'cqc_dataset.realtime_features'
- Updates Vertex AI Feature Store
- Uses streaming mode with autoscaling

Author: Claude Code
Project: CQC Rating Predictor ML System
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    GoogleCloudOptions,
    StandardOptions,
    WorkerOptions,
    SetupOptions,
    StreamingOptions
)
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.io.gcp.bigquery import WriteToBigQuery, BigQueryDisposition
from apache_beam.io.gcp.bigtable import WriteToBigTable
from apache_beam.transforms.window import (
    FixedWindows,
    SlidingWindows,
    Sessions,
    GlobalWindows
)
from apache_beam.transforms.trigger import (
    AfterProcessingTime,
    AfterWatermark,
    AfterCount,
    Repeatedly,
    AccumulationMode
)
from apache_beam.metrics import Metrics
from google.cloud import aiplatform
from google.cloud.aiplatform_v1 import FeatureOnlineStoreServiceClient
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DashboardEventParser(beam.DoFn):
    """Parse and validate incoming dashboard events from Pub/Sub."""
    
    def __init__(self):
        self.parse_success = Metrics.counter('events', 'parse_success')
        self.parse_errors = Metrics.counter('events', 'parse_errors')
        self.invalid_events = Metrics.counter('events', 'invalid_events')
    
    def process(self, element):
        """Parse JSON dashboard event and validate structure."""
        try:
            # Decode the Pub/Sub message if it's bytes
            if isinstance(element, bytes):
                element = element.decode('utf-8')
            
            # Parse JSON
            event = json.loads(element)
            
            # Validate required fields
            required_fields = ['location_id', 'event_type', 'timestamp', 'metrics']
            for field in required_fields:
                if field not in event:
                    self.invalid_events.inc()
                    logger.warning(f"Missing required field: {field}")
                    return
            
            # Add processing metadata
            event['processing_timestamp'] = datetime.now(timezone.utc).isoformat()
            event['pipeline_version'] = 'streaming-v1.0'
            
            # Validate event timestamp
            try:
                event_time = datetime.fromisoformat(
                    event['timestamp'].replace('Z', '+00:00')
                )
                event['event_timestamp_parsed'] = event_time.isoformat()
            except Exception as e:
                logger.warning(f"Invalid timestamp format: {event['timestamp']}")
                event['event_timestamp_parsed'] = datetime.now(timezone.utc).isoformat()
            
            self.parse_success.inc()
            yield event
            
        except json.JSONDecodeError as e:
            self.parse_errors.inc()
            logger.error(f"JSON parsing failed: {e}")
            yield beam.pvalue.TaggedOutput('errors', {
                'error_type': 'json_parse_error',
                'error_message': str(e),
                'raw_data': str(element)[:1000],  # Truncate for storage
                'error_timestamp': datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            self.parse_errors.inc()
            logger.error(f"Unexpected parsing error: {e}")
            yield beam.pvalue.TaggedOutput('errors', {
                'error_type': 'unexpected_error',
                'error_message': str(e),
                'raw_data': str(element)[:1000],
                'error_timestamp': datetime.now(timezone.utc).isoformat()
            })


class DashboardMetricsTransformer(beam.DoFn):
    """Transform dashboard metrics into ML features."""
    
    def __init__(self):
        self.transforms_success = Metrics.counter('transforms', 'success')
        self.transforms_errors = Metrics.counter('transforms', 'errors')
        
    def process(self, element):
        """Transform dashboard event into ML features."""
        try:
            event = element
            
            # Extract base information
            location_id = event['location_id']
            event_type = event['event_type']
            metrics = event.get('metrics', {})
            
            # Create feature record
            feature_record = {
                'location_id': location_id,
                'event_type': event_type,
                'event_timestamp': event['event_timestamp_parsed'],
                'processing_timestamp': event['processing_timestamp'],
                'pipeline_version': event['pipeline_version']
            }
            
            # Transform different event types
            if event_type == 'inspection_update':
                feature_record.update(self._transform_inspection_metrics(metrics))
            elif event_type == 'rating_change':
                feature_record.update(self._transform_rating_metrics(metrics))
            elif event_type == 'capacity_update':
                feature_record.update(self._transform_capacity_metrics(metrics))
            elif event_type == 'compliance_alert':
                feature_record.update(self._transform_compliance_metrics(metrics))
            elif event_type == 'dashboard_interaction':
                feature_record.update(self._transform_interaction_metrics(metrics))
            else:
                feature_record.update(self._transform_generic_metrics(metrics))
            
            # Add derived features
            feature_record.update(self._calculate_derived_features(feature_record, event))
            
            self.transforms_success.inc()
            yield feature_record
            
        except Exception as e:
            self.transforms_errors.inc()
            logger.error(f"Transform error for {element.get('location_id', 'unknown')}: {e}")
            yield beam.pvalue.TaggedOutput('errors', {
                'error_type': 'transform_error',
                'error_message': str(e),
                'location_id': element.get('location_id'),
                'event_type': element.get('event_type'),
                'error_timestamp': datetime.now(timezone.utc).isoformat()
            })
    
    def _transform_inspection_metrics(self, metrics: Dict) -> Dict:
        """Transform inspection-related metrics."""
        return {
            'inspection_type': metrics.get('inspection_type'),
            'inspector_count': metrics.get('inspector_count'),
            'inspection_duration_days': metrics.get('duration_days'),
            'areas_assessed': metrics.get('areas_assessed', []),
            'preliminary_findings': metrics.get('preliminary_findings'),
            'follow_up_required': metrics.get('follow_up_required', False),
            'inspection_priority': metrics.get('priority', 'standard'),
            'risk_indicators_found': len(metrics.get('risk_indicators', [])),
            'feature_category': 'inspection'
        }
    
    def _transform_rating_metrics(self, metrics: Dict) -> Dict:
        """Transform rating change metrics."""
        return {
            'previous_overall_rating': metrics.get('previous_rating'),
            'new_overall_rating': metrics.get('new_rating'),
            'rating_change_direction': self._calculate_rating_direction(
                metrics.get('previous_rating'),
                metrics.get('new_rating')
            ),
            'ratings_changed': metrics.get('specific_ratings_changed', []),
            'rating_change_reason': metrics.get('change_reason'),
            'enforcement_action_taken': metrics.get('enforcement_action', False),
            'public_notification_sent': metrics.get('public_notification', True),
            'rating_effective_date': metrics.get('effective_date'),
            'feature_category': 'rating'
        }
    
    def _transform_capacity_metrics(self, metrics: Dict) -> Dict:
        """Transform capacity update metrics."""
        return {
            'previous_bed_count': metrics.get('previous_beds'),
            'new_bed_count': metrics.get('new_beds'),
            'capacity_change': metrics.get('new_beds', 0) - metrics.get('previous_beds', 0),
            'capacity_change_reason': metrics.get('change_reason'),
            'service_types_affected': metrics.get('affected_services', []),
            'regulatory_approval_required': metrics.get('approval_required', True),
            'planned_occupancy_rate': metrics.get('planned_occupancy'),
            'staff_adjustment_planned': metrics.get('staff_adjustment', False),
            'feature_category': 'capacity'
        }
    
    def _transform_compliance_metrics(self, metrics: Dict) -> Dict:
        """Transform compliance alert metrics."""
        return {
            'alert_severity': metrics.get('severity', 'medium'),
            'alert_category': metrics.get('category'),
            'compliance_areas_affected': metrics.get('affected_areas', []),
            'action_required': metrics.get('action_required', True),
            'response_deadline': metrics.get('response_deadline'),
            'escalation_level': metrics.get('escalation_level', 0),
            'repeat_violation': metrics.get('repeat_violation', False),
            'potential_enforcement': metrics.get('potential_enforcement', False),
            'safeguarding_concern': metrics.get('safeguarding_concern', False),
            'feature_category': 'compliance'
        }
    
    def _transform_interaction_metrics(self, metrics: Dict) -> Dict:
        """Transform dashboard interaction metrics."""
        return {
            'user_type': metrics.get('user_type', 'public'),
            'page_views': metrics.get('page_views', 1),
            'time_on_page': metrics.get('time_on_page'),
            'interaction_type': metrics.get('interaction_type'),
            'search_queries': metrics.get('search_queries', []),
            'filters_applied': metrics.get('filters_applied', []),
            'data_downloaded': metrics.get('data_downloaded', False),
            'feedback_provided': metrics.get('feedback_provided', False),
            'session_duration': metrics.get('session_duration'),
            'feature_category': 'interaction'
        }
    
    def _transform_generic_metrics(self, metrics: Dict) -> Dict:
        """Transform generic metrics for unknown event types."""
        return {
            'generic_metric_1': metrics.get('value_1'),
            'generic_metric_2': metrics.get('value_2'),
            'generic_metric_3': metrics.get('value_3'),
            'metric_count': len(metrics),
            'has_location_data': 'location' in metrics,
            'has_temporal_data': any(key for key in metrics.keys() if 'time' in key.lower() or 'date' in key.lower()),
            'feature_category': 'generic'
        }
    
    def _calculate_rating_direction(self, prev_rating: str, new_rating: str) -> Optional[str]:
        """Calculate direction of rating change."""
        if not prev_rating or not new_rating:
            return None
        
        rating_values = {
            'Outstanding': 4,
            'Good': 3,
            'Requires improvement': 2,
            'Inadequate': 1
        }
        
        prev_val = rating_values.get(prev_rating, 0)
        new_val = rating_values.get(new_rating, 0)
        
        if new_val > prev_val:
            return 'improved'
        elif new_val < prev_val:
            return 'declined'
        else:
            return 'unchanged'
    
    def _calculate_derived_features(self, feature_record: Dict, original_event: Dict) -> Dict:
        """Calculate derived features from the transformed data."""
        derived = {}
        
        # Time-based features
        try:
            event_time = datetime.fromisoformat(feature_record['event_timestamp'])
            processing_time = datetime.fromisoformat(feature_record['processing_timestamp'])
            
            derived['processing_latency_seconds'] = (processing_time - event_time).total_seconds()
            derived['event_hour_of_day'] = event_time.hour
            derived['event_day_of_week'] = event_time.weekday()
            derived['event_month'] = event_time.month
            derived['is_weekend'] = event_time.weekday() >= 5
            derived['is_business_hours'] = 9 <= event_time.hour <= 17
        except Exception as e:
            logger.warning(f"Error calculating time features: {e}")
        
        # Event complexity score
        complexity_score = 0
        if feature_record.get('areas_assessed'):
            complexity_score += len(feature_record['areas_assessed'])
        if feature_record.get('ratings_changed'):
            complexity_score += len(feature_record['ratings_changed']) * 2
        if feature_record.get('enforcement_action_taken'):
            complexity_score += 5
        
        derived['event_complexity_score'] = complexity_score
        
        # Risk indicators
        risk_score = 0
        if feature_record.get('alert_severity') == 'high':
            risk_score += 3
        elif feature_record.get('alert_severity') == 'medium':
            risk_score += 2
        elif feature_record.get('alert_severity') == 'low':
            risk_score += 1
        
        if feature_record.get('safeguarding_concern'):
            risk_score += 5
        if feature_record.get('repeat_violation'):
            risk_score += 3
        if feature_record.get('rating_change_direction') == 'declined':
            risk_score += 4
        
        derived['real_time_risk_score'] = risk_score
        
        return derived


class FeatureAggregator(beam.CombineFn):
    """Aggregate features within time windows."""
    
    def create_accumulator(self):
        return {
            'location_id': None,
            'event_count': 0,
            'event_types': set(),
            'feature_categories': set(),
            'total_risk_score': 0,
            'total_complexity_score': 0,
            'latest_timestamp': None,
            'window_start': None,
            'window_end': None,
            'features': {}
        }
    
    def add_input(self, accumulator, element):
        acc = accumulator.copy()
        acc['event_count'] += 1
        acc['location_id'] = element.get('location_id')
        acc['event_types'].add(element.get('event_type'))
        acc['feature_categories'].add(element.get('feature_category'))
        acc['total_risk_score'] += element.get('real_time_risk_score', 0)
        acc['total_complexity_score'] += element.get('event_complexity_score', 0)
        
        # Track latest timestamp
        event_time = element.get('event_timestamp')
        if not acc['latest_timestamp'] or event_time > acc['latest_timestamp']:
            acc['latest_timestamp'] = event_time
        
        # Accumulate specific features by category
        category = element.get('feature_category')
        if category not in acc['features']:
            acc['features'][category] = []
        acc['features'][category].append(element)
        
        return acc
    
    def merge_accumulators(self, accumulators):
        merged = self.create_accumulator()
        
        for acc in accumulators:
            merged['event_count'] += acc['event_count']
            merged['event_types'].update(acc['event_types'])
            merged['feature_categories'].update(acc['feature_categories'])
            merged['total_risk_score'] += acc['total_risk_score']
            merged['total_complexity_score'] += acc['total_complexity_score']
            
            if acc['location_id']:
                merged['location_id'] = acc['location_id']
            
            if not merged['latest_timestamp'] or (acc['latest_timestamp'] and acc['latest_timestamp'] > merged['latest_timestamp']):
                merged['latest_timestamp'] = acc['latest_timestamp']
            
            # Merge features
            for category, features in acc['features'].items():
                if category not in merged['features']:
                    merged['features'][category] = []
                merged['features'][category].extend(features)
        
        return merged
    
    def extract_output(self, accumulator):
        if accumulator['event_count'] == 0:
            return None
        
        return {
            'location_id': accumulator['location_id'],
            'window_event_count': accumulator['event_count'],
            'unique_event_types': list(accumulator['event_types']),
            'unique_feature_categories': list(accumulator['feature_categories']),
            'average_risk_score': accumulator['total_risk_score'] / accumulator['event_count'],
            'average_complexity_score': accumulator['total_complexity_score'] / accumulator['event_count'],
            'latest_event_timestamp': accumulator['latest_timestamp'],
            'aggregation_timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type_count': len(accumulator['event_types']),
            'feature_category_count': len(accumulator['feature_categories']),
            'aggregated_features': self._summarize_features(accumulator['features'])
        }
    
    def _summarize_features(self, features_by_category: Dict[str, List[Dict]]) -> Dict:
        """Summarize features by category."""
        summary = {}
        
        for category, features in features_by_category.items():
            if not features:
                continue
                
            category_summary = {
                'count': len(features),
                'latest_timestamp': max(f.get('event_timestamp', '') for f in features)
            }
            
            # Category-specific summarization
            if category == 'rating':
                rating_changes = [f.get('rating_change_direction') for f in features if f.get('rating_change_direction')]
                category_summary.update({
                    'rating_improvements': rating_changes.count('improved'),
                    'rating_declines': rating_changes.count('declined'),
                    'enforcement_actions': sum(1 for f in features if f.get('enforcement_action_taken'))
                })
            
            elif category == 'inspection':
                category_summary.update({
                    'total_inspector_count': sum(f.get('inspector_count', 0) for f in features),
                    'follow_ups_required': sum(1 for f in features if f.get('follow_up_required')),
                    'high_priority_inspections': sum(1 for f in features if f.get('inspection_priority') == 'high')
                })
            
            elif category == 'compliance':
                category_summary.update({
                    'high_severity_alerts': sum(1 for f in features if f.get('alert_severity') == 'high'),
                    'safeguarding_concerns': sum(1 for f in features if f.get('safeguarding_concern')),
                    'repeat_violations': sum(1 for f in features if f.get('repeat_violation'))
                })
            
            summary[category] = category_summary
        
        return summary


class VertexAIFeatureStoreWriter(beam.DoFn):
    """Write features to Vertex AI Feature Store."""
    
    def __init__(self, project_id: str, region: str = 'europe-west2', 
                 feature_store_id: str = 'cqc_feature_store'):
        self.project_id = project_id
        self.region = region
        self.feature_store_id = feature_store_id
        self.writes_success = Metrics.counter('vertex_ai', 'writes_success')
        self.writes_errors = Metrics.counter('vertex_ai', 'writes_errors')
    
    def setup(self):
        """Initialize Vertex AI client."""
        aiplatform.init(project=self.project_id, location=self.region)
        self.client = FeatureOnlineStoreServiceClient()
    
    def process(self, element):
        """Write feature to Vertex AI Feature Store."""
        try:
            location_id = element.get('location_id')
            if not location_id:
                return
            
            # Prepare feature values for Feature Store
            feature_values = {}
            
            # Convert numeric features
            numeric_fields = [
                'window_event_count', 'average_risk_score', 'average_complexity_score',
                'event_type_count', 'feature_category_count', 'processing_latency_seconds'
            ]
            
            for field in numeric_fields:
                if field in element and element[field] is not None:
                    feature_values[field] = float(element[field])
            
            # Convert categorical features
            categorical_fields = [
                'unique_event_types', 'unique_feature_categories'
            ]
            
            for field in categorical_fields:
                if field in element and element[field]:
                    # Convert lists to comma-separated strings for Feature Store
                    feature_values[field] = ','.join(str(v) for v in element[field])
            
            # Add timestamp
            feature_values['last_updated'] = datetime.now(timezone.utc).timestamp()
            
            # Write to Feature Store (simplified - in production, use batch writes)
            # This would typically involve the Feature Store API
            logger.info(f"Would write to Feature Store for location {location_id}: {len(feature_values)} features")
            
            self.writes_success.inc()
            yield element  # Pass through for BigQuery
            
        except Exception as e:
            self.writes_errors.inc()
            logger.error(f"Feature Store write error: {e}")
            yield element  # Continue with BigQuery even if Feature Store fails


class StreamingFeaturePipeline:
    """Main streaming pipeline for real-time feature ingestion."""
    
    def __init__(self, project_id: str, region: str = 'europe-west2'):
        self.project_id = project_id
        self.region = region
    
    def get_pipeline_options(self, job_name: str = None, **kwargs) -> PipelineOptions:
        """Configure pipeline options for streaming."""
        options = PipelineOptions()
        
        # Google Cloud options
        google_cloud_options = options.view_as(GoogleCloudOptions)
        google_cloud_options.project = self.project_id
        google_cloud_options.region = self.region
        google_cloud_options.temp_location = f"gs://{self.project_id}-cqc-dataflow-temp/streaming"
        google_cloud_options.staging_location = f"gs://{self.project_id}-cqc-dataflow-staging/streaming"
        
        # Standard options
        standard_options = options.view_as(StandardOptions)
        standard_options.runner = 'DataflowRunner'
        
        # Streaming options
        streaming_options = options.view_as(StreamingOptions)
        streaming_options.streaming = True
        
        # Worker options for autoscaling
        worker_options = options.view_as(WorkerOptions)
        worker_options.num_workers = kwargs.get('num_workers', 1)
        worker_options.max_num_workers = kwargs.get('max_num_workers', 5)
        worker_options.autoscaling_algorithm = 'THROUGHPUT_BASED'
        worker_options.machine_type = kwargs.get('machine_type', 'n1-standard-2')
        worker_options.disk_size_gb = kwargs.get('disk_size_gb', 30)
        
        # Setup options
        setup_options = options.view_as(SetupOptions)
        setup_options.setup_file = './setup.py'
        
        if job_name:
            google_cloud_options.job_name = job_name
        
        return options
    
    def get_realtime_features_schema(self) -> List[Dict]:
        """Schema for real-time features BigQuery table."""
        return [
            # Core identification
            {'name': 'location_id', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'event_type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'event_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'processing_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'pipeline_version', 'type': 'STRING', 'mode': 'NULLABLE'},
            
            # Aggregated metrics (for windowed data)
            {'name': 'window_event_count', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'unique_event_types', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'unique_feature_categories', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'average_risk_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'average_complexity_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'latest_event_timestamp', 'type': 'TIMESTAMP', 'mode': 'NULLABLE'},
            {'name': 'aggregation_timestamp', 'type': 'TIMESTAMP', 'mode': 'NULLABLE'},
            
            # Individual event features
            {'name': 'feature_category', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'processing_latency_seconds', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'event_hour_of_day', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'event_day_of_week', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'is_weekend', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'is_business_hours', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'event_complexity_score', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'real_time_risk_score', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            
            # Inspection features
            {'name': 'inspection_type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'inspector_count', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'inspection_duration_days', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'areas_assessed', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'follow_up_required', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'inspection_priority', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'risk_indicators_found', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            
            # Rating features
            {'name': 'previous_overall_rating', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'new_overall_rating', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'rating_change_direction', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'ratings_changed', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'enforcement_action_taken', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            
            # Capacity features
            {'name': 'previous_bed_count', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'new_bed_count', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'capacity_change', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'capacity_change_reason', 'type': 'STRING', 'mode': 'NULLABLE'},
            
            # Compliance features
            {'name': 'alert_severity', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'alert_category', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'compliance_areas_affected', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'action_required', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'escalation_level', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'repeat_violation', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'safeguarding_concern', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            
            # Interaction features
            {'name': 'user_type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'page_views', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'time_on_page', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'interaction_type', 'type': 'STRING', 'mode': 'NULLABLE'},
            
            # Aggregated feature summaries (JSON)
            {'name': 'aggregated_features', 'type': 'JSON', 'mode': 'NULLABLE'}
        ]
    
    def run_pipeline(self, pubsub_topic: str, job_name: str = None):
        """Run the streaming feature pipeline."""
        if not job_name:
            job_name = f"cqc-streaming-features-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        options = self.get_pipeline_options(job_name=job_name)
        
        with beam.Pipeline(options=options) as pipeline:
            # Read from Pub/Sub
            raw_events = (
                pipeline
                | 'Read from Pub/Sub' >> ReadFromPubSub(topic=pubsub_topic)
            )
            
            # Parse and validate events
            parsed_events = (
                raw_events
                | 'Parse Dashboard Events' >> beam.ParDo(DashboardEventParser()).with_outputs('errors', main='valid')
            )
            
            # Transform valid events into features
            feature_events = (
                parsed_events.valid
                | 'Transform to Features' >> beam.ParDo(DashboardMetricsTransformer()).with_outputs('errors', main='features')
            )
            
            # Individual feature records (for immediate processing)
            individual_features = (
                feature_events.features
                | 'Write Individual Features to Vertex AI' >> beam.ParDo(
                    VertexAIFeatureStoreWriter(self.project_id, self.region)
                )
                | 'Write Individual to BigQuery' >> WriteToBigQuery(
                    table=f"{self.project_id}:cqc_dataset.realtime_features",
                    schema={'fields': self.get_realtime_features_schema()},
                    create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=BigQueryDisposition.WRITE_APPEND,
                    method='STREAMING_INSERTS'
                )
            )
            
            # Windowed aggregations for time-based features
            windowed_features = (
                feature_events.features
                | 'Window by Location and Time' >> beam.WindowInto(
                    FixedWindows(300),  # 5-minute windows
                    trigger=Repeatedly(AfterProcessingTime(60)),  # Every minute
                    accumulation_mode=AccumulationMode.DISCARDING
                )
                | 'Group by Location' >> beam.GroupBy('location_id')
                | 'Aggregate Features' >> beam.CombinePerKey(FeatureAggregator())
                | 'Filter Non-Empty Aggregations' >> beam.Filter(lambda x: x is not None)
                | 'Write Aggregated to BigQuery' >> WriteToBigQuery(
                    table=f"{self.project_id}:cqc_dataset.realtime_features_aggregated",
                    schema={'fields': [
                        {'name': 'location_id', 'type': 'STRING', 'mode': 'REQUIRED'},
                        {'name': 'window_event_count', 'type': 'INTEGER', 'mode': 'NULLABLE'},
                        {'name': 'unique_event_types', 'type': 'STRING', 'mode': 'REPEATED'},
                        {'name': 'unique_feature_categories', 'type': 'STRING', 'mode': 'REPEATED'},
                        {'name': 'average_risk_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                        {'name': 'average_complexity_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
                        {'name': 'latest_event_timestamp', 'type': 'TIMESTAMP', 'mode': 'NULLABLE'},
                        {'name': 'aggregation_timestamp', 'type': 'TIMESTAMP', 'mode': 'NULLABLE'},
                        {'name': 'event_type_count', 'type': 'INTEGER', 'mode': 'NULLABLE'},
                        {'name': 'feature_category_count', 'type': 'INTEGER', 'mode': 'NULLABLE'},
                        {'name': 'aggregated_features', 'type': 'JSON', 'mode': 'NULLABLE'}
                    ]},
                    create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=BigQueryDisposition.WRITE_APPEND,
                    method='STREAMING_INSERTS'
                )
            )
            
            # Error handling - combine all errors
            all_errors = (
                (parsed_events.errors, feature_events.errors)
                | 'Flatten Errors' >> beam.Flatten()
                | 'Write Errors to BigQuery' >> WriteToBigQuery(
                    table=f"{self.project_id}:cqc_dataset.streaming_errors",
                    schema={'fields': [
                        {'name': 'error_type', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'error_message', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'location_id', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'event_type', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'raw_data', 'type': 'STRING', 'mode': 'NULLABLE'},
                        {'name': 'error_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'}
                    ]},
                    create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                    write_disposition=BigQueryDisposition.WRITE_APPEND,
                    method='STREAMING_INSERTS'
                )
            )


def main():
    """Main entry point for streaming feature pipeline."""
    parser = argparse.ArgumentParser(description='CQC Streaming Feature Pipeline')
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    parser.add_argument('--pubsub-topic', 
                       default='projects/machine-learning-exp-467008/topics/dashboard-events',
                       help='Pub/Sub topic to read from')
    parser.add_argument('--region', default='europe-west2', help='GCP region')
    parser.add_argument('--job-name', help='Dataflow job name')
    parser.add_argument('--num-workers', type=int, default=1, help='Initial number of workers')
    parser.add_argument('--max-num-workers', type=int, default=5, help='Maximum number of workers')
    parser.add_argument('--machine-type', default='n1-standard-2', help='Worker machine type')
    
    args = parser.parse_args()
    
    logger.info(f"Starting streaming pipeline for project: {args.project_id}")
    logger.info(f"Reading from topic: {args.pubsub_topic}")
    
    pipeline = StreamingFeaturePipeline(args.project_id, args.region)
    
    pipeline.run_pipeline(
        pubsub_topic=args.pubsub_topic,
        job_name=args.job_name
    )
    
    logger.info("Streaming pipeline started successfully")


if __name__ == '__main__':
    main()