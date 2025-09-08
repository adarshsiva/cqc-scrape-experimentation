"""
Simple validation tests for the CQC Streaming Feature Pipeline.
These tests validate pipeline construction and basic functionality.
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

# Import pipeline components
from streaming_feature_pipeline import (
    DashboardEventParser,
    DashboardMetricsTransformer,
    FeatureAggregator,
    StreamingFeaturePipeline
)


class TestDashboardEventParser:
    """Test the dashboard event parser."""
    
    def test_valid_event_parsing(self):
        """Test parsing of valid dashboard events."""
        parser = DashboardEventParser()
        
        # Create a valid test event
        test_event = {
            'location_id': 'test-123',
            'event_type': 'dashboard_interaction',
            'timestamp': '2024-01-01T12:00:00Z',
            'metrics': {
                'user_type': 'public',
                'page_views': 1
            }
        }
        
        event_json = json.dumps(test_event)
        
        # Process the event
        results = list(parser.process(event_json))
        
        assert len(results) == 1
        result = results[0]
        assert result['location_id'] == 'test-123'
        assert result['event_type'] == 'dashboard_interaction'
        assert 'processing_timestamp' in result
        assert 'pipeline_version' in result
    
    def test_invalid_json_parsing(self):
        """Test handling of invalid JSON."""
        parser = DashboardEventParser()
        
        # Invalid JSON
        invalid_json = '{"location_id": "test", invalid}'
        
        # Process the invalid event
        results = list(parser.process(invalid_json))
        
        # Should produce an error output
        assert len(results) == 1
        # This would be tagged as an error in the actual pipeline
    
    def test_missing_required_fields(self):
        """Test handling of events missing required fields."""
        parser = DashboardEventParser()
        
        # Event missing required fields
        incomplete_event = {
            'location_id': 'test-123',
            # missing event_type, timestamp, metrics
        }
        
        event_json = json.dumps(incomplete_event)
        results = list(parser.process(event_json))
        
        # Should not produce valid output due to validation
        assert len(results) == 0


class TestDashboardMetricsTransformer:
    """Test the metrics transformer."""
    
    def test_inspection_metrics_transform(self):
        """Test transformation of inspection metrics."""
        transformer = DashboardMetricsTransformer()
        
        test_event = {
            'location_id': 'test-123',
            'event_type': 'inspection_update',
            'event_timestamp_parsed': '2024-01-01T12:00:00+00:00',
            'processing_timestamp': '2024-01-01T12:01:00+00:00',
            'pipeline_version': 'test',
            'metrics': {
                'inspection_type': 'comprehensive',
                'inspector_count': 2,
                'duration_days': 3,
                'areas_assessed': ['safe', 'effective'],
                'follow_up_required': True,
                'priority': 'high'
            }
        }
        
        results = list(transformer.process(test_event))
        
        assert len(results) == 1
        result = results[0]
        assert result['location_id'] == 'test-123'
        assert result['feature_category'] == 'inspection'
        assert result['inspection_type'] == 'comprehensive'
        assert result['inspector_count'] == 2
        assert result['follow_up_required'] == True
        assert 'real_time_risk_score' in result
        assert 'event_complexity_score' in result
    
    def test_rating_change_transform(self):
        """Test transformation of rating change metrics."""
        transformer = DashboardMetricsTransformer()
        
        test_event = {
            'location_id': 'test-456',
            'event_type': 'rating_change', 
            'event_timestamp_parsed': '2024-01-01T12:00:00+00:00',
            'processing_timestamp': '2024-01-01T12:01:00+00:00',
            'pipeline_version': 'test',
            'metrics': {
                'previous_rating': 'Good',
                'new_rating': 'Requires improvement',
                'specific_ratings_changed': ['safe', 'well_led'],
                'enforcement_action': True
            }
        }
        
        results = list(transformer.process(test_event))
        
        assert len(results) == 1
        result = results[0]
        assert result['feature_category'] == 'rating'
        assert result['previous_overall_rating'] == 'Good'
        assert result['new_overall_rating'] == 'Requires improvement'
        assert result['rating_change_direction'] == 'declined'
        assert result['enforcement_action_taken'] == True
    
    def test_compliance_alert_transform(self):
        """Test transformation of compliance alert metrics."""
        transformer = DashboardMetricsTransformer()
        
        test_event = {
            'location_id': 'test-789',
            'event_type': 'compliance_alert',
            'event_timestamp_parsed': '2024-01-01T12:00:00+00:00', 
            'processing_timestamp': '2024-01-01T12:01:00+00:00',
            'pipeline_version': 'test',
            'metrics': {
                'severity': 'high',
                'category': 'safeguarding',
                'affected_areas': ['care_planning'],
                'safeguarding_concern': True,
                'repeat_violation': True,
                'escalation_level': 2
            }
        }
        
        results = list(transformer.process(test_event))
        
        assert len(results) == 1
        result = results[0]
        assert result['feature_category'] == 'compliance'
        assert result['alert_severity'] == 'high'
        assert result['safeguarding_concern'] == True
        assert result['repeat_violation'] == True
        assert result['real_time_risk_score'] > 5  # Should be high due to severity and safeguarding


class TestFeatureAggregator:
    """Test the feature aggregator."""
    
    def test_aggregation_creation(self):
        """Test creating aggregation accumulator."""
        aggregator = FeatureAggregator()
        
        acc = aggregator.create_accumulator()
        
        assert acc['event_count'] == 0
        assert acc['location_id'] is None
        assert len(acc['event_types']) == 0
        assert acc['total_risk_score'] == 0
    
    def test_adding_inputs(self):
        """Test adding inputs to aggregator."""
        aggregator = FeatureAggregator()
        
        acc = aggregator.create_accumulator()
        
        test_element = {
            'location_id': 'test-123',
            'event_type': 'inspection_update',
            'feature_category': 'inspection',
            'real_time_risk_score': 5,
            'event_complexity_score': 3,
            'event_timestamp': '2024-01-01T12:00:00+00:00'
        }
        
        acc = aggregator.add_input(acc, test_element)
        
        assert acc['event_count'] == 1
        assert acc['location_id'] == 'test-123'
        assert 'inspection_update' in acc['event_types']
        assert 'inspection' in acc['feature_categories']
        assert acc['total_risk_score'] == 5
        assert acc['total_complexity_score'] == 3
    
    def test_extract_output(self):
        """Test extracting final output from aggregator."""
        aggregator = FeatureAggregator()
        
        acc = aggregator.create_accumulator()
        
        # Add some test data
        test_element = {
            'location_id': 'test-123',
            'event_type': 'compliance_alert',
            'feature_category': 'compliance', 
            'real_time_risk_score': 8,
            'event_complexity_score': 2,
            'event_timestamp': '2024-01-01T12:00:00+00:00'
        }
        
        acc = aggregator.add_input(acc, test_element)
        
        output = aggregator.extract_output(acc)
        
        assert output['location_id'] == 'test-123'
        assert output['window_event_count'] == 1
        assert 'compliance_alert' in output['unique_event_types']
        assert output['average_risk_score'] == 8.0
        assert output['average_complexity_score'] == 2.0
        assert 'aggregation_timestamp' in output


class TestStreamingFeaturePipeline:
    """Test the main streaming pipeline class."""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = StreamingFeaturePipeline(
            project_id='test-project',
            region='europe-west2'
        )
        
        assert pipeline.project_id == 'test-project'
        assert pipeline.region == 'europe-west2'
    
    def test_pipeline_options_creation(self):
        """Test creation of pipeline options."""
        pipeline = StreamingFeaturePipeline(
            project_id='test-project',
            region='europe-west2'
        )
        
        options = pipeline.get_pipeline_options(job_name='test-job')
        
        # Verify some key options are set
        google_cloud_options = options.view_as(pipeline.get_pipeline_options().__class__.GoogleCloudOptions)
        assert hasattr(options, '_all_options')
    
    def test_schema_definition(self):
        """Test BigQuery schema definition."""
        pipeline = StreamingFeaturePipeline(
            project_id='test-project'
        )
        
        schema = pipeline.get_realtime_features_schema()
        
        # Check required fields exist
        field_names = [field['name'] for field in schema]
        assert 'location_id' in field_names
        assert 'event_type' in field_names
        assert 'event_timestamp' in field_names
        assert 'processing_timestamp' in field_names
        
        # Check location_id is required
        location_field = next(f for f in schema if f['name'] == 'location_id')
        assert location_field['mode'] == 'REQUIRED'


def run_tests():
    """Run all tests manually (for environments without pytest)."""
    print("Running CQC Streaming Pipeline Tests...")
    
    # Test event parser
    print("\n1. Testing DashboardEventParser...")
    parser_test = TestDashboardEventParser()
    try:
        parser_test.test_valid_event_parsing()
        parser_test.test_missing_required_fields()
        print("   ✅ DashboardEventParser tests passed")
    except Exception as e:
        print(f"   ❌ DashboardEventParser tests failed: {e}")
    
    # Test metrics transformer
    print("\n2. Testing DashboardMetricsTransformer...")
    transformer_test = TestDashboardMetricsTransformer()
    try:
        transformer_test.test_inspection_metrics_transform()
        transformer_test.test_rating_change_transform()
        transformer_test.test_compliance_alert_transform()
        print("   ✅ DashboardMetricsTransformer tests passed")
    except Exception as e:
        print(f"   ❌ DashboardMetricsTransformer tests failed: {e}")
    
    # Test feature aggregator
    print("\n3. Testing FeatureAggregator...")
    aggregator_test = TestFeatureAggregator()
    try:
        aggregator_test.test_aggregation_creation()
        aggregator_test.test_adding_inputs()
        aggregator_test.test_extract_output()
        print("   ✅ FeatureAggregator tests passed")
    except Exception as e:
        print(f"   ❌ FeatureAggregator tests failed: {e}")
    
    # Test pipeline
    print("\n4. Testing StreamingFeaturePipeline...")
    pipeline_test = TestStreamingFeaturePipeline()
    try:
        pipeline_test.test_pipeline_initialization()
        pipeline_test.test_schema_definition()
        print("   ✅ StreamingFeaturePipeline tests passed")
    except Exception as e:
        print(f"   ❌ StreamingFeaturePipeline tests failed: {e}")
    
    print("\n🎉 All tests completed!")


if __name__ == '__main__':
    run_tests()