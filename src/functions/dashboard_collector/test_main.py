"""
Unit tests for the dashboard metrics collector Cloud Function.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from main import collect_dashboard_metrics, DashboardMetricsCollector


@pytest.fixture
def mock_request():
    """Create a mock Flask request object."""
    request = Mock()
    request.get_json = Mock(return_value={
        "provider_id": "test-provider-123",
        "location_id": "test-location-456",
        "dashboard_type": "incidents",
        "collect_all": False
    })
    return request


@pytest.fixture
def sample_incident_metrics():
    """Sample incident metrics data for testing."""
    return [{
        "metric_type": "incidents",
        "provider_id": "test-provider-123",
        "location_id": "test-location-456",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_metrics": {
            "total_incidents": 3,
            "open_incidents": 1,
            "avg_resolution_time_hours": 8.5,
            "severity_distribution": {
                "high": 1,
                "medium": 1,
                "low": 1
            }
        },
        "derived_metrics": {
            "incident_rate_per_bed": 0.06,
            "high_severity_percentage": 33.33,
            "resolution_efficiency_score": 82.3
        }
    }]


@pytest.fixture
def sample_staffing_metrics():
    """Sample staffing metrics data for testing."""
    return [{
        "metric_type": "staffing",
        "provider_id": "test-provider-123",
        "location_id": "test-location-456",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "raw_metrics": {
            "total_staff": 20,
            "registered_nurses": 6,
            "care_assistants": 14,
            "vacancy_count": 2,
            "turnover_rate": 15.5,
            "absence_rate": 8.2,
            "overtime_hours": 45
        },
        "derived_metrics": {
            "staff_to_bed_ratio": 0.4,
            "nurse_to_bed_ratio": 0.12,
            "staffing_adequacy_score": 85.5,
            "staff_stability_score": 78.2
        }
    }]


class TestCollectDashboardMetrics:
    """Test the main Cloud Function entry point."""
    
    def test_successful_incident_collection(self, mock_request):
        """Test successful collection of incident metrics."""
        with patch('main.DashboardMetricsCollector') as mock_collector_class:
            mock_collector = Mock()
            mock_collector_class.return_value = mock_collector
            mock_collector.collect_specific_metrics.return_value = {
                "metrics_count": 1,
                "events_published": 1,
                "feature_updates": 1,
                "timestamp": "2024-01-15T10:30:00Z"
            }
            
            response = collect_dashboard_metrics(mock_request)
            
            assert response["status"] == "success"
            assert response["metrics_collected"] == 1
            assert response["events_published"] == 1
            assert response["feature_store_updates"] == 1
            mock_collector.collect_specific_metrics.assert_called_once_with(
                "test-provider-123", "test-location-456", "incidents"
            )
    
    def test_collect_all_metrics(self):
        """Test collecting all metric types."""
        request = Mock()
        request.get_json = Mock(return_value={
            "provider_id": "test-provider-123",
            "collect_all": True
        })
        
        with patch('main.DashboardMetricsCollector') as mock_collector_class:
            mock_collector = Mock()
            mock_collector_class.return_value = mock_collector
            mock_collector.collect_all_metrics.return_value = {
                "metrics_count": 3,
                "events_published": 3,
                "feature_updates": 2,
                "timestamp": "2024-01-15T10:30:00Z"
            }
            
            response = collect_dashboard_metrics(request)
            
            assert response["status"] == "success"
            assert response["metrics_collected"] == 3
            mock_collector.collect_all_metrics.assert_called_once_with("test-provider-123", None)
    
    def test_missing_json_body(self):
        """Test handling of missing JSON body."""
        request = Mock()
        request.get_json = Mock(return_value=None)
        
        response, status_code = collect_dashboard_metrics(request)
        
        assert status_code == 400
        assert "No JSON body provided" in response["error"]
    
    def test_missing_provider_and_location_id(self):
        """Test handling of missing both provider and location ID."""
        request = Mock()
        request.get_json = Mock(return_value={
            "dashboard_type": "incidents"
        })
        
        response, status_code = collect_dashboard_metrics(request)
        
        assert status_code == 400
        assert "Either provider_id or location_id must be provided" in response["error"]
    
    def test_exception_handling(self, mock_request):
        """Test exception handling in the main function."""
        with patch('main.DashboardMetricsCollector') as mock_collector_class:
            mock_collector_class.side_effect = Exception("Test error")
            
            response, status_code = collect_dashboard_metrics(mock_request)
            
            assert status_code == 500
            assert "Test error" in response["error"]


class TestDashboardMetricsCollector:
    """Test the DashboardMetricsCollector class."""
    
    @patch('main.pubsub_v1.PublisherClient')
    @patch('main.bigquery.Client')
    @patch('main.aiplatform.init')
    def setUp(self, mock_vertex_init, mock_bq_client, mock_pubsub_client):
        """Set up test fixtures."""
        self.collector = DashboardMetricsCollector()
        self.collector.bigquery_client = Mock()
        self.collector.publisher = Mock()
        self.collector.topic_path = "projects/test-project/topics/dashboard-events"
    
    def test_init(self):
        """Test collector initialization."""
        with patch('main.pubsub_v1.PublisherClient') as mock_pubsub, \
             patch('main.bigquery.Client') as mock_bq, \
             patch('main.aiplatform.init') as mock_vertex:
            
            collector = DashboardMetricsCollector()
            
            assert collector.project_id == "machine-learning-exp-467008"
            assert collector.region == "europe-west2"
            mock_pubsub.assert_called_once()
            mock_bq.assert_called_once()
            mock_vertex.assert_called_once()
    
    def test_collect_incident_metrics_with_data(self, sample_incident_metrics):
        """Test collecting incident metrics when data exists."""
        self.setUp()
        
        # Mock BigQuery response
        mock_row = Mock()
        mock_row.provider_id = "test-provider-123"
        mock_row.location_id = "test-location-456"
        mock_row.incident_count = 3
        mock_row.high_severity_count = 1
        mock_row.medium_severity_count = 1
        mock_row.low_severity_count = 1
        mock_row.avg_resolution_time = 8.5
        mock_row.open_incidents = 1
        
        mock_query_job = Mock()
        mock_query_job.result.return_value = [mock_row]
        self.collector.bigquery_client.query.return_value = mock_query_job
        
        # Mock bed count query
        with patch.object(self.collector, '_get_bed_count', return_value=50):
            metrics = self.collector._collect_incident_metrics("test-provider-123", "test-location-456")
        
        assert len(metrics) == 1
        metric = metrics[0]
        assert metric["metric_type"] == "incidents"
        assert metric["provider_id"] == "test-provider-123"
        assert metric["location_id"] == "test-location-456"
        assert metric["raw_metrics"]["total_incidents"] == 3
        assert metric["derived_metrics"]["incident_rate_per_bed"] == 0.06  # 3/50
    
    def test_collect_incident_metrics_with_error(self):
        """Test collecting incident metrics when BigQuery fails."""
        self.setUp()
        
        # Mock BigQuery to raise an exception
        self.collector.bigquery_client.query.side_effect = Exception("BigQuery error")
        
        metrics = self.collector._collect_incident_metrics("test-provider-123", "test-location-456")
        
        # Should return synthetic data when error occurs
        assert len(metrics) == 1
        assert metrics[0]["metric_type"] == "incidents"
        assert "synthetic" in metrics[0]["provider_id"]
    
    def test_collect_staffing_metrics(self, sample_staffing_metrics):
        """Test collecting staffing metrics."""
        self.setUp()
        
        # Mock BigQuery response
        mock_row = Mock()
        mock_row.provider_id = "test-provider-123"
        mock_row.location_id = "test-location-456"
        mock_row.total_staff_count = 20
        mock_row.registered_nurses_count = 6
        mock_row.care_assistants_count = 14
        mock_row.vacancy_count = 2
        mock_row.turnover_rate_percent = 15.5
        mock_row.absence_rate_percent = 8.2
        mock_row.overtime_hours_total = 45
        
        mock_query_job = Mock()
        mock_query_job.result.return_value = [mock_row]
        self.collector.bigquery_client.query.return_value = mock_query_job
        
        with patch.object(self.collector, '_get_bed_count', return_value=50):
            metrics = self.collector._collect_staffing_metrics("test-provider-123", "test-location-456")
        
        assert len(metrics) == 1
        metric = metrics[0]
        assert metric["metric_type"] == "staffing"
        assert metric["raw_metrics"]["total_staff"] == 20
        assert metric["derived_metrics"]["staff_to_bed_ratio"] == 0.4  # 20/50
    
    def test_collect_care_quality_metrics(self):
        """Test collecting care quality metrics."""
        self.setUp()
        
        # Mock BigQuery response
        mock_row = Mock()
        mock_row.provider_id = "test-provider-123"
        mock_row.location_id = "test-location-456"
        mock_row.medication_errors_count = 1
        mock_row.falls_count = 2
        mock_row.pressure_ulcers_count = 0
        mock_row.resident_satisfaction_score = 8.5
        mock_row.family_satisfaction_score = 8.2
        mock_row.complaint_count = 1
        mock_row.compliment_count = 5
        mock_row.care_plan_compliance_percent = 92.5
        
        mock_query_job = Mock()
        mock_query_job.result.return_value = [mock_row]
        self.collector.bigquery_client.query.return_value = mock_query_job
        
        metrics = self.collector._collect_care_quality_metrics("test-provider-123", "test-location-456")
        
        assert len(metrics) == 1
        metric = metrics[0]
        assert metric["metric_type"] == "care_quality"
        assert metric["raw_metrics"]["medication_errors"] == 1
        assert metric["derived_metrics"]["safety_incident_rate"] == 3  # 1+2+0
    
    def test_publish_metrics_to_pubsub(self, sample_incident_metrics):
        """Test publishing metrics to Pub/Sub."""
        self.setUp()
        
        # Mock successful publish
        future = Mock()
        future.result.return_value = None
        self.collector.publisher.publish.return_value = future
        
        published_count = self.collector._publish_metrics_to_pubsub(sample_incident_metrics, "incidents")
        
        assert published_count == 1
        self.collector.publisher.publish.assert_called_once()
    
    def test_publish_metrics_to_pubsub_with_error(self, sample_incident_metrics):
        """Test publishing metrics when Pub/Sub fails."""
        self.setUp()
        
        # Mock publish failure
        future = Mock()
        future.result.side_effect = Exception("Pub/Sub error")
        self.collector.publisher.publish.return_value = future
        
        published_count = self.collector._publish_metrics_to_pubsub(sample_incident_metrics, "incidents")
        
        assert published_count == 0  # Should handle error gracefully
    
    def test_store_metrics_in_bigquery(self, sample_incident_metrics):
        """Test storing metrics in BigQuery."""
        self.setUp()
        
        # Mock BigQuery table and dataset
        mock_table = Mock()
        mock_dataset = Mock()
        mock_dataset.table.return_value = mock_table
        self.collector.bigquery_client.dataset.return_value = mock_dataset
        self.collector.bigquery_client.insert_rows_json.return_value = []  # No errors
        
        self.collector._store_metrics_in_bigquery(sample_incident_metrics)
        
        self.collector.bigquery_client.insert_rows_json.assert_called_once()
        # Verify the data structure
        call_args = self.collector.bigquery_client.insert_rows_json.call_args
        rows = call_args[0][1]  # Second argument is the rows
        assert len(rows) == 1
        assert rows[0]["metric_type"] == "incidents"
        assert "raw_metrics" in rows[0]
        assert "derived_metrics" in rows[0]
    
    def test_build_where_clause(self):
        """Test building WHERE clauses for BigQuery."""
        self.setUp()
        
        # Test with both provider and location
        clause = self.collector._build_where_clause("provider-123", "location-456")
        assert "AND provider_id = 'provider-123'" in clause
        assert "AND location_id = 'location-456'" in clause
        
        # Test with only provider
        clause = self.collector._build_where_clause("provider-123", None)
        assert "AND provider_id = 'provider-123'" in clause
        assert "location_id" not in clause
        
        # Test with neither
        clause = self.collector._build_where_clause(None, None)
        assert clause == ""
    
    def test_get_bed_count(self):
        """Test getting bed count for a location."""
        self.setUp()
        
        # Mock BigQuery response
        mock_row = Mock()
        mock_row.numberOfBeds = 45
        
        mock_query_job = Mock()
        mock_query_job.result.return_value = [mock_row]
        self.collector.bigquery_client.query.return_value = mock_query_job
        
        bed_count = self.collector._get_bed_count("provider-123", "location-456")
        
        assert bed_count == 45
    
    def test_get_bed_count_with_error(self):
        """Test getting bed count when query fails."""
        self.setUp()
        
        # Mock BigQuery to raise an exception
        self.collector.bigquery_client.query.side_effect = Exception("Query error")
        
        bed_count = self.collector._get_bed_count("provider-123", "location-456")
        
        assert bed_count == 50  # Default value
    
    def test_calculate_resolution_score(self):
        """Test resolution efficiency score calculation."""
        self.setUp()
        
        # Test immediate resolution (should be close to 100)
        score = self.collector._calculate_resolution_score(0.5)
        assert score > 95
        
        # Test 24 hour resolution (should be around 50)
        score = self.collector._calculate_resolution_score(24.0)
        assert 45 <= score <= 55
        
        # Test very slow resolution (should be close to 0)
        score = self.collector._calculate_resolution_score(72.0)
        assert score < 10
        
        # Test None input
        score = self.collector._calculate_resolution_score(None)
        assert score == 50.0
    
    def test_calculate_staffing_score(self):
        """Test staffing adequacy score calculation."""
        self.setUp()
        
        # Test optimal staffing (1:3 staff ratio, 1:10 nurse ratio)
        score = self.collector._calculate_staffing_score(0.33, 0.10)
        assert score >= 90
        
        # Test understaffed
        score = self.collector._calculate_staffing_score(0.15, 0.05)
        assert score < 60
        
        # Test overstaffed (should cap at 100)
        score = self.collector._calculate_staffing_score(1.0, 0.50)
        assert score == 100
    
    def test_calculate_care_quality_score(self):
        """Test care quality score calculation."""
        self.setUp()
        
        # Test high quality (no incidents, high satisfaction, high compliance)
        score = self.collector._calculate_care_quality_score(0, 0, 0, 9.0, 9.0, 95.0)
        assert score > 90
        
        # Test low quality (many incidents, low satisfaction, low compliance)
        score = self.collector._calculate_care_quality_score(5, 3, 2, 6.0, 6.0, 70.0)
        assert score < 60
    
    def test_collect_all_metrics(self):
        """Test collecting all metric types."""
        self.setUp()
        
        with patch.object(self.collector, '_collect_metrics_by_type') as mock_collect, \
             patch.object(self.collector, '_publish_metrics_to_pubsub', return_value=1) as mock_publish, \
             patch.object(self.collector, '_update_feature_store', return_value=1) as mock_feature, \
             patch.object(self.collector, '_store_metrics_in_bigquery') as mock_store:
            
            # Mock return different metrics for each type
            mock_collect.side_effect = [
                [{"metric_type": "incidents"}],  # incidents
                [{"metric_type": "staffing"}],   # staffing
                [{"metric_type": "care_quality"}]  # care_quality
            ]
            
            result = self.collector.collect_all_metrics("provider-123", "location-456")
            
            # Should call collect_metrics_by_type 3 times (for each metric type)
            assert mock_collect.call_count == 3
            
            # Should publish all metric types
            assert mock_publish.call_count == 3
            
            # Should update feature store for incidents and care_quality (2 times)
            assert mock_feature.call_count == 2
            
            # Should store all metrics
            mock_store.assert_called_once()
            
            assert result["metrics_count"] == 3
            assert result["events_published"] == 3
            assert result["feature_updates"] == 2


class TestSyntheticDataGeneration:
    """Test synthetic data generation for testing purposes."""
    
    @patch('main.pubsub_v1.PublisherClient')
    @patch('main.bigquery.Client')
    @patch('main.aiplatform.init')
    def setUp(self, mock_vertex_init, mock_bq_client, mock_pubsub_client):
        """Set up test fixtures."""
        self.collector = DashboardMetricsCollector()
    
    def test_generate_synthetic_incident_metrics(self):
        """Test generation of synthetic incident metrics."""
        self.setUp()
        
        metric = self.collector._generate_synthetic_incident_metrics("test-provider", "test-location")
        
        assert metric["metric_type"] == "incidents"
        assert metric["provider_id"] == "test-provider"
        assert metric["location_id"] == "test-location"
        assert "raw_metrics" in metric
        assert "derived_metrics" in metric
        assert isinstance(metric["raw_metrics"]["total_incidents"], int)
        assert metric["raw_metrics"]["total_incidents"] >= 0
    
    def test_generate_synthetic_staffing_metrics(self):
        """Test generation of synthetic staffing metrics."""
        self.setUp()
        
        metric = self.collector._generate_synthetic_staffing_metrics("test-provider", "test-location")
        
        assert metric["metric_type"] == "staffing"
        assert metric["provider_id"] == "test-provider"
        assert metric["location_id"] == "test-location"
        assert metric["raw_metrics"]["total_staff"] >= metric["raw_metrics"]["registered_nurses"]
    
    def test_generate_synthetic_care_quality_metrics(self):
        """Test generation of synthetic care quality metrics."""
        self.setUp()
        
        metric = self.collector._generate_synthetic_care_quality_metrics("test-provider", "test-location")
        
        assert metric["metric_type"] == "care_quality"
        assert metric["provider_id"] == "test-provider"
        assert metric["location_id"] == "test-location"
        assert 0 <= metric["raw_metrics"]["resident_satisfaction"] <= 10
        assert 0 <= metric["raw_metrics"]["family_satisfaction"] <= 10


if __name__ == "__main__":
    pytest.main([__file__])