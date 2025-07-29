"""
Test script for CQC notification service

This script provides unit tests and integration tests for the notification service.
"""

import os
import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
from notification_service import (
    NotificationService,
    NotificationChannel,
    NotificationConfig,
    AlertMessage
)


class TestNotificationService(unittest.TestCase):
    """Test cases for NotificationService"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.project_id = "test-project"
        self.dataset_id = "test_dataset"
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'ENABLE_EMAIL_NOTIFICATIONS': 'true',
            'SMTP_HOST': 'smtp.test.com',
            'SMTP_PORT': '587',
            'SENDER_EMAIL': 'test@example.com',
            'ALERT_RECIPIENTS': 'recipient1@test.com,recipient2@test.com',
            'ENABLE_WEBHOOK_NOTIFICATIONS': 'true',
            'WEBHOOK_URL': 'https://test-webhook.com/alerts'
        })
        self.env_patcher.start()
        
        # Mock BigQuery client
        self.mock_bq_client = Mock()
        self.mock_secret_client = Mock()
        
        with patch('google.cloud.bigquery.Client', return_value=self.mock_bq_client):
            with patch('google.cloud.secretmanager.SecretManagerServiceClient', 
                       return_value=self.mock_secret_client):
                self.service = NotificationService(self.project_id, self.dataset_id)
    
    def tearDown(self):
        """Clean up after tests"""
        self.env_patcher.stop()
    
    def test_initialization(self):
        """Test service initialization"""
        self.assertEqual(self.service.project_id, self.project_id)
        self.assertEqual(self.service.dataset_id, self.dataset_id)
        self.assertIsNotNone(self.service.notification_configs)
        self.assertIsNotNone(self.service.templates)
    
    def test_load_notification_configs(self):
        """Test loading notification configurations"""
        configs = self.service._load_notification_configs()
        
        # Check email config
        email_config = next(c for c in configs if c.channel == NotificationChannel.EMAIL)
        self.assertTrue(email_config.enabled)
        self.assertEqual(email_config.config['smtp_host'], 'smtp.test.com')
        self.assertEqual(email_config.config['recipients'], 
                         ['recipient1@test.com', 'recipient2@test.com'])
        
        # Check webhook config
        webhook_config = next(c for c in configs if c.channel == NotificationChannel.WEBHOOK)
        self.assertTrue(webhook_config.enabled)
        self.assertEqual(webhook_config.config['webhook_url'], 
                         'https://test-webhook.com/alerts')
    
    def test_prepare_alert_message(self):
        """Test alert message preparation"""
        # Mock query result
        row = pd.Series({
            'locationId': 'LOC123',
            'locationName': 'Test Care Home',
            'riskScore': 85.5,
            'riskLevel': 'HIGH',
            'topRiskFactors': json.dumps([
                {'factor': 'Staffing', 'impact': 25.0},
                {'factor': 'Compliance', 'impact': 20.0}
            ]),
            'recommendations': json.dumps([
                'Increase staffing levels',
                'Review compliance procedures'
            ]),
            'assessmentDate': '2024-01-15',
            'providerName': 'Test Provider Ltd',
            'postalAddressLine1': '123 Test Street',
            'postalAddressTownCity': 'Test City',
            'postalCode': 'TS1 2AB'
        })
        
        alert = self.service._prepare_alert_message(row)
        
        self.assertEqual(alert.location_id, 'LOC123')
        self.assertEqual(alert.location_name, 'Test Care Home')
        self.assertEqual(alert.risk_score, 85.5)
        self.assertEqual(alert.risk_level, 'HIGH')
        self.assertEqual(len(alert.risk_factors), 2)
        self.assertEqual(alert.risk_factors[0]['factor'], 'Staffing')
        self.assertIn('Test Street', alert.address)
    
    def test_query_high_risk_locations(self):
        """Test querying high-risk locations"""
        # Mock query result
        mock_df = pd.DataFrame([
            {
                'locationId': 'LOC123',
                'riskScore': 85.5,
                'riskLevel': 'HIGH'
            },
            {
                'locationId': 'LOC456',
                'riskScore': 72.3,
                'riskLevel': 'HIGH'
            }
        ])
        
        mock_query_job = Mock()
        mock_query_job.to_dataframe.return_value = mock_df
        self.mock_bq_client.query.return_value = mock_query_job
        
        # Test query
        result = self.service.query_high_risk_locations(risk_threshold=70.0)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]['locationId'], 'LOC123')
        self.mock_bq_client.query.assert_called_once()
    
    @patch('smtplib.SMTP')
    def test_send_email_alert(self, mock_smtp):
        """Test sending email alerts"""
        # Mock SMTP
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        # Mock secret retrieval
        self.mock_secret_client.access_secret_version.return_value = Mock(
            payload=Mock(data=b'test-password')
        )
        
        # Create test alert
        alert = AlertMessage(
            location_id='LOC123',
            location_name='Test Care Home',
            risk_score=85.5,
            risk_level='HIGH',
            risk_factors=[{'factor': 'Staffing', 'impact': 25.0}],
            recommendations=['Increase staffing levels'],
            assessment_date='2024-01-15',
            provider_name='Test Provider',
            address='123 Test Street, Test City'
        )
        
        # Test email sending
        config = {
            'smtp_host': 'smtp.test.com',
            'smtp_port': 587,
            'use_tls': True,
            'sender_email': 'test@example.com',
            'recipients': ['recipient@test.com']
        }
        
        result = self.service._send_email_alert(alert, config)
        
        self.assertTrue(result)
        mock_server.send_message.assert_called_once()
    
    @patch('requests.post')
    def test_send_webhook_alert(self, mock_post):
        """Test sending webhook alerts"""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Create test alert
        alert = AlertMessage(
            location_id='LOC123',
            location_name='Test Care Home',
            risk_score=85.5,
            risk_level='HIGH',
            risk_factors=[{'factor': 'Staffing', 'impact': 25.0}],
            recommendations=['Increase staffing levels'],
            assessment_date='2024-01-15'
        )
        
        # Test webhook sending
        config = {
            'webhook_url': 'https://test-webhook.com/alerts',
            'headers': {'Content-Type': 'application/json'},
            'timeout': 30
        }
        
        result = self.service._send_webhook_alert(alert, config)
        
        self.assertTrue(result)
        mock_post.assert_called_once()
        
        # Verify webhook payload
        call_args = mock_post.call_args
        self.assertEqual(call_args.kwargs['json']['location']['id'], 'LOC123')
        self.assertEqual(call_args.kwargs['json']['risk_assessment']['score'], 85.5)
    
    def test_send_high_risk_alerts_dry_run(self):
        """Test dry run mode"""
        # Mock query result
        mock_df = pd.DataFrame([{
            'locationId': 'LOC123',
            'locationName': 'Test Care Home',
            'riskScore': 85.5,
            'riskLevel': 'HIGH',
            'topRiskFactors': json.dumps([{'factor': 'Staffing', 'impact': 25.0}]),
            'recommendations': json.dumps(['Increase staffing']),
            'assessmentDate': '2024-01-15'
        }])
        
        mock_query_job = Mock()
        mock_query_job.to_dataframe.return_value = mock_df
        self.mock_bq_client.query.return_value = mock_query_job
        
        # Test dry run
        summary = self.service.send_high_risk_alerts(dry_run=True)
        
        self.assertEqual(summary['locations_found'], 1)
        self.assertEqual(summary['alerts_sent'], 0)
        self.assertEqual(summary['errors'], 0)
        self.assertTrue(summary['details'][0]['dry_run'])
    
    def test_send_high_risk_alerts_no_results(self):
        """Test when no high-risk locations found"""
        # Mock empty query result
        mock_query_job = Mock()
        mock_query_job.to_dataframe.return_value = pd.DataFrame()
        self.mock_bq_client.query.return_value = mock_query_job
        
        # Test
        summary = self.service.send_high_risk_alerts()
        
        self.assertEqual(summary['locations_found'], 0)
        self.assertEqual(summary['alerts_sent'], 0)
        self.assertEqual(summary['errors'], 0)


class TestAlertTemplates(unittest.TestCase):
    """Test cases for alert templates"""
    
    def setUp(self):
        """Set up test fixtures"""
        with patch('google.cloud.bigquery.Client'):
            with patch('google.cloud.secretmanager.SecretManagerServiceClient'):
                self.service = NotificationService('test-project', 'test-dataset')
        
        self.test_alert = AlertMessage(
            location_id='LOC123',
            location_name='Test Care Home',
            risk_score=85.5,
            risk_level='HIGH',
            risk_factors=[
                {'factor': 'Staffing', 'impact': 25.0},
                {'factor': 'Compliance', 'impact': 20.0}
            ],
            recommendations=[
                'Increase staffing levels',
                'Review compliance procedures'
            ],
            assessment_date='2024-01-15',
            provider_name='Test Provider Ltd',
            address='123 Test Street, Test City, TS1 2AB'
        )
    
    def test_email_template(self):
        """Test email template rendering"""
        # Test HTML template
        html_output = self.service.templates['email_html'].render(**self.test_alert.__dict__)
        self.assertIn('Test Care Home', html_output)
        self.assertIn('85.5%', html_output)
        self.assertIn('Staffing', html_output)
        self.assertIn('Increase staffing levels', html_output)
        
        # Test text template
        text_output = self.service.templates['email_text'].render(**self.test_alert.__dict__)
        self.assertIn('Test Care Home', text_output)
        self.assertIn('85.5%', text_output)
    
    def test_webhook_template(self):
        """Test webhook template rendering"""
        webhook_output = self.service.templates['webhook'].render(
            timestamp=datetime.utcnow().isoformat(),
            **self.test_alert.__dict__
        )
        
        # Parse JSON to verify structure
        webhook_data = json.loads(webhook_output)
        self.assertEqual(webhook_data['alert_type'], 'high_risk_location')
        self.assertEqual(webhook_data['location']['id'], 'LOC123')
        self.assertEqual(webhook_data['risk_assessment']['score'], 85.5)
        self.assertEqual(len(webhook_data['risk_assessment']['factors']), 2)
    
    def test_slack_template(self):
        """Test Slack template rendering"""
        slack_output = self.service.templates['slack'].render(
            timestamp_unix=int(datetime.utcnow().timestamp()),
            **self.test_alert.__dict__
        )
        
        # Parse JSON to verify structure
        slack_data = json.loads(slack_output)
        self.assertIn('High-Risk CQC Location Alert', slack_data['text'])
        self.assertEqual(slack_data['attachments'][0]['color'], 'danger')
        self.assertIn('Test Care Home', str(slack_data))
    
    def test_teams_template(self):
        """Test Teams template rendering"""
        teams_output = self.service.templates['teams'].render(**self.test_alert.__dict__)
        
        # Parse JSON to verify structure
        teams_data = json.loads(teams_output)
        self.assertEqual(teams_data['@type'], 'MessageCard')
        self.assertEqual(teams_data['themeColor'], 'FF0000')
        self.assertIn('Test Care Home', teams_data['summary'])


if __name__ == '__main__':
    unittest.main()