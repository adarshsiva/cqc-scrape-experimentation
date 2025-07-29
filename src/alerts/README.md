# CQC Alert Notification Service

This module implements a comprehensive notification system for alerting stakeholders about high-risk CQC locations based on risk assessments stored in BigQuery.

## Features

- **Multi-channel notifications**: Email, Webhooks, Slack, and Microsoft Teams
- **Configurable risk thresholds**: Alert on locations exceeding specified risk scores
- **Rich alert content**: Including risk factors, scores, and recommendations
- **Template-based messaging**: Customizable templates for each channel
- **Cloud Function deployment**: Ready for serverless execution
- **Comprehensive error handling**: With retry mechanisms and logging

## Architecture

```
BigQuery (Risk Assessments) → Notification Service → Multiple Channels
                                                    ├── Email (SMTP)
                                                    ├── Webhooks (HTTP)
                                                    ├── Slack
                                                    └── Teams
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Configure secrets in GCP Secret Manager:
```bash
# Store SMTP password
echo -n "your-smtp-password" | gcloud secrets create smtp-password --data-file=-

# Store webhook API key
echo -n "your-api-key" | gcloud secrets create webhook-api-key --data-file=-
```

## Usage

### Command Line

```bash
# Send alerts for current date
python notification_service.py \
  --project-id your-project-id \
  --dataset-id cqc_data \
  --threshold 70.0

# Dry run mode (query only, no alerts sent)
python notification_service.py \
  --project-id your-project-id \
  --dry-run

# Send alerts for specific date
python notification_service.py \
  --project-id your-project-id \
  --date 2024-01-15
```

### As a Module

```python
from notification_service import NotificationService

# Initialize service
service = NotificationService('your-project-id', 'cqc_data')

# Send alerts
summary = service.send_high_risk_alerts(
    risk_threshold=70.0,
    dry_run=False
)

print(f"Sent {summary['alerts_sent']} alerts for {summary['locations_found']} locations")
```

### Cloud Function Deployment

```bash
# Deploy to Google Cloud Functions
./deploy.sh

# Create scheduled job (daily at 9 AM)
gcloud scheduler jobs create http cqc-daily-alerts \
  --location=europe-west2 \
  --schedule='0 9 * * *' \
  --uri=https://your-function-url \
  --http-method=POST \
  --message-body='{"dry_run": false}'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GCP_PROJECT` | Google Cloud project ID | Required |
| `BIGQUERY_DATASET` | BigQuery dataset name | `cqc_data` |
| `ENABLE_EMAIL_NOTIFICATIONS` | Enable email alerts | `true` |
| `SMTP_HOST` | SMTP server hostname | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP server port | `587` |
| `SENDER_EMAIL` | Sender email address | Required for email |
| `ALERT_RECIPIENTS` | Comma-separated recipient emails | Required for email |
| `ENABLE_WEBHOOK_NOTIFICATIONS` | Enable webhook alerts | `true` |
| `WEBHOOK_URL` | Webhook endpoint URL | Required for webhooks |
| `HIGH_RISK_THRESHOLD` | Minimum risk score for alerts | `70.0` |

### Notification Channels

#### Email Configuration
- Supports SMTP with TLS
- HTML and plain text templates
- Multiple recipients with CC support

#### Webhook Configuration
- JSON payload with full alert details
- Custom headers support
- Configurable timeout and retry

#### Slack Configuration
- Rich message formatting
- Color-coded alerts by severity
- Thread support for discussions

#### Teams Configuration
- Adaptive card format
- Interactive elements
- Deep linking to systems

## Alert Message Structure

Each alert contains:
- **Location Information**: ID, name, address, provider
- **Risk Assessment**: Score, level, assessment date
- **Risk Factors**: Top 5 factors with impact percentages
- **Recommendations**: Top 5 actionable recommendations

## Testing

Run the test suite:
```bash
python -m pytest test_notification_service.py -v
```

Test specific components:
```bash
# Test email functionality
python -m pytest test_notification_service.py::TestNotificationService::test_send_email_alert -v

# Test templates
python -m pytest test_notification_service.py::TestAlertTemplates -v
```

## Monitoring

The service logs all operations with appropriate levels:
- **INFO**: Normal operations (alerts sent, queries executed)
- **WARNING**: Non-critical issues (missing optional config)
- **ERROR**: Failed operations (send failures, query errors)

Monitor Cloud Function logs:
```bash
gcloud functions logs read cqc-alert-notifications --limit 50
```

## Security

- **Secret Manager**: Sensitive credentials stored securely
- **Service Account**: Minimal required permissions
- **HTTPS Only**: All external communications encrypted
- **Input Validation**: All inputs sanitized and validated

## Troubleshooting

### Common Issues

1. **No alerts sent**: Check risk threshold and date parameters
2. **Email failures**: Verify SMTP credentials and recipient addresses
3. **Webhook timeouts**: Increase timeout or check endpoint availability
4. **Permission errors**: Ensure service account has required roles

### Debug Mode

Enable detailed logging:
```bash
export DEBUG_MODE=true
python notification_service.py --project-id your-project
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add tests for new features
3. Update documentation
4. Test locally before deploying

## License

This module is part of the CQC Rating Predictor ML System.