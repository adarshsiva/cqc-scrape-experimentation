"""
CQC High-Risk Location Notification Service

This module implements a notification system that queries BigQuery for high-risk
locations and sends alerts to stakeholders via email and webhooks.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from google.cloud import bigquery
from google.cloud import secretmanager
import pandas as pd
from jinja2 import Template

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    """Supported notification channels"""
    EMAIL = "email"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TEAMS = "teams"


@dataclass
class NotificationConfig:
    """Configuration for notification channels"""
    channel: NotificationChannel
    enabled: bool
    config: Dict[str, Any]


@dataclass
class AlertMessage:
    """Structure for alert messages"""
    location_id: str
    location_name: str
    risk_score: float
    risk_level: str
    risk_factors: List[Dict[str, Any]]
    recommendations: List[str]
    assessment_date: str
    provider_name: Optional[str] = None
    address: Optional[str] = None


class NotificationService:
    """Service for sending high-risk location alerts"""
    
    def __init__(self, project_id: str, dataset_id: str = "cqc_data"):
        """
        Initialize the notification service
        
        Args:
            project_id: GCP project ID
            dataset_id: BigQuery dataset ID
        """
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.bigquery_client = bigquery.Client(project=project_id)
        self.secret_client = secretmanager.SecretManagerServiceClient()
        
        # Load notification configurations
        self.notification_configs = self._load_notification_configs()
        
        # Load message templates
        self.templates = self._load_templates()
    
    def _load_notification_configs(self) -> List[NotificationConfig]:
        """Load notification channel configurations"""
        configs = []
        
        # Email configuration
        email_config = NotificationConfig(
            channel=NotificationChannel.EMAIL,
            enabled=os.getenv("ENABLE_EMAIL_NOTIFICATIONS", "true").lower() == "true",
            config={
                "smtp_host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
                "smtp_port": int(os.getenv("SMTP_PORT", "587")),
                "use_tls": True,
                "sender_email": os.getenv("SENDER_EMAIL"),
                "recipients": os.getenv("ALERT_RECIPIENTS", "").split(","),
                "cc_recipients": os.getenv("CC_RECIPIENTS", "").split(",") if os.getenv("CC_RECIPIENTS") else []
            }
        )
        configs.append(email_config)
        
        # Webhook configuration
        webhook_config = NotificationConfig(
            channel=NotificationChannel.WEBHOOK,
            enabled=os.getenv("ENABLE_WEBHOOK_NOTIFICATIONS", "true").lower() == "true",
            config={
                "webhook_url": os.getenv("WEBHOOK_URL"),
                "headers": {
                    "Content-Type": "application/json",
                    "X-CQC-Alert": "high-risk-location"
                },
                "timeout": 30
            }
        )
        configs.append(webhook_config)
        
        # Slack configuration
        slack_config = NotificationConfig(
            channel=NotificationChannel.SLACK,
            enabled=os.getenv("ENABLE_SLACK_NOTIFICATIONS", "false").lower() == "true",
            config={
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL"),
                "channel": os.getenv("SLACK_CHANNEL", "#cqc-alerts"),
                "username": "CQC Risk Monitor"
            }
        )
        configs.append(slack_config)
        
        # Teams configuration
        teams_config = NotificationConfig(
            channel=NotificationChannel.TEAMS,
            enabled=os.getenv("ENABLE_TEAMS_NOTIFICATIONS", "false").lower() == "true",
            config={
                "webhook_url": os.getenv("TEAMS_WEBHOOK_URL")
            }
        )
        configs.append(teams_config)
        
        return configs
    
    def _load_templates(self) -> Dict[str, Template]:
        """Load message templates for different channels"""
        templates = {}
        
        # Email HTML template
        templates["email_html"] = Template("""
<!DOCTYPE html>
<html>
<head>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }
        .alert-header { background-color: #d32f2f; color: white; padding: 20px; border-radius: 5px 5px 0 0; }
        .alert-body { background-color: #f5f5f5; padding: 20px; border-radius: 0 0 5px 5px; }
        .risk-score { font-size: 24px; font-weight: bold; color: #d32f2f; }
        .risk-factors { background-color: white; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .recommendations { background-color: #e3f2fd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .footer { margin-top: 20px; font-size: 12px; color: #666; }
    </style>
</head>
<body>
    <div class="alert-header">
        <h2>🚨 High-Risk CQC Location Alert</h2>
    </div>
    <div class="alert-body">
        <h3>{{ location_name }}</h3>
        <p><strong>Location ID:</strong> {{ location_id }}</p>
        {% if provider_name %}<p><strong>Provider:</strong> {{ provider_name }}</p>{% endif %}
        {% if address %}<p><strong>Address:</strong> {{ address }}</p>{% endif %}
        <p><strong>Assessment Date:</strong> {{ assessment_date }}</p>
        
        <div class="risk-score">
            Risk Score: {{ "%.1f" | format(risk_score) }}% ({{ risk_level }})
        </div>
        
        <div class="risk-factors">
            <h4>Top Risk Factors:</h4>
            <ul>
            {% for factor in risk_factors %}
                <li><strong>{{ factor.factor }}:</strong> Impact {{ "%.1f" | format(factor.impact) }}%</li>
            {% endfor %}
            </ul>
        </div>
        
        <div class="recommendations">
            <h4>Recommendations:</h4>
            <ul>
            {% for rec in recommendations %}
                <li>{{ rec }}</li>
            {% endfor %}
            </ul>
        </div>
        
        <div class="footer">
            <p>This is an automated alert from the CQC Risk Monitoring System. 
            Please review the location urgently and take appropriate action.</p>
        </div>
    </div>
</body>
</html>
        """)
        
        # Email text template
        templates["email_text"] = Template("""
HIGH-RISK CQC LOCATION ALERT

Location: {{ location_name }}
Location ID: {{ location_id }}
{% if provider_name %}Provider: {{ provider_name }}{% endif %}
{% if address %}Address: {{ address }}{% endif %}
Assessment Date: {{ assessment_date }}

Risk Score: {{ "%.1f" | format(risk_score) }}% ({{ risk_level }})

TOP RISK FACTORS:
{% for factor in risk_factors %}
- {{ factor.factor }}: Impact {{ "%.1f" | format(factor.impact) }}%
{% endfor %}

RECOMMENDATIONS:
{% for rec in recommendations %}
- {{ rec }}
{% endfor %}

This is an automated alert from the CQC Risk Monitoring System.
Please review the location urgently and take appropriate action.
        """)
        
        # Webhook JSON template
        templates["webhook"] = Template("""
{
    "alert_type": "high_risk_location",
    "timestamp": "{{ timestamp }}",
    "location": {
        "id": "{{ location_id }}",
        "name": "{{ location_name }}",
        {% if provider_name %}"provider": "{{ provider_name }}",{% endif %}
        {% if address %}"address": "{{ address }}",{% endif %}
        "assessment_date": "{{ assessment_date }}"
    },
    "risk_assessment": {
        "score": {{ risk_score }},
        "level": "{{ risk_level }}",
        "factors": {{ risk_factors | tojson }},
        "recommendations": {{ recommendations | tojson }}
    }
}
        """)
        
        # Slack template
        templates["slack"] = Template("""
{
    "text": "🚨 High-Risk CQC Location Alert",
    "attachments": [
        {
            "color": "danger",
            "fields": [
                {
                    "title": "Location",
                    "value": "{{ location_name }}",
                    "short": false
                },
                {
                    "title": "Risk Score",
                    "value": "{{ "%.1f" | format(risk_score) }}% ({{ risk_level }})",
                    "short": true
                },
                {
                    "title": "Location ID",
                    "value": "{{ location_id }}",
                    "short": true
                }
            ],
            "footer": "CQC Risk Monitor",
            "ts": {{ timestamp_unix }}
        },
        {
            "color": "warning",
            "title": "Top Risk Factors",
            "text": "{% for factor in risk_factors %}• {{ factor.factor }}: {{ "%.1f" | format(factor.impact) }}%\\n{% endfor %}"
        },
        {
            "color": "good",
            "title": "Recommendations",
            "text": "{% for rec in recommendations %}• {{ rec }}\\n{% endfor %}"
        }
    ]
}
        """)
        
        # Teams template
        templates["teams"] = Template("""
{
    "@type": "MessageCard",
    "@context": "https://schema.org/extensions",
    "themeColor": "FF0000",
    "summary": "High-Risk CQC Location Alert: {{ location_name }}",
    "sections": [
        {
            "activityTitle": "🚨 High-Risk CQC Location Alert",
            "activitySubtitle": "{{ location_name }}",
            "facts": [
                {
                    "name": "Location ID",
                    "value": "{{ location_id }}"
                },
                {
                    "name": "Risk Score",
                    "value": "{{ "%.1f" | format(risk_score) }}% ({{ risk_level }})"
                },
                {
                    "name": "Assessment Date",
                    "value": "{{ assessment_date }}"
                }
            ]
        },
        {
            "title": "Top Risk Factors",
            "text": "{% for factor in risk_factors %}• **{{ factor.factor }}**: {{ "%.1f" | format(factor.impact) }}%  \\n{% endfor %}"
        },
        {
            "title": "Recommendations",
            "text": "{% for rec in recommendations %}• {{ rec }}  \\n{% endfor %}"
        }
    ]
}
        """)
        
        return templates
    
    def _get_secret(self, secret_name: str) -> str:
        """Retrieve secret from Secret Manager"""
        try:
            name = f"projects/{self.project_id}/secrets/{secret_name}/versions/latest"
            response = self.secret_client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.warning(f"Failed to retrieve secret {secret_name}: {e}")
            return ""
    
    def query_high_risk_locations(self, 
                                  assessment_date: Optional[str] = None,
                                  risk_threshold: float = 70.0) -> pd.DataFrame:
        """
        Query BigQuery for high-risk locations
        
        Args:
            assessment_date: Date to query (defaults to current date)
            risk_threshold: Minimum risk score threshold
            
        Returns:
            DataFrame of high-risk locations
        """
        if not assessment_date:
            assessment_date = "CURRENT_DATE()"
        else:
            assessment_date = f"'{assessment_date}'"
        
        query = f"""
        SELECT 
            ra.locationId,
            ra.riskScore,
            ra.riskLevel,
            ra.topRiskFactors,
            ra.recommendations,
            ra.assessmentDate,
            l.locationName,
            l.postalAddressLine1,
            l.postalAddressLine2,
            l.postalAddressTownCity,
            l.postalAddressCounty,
            l.postalCode,
            p.name as providerName
        FROM `{self.project_id}.{self.dataset_id}.risk_assessments` ra
        LEFT JOIN `{self.project_id}.{self.dataset_id}.locations` l
            ON ra.locationId = l.locationId
        LEFT JOIN `{self.project_id}.{self.dataset_id}.providers` p
            ON l.providerId = p.providerId
        WHERE ra.assessmentDate = {assessment_date}
        AND ra.riskLevel = 'HIGH'
        AND ra.riskScore >= {risk_threshold}
        ORDER BY ra.riskScore DESC
        """
        
        logger.info(f"Querying for high-risk locations on {assessment_date}")
        
        try:
            df = self.bigquery_client.query(query).to_dataframe()
            logger.info(f"Found {len(df)} high-risk locations")
            return df
        except Exception as e:
            logger.error(f"Failed to query high-risk locations: {e}")
            raise
    
    def _prepare_alert_message(self, row: pd.Series) -> AlertMessage:
        """Prepare alert message from query result"""
        # Parse risk factors and recommendations
        risk_factors = json.loads(row['topRiskFactors']) if isinstance(row['topRiskFactors'], str) else row['topRiskFactors']
        recommendations = json.loads(row['recommendations']) if isinstance(row['recommendations'], str) else row['recommendations']
        
        # Build address
        address_parts = []
        for field in ['postalAddressLine1', 'postalAddressLine2', 
                      'postalAddressTownCity', 'postalAddressCounty', 'postalCode']:
            if pd.notna(row.get(field)) and row.get(field):
                address_parts.append(str(row[field]))
        address = ", ".join(address_parts) if address_parts else None
        
        return AlertMessage(
            location_id=row['locationId'],
            location_name=row.get('locationName', 'Unknown Location'),
            risk_score=row['riskScore'],
            risk_level=row['riskLevel'],
            risk_factors=risk_factors[:5],  # Top 5 factors
            recommendations=recommendations[:5],  # Top 5 recommendations
            assessment_date=str(row['assessmentDate']),
            provider_name=row.get('providerName'),
            address=address
        )
    
    def _send_email_alert(self, alert: AlertMessage, config: Dict[str, Any]) -> bool:
        """Send email alert"""
        try:
            # Get email credentials from Secret Manager
            sender_password = self._get_secret("smtp-password")
            
            if not config.get('sender_email') or not sender_password:
                logger.warning("Email credentials not configured")
                return False
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"🚨 High-Risk Alert: {alert.location_name}"
            msg['From'] = config['sender_email']
            msg['To'] = ", ".join(config['recipients'])
            if config.get('cc_recipients'):
                msg['Cc'] = ", ".join(config['cc_recipients'])
            
            # Render templates
            text_content = self.templates['email_text'].render(**alert.__dict__)
            html_content = self.templates['email_html'].render(**alert.__dict__)
            
            # Attach parts
            msg.attach(MIMEText(text_content, 'plain'))
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(config['smtp_host'], config['smtp_port']) as server:
                if config.get('use_tls'):
                    server.starttls()
                server.login(config['sender_email'], sender_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for location {alert.location_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    def _send_webhook_alert(self, alert: AlertMessage, config: Dict[str, Any]) -> bool:
        """Send webhook alert"""
        try:
            if not config.get('webhook_url'):
                logger.warning("Webhook URL not configured")
                return False
            
            # Prepare webhook data
            webhook_data = json.loads(self.templates['webhook'].render(
                timestamp=datetime.utcnow().isoformat(),
                **alert.__dict__
            ))
            
            # Send webhook
            response = requests.post(
                config['webhook_url'],
                json=webhook_data,
                headers=config.get('headers', {}),
                timeout=config.get('timeout', 30)
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for location {alert.location_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    def _send_slack_alert(self, alert: AlertMessage, config: Dict[str, Any]) -> bool:
        """Send Slack alert"""
        try:
            if not config.get('webhook_url'):
                logger.warning("Slack webhook URL not configured")
                return False
            
            # Prepare Slack message
            slack_data = json.loads(self.templates['slack'].render(
                timestamp_unix=int(datetime.utcnow().timestamp()),
                **alert.__dict__
            ))
            slack_data['channel'] = config.get('channel', '#general')
            slack_data['username'] = config.get('username', 'CQC Alert Bot')
            
            # Send to Slack
            response = requests.post(
                config['webhook_url'],
                json=slack_data,
                timeout=30
            )
            response.raise_for_status()
            
            logger.info(f"Slack alert sent for location {alert.location_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    def _send_teams_alert(self, alert: AlertMessage, config: Dict[str, Any]) -> bool:
        """Send Microsoft Teams alert"""
        try:
            if not config.get('webhook_url'):
                logger.warning("Teams webhook URL not configured")
                return False
            
            # Prepare Teams message
            teams_data = json.loads(self.templates['teams'].render(**alert.__dict__))
            
            # Send to Teams
            response = requests.post(
                config['webhook_url'],
                json=teams_data,
                timeout=30
            )
            response.raise_for_status()
            
            logger.info(f"Teams alert sent for location {alert.location_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Teams alert: {e}")
            return False
    
    def send_alert(self, alert: AlertMessage) -> Dict[str, bool]:
        """
        Send alert through all configured channels
        
        Args:
            alert: Alert message to send
            
        Returns:
            Dictionary of channel results
        """
        results = {}
        
        for config in self.notification_configs:
            if not config.enabled:
                continue
            
            try:
                if config.channel == NotificationChannel.EMAIL:
                    results['email'] = self._send_email_alert(alert, config.config)
                elif config.channel == NotificationChannel.WEBHOOK:
                    results['webhook'] = self._send_webhook_alert(alert, config.config)
                elif config.channel == NotificationChannel.SLACK:
                    results['slack'] = self._send_slack_alert(alert, config.config)
                elif config.channel == NotificationChannel.TEAMS:
                    results['teams'] = self._send_teams_alert(alert, config.config)
            except Exception as e:
                logger.error(f"Error sending {config.channel.value} alert: {e}")
                results[config.channel.value] = False
        
        return results
    
    def send_high_risk_alerts(self, 
                              assessment_date: Optional[str] = None,
                              risk_threshold: float = 70.0,
                              dry_run: bool = False) -> Dict[str, Any]:
        """
        Main method to query and send alerts for high-risk locations
        
        Args:
            assessment_date: Date to query (defaults to current date)
            risk_threshold: Minimum risk score threshold
            dry_run: If True, only query but don't send alerts
            
        Returns:
            Summary of alerts sent
        """
        # Query high-risk locations
        high_risk_df = self.query_high_risk_locations(assessment_date, risk_threshold)
        
        if high_risk_df.empty:
            logger.info("No high-risk locations found")
            return {
                "locations_found": 0,
                "alerts_sent": 0,
                "errors": 0
            }
        
        # Prepare summary
        summary = {
            "locations_found": len(high_risk_df),
            "alerts_sent": 0,
            "errors": 0,
            "details": []
        }
        
        # Process each high-risk location
        for _, row in high_risk_df.iterrows():
            try:
                # Prepare alert message
                alert = self._prepare_alert_message(row)
                
                if dry_run:
                    logger.info(f"[DRY RUN] Would send alert for {alert.location_id}")
                    summary['details'].append({
                        "location_id": alert.location_id,
                        "location_name": alert.location_name,
                        "risk_score": alert.risk_score,
                        "dry_run": True
                    })
                else:
                    # Send alerts
                    results = self.send_alert(alert)
                    
                    # Update summary
                    if any(results.values()):
                        summary['alerts_sent'] += 1
                    else:
                        summary['errors'] += 1
                    
                    summary['details'].append({
                        "location_id": alert.location_id,
                        "location_name": alert.location_name,
                        "risk_score": alert.risk_score,
                        "channels": results
                    })
                    
            except Exception as e:
                logger.error(f"Failed to process alert for location {row.get('locationId', 'unknown')}: {e}")
                summary['errors'] += 1
        
        # Log summary
        logger.info(f"Alert summary: {summary['locations_found']} locations found, "
                    f"{summary['alerts_sent']} alerts sent, {summary['errors']} errors")
        
        return summary


def main():
    """Main function for command-line execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Send CQC high-risk location alerts")
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--dataset-id", default="cqc_data", help="BigQuery dataset ID")
    parser.add_argument("--date", help="Assessment date (YYYY-MM-DD), defaults to current date")
    parser.add_argument("--threshold", type=float, default=70.0, help="Risk score threshold (default: 70.0)")
    parser.add_argument("--dry-run", action="store_true", help="Query only, don't send alerts")
    
    args = parser.parse_args()
    
    # Initialize service
    service = NotificationService(args.project_id, args.dataset_id)
    
    # Send alerts
    summary = service.send_high_risk_alerts(
        assessment_date=args.date,
        risk_threshold=args.threshold,
        dry_run=args.dry_run
    )
    
    # Print summary
    print(json.dumps(summary, indent=2))
    
    # Exit with error code if there were errors
    if summary['errors'] > 0:
        exit(1)


if __name__ == "__main__":
    main()