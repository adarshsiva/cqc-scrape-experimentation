"""
Alert utilities for CQC Airflow DAGs

This module provides functions for creating and sending various types of alerts.
"""

from typing import List, Dict, Any
from datetime import datetime
import json
import logging


def create_risk_alert_html(high_risk_locations: List[Dict[str, Any]]) -> str:
    """
    Create HTML content for high-risk location alerts
    
    Args:
        high_risk_locations: List of dictionaries containing location risk information
        
    Returns:
        HTML formatted alert content
    """
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .critical { color: #d32f2f; font-weight: bold; }
            .high { color: #f57c00; font-weight: bold; }
            .medium { color: #fbc02d; }
            .footer { margin-top: 30px; font-size: 12px; color: #666; }
        </style>
    </head>
    <body>
        <h2>CQC High Risk Location Alerts</h2>
        <p>Generated: {timestamp}</p>
        <p>The following locations have been identified as high-risk based on the latest assessment:</p>
        
        <table>
            <tr>
                <th>Location Name</th>
                <th>Location ID</th>
                <th>Provider Name</th>
                <th>Risk Level</th>
                <th>Risk Score</th>
                <th>Key Risk Factors</th>
                <th>Days Since Inspection</th>
            </tr>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    for location in high_risk_locations:
        risk_class = location.get('risk_level', '').lower()
        html_content += f"""
            <tr>
                <td>{location.get('location_name', 'N/A')}</td>
                <td>{location.get('location_id', 'N/A')}</td>
                <td>{location.get('provider_name', 'N/A')}</td>
                <td class="{risk_class}">{location.get('risk_level', 'N/A')}</td>
                <td>{location.get('risk_score', 0):.3f}</td>
                <td>{location.get('risk_factors', 'N/A')}</td>
                <td>{location.get('days_since_inspection', 'N/A')}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h3>Recommended Actions</h3>
        <ul>
            <li>Review and prioritize inspections for CRITICAL risk locations</li>
            <li>Allocate additional resources to HIGH risk locations</li>
            <li>Update risk mitigation plans for all flagged locations</li>
            <li>Monitor these locations more frequently</li>
        </ul>
        
        <div class="footer">
            <p>This is an automated alert from the CQC Risk Assessment System.</p>
            <p>For questions or concerns, please contact the data team.</p>
        </div>
    </body>
    </html>
    """
    
    return html_content


def create_monitoring_alert_html(issues: List[str], metrics: Dict[str, Any]) -> str:
    """
    Create HTML content for system monitoring alerts
    
    Args:
        issues: List of identified issues
        metrics: Dictionary of system metrics
        
    Returns:
        HTML formatted monitoring alert
    """
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; }
            .issue { color: #d32f2f; font-weight: bold; }
            .metric-table { border-collapse: collapse; margin-top: 20px; }
            .metric-table th, .metric-table td { border: 1px solid #ddd; padding: 8px; }
            .metric-table th { background-color: #f44336; color: white; }
            .warning { background-color: #fff3cd; padding: 10px; border: 1px solid #ffeaa7; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h2>CQC Pipeline System Alert</h2>
        <p>Generated: {timestamp}</p>
        
        <div class="warning">
            <h3>Issues Detected:</h3>
            <ul>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    for issue in issues:
        html_content += f'<li class="issue">{issue}</li>\n'
    
    html_content += """
            </ul>
        </div>
        
        <h3>Current System Metrics</h3>
        <table class="metric-table">
            <tr>
                <th>Metric</th>
                <th>Current Value</th>
                <th>Status</th>
            </tr>
    """
    
    for metric_name, metric_value in metrics.items():
        status = "🔴 Alert" if metric_name in str(issues) else "🟢 OK"
        html_content += f"""
            <tr>
                <td>{metric_name.replace('_', ' ').title()}</td>
                <td>{metric_value}</td>
                <td>{status}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h3>Immediate Actions Required</h3>
        <ul>
            <li>Review the identified issues and take corrective action</li>
            <li>Check system logs for more details</li>
            <li>Monitor the affected components closely</li>
            <li>Escalate to senior team if issues persist</li>
        </ul>
        
        <p><em>This alert was triggered because one or more system metrics exceeded their thresholds.</em></p>
    </body>
    </html>
    """
    
    return html_content


def create_data_quality_alert_html(quality_issues: Dict[str, Any]) -> str:
    """
    Create HTML content for data quality alerts
    
    Args:
        quality_issues: Dictionary containing data quality problems
        
    Returns:
        HTML formatted data quality alert
    """
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; }
            .quality-table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            .quality-table th, .quality-table td { border: 1px solid #ddd; padding: 8px; }
            .quality-table th { background-color: #ff9800; color: white; }
            .poor { color: #d32f2f; }
            .fair { color: #f57c00; }
            .good { color: #388e3c; }
        </style>
    </head>
    <body>
        <h2>CQC Data Quality Alert</h2>
        <p>Generated: {timestamp}</p>
        <p>Data quality issues have been detected in the following areas:</p>
        
        <table class="quality-table">
            <tr>
                <th>Table</th>
                <th>Issue Type</th>
                <th>Affected Records</th>
                <th>Percentage</th>
                <th>Severity</th>
            </tr>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    for table, issues in quality_issues.items():
        for issue_type, details in issues.items():
            severity = "poor" if details['percentage'] > 5 else "fair" if details['percentage'] > 1 else "good"
            html_content += f"""
            <tr>
                <td>{table}</td>
                <td>{issue_type}</td>
                <td>{details['count']:,}</td>
                <td class="{severity}">{details['percentage']:.2f}%</td>
                <td class="{severity}">{severity.upper()}</td>
            </tr>
            """
    
    html_content += """
        </table>
        
        <h3>Data Quality Standards</h3>
        <ul>
            <li><span class="good">GOOD</span>: Less than 1% affected records</li>
            <li><span class="fair">FAIR</span>: 1-5% affected records</li>
            <li><span class="poor">POOR</span>: More than 5% affected records</li>
        </ul>
        
        <h3>Recommended Actions</h3>
        <ul>
            <li>Investigate the source of data quality issues</li>
            <li>Update data validation rules if necessary</li>
            <li>Consider reprocessing affected records</li>
            <li>Document any data quality exceptions</li>
        </ul>
        
        <p><em>Regular data quality monitoring helps maintain the integrity of the CQC pipeline.</em></p>
    </body>
    </html>
    """
    
    return html_content


def create_model_performance_alert_html(performance_metrics: Dict[str, Any]) -> str:
    """
    Create HTML content for model performance alerts
    
    Args:
        performance_metrics: Dictionary containing model performance metrics
        
    Returns:
        HTML formatted model performance alert
    """
    html_content = """
    <html>
    <head>
        <style>
            body { font-family: Arial, sans-serif; }
            .metric-card { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .metric-value { font-size: 24px; font-weight: bold; }
            .below-threshold { color: #d32f2f; }
            .above-threshold { color: #388e3c; }
        </style>
    </head>
    <body>
        <h2>CQC Model Performance Alert</h2>
        <p>Generated: {timestamp}</p>
        <p>Model performance has degraded below acceptable thresholds:</p>
        
        <div class="metric-card">
            <h3>Current Model Performance</h3>
    """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    for metric, value in performance_metrics.items():
        threshold = 0.85 if metric == 'accuracy' else 0.8
        status_class = 'above-threshold' if value >= threshold else 'below-threshold'
        html_content += f"""
            <p>{metric.replace('_', ' ').title()}: 
            <span class="metric-value {status_class}">{value:.3f}</span> 
            (Threshold: {threshold:.3f})</p>
        """
    
    html_content += """
        </div>
        
        <h3>Potential Causes</h3>
        <ul>
            <li>Data drift - the distribution of input data has changed</li>
            <li>Concept drift - the relationship between features and target has changed</li>
            <li>New patterns in CQC ratings not captured by current model</li>
            <li>Insufficient training data for recent time periods</li>
        </ul>
        
        <h3>Recommended Actions</h3>
        <ul>
            <li>Trigger immediate model retraining with recent data</li>
            <li>Analyze feature importance changes</li>
            <li>Review recent predictions for systematic errors</li>
            <li>Consider updating feature engineering pipeline</li>
            <li>Schedule meeting to discuss model architecture updates</li>
        </ul>
        
        <p><em>Model performance is critical for accurate risk assessments. Please take immediate action.</em></p>
    </body>
    </html>
    """
    
    return html_content


def send_slack_notification(webhook_url: str, message: str, alert_type: str = "info"):
    """
    Send a notification to Slack
    
    Args:
        webhook_url: Slack webhook URL
        message: Message to send
        alert_type: Type of alert (info, warning, error)
    """
    import requests
    
    color_map = {
        "info": "#36a64f",
        "warning": "#ff9800",
        "error": "#ff0000"
    }
    
    payload = {
        "attachments": [{
            "color": color_map.get(alert_type, "#36a64f"),
            "title": "CQC Pipeline Alert",
            "text": message,
            "footer": "CQC Monitoring System",
            "ts": int(datetime.now().timestamp())
        }]
    }
    
    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        logging.info(f"Slack notification sent successfully: {alert_type}")
    except Exception as e:
        logging.error(f"Failed to send Slack notification: {str(e)}")


def format_metric_for_display(metric_name: str, metric_value: Any) -> str:
    """
    Format a metric value for display in alerts
    
    Args:
        metric_name: Name of the metric
        metric_value: Value of the metric
        
    Returns:
        Formatted string for display
    """
    if metric_name.endswith('_gb'):
        return f"{metric_value:.2f} GB"
    elif metric_name.endswith('_count'):
        return f"{metric_value:,}"
    elif metric_name.endswith('_rate') or metric_name.endswith('_score'):
        return f"{metric_value:.2%}"
    elif metric_name.endswith('_cost'):
        return f"${metric_value:.2f}"
    elif metric_name.endswith('_seconds') or metric_name.endswith('_latency'):
        return f"{metric_value:.3f}s"
    else:
        return str(metric_value)