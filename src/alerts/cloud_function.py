"""
Cloud Function for CQC High-Risk Alerts

This function can be deployed to Google Cloud Functions and triggered
by Cloud Scheduler to send periodic alerts for high-risk locations.
"""

import os
import json
import logging
from datetime import datetime
from flask import Request, Response
from notification_service import NotificationService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def send_cqc_alerts(request: Request) -> Response:
    """
    Cloud Function entry point for sending CQC high-risk alerts
    
    Args:
        request: Flask Request object
        
    Returns:
        Flask Response object with execution summary
    """
    try:
        # Parse request parameters
        request_json = request.get_json(silent=True)
        request_args = request.args
        
        # Get configuration from request or environment
        project_id = (request_json or {}).get('project_id') or \
                     request_args.get('project_id') or \
                     os.environ.get('GCP_PROJECT')
        
        dataset_id = (request_json or {}).get('dataset_id') or \
                     request_args.get('dataset_id') or \
                     os.environ.get('BIGQUERY_DATASET', 'cqc_data')
        
        assessment_date = (request_json or {}).get('assessment_date') or \
                          request_args.get('assessment_date')
        
        risk_threshold = float((request_json or {}).get('risk_threshold') or \
                               request_args.get('risk_threshold') or \
                               os.environ.get('HIGH_RISK_THRESHOLD', '70.0'))
        
        dry_run = ((request_json or {}).get('dry_run') or \
                   request_args.get('dry_run', '').lower() == 'true' or \
                   os.environ.get('DRY_RUN', 'false').lower() == 'true')
        
        # Validate required parameters
        if not project_id:
            return Response(
                json.dumps({
                    'error': 'Missing required parameter: project_id',
                    'status': 'error'
                }),
                status=400,
                mimetype='application/json'
            )
        
        # Log execution details
        logger.info(f"Starting CQC alert service: project={project_id}, "
                    f"dataset={dataset_id}, date={assessment_date}, "
                    f"threshold={risk_threshold}, dry_run={dry_run}")
        
        # Initialize notification service
        service = NotificationService(project_id, dataset_id)
        
        # Send alerts
        summary = service.send_high_risk_alerts(
            assessment_date=assessment_date,
            risk_threshold=risk_threshold,
            dry_run=dry_run
        )
        
        # Add metadata to summary
        summary['execution_time'] = datetime.utcnow().isoformat()
        summary['parameters'] = {
            'project_id': project_id,
            'dataset_id': dataset_id,
            'assessment_date': assessment_date or 'current',
            'risk_threshold': risk_threshold,
            'dry_run': dry_run
        }
        
        # Determine response status
        if summary['errors'] > 0:
            status_code = 207  # Multi-status (partial success)
        else:
            status_code = 200
        
        return Response(
            json.dumps(summary),
            status=status_code,
            mimetype='application/json'
        )
        
    except Exception as e:
        logger.error(f"Error in send_cqc_alerts: {str(e)}", exc_info=True)
        
        return Response(
            json.dumps({
                'error': str(e),
                'status': 'error',
                'execution_time': datetime.utcnow().isoformat()
            }),
            status=500,
            mimetype='application/json'
        )


def check_alert_status(request: Request) -> Response:
    """
    Health check endpoint for the alert service
    
    Args:
        request: Flask Request object
        
    Returns:
        Flask Response object with service status
    """
    try:
        # Basic health check
        status = {
            'service': 'cqc-alert-notifications',
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'environment': {
                'project_id': os.environ.get('GCP_PROJECT', 'not-set'),
                'dataset_id': os.environ.get('BIGQUERY_DATASET', 'cqc_data'),
                'email_enabled': os.environ.get('ENABLE_EMAIL_NOTIFICATIONS', 'false'),
                'webhook_enabled': os.environ.get('ENABLE_WEBHOOK_NOTIFICATIONS', 'false'),
                'slack_enabled': os.environ.get('ENABLE_SLACK_NOTIFICATIONS', 'false'),
                'teams_enabled': os.environ.get('ENABLE_TEAMS_NOTIFICATIONS', 'false')
            }
        }
        
        return Response(
            json.dumps(status),
            status=200,
            mimetype='application/json'
        )
        
    except Exception as e:
        return Response(
            json.dumps({
                'service': 'cqc-alert-notifications',
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }),
            status=500,
            mimetype='application/json'
        )


# Main entry point for Cloud Functions
def main(request: Request) -> Response:
    """
    Main Cloud Function entry point with routing
    
    Routes:
        /send or default: Send high-risk alerts
        /health: Health check endpoint
    """
    # Route based on path
    path = request.path.rstrip('/')
    
    if path.endswith('/health'):
        return check_alert_status(request)
    else:
        return send_cqc_alerts(request)


# For local testing
if __name__ == "__main__":
    from flask import Flask, request
    
    app = Flask(__name__)
    
    @app.route('/', methods=['GET', 'POST'])
    @app.route('/send', methods=['GET', 'POST'])
    @app.route('/health', methods=['GET'])
    def handle_request():
        return main(request)
    
    app.run(host='0.0.0.0', port=8080, debug=True)