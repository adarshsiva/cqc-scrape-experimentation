import json
import functions_framework
from google.cloud import storage
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@functions_framework.http
def convert_to_ndjson(request):
    """Convert JSON array files to newline-delimited JSON for BigQuery."""
    try:
        # Parse request
        request_json = request.get_json() if request.method == 'POST' else {}
        source_bucket = request_json.get('source_bucket', 'machine-learning-exp-467008-cqc-raw-data')
        source_prefix = request_json.get('source_prefix', 'raw/')
        dest_prefix = request_json.get('dest_prefix', 'processed/')
        
        client = storage.Client()
        bucket = client.bucket(source_bucket)
        
        converted_files = []
        
        # List all JSON files in the source prefix
        blobs = bucket.list_blobs(prefix=source_prefix)
        
        for blob in blobs:
            if blob.name.endswith('.json') and not blob.name.startswith(dest_prefix):
                logger.info(f"Converting {blob.name}")
                
                # Download JSON array
                json_content = blob.download_as_text()
                data = json.loads(json_content)
                
                # Convert to NDJSON
                ndjson_lines = []
                if isinstance(data, list):
                    for item in data:
                        ndjson_lines.append(json.dumps(item))
                else:
                    ndjson_lines.append(json.dumps(data))
                
                # Upload NDJSON
                dest_blob_name = blob.name.replace(source_prefix, dest_prefix).replace('.json', '.ndjson')
                dest_blob = bucket.blob(dest_blob_name)
                dest_blob.upload_from_string('\n'.join(ndjson_lines))
                
                converted_files.append({
                    'source': blob.name,
                    'destination': dest_blob_name,
                    'records': len(ndjson_lines)
                })
                
                logger.info(f"Converted {blob.name} to {dest_blob_name}")
        
        return {
            'status': 'success',
            'converted_files': converted_files,
            'timestamp': datetime.utcnow().isoformat()
        }, 200
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, 500