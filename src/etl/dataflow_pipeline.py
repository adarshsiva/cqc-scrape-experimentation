import argparse
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, GoogleCloudOptions, StandardOptions
from apache_beam.io import ReadFromText, WriteToBigQuery
from apache_beam.io.gcp.bigquery import BigQueryDisposition

try:
    from transforms import (
        ParseJsonFn,
        ExtractLocationFeatures,
        CalculateDerivedFeatures,
        FilterInvalidRecords
    )
except ImportError:
    # When running on Dataflow, the module might be in a different path
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transforms import (
        ParseJsonFn,
        ExtractLocationFeatures,
        CalculateDerivedFeatures,
        FilterInvalidRecords
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CQCETLPipeline:
    """ETL Pipeline for CQC data processing."""
    
    def __init__(self, project_id: str, dataset_id: str, temp_location: str):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.temp_location = temp_location
        
    def get_pipeline_options(self, runner: str = 'DataflowRunner') -> PipelineOptions:
        """Configure pipeline options."""
        options = PipelineOptions()
        
        google_cloud_options = options.view_as(GoogleCloudOptions)
        google_cloud_options.project = self.project_id
        google_cloud_options.temp_location = self.temp_location
        google_cloud_options.staging_location = f"{self.temp_location}/staging"
        
        standard_options = options.view_as(StandardOptions)
        standard_options.runner = runner
        
        return options
    
    def get_location_table_schema(self) -> List[Dict]:
        """Define BigQuery schema for locations table."""
        return [
            {'name': 'location_id', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'provider_id', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'name', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'postal_code', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'region', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'local_authority', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'registration_date', 'type': 'DATE', 'mode': 'NULLABLE'},
            {'name': 'last_inspection_date', 'type': 'DATE', 'mode': 'NULLABLE'},
            {'name': 'overall_rating', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'regulated_activities', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'service_types', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'specialisms', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'days_since_last_inspection', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'days_since_registration', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'num_regulated_activities', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'num_service_types', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'has_specialisms', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'ingestion_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'processing_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'}
        ]
    
    def get_provider_table_schema(self) -> List[Dict]:
        """Define BigQuery schema for providers table."""
        return [
            {'name': 'provider_id', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'name', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'postal_code', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'region', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'local_authority', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'registration_date', 'type': 'DATE', 'mode': 'NULLABLE'},
            {'name': 'ownership_type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'ingestion_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'processing_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'}
        ]
    
    def run_locations_pipeline(self, input_path: str):
        """Run ETL pipeline for locations data."""
        options = self.get_pipeline_options()
        
        with beam.Pipeline(options=options) as pipeline:
            locations = (
                pipeline
                | 'Read JSON Files' >> ReadFromText(input_path)
                | 'Parse JSON' >> beam.ParDo(ParseJsonFn())
                | 'Extract Features' >> beam.ParDo(ExtractLocationFeatures())
                | 'Calculate Derived Features' >> beam.ParDo(CalculateDerivedFeatures())
                | 'Filter Invalid Records' >> beam.ParDo(FilterInvalidRecords())
                | 'Add Processing Timestamp' >> beam.Map(
                    lambda x: {**x, 'processing_timestamp': datetime.utcnow().isoformat()}
                )
            )
            
            # Write to BigQuery
            locations | 'Write to BigQuery' >> WriteToBigQuery(
                table=f"{self.project_id}:{self.dataset_id}.locations",
                schema={'fields': self.get_location_table_schema()},
                create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=BigQueryDisposition.WRITE_APPEND
            )
            
            # Log statistics
            stats = (
                locations
                | 'Count Records' >> beam.combiners.Count.Globally()
                | 'Log Stats' >> beam.Map(
                    lambda count: logger.info(f"Processed {count} location records")
                )
            )
    
    def run_providers_pipeline(self, input_path: str):
        """Run ETL pipeline for providers data."""
        options = self.get_pipeline_options()
        
        with beam.Pipeline(options=options) as pipeline:
            providers = (
                pipeline
                | 'Read JSON Files' >> ReadFromText(input_path)
                | 'Parse JSON' >> beam.ParDo(ParseJsonFn())
                | 'Extract Provider Info' >> beam.Map(self._extract_provider_info)
                | 'Add Processing Timestamp' >> beam.Map(
                    lambda x: {**x, 'processing_timestamp': datetime.utcnow().isoformat()}
                )
            )
            
            # Write to BigQuery
            providers | 'Write to BigQuery' >> WriteToBigQuery(
                table=f"{self.project_id}:{self.dataset_id}.providers",
                schema={'fields': self.get_provider_table_schema()},
                create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=BigQueryDisposition.WRITE_APPEND
            )
    
    def _extract_provider_info(self, provider: Dict) -> Dict:
        """Extract relevant fields from provider data."""
        return {
            'provider_id': provider.get('providerId'),
            'name': provider.get('name'),
            'type': provider.get('type'),
            'postal_code': provider.get('postalCode'),
            'region': provider.get('region'),
            'local_authority': provider.get('localAuthority'),
            'registration_date': provider.get('registrationDate'),
            'ownership_type': provider.get('ownershipType'),
            'ingestion_timestamp': provider.get('_ingestion_timestamp', datetime.utcnow().isoformat())
        }


def main():
    """Main entry point for the ETL pipeline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    parser.add_argument('--dataset-id', required=True, help='BigQuery Dataset ID')
    parser.add_argument('--temp-location', required=True, help='GCS temp location')
    parser.add_argument('--input-path', required=True, help='Input file path pattern')
    parser.add_argument('--data-type', required=True, choices=['locations', 'providers'])
    parser.add_argument('--runner', default='DataflowRunner', help='Pipeline runner')
    
    args = parser.parse_args()
    
    pipeline = CQCETLPipeline(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        temp_location=args.temp_location
    )
    
    if args.data_type == 'locations':
        pipeline.run_locations_pipeline(args.input_path)
    else:
        pipeline.run_providers_pipeline(args.input_path)
    
    logger.info(f"Pipeline completed for {args.data_type}")


if __name__ == '__main__':
    main()