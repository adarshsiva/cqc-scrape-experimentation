# CQC Data Fetcher Scripts

This directory contains scripts for fetching and processing CQC (Care Quality Commission) data.

## fetch_detailed_cqc_data.py

An enhanced CQC data fetcher that retrieves detailed location information including:
- Current and historical ratings
- Inspection areas and compliance data
- Inspection reports
- Location metadata

### Features

- **Rate Limiting**: Implements delays between API requests to respect CQC API limits
- **Parallel Processing**: Uses ThreadPoolExecutor for concurrent fetching of location details
- **Retry Logic**: Automatic retry with exponential backoff for failed requests
- **Comprehensive Data**: Fetches ratings history, inspection areas, and reports for each location
- **Cloud Storage Integration**: Saves data directly to Google Cloud Storage
- **Summary Statistics**: Generates aggregated statistics about fetched data

### Usage

1. Set environment variables:
```bash
export GCP_PROJECT=your-project-id
export GCS_BUCKET=cqc-data-raw
export MAX_LOCATIONS=100  # Optional, defaults to 100
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the script:
```bash
python fetch_detailed_cqc_data.py
```

### Output Structure

The script saves data to GCS in the following structure:
```
gs://your-bucket/
├── detailed_locations/
│   ├── {location_id}/
│   │   └── {timestamp}.json  # Individual location details
│   ├── aggregated/
│   │   └── {timestamp}_all_locations.json  # All locations in one file
│   └── summary/
│       └── {timestamp}_summary.json  # Summary statistics
```

### Data Schema

Each detailed location record includes:
- Basic location information (ID, name, address, type)
- Current ratings (overall and by domain)
- Ratings history
- Inspection areas with ratings
- Recent inspection reports
- Provider information
- Regulated activities

### Configuration

- `max_locations`: Maximum number of locations to fetch (default: 100)
- `max_workers`: Number of parallel threads for fetching (default: 5)
- `rate_limit_delay`: Delay between API requests in seconds (default: 0.1)

### Error Handling

The script includes comprehensive error handling:
- Retries failed requests up to 3 times
- Logs all errors with detailed messages
- Continues processing even if individual locations fail
- Saves partial results if the process is interrupted