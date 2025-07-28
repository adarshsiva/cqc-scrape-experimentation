# CQC API Data Fetching Guide

This guide explains how to fetch real data from the Care Quality Commission (CQC) API instead of using sample data.

## Prerequisites

### 1. Register for CQC API Access

1. Visit [CQC API Registration](https://www.cqc.org.uk/about-us/transparency/using-cqc-data)
2. Click on "Register for API access"
3. Fill out the registration form with your details
4. You'll receive an email with your API subscription key

### 2. Understanding API Limits

The CQC API has rate limits:
- **Standard users**: 1,000 requests per hour
- **Partner users**: Higher limits (requires partner agreement with CQC)

## Using the API Scripts

### Test Your API Connection

First, verify your API key works:

```bash
cd /Users/adarsh/Documents/Dev/CQC_scrape
python scripts/fetch_cqc_api_data.py --api-key YOUR_API_KEY --test
```

This will show:
- Total number of providers and locations available
- Sample data from the API
- Confirmation that your key is working

### Fetch Data for Development

For development and testing, fetch a small sample:

```bash
# Fetch 100 providers and locations (basic data only)
python scripts/generate_sample_data.py \
    --mode api \
    --api-key YOUR_API_KEY \
    --limit 100

# Fetch with detailed information (slower but more complete)
python scripts/generate_sample_data.py \
    --mode api \
    --api-key YOUR_API_KEY \
    --limit 50 \
    --fetch-details
```

### Fetch Data for ML Training

To get a good sample for ML training with ratings:

```bash
# Fetch 1000 locations that have ratings
python scripts/fetch_cqc_api_data.py \
    --api-key YOUR_API_KEY \
    --ml-sample \
    --sample-size 1000
```

### Fetch Complete Dataset

**Warning**: The complete dataset contains ~50,000+ locations and will take several hours to download.

```bash
# Fetch all data (basic)
python scripts/generate_sample_data.py \
    --mode api \
    --api-key YOUR_API_KEY

# Fetch all data with details (not recommended - will take days)
python scripts/generate_sample_data.py \
    --mode api \
    --api-key YOUR_API_KEY \
    --fetch-details
```

## Data Structure

The CQC API returns data in the following structure:

### Providers
```json
{
  "providerId": "1-101641535",
  "name": "Example Care Provider Ltd",
  "type": "Organisation",
  "ownershipType": "Organisation",
  "registrationStatus": "Registered",
  "registrationDate": "2013-04-01",
  "locationIds": ["1-234567890", "1-234567891"]
}
```

### Locations (Basic)
```json
{
  "locationId": "1-234567890",
  "providerId": "1-101641535",
  "name": "Example Care Home",
  "type": "Social Care Org",
  "registrationStatus": "Registered",
  "currentRatings": {
    "overall": {
      "rating": "Good",
      "reportDate": "2023-06-15"
    }
  }
}
```

### Locations (Detailed)
The detailed endpoint includes:
- All rating categories (Safe, Effective, Caring, Responsive, Well-led)
- Regulated activities
- Specialisms
- Service types
- Inspection history
- Contact information
- And much more...

## Integration with ETL Pipeline

Once you've fetched the data, you can process it through the ETL pipeline:

```bash
# Set environment variables
export GCP_PROJECT=your-project-id
export GCS_BUCKET=your-bucket-name

# Run the ETL pipeline on API data
python src/etl/dataflow_pipeline.py \
    --input-path data/api/locations_api.json \
    --output-dataset your_dataset \
    --output-table locations_processed
```

## Using API Data vs Sample Data

### When to use API data:
- Production deployments
- Training final ML models
- Testing with real-world data distributions
- Compliance and regulatory requirements

### When to use sample data:
- Initial development
- Testing data pipelines
- CI/CD pipelines
- When API access is not available

## Best Practices

1. **Cache API responses**: Save fetched data locally to avoid repeated API calls
2. **Respect rate limits**: Use delays between requests
3. **Handle errors gracefully**: The API may return errors for some records
4. **Use incremental updates**: After initial fetch, only get updated records
5. **Monitor API usage**: Track your request count to avoid hitting limits

## Troubleshooting

### Authentication Error (401)
- Check your API key is correct
- Ensure your key is activated (check email confirmation)
- Make sure you're including the key in the header correctly

### Rate Limit Error (429)
- You've exceeded the hourly limit
- Wait for the time specified in the `Retry-After` header
- Consider applying for partner access if you need higher limits

### Timeout Errors
- The API can be slow for detailed requests
- Use `--limit` to fetch smaller batches
- Increase timeout values in the script if needed

### Missing Data
- Not all locations have ratings
- Some fields may be null or missing
- Use the detailed endpoint for complete information

## Example Workflow

Here's a complete workflow from API to ML training:

```bash
# 1. Test connection
python scripts/fetch_cqc_api_data.py --api-key YOUR_KEY --test

# 2. Fetch sample for development
python scripts/generate_sample_data.py \
    --mode api \
    --api-key YOUR_KEY \
    --limit 1000 \
    --output-dir data/dev

# 3. Process through ETL
python src/etl/dataflow_pipeline.py \
    --input-path data/dev/api/locations_api.json \
    --output-dataset cqc_dev \
    --output-table locations_features

# 4. Train ML model
python src/ml/pipeline/pipeline.py \
    --dataset cqc_dev \
    --table locations_features \
    --model-name cqc-predictor-dev

# 5. Test predictions
python src/prediction/test_local.py \
    --model-path models/cqc-predictor-dev
```

## Security Notes

1. **Never commit API keys**: Add them to `.gitignore`
2. **Use environment variables**: Store keys in `.env` files locally
3. **Use Secret Manager**: In production, use GCP Secret Manager
4. **Rotate keys regularly**: Change your API key periodically
5. **Monitor access**: Check CQC portal for API usage stats