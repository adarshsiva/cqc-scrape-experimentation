#!/bin/bash

# CQC Data Fetcher Runner Script

# Default values
DEFAULT_PROJECT="your-project-id"
DEFAULT_BUCKET="cqc-data-raw"
DEFAULT_MAX_LOCATIONS="100"

# Use environment variables or defaults
export GCP_PROJECT="${GCP_PROJECT:-$DEFAULT_PROJECT}"
export GCS_BUCKET="${GCS_BUCKET:-$DEFAULT_BUCKET}"
export MAX_LOCATIONS="${MAX_LOCATIONS:-$DEFAULT_MAX_LOCATIONS}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}CQC Data Fetcher${NC}"
echo "=================="
echo -e "Project ID: ${YELLOW}$GCP_PROJECT${NC}"
echo -e "Bucket Name: ${YELLOW}$GCS_BUCKET${NC}"
echo -e "Max Locations: ${YELLOW}$MAX_LOCATIONS${NC}"
echo ""

# Check if running in test mode
if [ "$1" == "test" ]; then
    echo -e "${YELLOW}Running in test mode...${NC}"
    python test_cqc_fetcher.py
    exit $?
fi

# Check for required environment
if [ "$GCP_PROJECT" == "your-project-id" ]; then
    echo -e "${RED}Error: Please set GCP_PROJECT environment variable${NC}"
    echo "Example: export GCP_PROJECT=my-gcp-project"
    exit 1
fi

# Install dependencies if needed
if ! python -c "import requests" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
fi

# Run the fetcher
echo -e "${GREEN}Starting CQC data fetch...${NC}"
python fetch_detailed_cqc_data.py

# Check exit status
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Data fetch completed successfully!${NC}"
else
    echo -e "${RED}Data fetch failed. Check the logs for details.${NC}"
    exit 1
fi