#!/bin/bash
# Run Proactive Risk Assessment API locally for development

echo "Starting Proactive Risk Assessment API locally..."

# Set environment variables
export GCP_PROJECT="machine-learning-exp-467008"
export MODEL_BUCKET="machine-learning-exp-467008-cqc-ml-artifacts"
export MODEL_PATH="models/proactive/model_package.pkl"
export PORT=8080

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "venv_proactive" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_proactive
fi

# Activate virtual environment
source venv_proactive/bin/activate

# Install/update dependencies
echo "Installing dependencies..."
pip install -r requirements_proactive.txt

# Run the application
echo "Starting Flask application on port ${PORT}..."
echo "API will be available at http://localhost:${PORT}"
echo "Press Ctrl+C to stop"
echo ""

python proactive_predictor.py