#!/usr/bin/env python3
import subprocess
import sys
import os

# Test the training script locally
def test_local_training():
    print("Testing ML training pipeline locally...")
    
    # Set up environment
    project_id = "machine-learning-exp-467008"
    input_path = "gs://machine-learning-exp-467008-cqc-ml-artifacts/training_data/"
    output_path = "gs://machine-learning-exp-467008-cqc-ml-artifacts/models/test/"
    
    # Run training script
    cmd = [
        sys.executable,
        "src/ml/trainer/task.py",
        "--project-id", project_id,
        "--input-path", input_path,
        "--output-path", output_path,
        "--model-type", "xgboost"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Training completed successfully!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    if test_local_training():
        print("\nLocal training test passed! Ready for Vertex AI deployment.")
    else:
        print("\nLocal training test failed. Please fix errors before deploying.")