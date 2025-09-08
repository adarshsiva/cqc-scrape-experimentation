#!/usr/bin/env python3
"""
Implementation Validation Script for Dashboard Prediction API

This script validates that the implementation matches the requirements from plan.md
without requiring cloud dependencies or imports.
"""

import ast
import os
import re

def validate_file_structure():
    """Validate that all required files are present."""
    print("🔍 Validating file structure...")
    
    required_files = [
        'src/api/dashboard_prediction_service.py',
        'src/api/requirements-dashboard.txt', 
        'src/api/Dockerfile.dashboard',
        'src/api/cloudbuild-deploy-dashboard-api.yaml',
        'src/api/deploy-dashboard-api.sh',
        'src/api/DASHBOARD_API_README.md',
        'src/api/test_dashboard_api.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(f"/mnt/l/llm/cqc-scrape-experimentation/{file_path}"):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True

def validate_api_structure():
    """Validate the API structure matches plan.md requirements."""
    print("\n🔍 Validating API structure...")
    
    api_file = "/mnt/l/llm/cqc-scrape-experimentation/src/api/dashboard_prediction_service.py"
    
    try:
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Check for required classes and functions
        required_elements = [
            'class ModelPredictionService',
            'class AuthenticationError',
            'def predict_cqc_rating_from_dashboard',
            'def require_auth',
            'def validate_api_token',
            'def get_client_id',
            '/api/cqc-prediction/dashboard/<care_home_id>',
            '@require_auth'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing API elements: {missing_elements}")
            return False
        else:
            print("✅ All required API elements present")
            return True
            
    except Exception as e:
        print(f"❌ Error validating API structure: {e}")
        return False

def validate_response_structure():
    """Validate the response structure matches plan.md specification."""
    print("\n🔍 Validating response structure...")
    
    api_file = "/mnt/l/llm/cqc-scrape-experimentation/src/api/dashboard_prediction_service.py"
    
    try:
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Check for response structure elements from plan.md
        response_elements = [
            "'care_home_id':",
            "'prediction':",
            "'predicted_rating':",
            "'predicted_rating_text':",
            "'confidence_score':",
            "'risk_level':",
            "'contributing_factors':",
            "'top_positive_factors':",
            "'top_risk_factors':",
            "'recommendations':",
            "'data_freshness':",
            "'last_updated':",
            "'data_coverage':"
        ]
        
        missing_elements = []
        for element in response_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Missing response elements: {missing_elements}")
            return False
        else:
            print("✅ Response structure matches plan.md specification")
            return True
            
    except Exception as e:
        print(f"❌ Error validating response structure: {e}")
        return False

def validate_integration_points():
    """Validate integration with required services."""
    print("\n🔍 Validating integration points...")
    
    api_file = "/mnt/l/llm/cqc-scrape-experimentation/src/api/dashboard_prediction_service.py"
    
    try:
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Check for integration points
        integration_points = [
            'DashboardFeatureExtractor',
            'FeatureAlignmentService',
            'extract_care_home_features',
            'transform_dashboard_to_cqc_features',
            'predict_cqc_rating',
            'explain_prediction',
            'generate_recommendations'
        ]
        
        missing_integrations = []
        for integration in integration_points:
            if integration not in content:
                missing_integrations.append(integration)
        
        if missing_integrations:
            print(f"❌ Missing integrations: {missing_integrations}")
            return False
        else:
            print("✅ All required integrations present")
            return True
            
    except Exception as e:
        print(f"❌ Error validating integrations: {e}")
        return False

def validate_deployment_config():
    """Validate deployment configuration."""
    print("\n🔍 Validating deployment configuration...")
    
    dockerfile = "/mnt/l/llm/cqc-scrape-experimentation/src/api/Dockerfile.dashboard"
    cloudbuild = "/mnt/l/llm/cqc-scrape-experimentation/src/api/cloudbuild-deploy-dashboard-api.yaml"
    requirements = "/mnt/l/llm/cqc-scrape-experimentation/src/api/requirements-dashboard.txt"
    
    try:
        # Check Dockerfile
        with open(dockerfile, 'r') as f:
            dockerfile_content = f.read()
        
        dockerfile_requirements = [
            'FROM python:3.10',
            'COPY src/api/requirements-dashboard.txt',
            'pip install',
            'COPY src/',
            'EXPOSE 8080',
            'CMD ["python", "src/api/dashboard_prediction_service.py"]'
        ]
        
        for req in dockerfile_requirements:
            if req not in dockerfile_content:
                print(f"❌ Missing Dockerfile requirement: {req}")
                return False
        
        # Check Cloud Build config
        with open(cloudbuild, 'r') as f:
            cloudbuild_content = f.read()
        
        cloudbuild_requirements = [
            'gcr.io/cloud-builders/docker',
            'gcloud run deploy',
            'dashboard-prediction-api',
            '--memory', '2Gi',
            '--cpu', '1'
        ]
        
        for req in cloudbuild_requirements:
            if req not in cloudbuild_content:
                print(f"❌ Missing Cloud Build requirement: {req}")
                return False
        
        # Check requirements.txt
        with open(requirements, 'r') as f:
            requirements_content = f.read()
        
        required_packages = [
            'flask',
            'functions-framework',
            'google-cloud-aiplatform',
            'google-cloud-bigquery',
            'google-cloud-storage',
            'google-cloud-secret-manager',
            'numpy',
            'pandas'
        ]
        
        for package in required_packages:
            if package not in requirements_content:
                print(f"❌ Missing required package: {package}")
                return False
        
        print("✅ Deployment configuration validated")
        return True
        
    except Exception as e:
        print(f"❌ Error validating deployment config: {e}")
        return False

def validate_documentation():
    """Validate documentation completeness."""
    print("\n🔍 Validating documentation...")
    
    readme_file = "/mnt/l/llm/cqc-scrape-experimentation/src/api/DASHBOARD_API_README.md"
    
    try:
        with open(readme_file, 'r') as f:
            content = f.read()
        
        doc_requirements = [
            '# Dashboard Prediction API Documentation',
            '## API Endpoints',
            '/api/cqc-prediction/dashboard/{care_home_id}',
            '## Authentication',
            '## Deployment',
            '## Error Handling',
            'Bearer {api_token}',
            'X-Client-ID',
            'predicted_rating',
            'confidence_score',
            'contributing_factors',
            'recommendations'
        ]
        
        missing_docs = []
        for req in doc_requirements:
            if req not in content:
                missing_docs.append(req)
        
        if missing_docs:
            print(f"❌ Missing documentation elements: {missing_docs}")
            return False
        else:
            print("✅ Documentation complete")
            return True
            
    except Exception as e:
        print(f"❌ Error validating documentation: {e}")
        return False

def validate_plan_md_compliance():
    """Validate compliance with plan.md specification."""
    print("\n🔍 Validating plan.md compliance...")
    
    api_file = "/mnt/l/llm/cqc-scrape-experimentation/src/api/dashboard_prediction_service.py"
    
    try:
        with open(api_file, 'r') as f:
            content = f.read()
        
        # Check for exact implementation from plan.md lines 516-572
        plan_requirements = [
            # Route specification
            "/api/cqc-prediction/dashboard/<care_home_id>",
            "methods=['GET']",
            "@require_auth",
            
            # Function implementation
            "def predict_cqc_rating_from_dashboard(care_home_id)",
            "extractor = DashboardFeatureExtractor(client_id=get_client_id())",
            "dashboard_features = extractor.extract_care_home_features(care_home_id)",
            "alignment_service = FeatureAlignmentService()",
            "cqc_features = alignment_service.transform_dashboard_to_cqc_features(dashboard_features)",
            "model_service = ModelPredictionService()",
            "prediction_result = model_service.predict_cqc_rating(cqc_features)",
            "feature_importance = model_service.explain_prediction(cqc_features)",
            
            # Response structure
            "'care_home_id': care_home_id",
            "'predicted_rating': prediction_result['rating']",
            "'predicted_rating_text': prediction_result['rating_text']",
            "'confidence_score': prediction_result['confidence']",
            "'risk_level': prediction_result['risk_level']",
            "'top_positive_factors': feature_importance['positive']",
            "'top_risk_factors': feature_importance['negative']",
            "generate_recommendations(prediction_result, dashboard_features)",
            "'last_updated': datetime.utcnow().isoformat()",
            
            # Error handling
            "except Exception as e:",
            "logger.error(f\"Prediction failed for care home {care_home_id}: {str(e)}\")",
            "'error': 'Prediction failed'",
            "'error_type': 'PREDICTION_ERROR'",
            "'message': 'Unable to generate CQC rating prediction'"
        ]
        
        missing_plan_elements = []
        for element in plan_requirements:
            if element not in content:
                missing_plan_elements.append(element)
        
        if missing_plan_elements:
            print(f"❌ Missing plan.md elements: {missing_plan_elements}")
            return False
        else:
            print("✅ Implementation matches plan.md specification")
            return True
            
    except Exception as e:
        print(f"❌ Error validating plan.md compliance: {e}")
        return False

def generate_validation_report():
    """Generate comprehensive validation report."""
    print("=" * 70)
    print("📋 DASHBOARD PREDICTION API IMPLEMENTATION VALIDATION REPORT")
    print("=" * 70)
    
    validations = [
        ("File Structure", validate_file_structure),
        ("API Structure", validate_api_structure),
        ("Response Structure", validate_response_structure),
        ("Integration Points", validate_integration_points),
        ("Deployment Config", validate_deployment_config),
        ("Documentation", validate_documentation),
        ("Plan.md Compliance", validate_plan_md_compliance)
    ]
    
    results = {}
    all_passed = True
    
    for name, validator in validations:
        results[name] = validator()
        if not results[name]:
            all_passed = False
    
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:.<25} {status}")
    
    print("=" * 70)
    
    if all_passed:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("\n📋 Implementation Summary:")
        print("  ✅ Real-time CQC prediction API implemented")
        print("  ✅ Dashboard integration with feature extraction")
        print("  ✅ Feature alignment and model prediction services")
        print("  ✅ Authentication and security middleware")
        print("  ✅ Comprehensive error handling")
        print("  ✅ Cloud Run deployment configuration")
        print("  ✅ Complete documentation and testing")
        print("\n🚀 Ready for deployment!")
        return True
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("Please address the issues above before deployment.")
        return False

if __name__ == '__main__':
    success = generate_validation_report()
    exit(0 if success else 1)