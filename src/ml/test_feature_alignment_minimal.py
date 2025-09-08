"""
Minimal test for Feature Alignment Service without external dependencies
"""

import sys
import os
import logging

# Mock the Google Cloud dependencies
class MockBigQueryClient:
    def __init__(self, project=None):
        pass

# Mock the google cloud module
import types
google = types.ModuleType('google')
google.cloud = types.ModuleType('google.cloud')
google.cloud.bigquery = types.ModuleType('google.cloud.bigquery')
google.cloud.bigquery.Client = MockBigQueryClient

sys.modules['google'] = google
sys.modules['google.cloud'] = google.cloud
sys.modules['google.cloud.bigquery'] = google.cloud.bigquery

# Now import the service
sys.path.append(os.path.dirname(__file__))
from feature_alignment import FeatureAlignmentService

def test_feature_alignment():
    """Test the Feature Alignment Service with sample data"""
    
    print("Testing Feature Alignment & Transformation Service")
    print("=" * 60)
    
    # Initialize service
    service = FeatureAlignmentService(project_id='test-project')
    
    # Sample dashboard features
    sample_dashboard_features = {
        'bed_capacity': 45,
        'occupancy_rate': 0.88,
        'facility_size_numeric': 3,
        'avg_care_complexity': 2.3,
        'activity_variety_score': 0.85,
        'incident_frequency_risk': 0.15,
        'medication_risk': 0.08,
        'safeguarding_risk': 0.0,
        'falls_risk': 0.12,
        'care_plan_overdue_risk': 0.05,
        'care_plan_compliance': 0.92,
        'resident_engagement': 0.78,
        'staff_compliance_score': 0.94,
        'staff_training_current': 0.88,
        'staff_incident_response': 0.3,
        'care_goal_achievement': 0.82,
        'social_isolation_risk': 0.15,
        'days_since_last_incident': 45,
        'postcode': 'SW1A 1AA',
        'regional_avg_beds': 38.0
    }
    
    print("\nInput Dashboard Features:")
    print("-" * 30)
    for key, value in sample_dashboard_features.items():
        print(f"{key}: {value}")
    
    # Transform features
    try:
        aligned_features = service.transform_dashboard_to_cqc_features(sample_dashboard_features)
        
        print(f"\nTransformed CQC-Compatible Features ({len(aligned_features)} features):")
        print("-" * 50)
        for key, value in sorted(aligned_features.items()):
            print(f"{key}: {value:.4f}")
        
        # Test individual methods
        print(f"\nTesting Individual Methods:")
        print("-" * 30)
        
        # Service complexity
        complexity = service._calculate_service_complexity(2.3, 0.85)
        print(f"Service Complexity Score: {complexity:.4f}")
        
        # Inspection risk
        inspection_risk = service._transform_to_inspection_risk(0.15, 0.05)
        print(f"Inspection Risk Score: {inspection_risk:.4f}")
        
        # Provider rating
        provider_rating = service._estimate_provider_rating(sample_dashboard_features)
        print(f"Estimated Provider Rating: {provider_rating:.4f}")
        
        # Regional risk
        regional_risk = service._lookup_regional_risk('SW1A 1AA')
        print(f"Regional Risk Rate: {regional_risk:.4f}")
        
        # Operational stability
        stability = service._calculate_operational_stability(sample_dashboard_features)
        print(f"Operational Stability: {stability:.4f}")
        
        # Care quality indicator
        quality = service._calculate_care_quality_indicator(sample_dashboard_features)
        print(f"Care Quality Indicator: {quality:.4f}")
        
        print(f"\nFeature Importance Mapping:")
        print("-" * 30)
        importance_map = service.get_feature_importance_mapping()
        for feature, description in list(importance_map.items())[:5]:  # Show first 5
            print(f"{feature}: {description}")
        
        print(f"\nTesting Edge Cases:")
        print("-" * 20)
        
        # Test with minimal features
        minimal_features = {'bed_capacity': 30, 'occupancy_rate': 0.8}
        minimal_aligned = service.transform_dashboard_to_cqc_features(minimal_features)
        print(f"Minimal features transformed: {len(minimal_aligned)} features generated")
        
        # Test with high-risk features
        high_risk_features = sample_dashboard_features.copy()
        high_risk_features.update({
            'incident_frequency_risk': 0.8,
            'medication_risk': 0.6,
            'safeguarding_risk': 1.0,
            'care_plan_compliance': 0.4
        })
        high_risk_aligned = service.transform_dashboard_to_cqc_features(high_risk_features)
        print(f"High-risk provider rating: {high_risk_aligned['provider_avg_rating']:.4f}")
        print(f"High-risk inspection risk: {high_risk_aligned['inspection_overdue_risk']:.4f}")
        
        print(f"\n✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during transformation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_ranges():
    """Test that all transformed features are within expected ranges"""
    print(f"\nTesting Feature Ranges:")
    print("-" * 25)
    
    service = FeatureAlignmentService(project_id='test-project')
    
    # Test with extreme values
    extreme_features = {
        'bed_capacity': 500,      # Very large
        'occupancy_rate': 1.5,    # Above 100%
        'avg_care_complexity': 5.0,  # Above scale
        'incident_frequency_risk': 2.0,  # Above 1.0
        'medication_risk': -0.5,  # Below 0
        'postcode': 'INVALID'
    }
    
    aligned = service.transform_dashboard_to_cqc_features(extreme_features)
    
    # Check ranges
    range_checks = {
        'bed_capacity': (1, 200),
        'occupancy_rate': (0.0, 1.0),
        'provider_avg_rating': (1.0, 4.0),
        'service_complexity_score': (1.0, 8.0),
        'inspection_overdue_risk': (0.0, 1.0),
        'regional_risk_rate': (0.0, 1.0)
    }
    
    all_passed = True
    for feature, (min_val, max_val) in range_checks.items():
        if feature in aligned:
            value = aligned[feature]
            in_range = min_val <= value <= max_val
            status = "✅" if in_range else "❌"
            print(f"{status} {feature}: {value:.4f} (expected: {min_val}-{max_val})")
            if not in_range:
                all_passed = False
    
    return all_passed

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Run tests
    test1_passed = test_feature_alignment()
    test2_passed = test_feature_ranges()
    
    print(f"\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("🎉 All Feature Alignment Tests PASSED!")
    else:
        print("⚠️  Some tests failed. Check output above.")
    print("=" * 60)