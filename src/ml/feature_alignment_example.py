"""
Example usage of the Feature Alignment & Transformation Service

This example demonstrates how to use the FeatureAlignmentService to transform
dashboard operational data into CQC training-compatible features.
"""

import sys
import os

# Mock Google Cloud dependencies for example
class MockBigQueryClient:
    def __init__(self, project=None):
        pass

import types
google = types.ModuleType('google')
google.cloud = types.ModuleType('google.cloud')
google.cloud.bigquery = types.ModuleType('google.cloud.bigquery')
google.cloud.bigquery.Client = MockBigQueryClient
sys.modules['google'] = google
sys.modules['google.cloud'] = google.cloud
sys.modules['google.cloud.bigquery'] = google.cloud.bigquery

from feature_alignment import FeatureAlignmentService


def example_care_home_prediction():
    """
    Example: Transform care home dashboard data for CQC rating prediction
    """
    print("🏥 CQC Rating Prediction - Feature Alignment Example")
    print("=" * 60)
    
    # Initialize the service
    service = FeatureAlignmentService(project_id='your-project-id')
    
    # Example 1: High-performing care home
    print("\n📊 Example 1: High-performing care home")
    print("-" * 40)
    
    high_performing_home = {
        'bed_capacity': 55,
        'occupancy_rate': 0.92,
        'facility_size_numeric': 3,  # Large facility
        'avg_care_complexity': 2.1,  # Moderate complexity
        'activity_variety_score': 0.95,  # Excellent variety
        
        # Low risk indicators
        'incident_frequency_risk': 0.05,
        'medication_risk': 0.02,
        'safeguarding_risk': 0.0,
        'falls_risk': 0.08,
        'care_plan_overdue_risk': 0.01,
        
        # High quality metrics
        'care_plan_compliance': 0.97,
        'resident_engagement': 0.89,
        'staff_compliance_score': 0.96,
        'staff_training_current': 0.94,
        'staff_incident_response': 0.2,  # Fast response
        'care_goal_achievement': 0.91,
        'social_isolation_risk': 0.08,  # Low isolation
        
        # Temporal factors
        'days_since_last_incident': 120,  # Long time since incident
        
        # Location
        'postcode': 'SW1A 1AA',  # Central London
        'regional_avg_beds': 42.0
    }
    
    aligned_features = service.transform_dashboard_to_cqc_features(high_performing_home)
    
    print(f"Provider Rating Estimate: {aligned_features['provider_avg_rating']:.2f}/4.0")
    print(f"Care Quality Score: {aligned_features['care_quality_indicator']:.2f}")
    print(f"Inspection Risk: {aligned_features['inspection_overdue_risk']:.2f}")
    print(f"Operational Stability: {aligned_features['operational_stability']:.2f}")
    print(f"Service Complexity: {aligned_features['service_complexity_score']:.1f}/8.0")
    
    # Example 2: At-risk care home
    print("\n⚠️  Example 2: At-risk care home")
    print("-" * 40)
    
    at_risk_home = {
        'bed_capacity': 28,
        'occupancy_rate': 0.75,  # Low occupancy
        'facility_size_numeric': 1,  # Small facility
        'avg_care_complexity': 2.8,  # High complexity
        'activity_variety_score': 0.45,  # Limited activities
        
        # High risk indicators
        'incident_frequency_risk': 0.65,
        'medication_risk': 0.35,
        'safeguarding_risk': 0.8,  # Safeguarding concerns
        'falls_risk': 0.45,
        'care_plan_overdue_risk': 0.25,
        
        # Lower quality metrics
        'care_plan_compliance': 0.68,
        'resident_engagement': 0.52,
        'staff_compliance_score': 0.71,
        'staff_training_current': 0.65,
        'staff_incident_response': 0.8,  # Slow response
        'care_goal_achievement': 0.58,
        'social_isolation_risk': 0.35,  # Higher isolation
        
        # Temporal factors
        'days_since_last_incident': 12,  # Recent incident
        
        # Location
        'postcode': 'M1 1AA',  # Manchester
        'regional_avg_beds': 32.0
    }
    
    at_risk_features = service.transform_dashboard_to_cqc_features(at_risk_home)
    
    print(f"Provider Rating Estimate: {at_risk_features['provider_avg_rating']:.2f}/4.0")
    print(f"Care Quality Score: {at_risk_features['care_quality_indicator']:.2f}")
    print(f"Inspection Risk: {at_risk_features['inspection_overdue_risk']:.2f}")
    print(f"Operational Stability: {at_risk_features['operational_stability']:.2f}")
    print(f"Service Complexity: {at_risk_features['service_complexity_score']:.1f}/8.0")
    
    # Compare the two homes
    print(f"\n🔍 Comparison Summary")
    print("-" * 25)
    
    rating_diff = aligned_features['provider_avg_rating'] - at_risk_features['provider_avg_rating']
    quality_diff = aligned_features['care_quality_indicator'] - at_risk_features['care_quality_indicator']
    risk_diff = at_risk_features['inspection_overdue_risk'] - aligned_features['inspection_overdue_risk']
    
    print(f"Rating Difference: {rating_diff:.2f} points higher for high-performing home")
    print(f"Quality Difference: {quality_diff:.2f} points higher for high-performing home")
    print(f"Risk Difference: {risk_diff:.2f} points higher risk for at-risk home")
    
    return aligned_features, at_risk_features


def demonstrate_feature_mapping():
    """
    Demonstrate the feature mapping capabilities
    """
    print(f"\n🗺️  Feature Mapping Demonstration")
    print("=" * 40)
    
    service = FeatureAlignmentService()
    
    # Show feature importance mapping
    mapping = service.get_feature_importance_mapping()
    
    print("Dashboard → CQC Feature Mappings:")
    print("-" * 35)
    
    for i, (feature, description) in enumerate(mapping.items(), 1):
        print(f"{i:2d}. {feature:<30} → {description}")
    
    print(f"\nTotal mapped features: {len(mapping)}")


def demonstrate_edge_cases():
    """
    Demonstrate how the service handles edge cases
    """
    print(f"\n🔧 Edge Case Handling")
    print("=" * 25)
    
    service = FeatureAlignmentService()
    
    # Test 1: Minimal data
    print("Test 1: Minimal dashboard data")
    minimal_data = {
        'bed_capacity': 20,
        'occupancy_rate': 0.5
    }
    
    minimal_features = service.transform_dashboard_to_cqc_features(minimal_data)
    print(f"  Features generated from minimal data: {len(minimal_features)}")
    print(f"  Provider rating (with defaults): {minimal_features['provider_avg_rating']:.2f}")
    
    # Test 2: Extreme values
    print("\nTest 2: Extreme values (should be normalized)")
    extreme_data = {
        'bed_capacity': 1000,  # Very large
        'occupancy_rate': 2.0,  # Over 100%
        'incident_frequency_risk': 5.0,  # Way above 1.0
        'avg_care_complexity': -1.0  # Below minimum
    }
    
    extreme_features = service.transform_dashboard_to_cqc_features(extreme_data)
    print(f"  Normalized bed capacity: {extreme_features['bed_capacity']}")
    print(f"  Normalized occupancy rate: {extreme_features['occupancy_rate']}")
    print(f"  Normalized incident risk: {extreme_features['incident_frequency_risk']}")
    
    # Test 3: Missing postcode
    print("\nTest 3: Invalid/missing postcode")
    no_postcode = {
        'bed_capacity': 30,
        'occupancy_rate': 0.8,
        'postcode': 'INVALID123'
    }
    
    no_pc_features = service.transform_dashboard_to_cqc_features(no_postcode)
    print(f"  Regional risk (default): {no_pc_features['regional_risk_rate']}")


if __name__ == "__main__":
    # Run examples
    high_perf, at_risk = example_care_home_prediction()
    demonstrate_feature_mapping()
    demonstrate_edge_cases()
    
    print(f"\n✨ Feature Alignment Examples Complete!")
    print("Ready for integration with CQC prediction models.")