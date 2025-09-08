# CQC Rating Predictor - Care Home Dashboard Integration Plan

*Last Updated: 2025-09-08*

## 🎯 System Overview

**Goal**: Train ML models on comprehensive CQC Syndication API data and deploy for real-time predictions using live care home dashboard data.

**Architecture**:
- **Training Data**: CQC Syndication API (all endpoints) → Real inspection ratings & outcomes
- **Prediction Data**: Care Home Dashboard (EAV system) → Live operational metrics
- **Output**: CQC rating predictions for care homes using their own operational data

---

## 📊 Data Source Analysis

### CQC Syndication API Training Data (Source of Truth)

#### Available Endpoints:
1. **Get Location By Id** → Detailed facility information
2. **Get Locations** → Bulk location listings
3. **Get Provider By Id** → Provider-level data
4. **Get Providers** → Bulk provider listings
5. **Get Location AssessmentServiceGroups** → Service complexity metrics
6. **Get Provider AssessmentServiceGroups** → Provider service patterns
7. **Get Location Inspection Areas** → Domain-specific ratings (Safe, Effective, Caring, etc.)
8. **Get Provider Inspection Areas By Location Id** → Historical inspection data
9. **Get Provider Inspection Areas By Provider Id** → Provider-level inspection patterns
10. **Get Inspection Areas** → Rating methodology data
11. **Get Reports** → Detailed inspection reports
12. **Get Changes Within Timeframe** → Recent updates

#### Key Training Features from CQC API:
```json
{
  "temporal_features": {
    "days_since_inspection": "From lastInspectionDate",
    "days_since_registration": "From registrationDate", 
    "inspection_overdue_risk": "Calculated risk score"
  },
  "operational_features": {
    "bed_capacity": "numberOfBeds",
    "facility_size": "Categorized bed count",
    "service_complexity": "regulatedActivities + gacServiceTypes + specialisms"
  },
  "quality_indicators": {
    "overall_rating": "TARGET VARIABLE (1-4 scale)",
    "domain_ratings": "Safe, Effective, Caring, Responsive, WellLed",
    "historical_trends": "Rating changes over time"
  },
  "geographic_risk": {
    "region": "Geographic location",
    "local_authority": "Local oversight area",
    "regional_risk_rate": "Regional poor performance rate"
  },
  "provider_patterns": {
    "provider_location_count": "Multi-location providers",
    "provider_avg_rating": "Provider performance",
    "provider_consistency": "Rating variance across locations"
  }
}
```

### Care Home Dashboard Prediction Data (Live Operations)

#### Available Dashboard Tables:
```sql
-- EAV Core System
entities            -- Care homes, residents, staff
attributes          -- Custom fields per entity type  
entity_values       -- Dynamic attribute values

-- Operational Data
residents           -- Current residents, care levels
incidents           -- Falls, medication errors, safeguarding
care_plans          -- Individual care plans, reviews
activities          -- Daily activities, engagement
users               -- Staff information, roles
audit_logs          -- All system actions, compliance

-- Vendor Integration
vendors             -- External system connections
vendor_mappings     -- Data field mappings
sync_logs           -- Data synchronization history
```

#### Dashboard-to-CQC Feature Mapping:
```python
dashboard_to_cqc_features = {
    # Operational Metrics
    'bed_capacity': 'COUNT(residents WHERE status=active)',
    'occupancy_rate': 'active_residents / total_capacity',
    'care_complexity': 'AVG(residents.care_level_numeric)',
    
    # Risk Indicators  
    'incident_risk_score': 'incident_frequency * severity_weights',
    'falls_per_resident': 'COUNT(incidents WHERE type=fall) / resident_count',
    'medication_error_rate': 'med_errors / total_med_administrations',
    'safeguarding_concerns': 'COUNT(incidents WHERE type=safeguarding)',
    
    # Care Quality Metrics
    'care_plan_compliance': 'on_time_reviews / total_care_plans',
    'care_plan_overdue_risk': 'overdue_reviews / total_care_plans',
    'care_goal_achievement': 'goals_met / total_goals',
    
    # Staff Performance
    'staff_incident_response': 'AVG(incident_resolution_time)',
    'staff_compliance_score': 'audit_compliance / total_audits',
    'staff_training_current': 'current_certifications / required_certifications',
    
    # Engagement & Activities
    'resident_engagement': 'activity_participation / total_activities',
    'social_isolation_risk': 'low_participation_residents / total_residents',
    'activity_variety_score': 'unique_activity_types / total_activities',
    
    # Temporal Features
    'days_since_last_incident': 'DATE_DIFF(NOW(), MAX(incident_date))',
    'care_plan_review_frequency': 'AVG(days_between_reviews)',
    'operational_stability': 'STDDEV(daily_incident_count)'
}
```

---

## 🏗️ Implementation Architecture

### Phase 1: Comprehensive CQC Data Collection

#### 1.1 Enhanced CQC API Data Extraction
```bash
# Fetch comprehensive training dataset
gcloud run jobs execute cqc-comprehensive-extractor \
  --region europe-west2 \
  --update-env-vars="
    ENDPOINTS=locations,providers,inspection_areas,reports,assessment_groups,
    MAX_LOCATIONS=50000,
    INCLUDE_HISTORICAL=true,
    FETCH_REPORTS=true,
    RATE_LIMIT=1800,
    PARALLEL_WORKERS=10" \
  --task-timeout=21600 --wait
```

#### 1.2 Comprehensive Feature Engineering
```sql
CREATE OR REPLACE TABLE `cqc_data.ml_training_features_comprehensive` AS
WITH location_data AS (
  -- Core location features from Get Location By Id
  SELECT 
    locationId, name, providerId,
    numberOfBeds, registrationDate, lastInspectionDate,
    overall_rating, safe_rating, effective_rating,
    caring_rating, responsive_rating, well_led_rating,
    region, localAuthority, organisationType,
    ARRAY_LENGTH(regulatedActivities) as service_complexity,
    DATE_DIFF(CURRENT_DATE(), DATE(lastInspectionDate), DAY) as days_since_inspection
  FROM `cqc_data.locations_comprehensive`
),
inspection_history AS (
  -- Historical patterns from Get Provider Inspection Areas  
  SELECT 
    locationId,
    COUNT(*) as inspection_count,
    AVG(overall_rating_numeric) as historical_avg_rating,
    STDDEV(overall_rating_numeric) as rating_volatility,
    COUNT(DISTINCT DATE(inspectionDate)) as unique_inspection_dates
  FROM `cqc_data.inspection_areas_history`
  GROUP BY locationId
),
service_assessment AS (
  -- Service complexity from Get Location AssessmentServiceGroups
  SELECT
    locationId,
    COUNT(DISTINCT serviceGroup) as service_group_count,
    COUNT(DISTINCT assessmentType) as assessment_type_count,
    AVG(riskScore) as avg_risk_score
  FROM `cqc_data.assessment_service_groups`
  GROUP BY locationId
),
provider_context AS (
  -- Provider-level patterns from Get Provider By Id
  SELECT
    providerId,
    COUNT(DISTINCT locationId) as provider_location_count,
    AVG(overall_rating_numeric) as provider_avg_rating,
    COUNT(DISTINCT region) as provider_geographic_spread,
    AVG(numberOfBeds) as provider_avg_capacity
  FROM location_data
  GROUP BY providerId
)

SELECT 
  l.*,
  -- Historical Context
  COALESCE(ih.inspection_count, 1) as inspection_history_count,
  COALESCE(ih.historical_avg_rating, 3.0) as historical_performance,
  COALESCE(ih.rating_volatility, 0.5) as performance_consistency,
  
  -- Service Complexity
  COALESCE(sa.service_group_count, 3) as service_diversity,
  COALESCE(sa.assessment_type_count, 5) as assessment_complexity,
  COALESCE(sa.avg_risk_score, 0.3) as inherent_risk_score,
  
  -- Provider Context
  COALESCE(pc.provider_location_count, 1) as provider_scale,
  COALESCE(pc.provider_avg_rating, 3.0) as provider_reputation,
  COALESCE(pc.provider_geographic_spread, 1) as provider_diversity,
  
  -- Risk Indicators
  CASE WHEN days_since_inspection > 730 THEN 1 ELSE 0 END as inspection_overdue,
  CASE WHEN ih.rating_volatility > 1.0 THEN 1 ELSE 0 END as performance_unstable,
  
  -- Target Variable
  CASE 
    WHEN overall_rating = 'Outstanding' THEN 4
    WHEN overall_rating = 'Good' THEN 3  
    WHEN overall_rating = 'Requires improvement' THEN 2
    WHEN overall_rating = 'Inadequate' THEN 1
  END as overall_rating_numeric

FROM location_data l
LEFT JOIN inspection_history ih USING(locationId)
LEFT JOIN service_assessment sa USING(locationId)  
LEFT JOIN provider_context pc USING(providerId)
WHERE l.overall_rating IS NOT NULL;
```

### Phase 2: Dashboard Feature Extraction Service

#### 2.1 Dashboard Feature Calculator
```python
# src/ml/dashboard_feature_extractor.py
class DashboardFeatureExtractor:
    def __init__(self, client_id):
        self.client_id = client_id
        self.db_service = DatabaseService()
    
    def extract_care_home_features(self, care_home_entity_id):
        """Extract ML features from dashboard data"""
        
        features = {}
        
        # 1. Operational Metrics
        features.update(self._calculate_operational_metrics(care_home_entity_id))
        
        # 2. Risk Indicators  
        features.update(self._calculate_risk_indicators(care_home_entity_id))
        
        # 3. Care Quality Metrics
        features.update(self._calculate_care_quality_metrics(care_home_entity_id))
        
        # 4. Staff Performance
        features.update(self._calculate_staff_performance(care_home_entity_id))
        
        # 5. Engagement Metrics
        features.update(self._calculate_engagement_metrics(care_home_entity_id))
        
        # 6. Temporal Features
        features.update(self._calculate_temporal_features(care_home_entity_id))
        
        return features
    
    def _calculate_operational_metrics(self, care_home_id):
        """Map dashboard data to CQC operational features"""
        query = """
        WITH resident_data AS (
            SELECT 
                COUNT(*) as active_residents,
                AVG(CASE 
                    WHEN ev.value_string = 'High' THEN 3
                    WHEN ev.value_string = 'Medium' THEN 2  
                    WHEN ev.value_string = 'Low' THEN 1
                    ELSE 2
                END) as avg_care_complexity
            FROM entities e
            JOIN entity_values ev ON e.id = ev.entity_id
            JOIN attributes a ON ev.attribute_id = a.id
            WHERE e.entity_type = 'resident' 
            AND e.status = 'active'
            AND a.name = 'care_level'
            AND e.client_id = %s
        ),
        capacity_data AS (
            SELECT CAST(ev.value_integer AS SIGNED) as bed_capacity
            FROM entities e
            JOIN entity_values ev ON e.id = ev.entity_id
            JOIN attributes a ON ev.attribute_id = a.id  
            WHERE e.id = %s
            AND a.name = 'bed_capacity'
        )
        
        SELECT 
            r.active_residents,
            r.avg_care_complexity,
            c.bed_capacity,
            CASE WHEN c.bed_capacity > 0 
                 THEN r.active_residents / c.bed_capacity 
                 ELSE 0 END as occupancy_rate,
            CASE 
                WHEN c.bed_capacity >= 60 THEN 4  -- Very Large
                WHEN c.bed_capacity >= 40 THEN 3  -- Large
                WHEN c.bed_capacity >= 20 THEN 2  -- Medium
                ELSE 1  -- Small
            END as facility_size_numeric
        FROM resident_data r
        CROSS JOIN capacity_data c
        """
        
        result = self.db_service.execute_query(query, [self.client_id, care_home_id])
        if result:
            return {
                'bed_capacity': result[0]['bed_capacity'] or 30,
                'occupancy_rate': result[0]['occupancy_rate'] or 0.85,
                'avg_care_complexity': result[0]['avg_care_complexity'] or 2.0,
                'facility_size_numeric': result[0]['facility_size_numeric'] or 2
            }
        return {}
    
    def _calculate_risk_indicators(self, care_home_id):
        """Calculate incident-based risk scores"""
        query = """
        WITH incident_analysis AS (
            SELECT 
                COUNT(*) as total_incidents,
                COUNT(DISTINCT DATE(incident_date)) as incident_days,
                AVG(CASE 
                    WHEN severity = 'Critical' THEN 4
                    WHEN severity = 'High' THEN 3
                    WHEN severity = 'Medium' THEN 2
                    WHEN severity = 'Low' THEN 1
                    ELSE 2
                END) as avg_severity,
                COUNTIF(incident_type = 'Fall') as falls_count,
                COUNTIF(incident_type = 'Medication Error') as med_errors,
                COUNTIF(incident_type = 'Safeguarding') as safeguarding_count,
                MAX(incident_date) as last_incident_date,
                COUNT(*) / COUNT(DISTINCT resident_id) as incidents_per_resident
            FROM incidents 
            WHERE incident_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
            AND care_home_id = %s
        )
        
        SELECT 
            total_incidents,
            avg_severity,
            falls_count,  
            med_errors,
            safeguarding_count,
            DATEDIFF(CURRENT_DATE, last_incident_date) as days_since_last_incident,
            incidents_per_resident,
            -- Risk scores (0-1 scale)
            LEAST(total_incidents / 50.0, 1.0) as incident_frequency_risk,
            LEAST(falls_count / 20.0, 1.0) as falls_risk,
            LEAST(med_errors / 10.0, 1.0) as medication_risk,
            CASE WHEN safeguarding_count > 0 THEN 1.0 ELSE 0.0 END as safeguarding_risk
        FROM incident_analysis
        """
        
        result = self.db_service.execute_query(query, [care_home_id])
        if result and result[0]['total_incidents']:
            return {
                'incident_frequency_risk': result[0]['incident_frequency_risk'] or 0.0,
                'falls_risk': result[0]['falls_risk'] or 0.0,
                'medication_risk': result[0]['medication_risk'] or 0.0,
                'safeguarding_risk': result[0]['safeguarding_risk'] or 0.0,
                'days_since_last_incident': result[0]['days_since_last_incident'] or 365,
                'avg_incident_severity': result[0]['avg_severity'] or 2.0
            }
        return {
            'incident_frequency_risk': 0.0,
            'falls_risk': 0.0, 
            'medication_risk': 0.0,
            'safeguarding_risk': 0.0,
            'days_since_last_incident': 365,
            'avg_incident_severity': 1.0
        }
```

#### 2.2 Feature Alignment & Transformation
```python
# src/ml/feature_alignment.py
class FeatureAlignmentService:
    """Align dashboard features with CQC training features"""
    
    def __init__(self):
        self.cqc_feature_ranges = self._load_training_feature_stats()
    
    def transform_dashboard_to_cqc_features(self, dashboard_features):
        """Transform dashboard metrics to match CQC training feature space"""
        
        aligned_features = {}
        
        # Direct mappings
        aligned_features['bed_capacity'] = dashboard_features.get('bed_capacity', 30)
        
        # Derived mappings with normalization
        aligned_features['service_complexity_score'] = self._calculate_service_complexity(
            dashboard_features.get('avg_care_complexity', 2.0),
            dashboard_features.get('activity_variety_score', 0.7)
        )
        
        # Risk indicator transformations
        aligned_features['inspection_overdue_risk'] = self._transform_to_inspection_risk(
            dashboard_features.get('incident_frequency_risk', 0.0),
            dashboard_features.get('care_plan_overdue_risk', 0.0)
        )
        
        # Provider-level approximations (single location assumption)
        aligned_features['provider_location_count'] = 1
        aligned_features['provider_avg_rating'] = self._estimate_provider_rating(dashboard_features)
        aligned_features['provider_rating_consistency'] = 0.3  # Default stability
        
        # Regional risk (lookup by postcode)
        aligned_features['regional_risk_rate'] = self._lookup_regional_risk(
            dashboard_features.get('postcode', 'SW1A 1AA')
        )
        
        # Interaction features
        aligned_features['complexity_scale_interaction'] = (
            aligned_features['service_complexity_score'] * 
            aligned_features['provider_location_count']
        )
        
        aligned_features['inspection_regional_risk'] = (
            aligned_features['inspection_overdue_risk'] * 
            aligned_features['regional_risk_rate']
        )
        
        return aligned_features
    
    def _estimate_provider_rating(self, dashboard_features):
        """Estimate provider rating from dashboard metrics"""
        
        # Weighted scoring based on key indicators
        risk_score = (
            dashboard_features.get('incident_frequency_risk', 0.0) * 0.3 +
            dashboard_features.get('medication_risk', 0.0) * 0.3 +
            dashboard_features.get('safeguarding_risk', 0.0) * 0.4
        )
        
        quality_score = (
            dashboard_features.get('care_plan_compliance', 0.8) * 0.4 +
            dashboard_features.get('resident_engagement', 0.7) * 0.3 +
            dashboard_features.get('staff_compliance_score', 0.9) * 0.3  
        )
        
        # Convert to CQC scale (1-4) with 3.0 as default "Good"
        estimated_rating = 3.0 + (quality_score - risk_score) * 1.5
        return max(1.0, min(4.0, estimated_rating))
```

### Phase 3: Unified ML Pipeline

#### 3.1 Enhanced Model Training
```python
# src/ml/unified_model_trainer.py
class UnifiedCQCModelTrainer:
    def __init__(self):
        self.feature_alignment = FeatureAlignmentService()
        
    def train_comprehensive_model(self):
        """Train on comprehensive CQC data with dashboard feature validation"""
        
        # Load comprehensive CQC training data
        training_data = self._load_comprehensive_cqc_data()
        
        # Feature engineering with dashboard compatibility
        features, targets = self._prepare_unified_features(training_data)
        
        # Train ensemble models
        models = self._train_ensemble_models(features, targets)
        
        # Validate feature alignment with sample dashboard data
        self._validate_feature_alignment(models)
        
        return models
    
    def _prepare_unified_features(self, training_data):
        """Prepare features compatible with both CQC and dashboard data"""
        
        feature_columns = [
            # Core operational (available in both)
            'bed_capacity', 'facility_size_numeric', 'occupancy_rate',
            
            # Risk indicators (CQC: historical, Dashboard: current)
            'inspection_overdue_risk', 'incident_frequency_risk', 
            'medication_risk', 'safeguarding_risk',
            
            # Quality metrics (CQC: ratings, Dashboard: compliance)
            'service_complexity_score', 'care_quality_indicator',
            
            # Temporal features
            'days_since_inspection', 'operational_stability',
            
            # Provider context
            'provider_location_count', 'provider_avg_rating',
            'provider_rating_consistency',
            
            # Regional context
            'regional_risk_rate', 'regional_avg_beds',
            
            # Interaction features
            'complexity_scale_interaction', 'inspection_regional_risk'
        ]
        
        # Extract and transform features
        X = training_data[feature_columns].fillna(method='ffill')
        y = training_data['overall_rating_numeric']
        
        return X, y
```

#### 3.2 Real-time Prediction API
```python
# src/api/dashboard_prediction_service.py
@app.route('/api/cqc-prediction/dashboard/<care_home_id>', methods=['GET'])
@require_auth
def predict_cqc_rating_from_dashboard(care_home_id):
    """Real-time CQC rating prediction using dashboard data"""
    
    try:
        # Extract features from dashboard
        extractor = DashboardFeatureExtractor(client_id=get_client_id())
        dashboard_features = extractor.extract_care_home_features(care_home_id)
        
        # Transform to CQC feature space
        alignment_service = FeatureAlignmentService()
        cqc_features = alignment_service.transform_dashboard_to_cqc_features(dashboard_features)
        
        # Load trained model
        model_service = ModelPredictionService()
        prediction_result = model_service.predict_cqc_rating(cqc_features)
        
        # Enhance with explanations
        feature_importance = model_service.explain_prediction(cqc_features)
        
        response = {
            'care_home_id': care_home_id,
            'prediction': {
                'predicted_rating': prediction_result['rating'],
                'predicted_rating_text': prediction_result['rating_text'],
                'confidence_score': prediction_result['confidence'],
                'risk_level': prediction_result['risk_level']
            },
            'contributing_factors': {
                'top_positive_factors': feature_importance['positive'][:3],
                'top_risk_factors': feature_importance['negative'][:3],
                'operational_score': dashboard_features.get('operational_score', 0.8),
                'quality_score': dashboard_features.get('care_quality_score', 0.7),
                'risk_score': dashboard_features.get('overall_risk_score', 0.2)
            },
            'recommendations': model_service.generate_recommendations(prediction_result, dashboard_features),
            'data_freshness': {
                'last_updated': datetime.utcnow().isoformat(),
                'data_coverage': self._calculate_data_coverage(dashboard_features)
            }
        }
        
        # Store prediction for tracking
        self._store_prediction_result(care_home_id, response)
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction failed for care home {care_home_id}: {str(e)}")
        return jsonify({
            'error': 'Prediction failed',
            'error_type': 'PREDICTION_ERROR',
            'message': 'Unable to generate CQC rating prediction'
        }), 500
```

### Phase 4: Dashboard Integration

#### 4.1 CQC Prediction Dashboard Widget
```javascript
// frontend/src/components/CQCPredictionWidget.js
const CQCPredictionWidget = ({ careHomeId }) => {
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(true);
    
    useEffect(() => {
        fetchPrediction();
    }, [careHomeId]);
    
    const fetchPrediction = async () => {
        try {
            const response = await makeAuthenticatedRequest(
                `/api/cqc-prediction/dashboard/${careHomeId}`
            );
            setPrediction(response.data);
        } catch (error) {
            console.error('Failed to fetch CQC prediction:', error);
        } finally {
            setLoading(false);
        }
    };
    
    const getRatingColor = (rating) => {
        const colors = {
            4: 'text-green-600 bg-green-100',  // Outstanding
            3: 'text-blue-600 bg-blue-100',    // Good
            2: 'text-orange-600 bg-orange-100', // Requires improvement
            1: 'text-red-600 bg-red-100'       // Inadequate
        };
        return colors[rating] || 'text-gray-600 bg-gray-100';
    };
    
    if (loading) return <div className="animate-pulse">Loading prediction...</div>;
    
    if (!prediction) return <div>Unable to load prediction</div>;
    
    return (
        <div className="bg-white rounded-lg shadow-md p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">
                CQC Rating Prediction
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Prediction Result */}
                <div className="space-y-3">
                    <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${getRatingColor(prediction.prediction.predicted_rating)}`}>
                        {prediction.prediction.predicted_rating_text}
                    </div>
                    
                    <div className="text-sm text-gray-600">
                        Confidence: {(prediction.prediction.confidence_score * 100).toFixed(1)}%
                    </div>
                    
                    <div className="text-sm">
                        Risk Level: <span className={`font-medium ${
                            prediction.prediction.risk_level === 'High' ? 'text-red-600' :
                            prediction.prediction.risk_level === 'Medium' ? 'text-orange-600' :
                            'text-green-600'
                        }`}>
                            {prediction.prediction.risk_level}
                        </span>
                    </div>
                </div>
                
                {/* Contributing Factors */}
                <div className="space-y-3">
                    <div>
                        <h4 className="text-sm font-medium text-green-600 mb-1">Positive Factors</h4>
                        <ul className="text-xs text-gray-600 space-y-1">
                            {prediction.contributing_factors.top_positive_factors.map((factor, idx) => (
                                <li key={idx}>• {factor.name}: {factor.impact}</li>
                            ))}
                        </ul>
                    </div>
                    
                    <div>
                        <h4 className="text-sm font-medium text-red-600 mb-1">Risk Factors</h4>
                        <ul className="text-xs text-gray-600 space-y-1">
                            {prediction.contributing_factors.top_risk_factors.map((factor, idx) => (
                                <li key={idx}>• {factor.name}: {factor.impact}</li>
                            ))}
                        </ul>
                    </div>
                </div>
            </div>
            
            {/* Recommendations */}
            {prediction.recommendations && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                    <h4 className="text-sm font-medium text-gray-900 mb-2">Recommendations</h4>
                    <ul className="text-xs text-gray-600 space-y-1">
                        {prediction.recommendations.map((rec, idx) => (
                            <li key={idx}>• {rec}</li>
                        ))}
                    </ul>
                </div>
            )}
            
            <div className="mt-4 text-xs text-gray-400">
                Last updated: {new Date(prediction.data_freshness.last_updated).toLocaleDateString()}
            </div>
        </div>
    );
};
```

---

## 🚀 Implementation Timeline

### Phase 1: Enhanced CQC Data Collection (1-2 days)
- Implement comprehensive CQC API data extraction
- Fetch from all available endpoints
- Build comprehensive training dataset

### Phase 2: Dashboard Feature Engineering (1-2 days) 
- Build dashboard feature extraction service
- Create feature alignment and transformation logic
- Validate feature compatibility between sources

### Phase 3: Model Training & Deployment (1 day)
- Train models on comprehensive CQC dataset  
- Deploy models with dashboard feature support
- Create prediction API endpoints

### Phase 4: Dashboard Integration (1-2 days)
- Build CQC prediction widget for dashboard
- Add prediction storage and tracking
- Create alerts and recommendations system

### Phase 5: Testing & Optimization (1 day)
- End-to-end testing with real dashboard data
- Model performance validation
- User acceptance testing

**Total Timeline: 5-8 days**

---

## 📈 Success Metrics

### Training Performance
- **Accuracy**: >75% on CQC validation data
- **Precision/Recall**: >70% for all rating classes
- **Feature Importance**: Clear attribution to regulatory domains

### Dashboard Integration
- **API Latency**: <500ms for real-time predictions
- **Feature Coverage**: >80% dashboard features mapped to CQC equivalents
- **Prediction Confidence**: Average confidence >70%

### Business Value
- **Early Warning**: Identify rating decline risk 3-6 months ahead
- **Actionable Insights**: Specific recommendations for improvement
- **Regulatory Preparation**: Support CQC inspection readiness

---

## 🔧 Key Implementation Files

```
src/ml/
├── comprehensive_cqc_extractor.py     # All CQC API endpoints
├── dashboard_feature_extractor.py     # Dashboard data extraction
├── feature_alignment.py               # Transform dashboard → CQC features
├── unified_model_trainer.py           # Enhanced training pipeline
└── model_prediction_service.py        # Real-time predictions

src/api/
├── cqc_prediction_routes.py           # Prediction endpoints
└── dashboard_integration_routes.py    # Dashboard-specific APIs

frontend/src/components/
├── CQCPredictionWidget.js             # Main prediction display
├── CQCTrendChart.js                   # Historical predictions
└── CQCRecommendations.js              # Actionable insights

sql/
├── cqc_comprehensive_features.sql     # Training feature extraction  
├── dashboard_feature_views.sql        # Dashboard data views
└── prediction_storage.sql             # Store predictions
```

This integrated approach ensures the ML model is trained on authoritative CQC data while making predictions using rich operational data from your care home dashboard, providing actionable insights for care quality improvement.