#!/usr/bin/env python3
"""
Dashboard Feature Extraction Service for CQC Rating Prediction

This module extracts ML features from dashboard operational data (EAV system) and maps them 
to CQC training feature space. Implements the Phase 2.1 specification from plan.md.

The service transforms care home dashboard metrics into features compatible with CQC training data:
- Operational metrics (occupancy, capacity, care complexity)
- Risk indicators (incidents, falls, medication errors)
- Care quality metrics (care plan compliance, goal achievement)  
- Staff performance (compliance, training, response times)
- Engagement metrics (resident participation, activity variety)
- Temporal features (incident patterns, operational stability)
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Google Cloud imports
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseService:
    """Database service for accessing dashboard EAV system data."""
    
    def __init__(self, project_id: Optional[str] = None):
        """Initialize database service with BigQuery client."""
        self.project_id = project_id or os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        self.client = bigquery.Client(project=self.project_id)
        self.dataset_id = 'dashboard_data'  # Dashboard EAV dataset
        logger.info(f"Initialized DatabaseService for project: {self.project_id}")
    
    def execute_query(self, query: str, parameters: List[Any] = None) -> List[Dict[str, Any]]:
        """Execute BigQuery query with optional parameters."""
        try:
            # Configure query parameters for BigQuery
            job_config = bigquery.QueryJobConfig()
            if parameters:
                # Convert parameters to BigQuery format
                query_parameters = []
                for i, param in enumerate(parameters):
                    query_parameters.append(bigquery.ScalarQueryParameter(
                        f"param_{i}", 
                        "STRING" if isinstance(param, str) else "INT64",
                        param
                    ))
                job_config.query_parameters = query_parameters
                
                # Replace %s with parameter references
                for i in range(len(parameters)):
                    query = query.replace('%s', f'@param_{i}', 1)
            
            # Execute query
            query_job = self.client.query(query, job_config=job_config)
            results = query_job.result()
            
            # Convert to list of dictionaries
            rows = []
            for row in results:
                rows.append(dict(row))
            
            logger.info(f"Query executed successfully, returned {len(rows)} rows")
            return rows
            
        except Exception as e:
            logger.error(f"Database query failed: {str(e)}")
            logger.error(f"Query: {query[:200]}...")
            return []

class DashboardFeatureExtractor:
    """Extract ML features from dashboard operational data for CQC prediction."""
    
    def __init__(self, client_id: str):
        """Initialize feature extractor for specific client."""
        self.client_id = client_id
        self.db_service = DatabaseService()
        logger.info(f"Initialized DashboardFeatureExtractor for client: {client_id}")
    
    def extract_care_home_features(self, care_home_entity_id: str) -> Dict[str, float]:
        """
        Extract comprehensive ML features from dashboard data.
        
        Implements the dashboard-to-CQC feature mapping from plan.md section:
        - Operational Metrics
        - Risk Indicators  
        - Care Quality Metrics
        - Staff Performance
        - Engagement & Activities
        - Temporal Features
        
        Args:
            care_home_entity_id: Unique identifier for care home entity
            
        Returns:
            Dictionary of extracted features ready for ML prediction
        """
        try:
            logger.info(f"Extracting features for care home: {care_home_entity_id}")
            
            features = {}
            
            # 1. Operational Metrics - maps to CQC facility size, capacity, complexity
            logger.info("Calculating operational metrics...")
            features.update(self._calculate_operational_metrics(care_home_entity_id))
            
            # 2. Risk Indicators - maps to CQC inspection risk, safety concerns
            logger.info("Calculating risk indicators...")
            features.update(self._calculate_risk_indicators(care_home_entity_id))
            
            # 3. Care Quality Metrics - maps to CQC care effectiveness, responsiveness
            logger.info("Calculating care quality metrics...")
            features.update(self._calculate_care_quality_metrics(care_home_entity_id))
            
            # 4. Staff Performance - maps to CQC well-led domain
            logger.info("Calculating staff performance metrics...")
            features.update(self._calculate_staff_performance(care_home_entity_id))
            
            # 5. Engagement Metrics - maps to CQC caring domain
            logger.info("Calculating engagement metrics...")
            features.update(self._calculate_engagement_metrics(care_home_entity_id))
            
            # 6. Temporal Features - maps to CQC historical patterns
            logger.info("Calculating temporal features...")
            features.update(self._calculate_temporal_features(care_home_entity_id))
            
            logger.info(f"Successfully extracted {len(features)} features")
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed for care home {care_home_entity_id}: {str(e)}")
            return self._get_default_features()
    
    def _calculate_operational_metrics(self, care_home_id: str) -> Dict[str, float]:
        """
        Map dashboard data to CQC operational features.
        
        Implements the dashboard-to-CQC mapping:
        - bed_capacity: COUNT(residents WHERE status=active)
        - occupancy_rate: active_residents / total_capacity  
        - care_complexity: AVG(residents.care_level_numeric)
        
        Following the exact SQL query from plan.md lines 266-308.
        """
        query = f"""
        WITH resident_data AS (
            SELECT 
                COUNT(*) as active_residents,
                AVG(CASE 
                    WHEN ev.value_string = 'High' THEN 3
                    WHEN ev.value_string = 'Medium' THEN 2  
                    WHEN ev.value_string = 'Low' THEN 1
                    ELSE 2
                END) as avg_care_complexity
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.entities` e
            JOIN `{self.db_service.project_id}.{self.db_service.dataset_id}.entity_values` ev ON e.id = ev.entity_id
            JOIN `{self.db_service.project_id}.{self.db_service.dataset_id}.attributes` a ON ev.attribute_id = a.id
            WHERE e.entity_type = 'resident' 
            AND e.status = 'active'
            AND a.name = 'care_level'
            AND e.client_id = @param_0
        ),
        capacity_data AS (
            SELECT CAST(ev.value_integer AS INT64) as bed_capacity
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.entities` e
            JOIN `{self.db_service.project_id}.{self.db_service.dataset_id}.entity_values` ev ON e.id = ev.entity_id
            JOIN `{self.db_service.project_id}.{self.db_service.dataset_id}.attributes` a ON ev.attribute_id = a.id  
            WHERE e.id = @param_1
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
        
        try:
            result = self.db_service.execute_query(query, [self.client_id, care_home_id])
            if result and len(result) > 0:
                row = result[0]
                return {
                    'bed_capacity': float(row.get('bed_capacity') or 30),
                    'occupancy_rate': float(row.get('occupancy_rate') or 0.85),
                    'avg_care_complexity': float(row.get('avg_care_complexity') or 2.0),
                    'facility_size_numeric': float(row.get('facility_size_numeric') or 2)
                }
        except Exception as e:
            logger.error(f"Failed to calculate operational metrics: {str(e)}")
        
        # Return defaults on failure
        return {
            'bed_capacity': 30.0,
            'occupancy_rate': 0.85,
            'avg_care_complexity': 2.0,
            'facility_size_numeric': 2.0
        }
    
    def _calculate_risk_indicators(self, care_home_id: str) -> Dict[str, float]:
        """
        Calculate incident-based risk scores.
        
        Implements the risk indicator mapping:
        - incident_risk_score: incident_frequency * severity_weights
        - falls_per_resident: COUNT(incidents WHERE type=fall) / resident_count
        - medication_error_rate: med_errors / total_med_administrations
        - safeguarding_concerns: COUNT(incidents WHERE type=safeguarding)
        
        Following the exact SQL query from plan.md lines 322-358.
        """
        query = f"""
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
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.incidents` 
            WHERE incident_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
            AND care_home_id = @param_0
        )
        
        SELECT 
            total_incidents,
            avg_severity,
            falls_count,  
            med_errors,
            safeguarding_count,
            DATE_DIFF(CURRENT_DATE, last_incident_date, DAY) as days_since_last_incident,
            incidents_per_resident,
            -- Risk scores (0-1 scale)
            LEAST(total_incidents / 50.0, 1.0) as incident_frequency_risk,
            LEAST(falls_count / 20.0, 1.0) as falls_risk,
            LEAST(med_errors / 10.0, 1.0) as medication_risk,
            CASE WHEN safeguarding_count > 0 THEN 1.0 ELSE 0.0 END as safeguarding_risk
        FROM incident_analysis
        """
        
        try:
            result = self.db_service.execute_query(query, [care_home_id])
            if result and len(result) > 0 and result[0].get('total_incidents'):
                row = result[0]
                return {
                    'incident_frequency_risk': float(row.get('incident_frequency_risk') or 0.0),
                    'falls_risk': float(row.get('falls_risk') or 0.0),
                    'medication_risk': float(row.get('medication_risk') or 0.0),
                    'safeguarding_risk': float(row.get('safeguarding_risk') or 0.0),
                    'days_since_last_incident': float(row.get('days_since_last_incident') or 365),
                    'avg_incident_severity': float(row.get('avg_severity') or 2.0)
                }
        except Exception as e:
            logger.error(f"Failed to calculate risk indicators: {str(e)}")
        
        # Return safe defaults for low-risk facilities
        return {
            'incident_frequency_risk': 0.0,
            'falls_risk': 0.0, 
            'medication_risk': 0.0,
            'safeguarding_risk': 0.0,
            'days_since_last_incident': 365.0,
            'avg_incident_severity': 1.0
        }
    
    def _calculate_care_quality_metrics(self, care_home_id: str) -> Dict[str, float]:
        """
        Calculate care quality and compliance metrics.
        
        Implements mapping:
        - care_plan_compliance: on_time_reviews / total_care_plans
        - care_plan_overdue_risk: overdue_reviews / total_care_plans  
        - care_goal_achievement: goals_met / total_goals
        """
        query = f"""
        WITH care_plan_analysis AS (
            SELECT 
                COUNT(*) as total_care_plans,
                COUNTIF(DATE_DIFF(next_review_date, CURRENT_DATE, DAY) >= 0) as on_time_plans,
                COUNTIF(DATE_DIFF(CURRENT_DATE, next_review_date, DAY) > 0) as overdue_plans,
                AVG(DATE_DIFF(CURRENT_DATE, last_review_date, DAY)) as avg_days_since_review
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.care_plans` 
            WHERE care_home_id = @param_0
            AND status = 'active'
        ),
        goal_analysis AS (
            SELECT 
                COUNT(*) as total_goals,
                COUNTIF(status = 'achieved') as goals_met,
                COUNTIF(status = 'in_progress') as goals_in_progress,
                AVG(CASE 
                    WHEN status = 'achieved' THEN 1.0
                    WHEN status = 'in_progress' THEN 0.5  
                    ELSE 0.0
                END) as goal_achievement_score
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.care_goals` cg
            JOIN `{self.db_service.project_id}.{self.db_service.dataset_id}.care_plans` cp ON cg.care_plan_id = cp.id
            WHERE cp.care_home_id = @param_0
        )
        
        SELECT 
            cp.total_care_plans,
            cp.on_time_plans,
            cp.overdue_plans,
            cp.avg_days_since_review,
            g.total_goals,
            g.goals_met,
            g.goal_achievement_score,
            CASE WHEN cp.total_care_plans > 0 
                 THEN cp.on_time_plans / cp.total_care_plans 
                 ELSE 1.0 END as care_plan_compliance,
            CASE WHEN cp.total_care_plans > 0 
                 THEN cp.overdue_plans / cp.total_care_plans 
                 ELSE 0.0 END as care_plan_overdue_risk,
            CASE WHEN g.total_goals > 0 
                 THEN g.goals_met / g.total_goals 
                 ELSE 0.8 END as care_goal_achievement
        FROM care_plan_analysis cp
        CROSS JOIN goal_analysis g
        """
        
        try:
            result = self.db_service.execute_query(query, [care_home_id])
            if result and len(result) > 0:
                row = result[0]
                return {
                    'care_plan_compliance': float(row.get('care_plan_compliance') or 0.8),
                    'care_plan_overdue_risk': float(row.get('care_plan_overdue_risk') or 0.1),
                    'care_goal_achievement': float(row.get('care_goal_achievement') or 0.7),
                    'avg_days_since_review': float(row.get('avg_days_since_review') or 30)
                }
        except Exception as e:
            logger.error(f"Failed to calculate care quality metrics: {str(e)}")
        
        return {
            'care_plan_compliance': 0.8,
            'care_plan_overdue_risk': 0.1,
            'care_goal_achievement': 0.7,
            'avg_days_since_review': 30.0
        }
    
    def _calculate_staff_performance(self, care_home_id: str) -> Dict[str, float]:
        """
        Calculate staff performance and compliance metrics.
        
        Implements mapping:
        - staff_incident_response: AVG(incident_resolution_time)
        - staff_compliance_score: audit_compliance / total_audits
        - staff_training_current: current_certifications / required_certifications
        """
        query = f"""
        WITH incident_response AS (
            SELECT 
                AVG(DATE_DIFF(resolution_date, incident_date, HOUR)) as avg_resolution_hours,
                COUNT(*) as total_incidents_resolved,
                COUNTIF(DATE_DIFF(resolution_date, incident_date, HOUR) <= 24) as incidents_resolved_24h
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.incidents`
            WHERE care_home_id = @param_0
            AND resolution_date IS NOT NULL
            AND incident_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
        ),
        audit_compliance AS (
            SELECT 
                COUNT(*) as total_audits,
                COUNTIF(compliance_status = 'compliant') as compliant_audits,
                AVG(compliance_score) as avg_compliance_score
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.audit_logs`
            WHERE entity_id IN (
                SELECT id FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.entities` 
                WHERE entity_type = 'care_home' AND parent_id = @param_0
            )
            AND audit_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
        ),
        staff_training AS (
            SELECT 
                COUNT(*) as total_staff,
                AVG(training_completion_rate) as avg_training_completion,
                COUNTIF(certification_current = true) as staff_with_current_certs
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.users`
            WHERE care_home_id = @param_0
            AND role IN ('care_assistant', 'senior_care_assistant', 'nurse', 'care_manager')
            AND status = 'active'
        )
        
        SELECT 
            ir.avg_resolution_hours,
            ir.incidents_resolved_24h,
            ir.total_incidents_resolved,
            ac.total_audits,
            ac.compliant_audits,
            ac.avg_compliance_score,
            st.total_staff,
            st.avg_training_completion,
            st.staff_with_current_certs,
            CASE WHEN ir.total_incidents_resolved > 0 
                 THEN ir.incidents_resolved_24h / ir.total_incidents_resolved 
                 ELSE 0.9 END as incident_response_score,
            CASE WHEN ac.total_audits > 0 
                 THEN ac.compliant_audits / ac.total_audits 
                 ELSE 0.85 END as staff_compliance_score,
            CASE WHEN st.total_staff > 0 
                 THEN st.staff_with_current_certs / st.total_staff 
                 ELSE 0.8 END as staff_training_current
        FROM incident_response ir
        CROSS JOIN audit_compliance ac
        CROSS JOIN staff_training st
        """
        
        try:
            result = self.db_service.execute_query(query, [care_home_id])
            if result and len(result) > 0:
                row = result[0]
                return {
                    'staff_incident_response': float(row.get('incident_response_score') or 0.9),
                    'staff_compliance_score': float(row.get('staff_compliance_score') or 0.85),
                    'staff_training_current': float(row.get('staff_training_current') or 0.8),
                    'avg_resolution_hours': float(row.get('avg_resolution_hours') or 12)
                }
        except Exception as e:
            logger.error(f"Failed to calculate staff performance: {str(e)}")
        
        return {
            'staff_incident_response': 0.9,
            'staff_compliance_score': 0.85,
            'staff_training_current': 0.8,
            'avg_resolution_hours': 12.0
        }
    
    def _calculate_engagement_metrics(self, care_home_id: str) -> Dict[str, float]:
        """
        Calculate resident engagement and activity metrics.
        
        Implements mapping:
        - resident_engagement: activity_participation / total_activities
        - social_isolation_risk: low_participation_residents / total_residents
        - activity_variety_score: unique_activity_types / total_activities
        """
        query = f"""
        WITH activity_participation AS (
            SELECT 
                COUNT(DISTINCT a.id) as total_activities,
                COUNT(DISTINCT ap.resident_id) as participating_residents,
                COUNT(*) as total_participations,
                COUNT(DISTINCT a.activity_type) as unique_activity_types,
                AVG(ap.participation_score) as avg_participation_score
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.activities` a
            LEFT JOIN `{self.db_service.project_id}.{self.db_service.dataset_id}.activity_participation` ap 
                ON a.id = ap.activity_id
            WHERE a.care_home_id = @param_0
            AND a.activity_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
        ),
        resident_engagement AS (
            SELECT 
                COUNT(DISTINCT r.id) as total_residents,
                COUNT(DISTINCT CASE WHEN participation_count >= 5 THEN r.id END) as engaged_residents,
                COUNT(DISTINCT CASE WHEN participation_count <= 2 THEN r.id END) as low_participation_residents
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.entities` r
            LEFT JOIN (
                SELECT 
                    resident_id,
                    COUNT(*) as participation_count
                FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.activity_participation`
                WHERE created_date >= DATE_SUB(CURRENT_DATE, INTERVAL 30 DAY)
                GROUP BY resident_id
            ) part ON r.id = part.resident_id
            WHERE r.entity_type = 'resident'
            AND r.status = 'active'
            AND r.client_id = @param_1
        )
        
        SELECT 
            ap.total_activities,
            ap.participating_residents,
            ap.total_participations,
            ap.unique_activity_types,
            ap.avg_participation_score,
            re.total_residents,
            re.engaged_residents,
            re.low_participation_residents,
            CASE WHEN ap.total_activities > 0 AND re.total_residents > 0
                 THEN ap.total_participations / (ap.total_activities * re.total_residents)
                 ELSE 0.7 END as resident_engagement,
            CASE WHEN re.total_residents > 0
                 THEN re.low_participation_residents / re.total_residents
                 ELSE 0.1 END as social_isolation_risk,
            CASE WHEN ap.total_activities > 0
                 THEN ap.unique_activity_types / ap.total_activities
                 ELSE 0.6 END as activity_variety_score
        FROM activity_participation ap
        CROSS JOIN resident_engagement re
        """
        
        try:
            result = self.db_service.execute_query(query, [care_home_id, self.client_id])
            if result and len(result) > 0:
                row = result[0]
                return {
                    'resident_engagement': float(row.get('resident_engagement') or 0.7),
                    'social_isolation_risk': float(row.get('social_isolation_risk') or 0.1),
                    'activity_variety_score': float(row.get('activity_variety_score') or 0.6),
                    'total_activities': float(row.get('total_activities') or 20)
                }
        except Exception as e:
            logger.error(f"Failed to calculate engagement metrics: {str(e)}")
        
        return {
            'resident_engagement': 0.7,
            'social_isolation_risk': 0.1,
            'activity_variety_score': 0.6,
            'total_activities': 20.0
        }
    
    def _calculate_temporal_features(self, care_home_id: str) -> Dict[str, float]:
        """
        Calculate temporal patterns and stability metrics.
        
        Implements mapping:
        - days_since_last_incident: DATE_DIFF(NOW(), MAX(incident_date))
        - care_plan_review_frequency: AVG(days_between_reviews)  
        - operational_stability: STDDEV(daily_incident_count)
        """
        query = f"""
        WITH incident_temporal AS (
            SELECT 
                MAX(incident_date) as last_incident_date,
                DATE_DIFF(CURRENT_DATE, MAX(incident_date), DAY) as days_since_last_incident,
                COUNT(*) as total_incidents_90d,
                STDDEV(daily_incident_count) as incident_stability
            FROM (
                SELECT 
                    DATE(incident_date) as incident_day,
                    incident_date,
                    COUNT(*) as daily_incident_count
                FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.incidents`
                WHERE care_home_id = @param_0
                AND incident_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
                GROUP BY DATE(incident_date), incident_date
            )
        ),
        care_plan_temporal AS (
            SELECT 
                AVG(DATE_DIFF(next_review_date, last_review_date, DAY)) as avg_review_frequency,
                COUNT(*) as total_reviews,
                STDDEV(DATE_DIFF(next_review_date, last_review_date, DAY)) as review_consistency
            FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.care_plans`
            WHERE care_home_id = @param_0
            AND last_review_date IS NOT NULL
            AND next_review_date IS NOT NULL
        ),
        operational_trends AS (
            SELECT 
                COUNT(DISTINCT DATE(created_date)) as active_days,
                COUNT(*) / COUNT(DISTINCT DATE(created_date)) as avg_daily_activities,
                STDDEV(daily_activity_count) as activity_stability
            FROM (
                SELECT 
                    DATE(created_date) as activity_day,
                    created_date,
                    COUNT(*) as daily_activity_count
                FROM `{self.db_service.project_id}.{self.db_service.dataset_id}.audit_logs`
                WHERE entity_id = @param_0
                AND created_date >= DATE_SUB(CURRENT_DATE, INTERVAL 90 DAY)
                GROUP BY DATE(created_date), created_date
            )
        )
        
        SELECT 
            it.days_since_last_incident,
            it.total_incidents_90d,
            it.incident_stability,
            cp.avg_review_frequency,
            cp.review_consistency,
            ot.active_days,
            ot.avg_daily_activities,
            ot.activity_stability,
            -- Derived stability score (lower is more stable)
            COALESCE(it.incident_stability, 0.5) + 
            COALESCE(cp.review_consistency, 5.0) / 30.0 + 
            COALESCE(ot.activity_stability, 2.0) / 10.0 as operational_stability
        FROM incident_temporal it
        CROSS JOIN care_plan_temporal cp
        CROSS JOIN operational_trends ot
        """
        
        try:
            result = self.db_service.execute_query(query, [care_home_id])
            if result and len(result) > 0:
                row = result[0]
                return {
                    'days_since_last_incident': float(row.get('days_since_last_incident') or 30),
                    'care_plan_review_frequency': float(row.get('avg_review_frequency') or 90),
                    'operational_stability': float(row.get('operational_stability') or 0.3),
                    'incident_trend_stability': float(row.get('incident_stability') or 0.5)
                }
        except Exception as e:
            logger.error(f"Failed to calculate temporal features: {str(e)}")
        
        return {
            'days_since_last_incident': 30.0,
            'care_plan_review_frequency': 90.0,
            'operational_stability': 0.3,
            'incident_trend_stability': 0.5
        }
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature set for error scenarios."""
        return {
            # Operational metrics
            'bed_capacity': 30.0,
            'occupancy_rate': 0.85,
            'avg_care_complexity': 2.0,
            'facility_size_numeric': 2.0,
            
            # Risk indicators  
            'incident_frequency_risk': 0.0,
            'falls_risk': 0.0,
            'medication_risk': 0.0,
            'safeguarding_risk': 0.0,
            'days_since_last_incident': 365.0,
            'avg_incident_severity': 1.0,
            
            # Care quality metrics
            'care_plan_compliance': 0.8,
            'care_plan_overdue_risk': 0.1,
            'care_goal_achievement': 0.7,
            'avg_days_since_review': 30.0,
            
            # Staff performance
            'staff_incident_response': 0.9,
            'staff_compliance_score': 0.85,
            'staff_training_current': 0.8,
            'avg_resolution_hours': 12.0,
            
            # Engagement metrics
            'resident_engagement': 0.7,
            'social_isolation_risk': 0.1,
            'activity_variety_score': 0.6,
            'total_activities': 20.0,
            
            # Temporal features
            'care_plan_review_frequency': 90.0,
            'operational_stability': 0.3,
            'incident_trend_stability': 0.5
        }

# Example usage and validation
if __name__ == "__main__":
    """
    Example usage of Dashboard Feature Extractor.
    For testing and validation purposes.
    """
    
    # Initialize for demo client
    client_id = "demo_client_123"
    care_home_id = "care_home_456"
    
    extractor = DashboardFeatureExtractor(client_id)
    
    try:
        # Extract features
        features = extractor.extract_care_home_features(care_home_id)
        
        print(f"\n=== Dashboard Feature Extraction Results ===")
        print(f"Client ID: {client_id}")
        print(f"Care Home ID: {care_home_id}")
        print(f"Total Features: {len(features)}")
        
        print(f"\n--- Operational Metrics ---")
        operational_keys = ['bed_capacity', 'occupancy_rate', 'avg_care_complexity', 'facility_size_numeric']
        for key in operational_keys:
            if key in features:
                print(f"{key}: {features[key]}")
        
        print(f"\n--- Risk Indicators ---")
        risk_keys = ['incident_frequency_risk', 'falls_risk', 'medication_risk', 'safeguarding_risk']
        for key in risk_keys:
            if key in features:
                print(f"{key}: {features[key]}")
        
        print(f"\n--- Care Quality Metrics ---")
        quality_keys = ['care_plan_compliance', 'care_goal_achievement', 'staff_compliance_score']
        for key in quality_keys:
            if key in features:
                print(f"{key}: {features[key]}")
                
        print(f"\n--- Engagement & Temporal ---")
        engagement_keys = ['resident_engagement', 'social_isolation_risk', 'operational_stability']
        for key in engagement_keys:
            if key in features:
                print(f"{key}: {features[key]}")
        
        print(f"\n=== Ready for CQC Prediction ===")
        
    except Exception as e:
        logger.error(f"Demo extraction failed: {str(e)}")
        print(f"Error: {str(e)}")