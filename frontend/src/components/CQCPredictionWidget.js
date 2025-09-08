import React, { useState, useEffect } from 'react';
import CQCTrendChart from './CQCTrendChart';
import CQCRecommendations from './CQCRecommendations';

const CQCPredictionWidget = ({ careHomeId }) => {
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [showHistory, setShowHistory] = useState(false);
    
    useEffect(() => {
        if (careHomeId) {
            fetchPrediction();
        }
    }, [careHomeId]);
    
    const makeAuthenticatedRequest = async (url) => {
        // This would be replaced with your actual authentication logic
        const response = await fetch(url, {
            headers: {
                'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return response.json();
    };
    
    const fetchPrediction = async () => {
        try {
            setLoading(true);
            setError(null);
            const data = await makeAuthenticatedRequest(
                `/api/cqc-prediction/dashboard/${careHomeId}`
            );
            setPrediction(data);
        } catch (error) {
            console.error('Failed to fetch CQC prediction:', error);
            setError('Failed to load prediction data. Please try again.');
        } finally {
            setLoading(false);
        }
    };
    
    const getRatingColor = (rating) => {
        const colors = {
            4: 'text-green-600 bg-green-100 border-green-200',  // Outstanding
            3: 'text-blue-600 bg-blue-100 border-blue-200',    // Good
            2: 'text-orange-600 bg-orange-100 border-orange-200', // Requires improvement
            1: 'text-red-600 bg-red-100 border-red-200'       // Inadequate
        };
        return colors[rating] || 'text-gray-600 bg-gray-100 border-gray-200';
    };
    
    const getRatingText = (rating) => {
        const texts = {
            4: 'Outstanding',
            3: 'Good', 
            2: 'Requires Improvement',
            1: 'Inadequate'
        };
        return texts[rating] || 'Unknown';
    };
    
    const getRiskLevelColor = (riskLevel) => {
        const colors = {
            'High': 'text-red-600',
            'Medium': 'text-orange-600',
            'Low': 'text-green-600'
        };
        return colors[riskLevel] || 'text-gray-600';
    };
    
    if (loading) {
        return (
            <div className="bg-white rounded-lg shadow-md p-6">
                <div className="animate-pulse">
                    <div className="h-6 bg-gray-200 rounded mb-4 w-1/3"></div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-3">
                            <div className="h-8 bg-gray-200 rounded w-2/3"></div>
                            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
                        </div>
                        <div className="space-y-3">
                            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                            <div className="h-4 bg-gray-200 rounded w-3/4"></div>
                        </div>
                    </div>
                </div>
            </div>
        );
    }
    
    if (error) {
        return (
            <div className="bg-white rounded-lg shadow-md p-6">
                <div className="text-center">
                    <div className="text-red-600 mb-2">
                        <svg className="w-8 h-8 mx-auto" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                        </svg>
                    </div>
                    <p className="text-gray-600 mb-4">{error}</p>
                    <button 
                        onClick={fetchPrediction}
                        className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }
    
    if (!prediction) {
        return (
            <div className="bg-white rounded-lg shadow-md p-6">
                <div className="text-center text-gray-500">
                    No prediction data available for this care home.
                </div>
            </div>
        );
    }
    
    return (
        <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-center mb-4">
                <h3 className="text-lg font-semibold text-gray-900">
                    CQC Rating Prediction
                </h3>
                <div className="flex gap-2">
                    <button 
                        onClick={() => setShowHistory(!showHistory)}
                        className="text-blue-600 hover:text-blue-800 text-sm font-medium transition-colors"
                        aria-label={showHistory ? "Hide historical data" : "Show historical data"}
                    >
                        {showHistory ? 'Hide History' : 'Show History'}
                    </button>
                    <button 
                        onClick={fetchPrediction}
                        className="text-gray-500 hover:text-gray-700 transition-colors"
                        aria-label="Refresh prediction"
                    >
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
                        </svg>
                    </button>
                </div>
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Prediction Result */}
                <div className="space-y-4">
                    <div className={`inline-flex items-center px-4 py-2 rounded-lg text-sm font-medium border ${getRatingColor(prediction.prediction.predicted_rating)}`}>
                        <span className="text-lg font-semibold mr-2">
                            {prediction.prediction.predicted_rating}
                        </span>
                        {prediction.prediction.predicted_rating_text || getRatingText(prediction.prediction.predicted_rating)}
                    </div>
                    
                    <div className="space-y-2">
                        <div className="flex items-center justify-between">
                            <span className="text-sm text-gray-600">Confidence Score</span>
                            <span className="text-sm font-medium text-gray-900">
                                {(prediction.prediction.confidence_score * 100).toFixed(1)}%
                            </span>
                        </div>
                        
                        <div className="w-full bg-gray-200 rounded-full h-2">
                            <div 
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{width: `${prediction.prediction.confidence_score * 100}%`}}
                            ></div>
                        </div>
                    </div>
                    
                    <div className="flex items-center justify-between pt-2">
                        <span className="text-sm text-gray-600">Risk Level</span>
                        <span className={`text-sm font-medium ${getRiskLevelColor(prediction.prediction.risk_level)}`}>
                            {prediction.prediction.risk_level}
                        </span>
                    </div>
                </div>
                
                {/* Contributing Factors */}
                <div className="space-y-4">
                    {prediction.contributing_factors?.top_positive_factors?.length > 0 && (
                        <div>
                            <h4 className="text-sm font-medium text-green-600 mb-2 flex items-center">
                                <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                                Positive Factors
                            </h4>
                            <ul className="text-sm text-gray-600 space-y-1">
                                {prediction.contributing_factors.top_positive_factors.map((factor, idx) => (
                                    <li key={idx} className="flex justify-between">
                                        <span>• {factor.name}</span>
                                        <span className="text-green-600 font-medium">+{factor.impact}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                    
                    {prediction.contributing_factors?.top_risk_factors?.length > 0 && (
                        <div>
                            <h4 className="text-sm font-medium text-red-600 mb-2 flex items-center">
                                <svg className="w-4 h-4 mr-1" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                                </svg>
                                Risk Factors
                            </h4>
                            <ul className="text-sm text-gray-600 space-y-1">
                                {prediction.contributing_factors.top_risk_factors.map((factor, idx) => (
                                    <li key={idx} className="flex justify-between">
                                        <span>• {factor.name}</span>
                                        <span className="text-red-600 font-medium">{factor.impact}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            </div>
            
            {/* Historical Trends */}
            {showHistory && (
                <div className="mt-6 pt-6 border-t border-gray-200">
                    <CQCTrendChart careHomeId={careHomeId} />
                </div>
            )}
            
            {/* Recommendations */}
            {prediction.recommendations && prediction.recommendations.length > 0 && (
                <div className="mt-6 pt-6 border-t border-gray-200">
                    <CQCRecommendations recommendations={prediction.recommendations} />
                </div>
            )}
            
            {/* Data Freshness */}
            {prediction.data_freshness && (
                <div className="mt-6 pt-4 border-t border-gray-100">
                    <div className="flex items-center justify-between text-xs text-gray-500">
                        <div className="flex items-center">
                            <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                            <span>Data last updated: {new Date(prediction.data_freshness.last_updated).toLocaleDateString()}</span>
                        </div>
                        <span>Prediction generated: {new Date().toLocaleDateString()}</span>
                    </div>
                </div>
            )}
        </div>
    );
};

export default CQCPredictionWidget;