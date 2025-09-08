import React, { useState, useEffect } from 'react';

const CQCTrendChart = ({ careHomeId }) => {
    const [trendData, setTrendData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [timeRange, setTimeRange] = useState('6m'); // 6 months default
    
    useEffect(() => {
        if (careHomeId) {
            fetchTrendData();
        }
    }, [careHomeId, timeRange]);
    
    const makeAuthenticatedRequest = async (url) => {
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
    
    const fetchTrendData = async () => {
        try {
            setLoading(true);
            setError(null);
            const data = await makeAuthenticatedRequest(
                `/api/cqc-prediction/trends/${careHomeId}?timeRange=${timeRange}`
            );
            setTrendData(data);
        } catch (error) {
            console.error('Failed to fetch trend data:', error);
            setError('Failed to load historical data');
        } finally {
            setLoading(false);
        }
    };
    
    const getRatingColor = (rating) => {
        const colors = {
            4: '#10B981', // green-500 - Outstanding
            3: '#3B82F6', // blue-500 - Good
            2: '#F59E0B', // amber-500 - Requires improvement
            1: '#EF4444'  // red-500 - Inadequate
        };
        return colors[rating] || '#6B7280'; // gray-500
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
    
    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleDateString('en-GB', {
            month: 'short',
            day: 'numeric'
        });
    };
    
    const calculateTrendDirection = () => {
        if (!trendData?.predictions || trendData.predictions.length < 2) return null;
        
        const recent = trendData.predictions.slice(-2);
        const change = recent[1].predicted_rating - recent[0].predicted_rating;
        
        if (change > 0) return 'improving';
        if (change < 0) return 'declining';
        return 'stable';
    };
    
    const getTrendIcon = (direction) => {
        switch (direction) {
            case 'improving':
                return (
                    <svg className="w-4 h-4 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M3.293 9.707a1 1 0 010-1.414l6-6a1 1 0 011.414 0l6 6a1 1 0 01-1.414 1.414L11 5.414V17a1 1 0 11-2 0V5.414L4.707 9.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
                    </svg>
                );
            case 'declining':
                return (
                    <svg className="w-4 h-4 text-red-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 10.293a1 1 0 010 1.414l-6 6a1 1 0 01-1.414 0l-6-6a1 1 0 111.414-1.414L9 14.586V3a1 1 0 012 0v11.586l4.293-4.293a1 1 0 011.414 0z" clipRule="evenodd" />
                    </svg>
                );
            default:
                return (
                    <svg className="w-4 h-4 text-gray-600" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clipRule="evenodd" />
                    </svg>
                );
        }
    };
    
    if (loading) {
        return (
            <div>
                <h4 className="text-sm font-medium text-gray-900 mb-4">Historical Predictions</h4>
                <div className="animate-pulse">
                    <div className="h-32 bg-gray-200 rounded"></div>
                </div>
            </div>
        );
    }
    
    if (error) {
        return (
            <div>
                <h4 className="text-sm font-medium text-gray-900 mb-4">Historical Predictions</h4>
                <div className="text-center text-gray-500 py-8">
                    <p>{error}</p>
                    <button 
                        onClick={fetchTrendData}
                        className="mt-2 text-blue-600 hover:text-blue-800 text-sm"
                    >
                        Try again
                    </button>
                </div>
            </div>
        );
    }
    
    if (!trendData?.predictions || trendData.predictions.length === 0) {
        return (
            <div>
                <h4 className="text-sm font-medium text-gray-900 mb-4">Historical Predictions</h4>
                <div className="text-center text-gray-500 py-8">
                    No historical data available
                </div>
            </div>
        );
    }
    
    const trendDirection = calculateTrendDirection();
    const maxRating = 4;
    const minRating = 1;
    
    return (
        <div>
            <div className="flex justify-between items-center mb-4">
                <div className="flex items-center gap-2">
                    <h4 className="text-sm font-medium text-gray-900">Historical Predictions</h4>
                    {trendDirection && (
                        <div className="flex items-center gap-1">
                            {getTrendIcon(trendDirection)}
                            <span className={`text-xs ${
                                trendDirection === 'improving' ? 'text-green-600' :
                                trendDirection === 'declining' ? 'text-red-600' : 'text-gray-600'
                            }`}>
                                {trendDirection}
                            </span>
                        </div>
                    )}
                </div>
                
                <div className="flex gap-1">
                    {['1m', '3m', '6m', '1y'].map((range) => (
                        <button
                            key={range}
                            onClick={() => setTimeRange(range)}
                            className={`px-2 py-1 text-xs rounded ${
                                timeRange === range 
                                    ? 'bg-blue-100 text-blue-700' 
                                    : 'text-gray-500 hover:text-gray-700'
                            }`}
                        >
                            {range}
                        </button>
                    ))}
                </div>
            </div>
            
            {/* Simple Chart Visualization */}
            <div className="relative h-32 bg-gray-50 rounded-lg p-4">
                <div className="absolute left-0 top-0 bottom-0 w-8 flex flex-col justify-between text-xs text-gray-500 py-4">
                    <span>4</span>
                    <span>3</span>
                    <span>2</span>
                    <span>1</span>
                </div>
                
                <div className="ml-8 h-full relative">
                    <svg viewBox={`0 0 ${trendData.predictions.length * 40} 100`} className="w-full h-full">
                        {/* Grid lines */}
                        {[25, 50, 75].map((y) => (
                            <line
                                key={y}
                                x1="0"
                                y1={y}
                                x2={trendData.predictions.length * 40}
                                y2={y}
                                stroke="#E5E7EB"
                                strokeWidth="1"
                            />
                        ))}
                        
                        {/* Trend line */}
                        {trendData.predictions.length > 1 && (
                            <polyline
                                fill="none"
                                stroke="#3B82F6"
                                strokeWidth="2"
                                points={trendData.predictions
                                    .map((pred, idx) => {
                                        const x = idx * 40 + 20;
                                        const y = 100 - ((pred.predicted_rating - minRating) / (maxRating - minRating)) * 100;
                                        return `${x},${y}`;
                                    })
                                    .join(' ')
                                }
                            />
                        )}
                        
                        {/* Data points */}
                        {trendData.predictions.map((pred, idx) => {
                            const x = idx * 40 + 20;
                            const y = 100 - ((pred.predicted_rating - minRating) / (maxRating - minRating)) * 100;
                            
                            return (
                                <g key={idx}>
                                    <circle
                                        cx={x}
                                        cy={y}
                                        r="4"
                                        fill={getRatingColor(pred.predicted_rating)}
                                        stroke="white"
                                        strokeWidth="2"
                                    />
                                    <title>
                                        {formatDate(pred.prediction_date)}: {getRatingText(pred.predicted_rating)} ({pred.confidence_score ? (pred.confidence_score * 100).toFixed(0) + '% confidence' : ''})
                                    </title>
                                </g>
                            );
                        })}
                    </svg>
                </div>
                
                {/* Date labels */}
                <div className="ml-8 flex justify-between text-xs text-gray-500 mt-2">
                    <span>{formatDate(trendData.predictions[0].prediction_date)}</span>
                    <span>{formatDate(trendData.predictions[trendData.predictions.length - 1].prediction_date)}</span>
                </div>
            </div>
            
            {/* Summary Statistics */}
            <div className="mt-4 grid grid-cols-3 gap-4 text-center">
                <div>
                    <div className="text-lg font-semibold text-gray-900">
                        {trendData.predictions.length}
                    </div>
                    <div className="text-xs text-gray-500">Predictions</div>
                </div>
                
                <div>
                    <div className="text-lg font-semibold text-gray-900">
                        {trendData.average_rating?.toFixed(1) || 'N/A'}
                    </div>
                    <div className="text-xs text-gray-500">Average Rating</div>
                </div>
                
                <div>
                    <div className="text-lg font-semibold text-gray-900">
                        {trendData.average_confidence ? (trendData.average_confidence * 100).toFixed(0) + '%' : 'N/A'}
                    </div>
                    <div className="text-xs text-gray-500">Avg Confidence</div>
                </div>
            </div>
            
            {/* Recent Predictions List */}
            {trendData.predictions.length > 0 && (
                <div className="mt-4">
                    <h5 className="text-xs font-medium text-gray-700 mb-2">Recent Predictions</h5>
                    <div className="space-y-2 max-h-32 overflow-y-auto">
                        {trendData.predictions.slice(-5).reverse().map((pred, idx) => (
                            <div key={idx} className="flex justify-between items-center text-xs">
                                <span className="text-gray-600">{formatDate(pred.prediction_date)}</span>
                                <div className="flex items-center gap-2">
                                    <div 
                                        className="w-3 h-3 rounded-full"
                                        style={{backgroundColor: getRatingColor(pred.predicted_rating)}}
                                    ></div>
                                    <span className="font-medium">{getRatingText(pred.predicted_rating)}</span>
                                    {pred.confidence_score && (
                                        <span className="text-gray-500">
                                            ({(pred.confidence_score * 100).toFixed(0)}%)
                                        </span>
                                    )}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
};

export default CQCTrendChart;