import React, { useState } from 'react';

const CQCRecommendations = ({ recommendations, careHomeId }) => {
    const [expandedItem, setExpandedItem] = useState(null);
    const [completedActions, setCompletedActions] = useState(new Set());
    
    // If recommendations is a simple array of strings, convert to structured format
    const structuredRecommendations = Array.isArray(recommendations) && typeof recommendations[0] === 'string' 
        ? recommendations.map((rec, idx) => ({
            id: idx,
            title: rec,
            category: categorizeRecommendation(rec),
            priority: 'medium',
            description: rec,
            actions: extractActions(rec)
        }))
        : recommendations || [];
    
    function categorizeRecommendation(recommendation) {
        const rec = recommendation.toLowerCase();
        
        if (rec.includes('staff') || rec.includes('training') || rec.includes('skill')) {
            return 'staffing';
        } else if (rec.includes('safety') || rec.includes('incident') || rec.includes('risk')) {
            return 'safety';
        } else if (rec.includes('care') || rec.includes('treatment') || rec.includes('assessment')) {
            return 'care_quality';
        } else if (rec.includes('management') || rec.includes('leadership') || rec.includes('oversight')) {
            return 'management';
        } else if (rec.includes('environment') || rec.includes('facility') || rec.includes('maintenance')) {
            return 'environment';
        } else if (rec.includes('record') || rec.includes('documentation') || rec.includes('policy')) {
            return 'documentation';
        }
        return 'general';
    }
    
    function extractActions(recommendation) {
        // Simple action extraction - in a real system this would be more sophisticated
        const actions = [];
        
        if (recommendation.includes('increase') || recommendation.includes('improve')) {
            actions.push('Review current processes');
            actions.push('Develop improvement plan');
        }
        if (recommendation.includes('training')) {
            actions.push('Schedule staff training');
            actions.push('Assess training effectiveness');
        }
        if (recommendation.includes('monitor') || recommendation.includes('review')) {
            actions.push('Set up regular monitoring');
            actions.push('Create review schedule');
        }
        
        return actions.length > 0 ? actions : ['Review and implement'];
    }
    
    const getCategoryIcon = (category) => {
        const icons = {
            staffing: (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" />
                </svg>
            ),
            safety: (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M3 6a3 3 0 013-3h10a1 1 0 01.8 1.6L14.25 8l2.55 3.4A1 1 0 0116 13H6a1 1 0 00-1 1v3a1 1 0 11-2 0V6z" clipRule="evenodd" />
                </svg>
            ),
            care_quality: (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M3.172 5.172a4 4 0 015.656 0L10 6.343l1.172-1.171a4 4 0 115.656 5.656L10 17.657l-6.828-6.829a4 4 0 010-5.656z" clipRule="evenodd" />
                </svg>
            ),
            management: (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M6 6V5a3 3 0 013-3h2a3 3 0 013 3v1h2a2 2 0 012 2v3.57A22.952 22.952 0 0110 13a22.95 22.95 0 01-8-1.43V8a2 2 0 012-2h2zm2-1a1 1 0 011-1h2a1 1 0 011 1v1H8V5zm1 5a1 1 0 011-1h.01a1 1 0 110 2H10a1 1 0 01-1-1z" clipRule="evenodd" />
                </svg>
            ),
            environment: (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4 4a2 2 0 00-2 2v8a2 2 0 002 2h12a2 2 0 002-2V6a2 2 0 00-2-2H4zm3 5a2 2 0 114 0v2H7V9z" clipRule="evenodd" />
                </svg>
            ),
            documentation: (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                </svg>
            ),
            general: (
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
            )
        };
        return icons[category] || icons.general;
    };
    
    const getCategoryColor = (category) => {
        const colors = {
            staffing: 'text-blue-600 bg-blue-50',
            safety: 'text-red-600 bg-red-50',
            care_quality: 'text-green-600 bg-green-50',
            management: 'text-purple-600 bg-purple-50',
            environment: 'text-yellow-600 bg-yellow-50',
            documentation: 'text-gray-600 bg-gray-50',
            general: 'text-indigo-600 bg-indigo-50'
        };
        return colors[category] || colors.general;
    };
    
    const getPriorityColor = (priority) => {
        const colors = {
            high: 'text-red-600 bg-red-100 border-red-200',
            medium: 'text-yellow-600 bg-yellow-100 border-yellow-200',
            low: 'text-green-600 bg-green-100 border-green-200'
        };
        return colors[priority] || colors.medium;
    };
    
    const toggleActionComplete = (recommendationId, actionIndex) => {
        const actionId = `${recommendationId}-${actionIndex}`;
        const newCompleted = new Set(completedActions);
        
        if (completedActions.has(actionId)) {
            newCompleted.delete(actionId);
        } else {
            newCompleted.add(actionId);
        }
        
        setCompletedActions(newCompleted);
    };
    
    const toggleExpanded = (itemId) => {
        setExpandedItem(expandedItem === itemId ? null : itemId);
    };
    
    if (!structuredRecommendations || structuredRecommendations.length === 0) {
        return (
            <div>
                <h4 className="text-sm font-medium text-gray-900 mb-4">Recommendations</h4>
                <div className="text-center text-gray-500 py-4">
                    No recommendations available at this time
                </div>
            </div>
        );
    }
    
    return (
        <div>
            <div className="flex justify-between items-center mb-4">
                <h4 className="text-sm font-medium text-gray-900">Recommendations</h4>
                <div className="text-xs text-gray-500">
                    {structuredRecommendations.length} items
                </div>
            </div>
            
            <div className="space-y-3">
                {structuredRecommendations.map((rec, idx) => {
                    const isExpanded = expandedItem === rec.id || expandedItem === idx;
                    const completedCount = rec.actions ? 
                        rec.actions.filter((_, actionIdx) => 
                            completedActions.has(`${rec.id || idx}-${actionIdx}`)
                        ).length : 0;
                    
                    return (
                        <div key={rec.id || idx} className="border border-gray-200 rounded-lg p-4 hover:border-gray-300 transition-colors">
                            <div className="flex items-start justify-between">
                                <div className="flex-1">
                                    <div className="flex items-center gap-2 mb-2">
                                        <div className={`p-1 rounded ${getCategoryColor(rec.category)}`}>
                                            {getCategoryIcon(rec.category)}
                                        </div>
                                        
                                        <div className="flex-1">
                                            <p className="text-sm font-medium text-gray-900">
                                                {rec.title || rec.description}
                                            </p>
                                        </div>
                                        
                                        {rec.priority && (
                                            <span className={`px-2 py-1 text-xs font-medium rounded-full border ${getPriorityColor(rec.priority)}`}>
                                                {rec.priority}
                                            </span>
                                        )}
                                    </div>
                                    
                                    {rec.description && rec.title !== rec.description && (
                                        <p className="text-xs text-gray-600 mb-2">
                                            {rec.description}
                                        </p>
                                    )}
                                    
                                    {/* Progress bar for actions */}
                                    {rec.actions && rec.actions.length > 0 && (
                                        <div className="mb-2">
                                            <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                                                <span>Progress</span>
                                                <span>{completedCount}/{rec.actions.length} completed</span>
                                            </div>
                                            <div className="w-full bg-gray-200 rounded-full h-1">
                                                <div 
                                                    className="bg-green-500 h-1 rounded-full transition-all duration-300"
                                                    style={{width: `${(completedCount / rec.actions.length) * 100}%`}}
                                                ></div>
                                            </div>
                                        </div>
                                    )}
                                </div>
                                
                                {rec.actions && rec.actions.length > 0 && (
                                    <button
                                        onClick={() => toggleExpanded(rec.id || idx)}
                                        className="ml-2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
                                        aria-label={isExpanded ? "Collapse actions" : "Expand actions"}
                                    >
                                        <svg className={`w-4 h-4 transform transition-transform ${isExpanded ? 'rotate-180' : ''}`} fill="currentColor" viewBox="0 0 20 20">
                                            <path fillRule="evenodd" d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" clipRule="evenodd" />
                                        </svg>
                                    </button>
                                )}
                            </div>
                            
                            {/* Action Items */}
                            {isExpanded && rec.actions && rec.actions.length > 0 && (
                                <div className="mt-4 pt-4 border-t border-gray-100">
                                    <h5 className="text-xs font-medium text-gray-700 mb-2">Action Items</h5>
                                    <ul className="space-y-2">
                                        {rec.actions.map((action, actionIdx) => {
                                            const actionId = `${rec.id || idx}-${actionIdx}`;
                                            const isCompleted = completedActions.has(actionId);
                                            
                                            return (
                                                <li key={actionIdx} className="flex items-start gap-3">
                                                    <button
                                                        onClick={() => toggleActionComplete(rec.id || idx, actionIdx)}
                                                        className={`mt-1 w-4 h-4 rounded border-2 flex items-center justify-center transition-colors ${
                                                            isCompleted 
                                                                ? 'bg-green-500 border-green-500 text-white' 
                                                                : 'border-gray-300 hover:border-gray-400'
                                                        }`}
                                                    >
                                                        {isCompleted && (
                                                            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                                                                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                                                            </svg>
                                                        )}
                                                    </button>
                                                    <span className={`text-xs flex-1 ${
                                                        isCompleted ? 'text-gray-500 line-through' : 'text-gray-700'
                                                    }`}>
                                                        {action}
                                                    </span>
                                                </li>
                                            );
                                        })}
                                    </ul>
                                </div>
                            )}
                        </div>
                    );
                })}
            </div>
            
            {/* Summary */}
            <div className="mt-4 p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between text-xs text-gray-600">
                    <span>
                        Total recommendations: {structuredRecommendations.length}
                    </span>
                    <div className="flex gap-4">
                        <span className="flex items-center gap-1">
                            <div className="w-2 h-2 bg-red-400 rounded-full"></div>
                            High Priority: {structuredRecommendations.filter(r => r.priority === 'high').length}
                        </span>
                        <span className="flex items-center gap-1">
                            <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                            Medium Priority: {structuredRecommendations.filter(r => r.priority === 'medium').length}
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default CQCRecommendations;