import React from 'react';
import { CQCPredictionWidget, CQCTrendChart, CQCRecommendations } from './index';

// Example usage of the CQC Dashboard components
const CQCDashboardExample = () => {
  const careHomeId = "12345";
  
  // Example recommendations data
  const sampleRecommendations = [
    {
      id: 1,
      title: "Increase Staff Training Frequency",
      category: "staffing",
      priority: "high",
      description: "Current staff training intervals are too long, leading to knowledge gaps",
      actions: [
        "Schedule monthly training sessions",
        "Create competency assessment program",
        "Implement peer mentoring system"
      ]
    },
    {
      id: 2,
      title: "Improve Safety Incident Reporting",
      category: "safety", 
      priority: "medium",
      description: "Enhance incident reporting procedures to better track and prevent safety issues",
      actions: [
        "Update incident reporting forms",
        "Train staff on new procedures",
        "Set up regular safety reviews"
      ]
    },
    {
      id: 3,
      title: "Enhance Documentation Practices",
      category: "documentation",
      priority: "medium", 
      description: "Care records need more consistent and detailed documentation",
      actions: [
        "Review documentation standards",
        "Provide documentation training",
        "Implement quality checks"
      ]
    }
  ];

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">
            CQC Dashboard - Care Home {careHomeId}
          </h1>
          <p className="mt-2 text-gray-600">
            Monitor CQC predictions, trends, and recommendations
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Main Prediction Widget */}
          <div className="lg:col-span-1">
            <CQCPredictionWidget careHomeId={careHomeId} />
          </div>

          {/* Standalone Trend Chart */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md p-6">
              <CQCTrendChart careHomeId={careHomeId} />
            </div>
          </div>

          {/* Standalone Recommendations */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow-md p-6">
              <CQCRecommendations 
                recommendations={sampleRecommendations}
                careHomeId={careHomeId}
              />
            </div>
          </div>
        </div>

        {/* Integration Examples */}
        <div className="mt-12">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Integration Examples
          </h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {/* Compact Widget */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Compact Layout
              </h3>
              <div className="scale-75 transform origin-top-left">
                <CQCPredictionWidget careHomeId={careHomeId} />
              </div>
            </div>

            {/* Widget with Custom Styling */}
            <div className="bg-gradient-to-br from-blue-50 to-indigo-100 rounded-lg shadow-md p-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Custom Background
              </h3>
              <CQCPredictionWidget careHomeId={careHomeId} />
            </div>

            {/* Mobile-Optimized */}
            <div className="bg-white rounded-lg shadow-md p-4">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Mobile Layout
              </h3>
              <div className="max-w-sm">
                <CQCPredictionWidget careHomeId={careHomeId} />
              </div>
            </div>
          </div>
        </div>

        {/* Usage Instructions */}
        <div className="mt-12 bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">
            Usage Instructions
          </h2>
          
          <div className="prose text-gray-600">
            <h3>Quick Setup</h3>
            <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`import { CQCPredictionWidget } from './components/CQCPredictionWidget';

function Dashboard() {
  return (
    <div>
      <CQCPredictionWidget careHomeId="your-care-home-id" />
    </div>
  );
}`}
            </pre>

            <h3>Authentication Setup</h3>
            <p>Ensure your app sets the authentication token:</p>
            <pre className="bg-gray-100 p-4 rounded text-sm overflow-x-auto">
{`// Set auth token after login
localStorage.setItem('authToken', 'your-bearer-token');`}
            </pre>

            <h3>API Endpoints</h3>
            <p>Configure these endpoints in your backend:</p>
            <ul className="list-disc pl-6">
              <li><code>GET /api/cqc-prediction/dashboard/{'{careHomeId}'}</code> - Main prediction data</li>
              <li><code>GET /api/cqc-prediction/trends/{'{careHomeId}'}?timeRange={'{range}'}</code> - Historical trends</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CQCDashboardExample;