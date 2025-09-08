# CQC Prediction Dashboard Components

React components for displaying CQC (Care Quality Commission) prediction data in healthcare dashboards.

## Components

### CQCPredictionWidget

Main widget component that displays CQC rating predictions with confidence scores, contributing factors, and actionable recommendations.

**Props:**
- `careHomeId` (string, required) - Unique identifier for the care home

**Features:**
- Color-coded rating predictions (Outstanding, Good, Requires Improvement, Inadequate)
- Confidence score visualization with progress bar
- Risk level indicators
- Contributing factors (positive and risk factors)
- Historical trends integration
- Data freshness indicators
- Error handling and loading states
- Responsive design with Tailwind CSS

**Usage:**
```jsx
import { CQCPredictionWidget } from './components/CQCPredictionWidget';

<CQCPredictionWidget careHomeId="12345" />
```

### CQCTrendChart

Component for displaying historical CQC prediction trends with interactive charts and time range selection.

**Props:**
- `careHomeId` (string, required) - Unique identifier for the care home

**Features:**
- SVG-based trend visualization
- Multiple time range options (1m, 3m, 6m, 1y)
- Trend direction indicators (improving, declining, stable)
- Summary statistics
- Recent predictions list
- Interactive data points with tooltips

**Usage:**
```jsx
import { CQCTrendChart } from './components/CQCTrendChart';

<CQCTrendChart careHomeId="12345" />
```

### CQCRecommendations

Component for displaying actionable insights and recommendations with progress tracking.

**Props:**
- `recommendations` (array, required) - Array of recommendation objects or strings
- `careHomeId` (string, optional) - Care home identifier for context

**Features:**
- Categorized recommendations (staffing, safety, care quality, etc.)
- Priority level indicators (high, medium, low)
- Expandable action items
- Progress tracking with checkboxes
- Category-specific icons and colors
- Summary statistics

**Usage:**
```jsx
import { CQCRecommendations } from './components/CQCRecommendations';

const recommendations = [
  "Increase staff training frequency",
  "Improve safety protocols",
  "Enhance documentation practices"
];

<CQCRecommendations 
  recommendations={recommendations} 
  careHomeId="12345" 
/>
```

## API Integration

The components expect these API endpoints:

### Prediction Data
- `GET /api/cqc-prediction/dashboard/{careHomeId}`

Expected response format:
```json
{
  "prediction": {
    "predicted_rating": 3,
    "predicted_rating_text": "Good",
    "confidence_score": 0.85,
    "risk_level": "Medium"
  },
  "contributing_factors": {
    "top_positive_factors": [
      {"name": "Staff Training", "impact": "+0.2"}
    ],
    "top_risk_factors": [
      {"name": "Incident Rate", "impact": "-0.1"}
    ]
  },
  "recommendations": [
    "Increase staff training frequency"
  ],
  "data_freshness": {
    "last_updated": "2024-01-15T10:00:00Z"
  }
}
```

### Trend Data
- `GET /api/cqc-prediction/trends/{careHomeId}?timeRange={range}`

Expected response format:
```json
{
  "predictions": [
    {
      "prediction_date": "2024-01-01T00:00:00Z",
      "predicted_rating": 3,
      "confidence_score": 0.82
    }
  ],
  "average_rating": 3.2,
  "average_confidence": 0.84
}
```

## Styling

Components use Tailwind CSS classes for styling. Required Tailwind colors:
- `green` (Outstanding ratings)
- `blue` (Good ratings, confidence bars)
- `orange/amber` (Requires improvement ratings)
- `red` (Inadequate ratings, high risk)
- `gray` (UI elements, loading states)

## Authentication

Components use a `makeAuthenticatedRequest` function that expects:
- Bearer token in localStorage (`authToken`)
- Standard Authorization header format

## Error Handling

All components include:
- Loading states with skeleton animations
- Error states with retry functionality  
- Graceful degradation for missing data
- Responsive design for mobile devices

## Accessibility

Components follow accessibility best practices:
- ARIA labels for interactive elements
- Keyboard navigation support
- Color contrast compliance
- Screen reader friendly content
- Focus management

## Browser Support

Compatible with all modern browsers supporting:
- ES6+ JavaScript features
- CSS Grid and Flexbox
- SVG rendering
- Local Storage API