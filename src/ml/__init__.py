"""
CQC ML Pipeline Package.

This package contains modules for feature engineering and Vertex AI pipeline components
for training and deploying CQC rating prediction models.
"""

from .features import CQCFeatureEngineer

__version__ = "0.1.0"
__all__ = ["CQCFeatureEngineer"]