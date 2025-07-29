"""
CQC Alerts Package

This package provides notification services for high-risk CQC locations.
"""

from .notification_service import (
    NotificationService,
    NotificationChannel,
    NotificationConfig,
    AlertMessage
)

__all__ = [
    'NotificationService',
    'NotificationChannel',
    'NotificationConfig',
    'AlertMessage'
]