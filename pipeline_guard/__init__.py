#!/usr/bin/env python3
"""
Self-Healing Pipeline Guard System
Enterprise-grade CI/CD pipeline monitoring, anomaly detection, and automated repair

Built on HDC Robot Controller infrastructure for advanced pattern recognition
and predictive failure analysis.

Author: Terry - Terragon Labs Autonomous Systems
"""

__version__ = "1.0.0"
__author__ = "Terry - Terragon Labs"

from .core import PipelineGuard, GuardStatus
from .monitoring import PipelineMonitor, MetricCollector  
from .detection import AnomalyDetector, FailurePredictor
from .repair import AutoRepair, RepairStrategy
from .dashboard import HealthDashboard

__all__ = [
    'PipelineGuard',
    'GuardStatus', 
    'PipelineMonitor',
    'MetricCollector',
    'AnomalyDetector',
    'FailurePredictor',
    'AutoRepair',
    'RepairStrategy',
    'HealthDashboard'
]