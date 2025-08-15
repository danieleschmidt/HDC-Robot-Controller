#!/usr/bin/env python3
"""
Advanced Anomaly Detection and Failure Prediction for Pipeline Guard
Leveraging HDC pattern recognition and machine learning for intelligent analysis

Features:
- Multi-dimensional anomaly detection using HDC similarity analysis
- Predictive failure analysis with temporal pattern recognition
- Adaptive thresholds based on historical data and seasonality
- Real-time statistical analysis and trend detection
- Integration with HDC hypervector pattern matching
"""

import time
import math
import numpy as np
import statistics
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import logging

# HDC imports for pattern recognition
from hdc_robot_controller.core.hypervector import HyperVector
from hdc_robot_controller.core.memory import HierarchicalMemory
from hdc_robot_controller.core.behavior_learner import BehaviorLearner

# Import pipeline guard types
from .core import PipelineMetrics, AlertSeverity, PipelinePhase

class AnomalyType(Enum):
    PERFORMANCE_DEGRADATION = "performance_degradation"
    RESOURCE_ANOMALY = "resource_anomaly" 
    ERROR_SPIKE = "error_spike"
    PATTERN_DEVIATION = "pattern_deviation"
    TEMPORAL_ANOMALY = "temporal_anomaly"
    QUALITY_REGRESSION = "quality_regression"

class DetectionMethod(Enum):
    STATISTICAL = "statistical"
    HDC_PATTERN = "hdc_pattern"
    MACHINE_LEARNING = "machine_learning"
    HYBRID = "hybrid"
    TEMPORAL = "temporal"

@dataclass
class AnomalyResult:
    """Result of anomaly detection analysis"""
    anomaly_type: AnomalyType
    severity: AlertSeverity
    confidence: float  # 0.0 - 1.0
    detection_method: DetectionMethod
    timestamp: datetime
    pipeline_id: str
    metrics: PipelineMetrics
    
    # Detailed analysis
    baseline_value: Optional[float] = None
    actual_value: Optional[float] = None
    deviation_score: float = 0.0
    pattern_similarity: float = 0.0
    
    # Explanatory information
    description: str = ""
    contributing_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass 
class PredictionResult:
    """Result of failure prediction analysis"""
    failure_probability: float  # 0.0 - 1.0
    predicted_failure_time: Optional[datetime] = None
    failure_mode: Optional[str] = None
    confidence: float = 0.0
    contributing_indicators: List[str] = field(default_factory=list)
    prevention_recommendations: List[str] = field(default_factory=list)

class AnomalyDetector:
    """
    Advanced anomaly detection using multiple approaches:
    1. Statistical analysis (z-score, IQR, trend analysis)
    2. HDC pattern matching against known good/bad patterns
    3. Temporal analysis for detecting cyclic anomalies
    4. Multi-dimensional analysis across metrics
    """
    
    def __init__(self, hdc_dimension: int = 10000, history_window: int = 100):
        self.hdc_dimension = hdc_dimension
        self.history_window = history_window
        
        # HDC components
        self.memory = HierarchicalMemory(hdc_dimension)
        self.behavior_learner = BehaviorLearner(hdc_dimension)
        
        # Historical data for analysis
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self.baseline_patterns: Dict[str, List[HyperVector]] = defaultdict(list)
        self.anomaly_patterns: Dict[str, List[HyperVector]] = defaultdict(list)
        
        # Statistical baselines
        self.baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Temporal patterns
        self.temporal_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.seasonality_data: Dict[str, List[float]] = defaultdict(list)
        
        self.logger = logging.getLogger('anomaly_detector')
        
        # Configuration
        self.config = {
            'statistical_threshold': 2.5,  # Z-score threshold
            'pattern_similarity_threshold': 0.8,
            'trend_window': 10,
            'seasonal_window': 168,  # 1 week in hours
            'min_history_for_detection': 10
        }
        
        self.logger.info("Anomaly detector initialized", 
                        hdc_dimension=hdc_dimension,
                        history_window=history_window)
    
    def detect_anomalies(self, metrics: PipelineMetrics) -> List[AnomalyResult]:
        """
        Comprehensive anomaly detection using multiple methods
        Returns list of detected anomalies with confidence scores
        """
        
        anomalies = []
        pipeline_id = metrics.pipeline_id
        
        # Store metrics in history
        self.metrics_history[pipeline_id].append(metrics)
        
        # Skip detection if insufficient history
        if len(self.metrics_history[pipeline_id]) < self.config['min_history_for_detection']:
            return anomalies
        
        # Update baselines and thresholds
        self._update_baselines(pipeline_id)
        
        # 1. Statistical anomaly detection
        statistical_anomalies = self._detect_statistical_anomalies(metrics)
        anomalies.extend(statistical_anomalies)
        
        # 2. HDC pattern-based anomaly detection
        pattern_anomalies = self._detect_pattern_anomalies(metrics)
        anomalies.extend(pattern_anomalies)
        
        # 3. Temporal anomaly detection
        temporal_anomalies = self._detect_temporal_anomalies(metrics)
        anomalies.extend(temporal_anomalies)
        
        # 4. Multi-dimensional correlation anomalies
        correlation_anomalies = self._detect_correlation_anomalies(metrics)
        anomalies.extend(correlation_anomalies)
        
        # 5. Quality regression detection
        quality_anomalies = self._detect_quality_anomalies(metrics)
        anomalies.extend(quality_anomalies)
        
        # Remove duplicates and rank by confidence
        anomalies = self._deduplicate_and_rank_anomalies(anomalies)
        
        # Learn from detection results
        self._learn_from_detection(metrics, anomalies)
        
        return anomalies
    
    def _detect_statistical_anomalies(self, metrics: PipelineMetrics) -> List[AnomalyResult]:
        """Detect anomalies using statistical analysis"""
        
        anomalies = []
        pipeline_id = metrics.pipeline_id
        history = list(self.metrics_history[pipeline_id])
        
        if len(history) < 5:
            return anomalies
        
        # Key metrics to analyze
        metric_extractors = {
            'duration': lambda m: m.duration_seconds,
            'error_count': lambda m: m.error_count,
            'cpu_usage': lambda m: m.cpu_usage,
            'memory_usage': lambda m: m.memory_usage,
            'health_score': lambda m: m.get_health_score(),
            'test_coverage': lambda m: m.test_coverage,
            'code_quality': lambda m: m.code_quality_score
        }
        
        for metric_name, extractor in metric_extractors.items():
            try:
                # Extract historical values
                historical_values = [extractor(m) for m in history[:-1]]  # Exclude current
                current_value = extractor(metrics)
                
                if len(historical_values) < 3:
                    continue
                
                # Statistical analysis
                mean_val = statistics.mean(historical_values)
                std_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
                
                if std_val == 0:
                    continue
                
                # Z-score analysis
                z_score = abs(current_value - mean_val) / std_val
                
                if z_score > self.config['statistical_threshold']:
                    # Determine severity based on z-score
                    if z_score > 4.0:
                        severity = AlertSeverity.CRITICAL
                    elif z_score > 3.0:
                        severity = AlertSeverity.WARNING
                    else:
                        severity = AlertSeverity.INFO
                    
                    # Calculate confidence based on z-score and sample size
                    confidence = min(1.0, z_score / 5.0) * min(1.0, len(historical_values) / 20.0)
                    
                    # Determine anomaly type
                    if metric_name in ['duration']:
                        anomaly_type = AnomalyType.PERFORMANCE_DEGRADATION
                    elif metric_name in ['cpu_usage', 'memory_usage']:
                        anomaly_type = AnomalyType.RESOURCE_ANOMALY
                    elif metric_name in ['error_count']:
                        anomaly_type = AnomalyType.ERROR_SPIKE
                    else:
                        anomaly_type = AnomalyType.QUALITY_REGRESSION
                    
                    # Create anomaly result
                    anomaly = AnomalyResult(
                        anomaly_type=anomaly_type,
                        severity=severity,
                        confidence=confidence,
                        detection_method=DetectionMethod.STATISTICAL,
                        timestamp=datetime.now(),
                        pipeline_id=pipeline_id,
                        metrics=metrics,
                        baseline_value=mean_val,
                        actual_value=current_value,
                        deviation_score=z_score,
                        description=f"{metric_name} deviation: {current_value:.2f} (baseline: {mean_val:.2f}, z-score: {z_score:.2f})",
                        contributing_factors=[f"Statistical deviation in {metric_name}"],
                        recommendations=self._generate_statistical_recommendations(metric_name, current_value, mean_val)
                    )
                    
                    anomalies.append(anomaly)
                    
            except Exception as e:
                self.logger.error(f"Error in statistical anomaly detection for {metric_name}: {e}")
        
        return anomalies
    
    def _detect_pattern_anomalies(self, metrics: PipelineMetrics) -> List[AnomalyResult]:
        """Detect anomalies using HDC pattern matching"""
        
        anomalies = []
        pipeline_id = metrics.pipeline_id
        
        if not metrics.pattern_signature:
            return anomalies
        
        # Compare against known baseline patterns
        baseline_patterns = self.baseline_patterns.get(pipeline_id, [])
        anomaly_patterns = self.anomaly_patterns.get(pipeline_id, [])
        
        if not baseline_patterns:
            return anomalies
        
        # Calculate similarity to baseline patterns
        baseline_similarities = [
            metrics.pattern_signature.similarity(pattern) 
            for pattern in baseline_patterns
        ]
        max_baseline_similarity = max(baseline_similarities) if baseline_similarities else 0.0
        
        # Calculate similarity to known anomaly patterns
        if anomaly_patterns:
            anomaly_similarities = [
                metrics.pattern_signature.similarity(pattern)
                for pattern in anomaly_patterns
            ]
            max_anomaly_similarity = max(anomaly_similarities) if anomaly_similarities else 0.0
        else:
            max_anomaly_similarity = 0.0
        
        # Detect pattern deviation
        if max_baseline_similarity < self.config['pattern_similarity_threshold']:
            # Low similarity to baseline patterns indicates anomaly
            confidence = 1.0 - max_baseline_similarity
            
            # Higher confidence if similar to known anomaly patterns
            if max_anomaly_similarity > 0.7:
                confidence = min(1.0, confidence + 0.3)
            
            severity = AlertSeverity.WARNING
            if confidence > 0.8:
                severity = AlertSeverity.CRITICAL
            elif confidence < 0.5:
                severity = AlertSeverity.INFO
            
            anomaly = AnomalyResult(
                anomaly_type=AnomalyType.PATTERN_DEVIATION,
                severity=severity,
                confidence=confidence,
                detection_method=DetectionMethod.HDC_PATTERN,
                timestamp=datetime.now(),
                pipeline_id=pipeline_id,
                metrics=metrics,
                pattern_similarity=max_baseline_similarity,
                description=f"Pattern deviation detected: {max_baseline_similarity:.2f} similarity to baseline",
                contributing_factors=["HDC pattern analysis shows deviation from normal behavior"],
                recommendations=[
                    "Review recent changes that may have altered pipeline behavior",
                    "Compare current execution with historical successful runs",
                    "Consider reverting to known good configuration"
                ]
            )
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def _detect_temporal_anomalies(self, metrics: PipelineMetrics) -> List[AnomalyResult]:
        """Detect anomalies based on temporal patterns and trends"""
        
        anomalies = []
        pipeline_id = metrics.pipeline_id
        history = list(self.metrics_history[pipeline_id])
        
        if len(history) < self.config['trend_window']:
            return anomalies
        
        # Trend analysis for key metrics
        trend_metrics = ['duration_seconds', 'error_count', 'cpu_usage', 'memory_usage']
        
        for metric_name in trend_metrics:
            try:
                # Extract recent values for trend analysis
                recent_values = [getattr(m, metric_name, 0) for m in history[-self.config['trend_window']:]]
                
                if len(recent_values) < 5:
                    continue
                
                # Calculate trend using linear regression
                x = list(range(len(recent_values)))
                slope = self._calculate_trend_slope(x, recent_values)
                
                # Detect significant trends
                current_value = getattr(metrics, metric_name, 0)
                baseline_value = statistics.mean(recent_values[:-1])
                
                # Trend significance thresholds
                if metric_name == 'duration_seconds':
                    # Increasing duration is bad
                    if slope > 5.0 and current_value > baseline_value * 1.2:
                        anomalies.append(self._create_temporal_anomaly(
                            metrics, AnomalyType.PERFORMANCE_DEGRADATION,
                            f"Increasing {metric_name} trend detected",
                            slope, current_value, baseline_value
                        ))
                
                elif metric_name == 'error_count':
                    # Increasing errors is bad
                    if slope > 0.1 and current_value > baseline_value + 1:
                        anomalies.append(self._create_temporal_anomaly(
                            metrics, AnomalyType.ERROR_SPIKE,
                            f"Increasing {metric_name} trend detected",
                            slope, current_value, baseline_value
                        ))
                
                elif metric_name in ['cpu_usage', 'memory_usage']:
                    # Increasing resource usage trends
                    if slope > 0.05 and current_value > baseline_value + 0.1:
                        anomalies.append(self._create_temporal_anomaly(
                            metrics, AnomalyType.RESOURCE_ANOMALY,
                            f"Increasing {metric_name} trend detected",
                            slope, current_value, baseline_value
                        ))
                
            except Exception as e:
                self.logger.error(f"Error in temporal anomaly detection for {metric_name}: {e}")
        
        return anomalies
    
    def _detect_correlation_anomalies(self, metrics: PipelineMetrics) -> List[AnomalyResult]:
        """Detect anomalies based on unusual correlations between metrics"""
        
        anomalies = []
        pipeline_id = metrics.pipeline_id
        history = list(self.metrics_history[pipeline_id])
        
        if len(history) < 10:
            return anomalies
        
        # Expected correlations
        expected_correlations = [
            ('duration_seconds', 'cpu_usage', 'positive'),
            ('error_count', 'success_rate', 'negative'),
            ('test_coverage', 'code_quality_score', 'positive'),
            ('memory_usage', 'duration_seconds', 'positive')
        ]
        
        for metric1, metric2, expected_direction in expected_correlations:
            try:
                # Extract historical correlation
                values1 = [getattr(m, metric1, 0) for m in history]
                values2 = [getattr(m, metric2, 0) for m in history]
                
                if len(values1) < 5 or len(values2) < 5:
                    continue
                
                # Calculate correlation coefficient
                correlation = self._calculate_correlation(values1, values2)
                
                # Check if correlation violates expectation
                current_val1 = getattr(metrics, metric1, 0)
                current_val2 = getattr(metrics, metric2, 0)
                
                # Predict expected value based on correlation
                if len(values1) > 1 and len(values2) > 1:
                    mean1, mean2 = statistics.mean(values1[:-1]), statistics.mean(values2[:-1])
                    std1, std2 = statistics.stdev(values1[:-1]), statistics.stdev(values2[:-1])
                    
                    if std1 > 0 and std2 > 0:
                        expected_val2 = mean2 + correlation * (std2 / std1) * (current_val1 - mean1)
                        deviation = abs(current_val2 - expected_val2) / (std2 + 1e-6)
                        
                        if deviation > 2.0:  # Significant correlation violation
                            anomaly = AnomalyResult(
                                anomaly_type=AnomalyType.PATTERN_DEVIATION,
                                severity=AlertSeverity.WARNING,
                                confidence=min(1.0, deviation / 3.0),
                                detection_method=DetectionMethod.STATISTICAL,
                                timestamp=datetime.now(),
                                pipeline_id=pipeline_id,
                                metrics=metrics,
                                deviation_score=deviation,
                                description=f"Correlation anomaly: {metric1} vs {metric2} (expected: {expected_val2:.2f}, actual: {current_val2:.2f})",
                                contributing_factors=[f"Unusual correlation between {metric1} and {metric2}"],
                                recommendations=["Investigate interdependent system behaviors"]
                            )
                            
                            anomalies.append(anomaly)
                            
            except Exception as e:
                self.logger.error(f"Error in correlation anomaly detection: {e}")
        
        return anomalies
    
    def _detect_quality_anomalies(self, metrics: PipelineMetrics) -> List[AnomalyResult]:
        """Detect quality-related anomalies"""
        
        anomalies = []
        pipeline_id = metrics.pipeline_id
        history = list(self.metrics_history[pipeline_id])
        
        if len(history) < 5:
            return anomalies
        
        # Quality thresholds
        quality_thresholds = {
            'test_coverage': 0.7,
            'code_quality_score': 0.8,
            'security_score': 0.85
        }
        
        for metric_name, threshold in quality_thresholds.items():
            current_value = getattr(metrics, metric_name, 1.0)
            
            if current_value < threshold:
                # Calculate severity based on how far below threshold
                severity_factor = (threshold - current_value) / threshold
                
                if severity_factor > 0.3:
                    severity = AlertSeverity.CRITICAL
                elif severity_factor > 0.15:
                    severity = AlertSeverity.WARNING
                else:
                    severity = AlertSeverity.INFO
                
                # Check if this is a regression from historical performance
                historical_values = [getattr(m, metric_name, 1.0) for m in history[-10:]]
                if historical_values:
                    avg_historical = statistics.mean(historical_values)
                    
                    if current_value < avg_historical * 0.9:  # 10% regression
                        anomaly = AnomalyResult(
                            anomaly_type=AnomalyType.QUALITY_REGRESSION,
                            severity=severity,
                            confidence=severity_factor,
                            detection_method=DetectionMethod.STATISTICAL,
                            timestamp=datetime.now(),
                            pipeline_id=pipeline_id,
                            metrics=metrics,
                            baseline_value=avg_historical,
                            actual_value=current_value,
                            description=f"Quality regression in {metric_name}: {current_value:.2f} (historical avg: {avg_historical:.2f})",
                            contributing_factors=[f"Quality metric {metric_name} below threshold and historical average"],
                            recommendations=self._generate_quality_recommendations(metric_name)
                        )
                        
                        anomalies.append(anomaly)
        
        return anomalies
    
    def _create_temporal_anomaly(self, metrics: PipelineMetrics, anomaly_type: AnomalyType,
                                description: str, slope: float, current_value: float, 
                                baseline_value: float) -> AnomalyResult:
        """Helper to create temporal anomaly result"""
        
        # Determine severity based on trend magnitude
        trend_magnitude = abs(slope)
        if trend_magnitude > 10.0:
            severity = AlertSeverity.CRITICAL
        elif trend_magnitude > 5.0:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.INFO
        
        confidence = min(1.0, trend_magnitude / 10.0)
        
        return AnomalyResult(
            anomaly_type=anomaly_type,
            severity=severity,
            confidence=confidence,
            detection_method=DetectionMethod.TEMPORAL,
            timestamp=datetime.now(),
            pipeline_id=metrics.pipeline_id,
            metrics=metrics,
            baseline_value=baseline_value,
            actual_value=current_value,
            deviation_score=slope,
            description=description,
            contributing_factors=["Temporal trend analysis"],
            recommendations=["Monitor trend closely", "Consider proactive intervention"]
        )
    
    def _calculate_trend_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate linear trend slope using least squares"""
        
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        try:
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] * x[i] for i in range(n))
            sum_y2 = sum(y[i] * y[i] for i in range(n))
            
            numerator = n * sum_xy - sum_x * sum_y
            denominator = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
            
            if denominator == 0:
                return 0.0
            
            return numerator / denominator
            
        except Exception:
            return 0.0
    
    def _generate_statistical_recommendations(self, metric_name: str, current_value: float, 
                                            baseline_value: float) -> List[str]:
        """Generate recommendations based on statistical anomaly"""
        
        recommendations = []
        
        if metric_name == 'duration':
            if current_value > baseline_value:
                recommendations.extend([
                    "Investigate performance bottlenecks",
                    "Check for resource constraints",
                    "Review recent code changes for performance regressions"
                ])
        elif metric_name == 'error_count':
            recommendations.extend([
                "Review error logs for common failure patterns",
                "Check for infrastructure issues",
                "Validate recent deployments"
            ])
        elif metric_name in ['cpu_usage', 'memory_usage']:
            recommendations.extend([
                "Scale resources if needed",
                "Profile application for resource leaks",
                "Optimize resource-intensive operations"
            ])
        
        return recommendations
    
    def _generate_quality_recommendations(self, metric_name: str) -> List[str]:
        """Generate recommendations for quality issues"""
        
        recommendations_map = {
            'test_coverage': [
                "Add missing unit tests",
                "Improve integration test coverage",
                "Review test strategy and identify gaps"
            ],
            'code_quality_score': [
                "Address code smells and technical debt",
                "Improve code documentation",
                "Refactor complex functions"
            ],
            'security_score': [
                "Fix identified security vulnerabilities",
                "Update dependencies with security patches",
                "Review security best practices"
            ]
        }
        
        return recommendations_map.get(metric_name, ["Review and improve quality metrics"])
    
    def _deduplicate_and_rank_anomalies(self, anomalies: List[AnomalyResult]) -> List[AnomalyResult]:
        """Remove duplicate anomalies and rank by confidence"""
        
        # Group similar anomalies
        grouped_anomalies = defaultdict(list)
        
        for anomaly in anomalies:
            key = (anomaly.anomaly_type, anomaly.pipeline_id)
            grouped_anomalies[key].append(anomaly)
        
        # Keep highest confidence anomaly from each group
        deduplicated = []
        for group in grouped_anomalies.values():
            best_anomaly = max(group, key=lambda a: a.confidence)
            deduplicated.append(best_anomaly)
        
        # Sort by confidence (highest first)
        deduplicated.sort(key=lambda a: a.confidence, reverse=True)
        
        return deduplicated
    
    def _update_baselines(self, pipeline_id: str):
        """Update statistical baselines and thresholds"""
        
        history = list(self.metrics_history[pipeline_id])
        if len(history) < 5:
            return
        
        # Update statistical baselines
        metrics_to_track = ['duration_seconds', 'error_count', 'cpu_usage', 'memory_usage', 'test_coverage']
        
        for metric_name in metrics_to_track:
            values = [getattr(m, metric_name, 0) for m in history]
            
            if values:
                self.baselines[pipeline_id][metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0,
                    'median': statistics.median(values),
                    'q25': np.percentile(values, 25),
                    'q75': np.percentile(values, 75)
                }
    
    def _learn_from_detection(self, metrics: PipelineMetrics, anomalies: List[AnomalyResult]):
        """Learn from detection results to improve future detection"""
        
        pipeline_id = metrics.pipeline_id
        
        if not metrics.pattern_signature:
            return
        
        # Store patterns based on detection results
        if not anomalies:
            # No anomalies detected - store as baseline pattern
            self.baseline_patterns[pipeline_id].append(metrics.pattern_signature)
            
            # Keep only recent baseline patterns
            if len(self.baseline_patterns[pipeline_id]) > 50:
                self.baseline_patterns[pipeline_id] = self.baseline_patterns[pipeline_id][-50:]
        
        else:
            # Anomalies detected - store as anomaly pattern
            for anomaly in anomalies:
                if anomaly.confidence > 0.7:  # High confidence anomalies
                    self.anomaly_patterns[pipeline_id].append(metrics.pattern_signature)
                    
                    # Keep only recent anomaly patterns
                    if len(self.anomaly_patterns[pipeline_id]) > 20:
                        self.anomaly_patterns[pipeline_id] = self.anomaly_patterns[pipeline_id][-20:]

class FailurePredictor:
    """
    Predictive failure analysis using pattern recognition and trend analysis
    Leverages HDC similarity matching and temporal analysis for early warning
    """
    
    def __init__(self, hdc_dimension: int = 10000, prediction_horizon: int = 60):
        self.hdc_dimension = hdc_dimension
        self.prediction_horizon = prediction_horizon  # minutes
        
        # HDC components
        self.memory = HierarchicalMemory(hdc_dimension)
        
        # Historical failure data
        self.failure_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.pre_failure_patterns: Dict[str, List[HyperVector]] = defaultdict(list)
        
        # Predictive models
        self.trend_models: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.pattern_models: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        self.logger = logging.getLogger('failure_predictor')
        self.logger.info("Failure predictor initialized")
    
    def predict_failure(self, metrics: PipelineMetrics, 
                       recent_metrics: List[PipelineMetrics]) -> PredictionResult:
        """
        Predict probability and timing of pipeline failure
        """
        
        pipeline_id = metrics.pipeline_id
        
        # Trend-based prediction
        trend_prediction = self._predict_from_trends(metrics, recent_metrics)
        
        # Pattern-based prediction
        pattern_prediction = self._predict_from_patterns(metrics)
        
        # Resource-based prediction
        resource_prediction = self._predict_from_resources(metrics, recent_metrics)
        
        # Combine predictions
        combined_prediction = self._combine_predictions([
            trend_prediction,
            pattern_prediction, 
            resource_prediction
        ])
        
        return combined_prediction
    
    def _predict_from_trends(self, metrics: PipelineMetrics, 
                           recent_metrics: List[PipelineMetrics]) -> PredictionResult:
        """Predict failure based on metric trends"""
        
        if len(recent_metrics) < 5:
            return PredictionResult(failure_probability=0.0)
        
        # Analyze trends in key failure indicators
        indicators = {
            'error_rate': [m.error_count / max(1, m.error_count + 50) for m in recent_metrics],
            'duration_trend': [m.duration_seconds for m in recent_metrics],
            'resource_pressure': [(m.cpu_usage + m.memory_usage) / 2 for m in recent_metrics],
            'quality_decline': [1.0 - m.get_health_score() for m in recent_metrics]
        }
        
        failure_probability = 0.0
        contributing_factors = []
        
        for indicator_name, values in indicators.items():
            if len(values) >= 3:
                # Calculate trend
                x = list(range(len(values)))
                slope = self._calculate_trend_slope(x, values)
                
                # Assess failure risk based on trend
                if indicator_name == 'error_rate' and slope > 0.01:
                    risk = min(0.8, slope * 10)
                    failure_probability = max(failure_probability, risk)
                    contributing_factors.append(f"Increasing error rate trend ({slope:.3f})")
                
                elif indicator_name == 'duration_trend' and slope > 10:
                    risk = min(0.6, slope / 100)
                    failure_probability = max(failure_probability, risk)
                    contributing_factors.append(f"Performance degradation trend ({slope:.1f}s/iteration)")
                
                elif indicator_name == 'resource_pressure' and slope > 0.05:
                    risk = min(0.7, slope * 5)
                    failure_probability = max(failure_probability, risk)
                    contributing_factors.append(f"Increasing resource pressure ({slope:.3f})")
                
                elif indicator_name == 'quality_decline' and slope > 0.02:
                    risk = min(0.5, slope * 10)
                    failure_probability = max(failure_probability, risk)
                    contributing_factors.append(f"Quality decline trend ({slope:.3f})")
        
        # Predict failure timing if probability is significant
        predicted_time = None
        if failure_probability > 0.3:
            # Estimate time to failure based on trend velocity
            time_to_critical = self._estimate_time_to_critical(recent_metrics)
            if time_to_critical:
                predicted_time = datetime.now() + timedelta(minutes=time_to_critical)
        
        return PredictionResult(
            failure_probability=failure_probability,
            predicted_failure_time=predicted_time,
            failure_mode="trend_based_degradation",
            confidence=min(1.0, len(recent_metrics) / 10.0),
            contributing_indicators=contributing_factors,
            prevention_recommendations=self._generate_trend_prevention_recommendations(contributing_factors)
        )
    
    def _predict_from_patterns(self, metrics: PipelineMetrics) -> PredictionResult:
        """Predict failure based on pattern similarity to historical failures"""
        
        if not metrics.pattern_signature:
            return PredictionResult(failure_probability=0.0)
        
        pipeline_id = metrics.pipeline_id
        pre_failure_patterns = self.pre_failure_patterns.get(pipeline_id, [])
        
        if not pre_failure_patterns:
            return PredictionResult(failure_probability=0.0)
        
        # Calculate similarity to pre-failure patterns
        similarities = [
            metrics.pattern_signature.similarity(pattern)
            for pattern in pre_failure_patterns
        ]
        
        max_similarity = max(similarities) if similarities else 0.0
        
        # Higher similarity to pre-failure patterns indicates higher failure risk
        failure_probability = max_similarity
        
        return PredictionResult(
            failure_probability=failure_probability,
            failure_mode="pattern_similarity",
            confidence=min(1.0, len(pre_failure_patterns) / 5.0),
            contributing_indicators=[f"High similarity ({max_similarity:.2f}) to historical pre-failure patterns"],
            prevention_recommendations=[
                "Review patterns that led to previous failures",
                "Consider proactive intervention based on historical data"
            ]
        )
    
    def _predict_from_resources(self, metrics: PipelineMetrics, 
                              recent_metrics: List[PipelineMetrics]) -> PredictionResult:
        """Predict failure based on resource utilization patterns"""
        
        current_resource_stress = (metrics.cpu_usage + metrics.memory_usage + metrics.disk_usage) / 3
        
        failure_probability = 0.0
        contributing_factors = []
        
        # Critical resource thresholds
        if metrics.cpu_usage > 0.95:
            failure_probability = max(failure_probability, 0.8)
            contributing_factors.append("Critical CPU usage")
        
        if metrics.memory_usage > 0.95:
            failure_probability = max(failure_probability, 0.9)
            contributing_factors.append("Critical memory usage")
        
        if metrics.disk_usage > 0.95:
            failure_probability = max(failure_probability, 0.7)
            contributing_factors.append("Critical disk usage")
        
        # Resource exhaustion trends
        if len(recent_metrics) >= 3:
            resource_trends = [(m.cpu_usage + m.memory_usage + m.disk_usage) / 3 for m in recent_metrics]
            if len(resource_trends) >= 3:
                trend_slope = self._calculate_trend_slope(list(range(len(resource_trends))), resource_trends)
                
                if trend_slope > 0.1 and current_resource_stress > 0.8:
                    failure_probability = max(failure_probability, 0.6)
                    contributing_factors.append("Accelerating resource consumption")
        
        return PredictionResult(
            failure_probability=failure_probability,
            failure_mode="resource_exhaustion",
            confidence=0.8,
            contributing_indicators=contributing_factors,
            prevention_recommendations=[
                "Scale resources immediately",
                "Identify and optimize resource-intensive processes",
                "Implement resource monitoring alerts"
            ]
        )
    
    def _combine_predictions(self, predictions: List[PredictionResult]) -> PredictionResult:
        """Combine multiple prediction results into a single result"""
        
        # Weight predictions by confidence
        weighted_probability = 0.0
        total_weight = 0.0
        all_indicators = []
        all_recommendations = []
        
        for pred in predictions:
            if pred.confidence > 0:
                weight = pred.confidence
                weighted_probability += pred.failure_probability * weight
                total_weight += weight
                all_indicators.extend(pred.contributing_indicators)
                all_recommendations.extend(pred.prevention_recommendations)
        
        if total_weight > 0:
            final_probability = weighted_probability / total_weight
        else:
            final_probability = 0.0
        
        # Select most likely failure mode
        failure_modes = [p.failure_mode for p in predictions if p.failure_mode and p.confidence > 0.5]
        primary_failure_mode = failure_modes[0] if failure_modes else None
        
        # Select earliest predicted time
        predicted_times = [p.predicted_failure_time for p in predictions if p.predicted_failure_time]
        earliest_time = min(predicted_times) if predicted_times else None
        
        return PredictionResult(
            failure_probability=final_probability,
            predicted_failure_time=earliest_time,
            failure_mode=primary_failure_mode,
            confidence=total_weight / len(predictions) if predictions else 0.0,
            contributing_indicators=list(set(all_indicators)),
            prevention_recommendations=list(set(all_recommendations))
        )
    
    def _calculate_trend_slope(self, x: List[float], y: List[float]) -> float:
        """Calculate linear trend slope"""
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] * x[i] for i in range(n))
        
        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return slope
    
    def _estimate_time_to_critical(self, recent_metrics: List[PipelineMetrics]) -> Optional[int]:
        """Estimate minutes until system reaches critical state"""
        
        if len(recent_metrics) < 3:
            return None
        
        # Analyze trends in critical metrics
        error_counts = [m.error_count for m in recent_metrics]
        durations = [m.duration_seconds for m in recent_metrics]
        
        # Estimate time to reach failure thresholds
        times_to_failure = []
        
        # Error count trend
        if len(error_counts) >= 3:
            slope = self._calculate_trend_slope(list(range(len(error_counts))), error_counts)
            if slope > 0:
                current_errors = error_counts[-1]
                critical_errors = 10  # Threshold for critical error count
                
                if current_errors < critical_errors:
                    time_to_critical = (critical_errors - current_errors) / slope
                    times_to_failure.append(time_to_critical * 30)  # Convert to minutes (assuming 30min intervals)
        
        # Duration trend
        if len(durations) >= 3:
            slope = self._calculate_trend_slope(list(range(len(durations))), durations)
            if slope > 0:
                current_duration = durations[-1]
                critical_duration = current_duration * 2  # 100% increase is critical
                
                time_to_critical = (critical_duration - current_duration) / slope
                times_to_failure.append(time_to_critical * 30)  # Convert to minutes
        
        if times_to_failure:
            return min(times_to_failure)  # Most urgent prediction
        
        return None
    
    def _generate_trend_prevention_recommendations(self, factors: List[str]) -> List[str]:
        """Generate recommendations based on trend factors"""
        
        recommendations = []
        
        for factor in factors:
            if "error rate" in factor:
                recommendations.extend([
                    "Investigate root cause of increasing errors",
                    "Implement error monitoring and alerting",
                    "Consider rolling back recent changes"
                ])
            elif "performance" in factor:
                recommendations.extend([
                    "Profile application for performance bottlenecks",
                    "Scale computational resources",
                    "Optimize critical path operations"
                ])
            elif "resource pressure" in factor:
                recommendations.extend([
                    "Scale infrastructure resources",
                    "Implement resource usage monitoring",
                    "Optimize resource-intensive processes"
                ])
            elif "quality decline" in factor:
                recommendations.extend([
                    "Review code quality metrics",
                    "Increase testing coverage",
                    "Address technical debt"
                ])
        
        return list(set(recommendations))  # Remove duplicates
    
    def record_failure(self, pipeline_id: str, failure_time: datetime, 
                      pre_failure_metrics: List[PipelineMetrics]):
        """Record a failure event for learning"""
        
        failure_record = {
            'failure_time': failure_time,
            'pre_failure_window': len(pre_failure_metrics),
            'failure_indicators': self._extract_failure_indicators(pre_failure_metrics)
        }
        
        self.failure_history[pipeline_id].append(failure_record)
        
        # Store pre-failure patterns for future prediction
        for metrics in pre_failure_metrics[-5:]:  # Last 5 data points before failure
            if metrics.pattern_signature:
                self.pre_failure_patterns[pipeline_id].append(metrics.pattern_signature)
        
        self.logger.info(f"Recorded failure for pipeline {pipeline_id}")
    
    def _extract_failure_indicators(self, metrics_list: List[PipelineMetrics]) -> Dict[str, Any]:
        """Extract key indicators that preceded failure"""
        
        if not metrics_list:
            return {}
        
        indicators = {}
        
        # Calculate trends leading to failure
        error_counts = [m.error_count for m in metrics_list]
        durations = [m.duration_seconds for m in metrics_list]
        health_scores = [m.get_health_score() for m in metrics_list]
        
        if len(error_counts) >= 2:
            indicators['error_trend'] = error_counts[-1] - error_counts[0]
        
        if len(durations) >= 2:
            indicators['duration_trend'] = durations[-1] - durations[0]
        
        if len(health_scores) >= 2:
            indicators['health_decline'] = health_scores[0] - health_scores[-1]
        
        # Final state before failure
        final_metrics = metrics_list[-1]
        indicators['final_cpu_usage'] = final_metrics.cpu_usage
        indicators['final_memory_usage'] = final_metrics.memory_usage
        indicators['final_error_count'] = final_metrics.error_count
        
        return indicators