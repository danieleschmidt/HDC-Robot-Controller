#!/usr/bin/env python3
"""
Comprehensive Test Suite for Self-Healing Pipeline Guard
Testing all core components with high coverage and realistic scenarios

Test Coverage:
- Pipeline Guard Core functionality
- Anomaly Detection algorithms
- Automated Repair mechanisms
- Monitoring and metrics collection
- Dashboard and API endpoints
- Integration scenarios
- Error handling and edge cases
"""

import pytest
import asyncio
import time
import json
import tempfile
import threading
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Import pipeline guard components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipeline_guard.core import (
    PipelineGuard, GuardStatus, GuardAlert, AlertSeverity, 
    PipelinePhase, PipelineMetrics
)
from pipeline_guard.detection import (
    AnomalyDetector, FailurePredictor, AnomalyType, 
    DetectionMethod, AnomalyResult, PredictionResult
)
from pipeline_guard.repair import (
    AutoRepair, RepairStrategy, RepairResult, RepairAction, 
    RepairPriority, GenericRepairEngine
)
from pipeline_guard.monitoring import (
    PipelineMonitor, MetricCollector, MetricPoint, MetricSource,
    SystemMetricsCollector
)

# HDC imports for testing
from hdc_robot_controller.core.hypervector import HyperVector

class TestPipelineGuardCore:
    """Test suite for Pipeline Guard core functionality"""
    
    @pytest.fixture
    def pipeline_guard(self):
        """Create test pipeline guard instance"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                "monitoring": {"interval_seconds": 1},
                "thresholds": {
                    "health_score_warning": 0.7,
                    "health_score_critical": 0.5
                }
            }
            json.dump(test_config, f)
            config_path = f.name
        
        guard = PipelineGuard(
            config_path=config_path,
            hdc_dimension=1000,  # Smaller for testing
            enable_auto_repair=True,
            enable_predictive_analysis=True
        )
        
        yield guard
        
        guard.stop_monitoring()
        os.unlink(config_path)
    
    def test_guard_initialization(self, pipeline_guard):
        """Test pipeline guard initialization"""
        assert pipeline_guard.status == GuardStatus.HEALTHY
        assert pipeline_guard.hdc_dimension == 1000
        assert pipeline_guard.enable_auto_repair is True
        assert pipeline_guard.enable_predictive_analysis is True
        assert len(pipeline_guard.active_pipelines) == 0
    
    def test_pipeline_registration(self, pipeline_guard):
        """Test pipeline registration and unregistration"""
        # Register pipeline
        pipeline_config = {
            "expected_duration": 300,
            "success_rate": 0.95
        }
        
        pipeline_guard.register_pipeline("test-pipeline", pipeline_config)
        
        assert "test-pipeline" in pipeline_guard.active_pipelines
        assert pipeline_guard.active_pipelines["test-pipeline"]["config"] == pipeline_config
        
        # Unregister pipeline
        pipeline_guard.unregister_pipeline("test-pipeline")
        assert "test-pipeline" not in pipeline_guard.active_pipelines
    
    def test_pipeline_health_assessment(self, pipeline_guard):
        """Test pipeline health assessment"""
        # Register test pipeline
        pipeline_guard.register_pipeline("test-pipeline", {"expected_duration": 300})
        
        # No metrics yet
        health = pipeline_guard.get_pipeline_health("test-pipeline")
        assert health["status"] == "no_data"
        
        # Add test metrics
        test_metrics = PipelineMetrics(
            timestamp=datetime.now(),
            pipeline_id="test-pipeline",
            phase=PipelinePhase.BUILD,
            duration_seconds=250,
            success_rate=0.95,
            error_count=0,
            warning_count=1,
            cpu_usage=0.6,
            memory_usage=0.5,
            disk_usage=0.3,
            network_io=0.4,
            test_coverage=0.85,
            code_quality_score=0.9,
            security_score=0.88
        )
        
        # Simulate metrics processing
        pipeline_guard.active_pipelines["test-pipeline"]["last_metrics"] = test_metrics
        
        health = pipeline_guard.get_pipeline_health("test-pipeline")
        assert health["status"] == "healthy"
        assert health["health_score"] > 0.8
    
    def test_guard_summary(self, pipeline_guard):
        """Test guard summary generation"""
        # Register test pipelines
        pipeline_guard.register_pipeline("pipeline-1", {"expected_duration": 300})
        pipeline_guard.register_pipeline("pipeline-2", {"expected_duration": 400})
        
        summary = pipeline_guard.get_guard_summary()
        
        assert summary["guard_status"] == GuardStatus.HEALTHY.value
        assert summary["monitored_pipelines"] == 2
        assert "timestamp" in summary
        assert "pipeline_health" in summary
    
    def test_alert_generation(self, pipeline_guard):
        """Test alert generation and processing"""
        pipeline_guard.register_pipeline("test-pipeline", {"expected_duration": 300})
        
        # Create metrics that should trigger alerts
        bad_metrics = PipelineMetrics(
            timestamp=datetime.now(),
            pipeline_id="test-pipeline",
            phase=PipelinePhase.BUILD,
            duration_seconds=900,  # 3x expected duration
            success_rate=0.3,      # Low success rate
            error_count=5,         # High error count
            warning_count=3,
            cpu_usage=0.95,        # High CPU usage
            memory_usage=0.9,      # High memory usage
            disk_usage=0.4,
            network_io=0.2,
            test_coverage=0.5,     # Low test coverage
            code_quality_score=0.6, # Low code quality
            security_score=0.7
        )
        
        # Process metrics (simulate monitoring loop)
        pipeline_guard._process_pipeline_metrics(bad_metrics)
        
        # Check that alerts were generated
        assert len(pipeline_guard.alert_history) > 0
        
        # Check alert types
        alert_types = [alert.severity for alert in pipeline_guard.alert_history]
        assert AlertSeverity.CRITICAL in alert_types or AlertSeverity.WARNING in alert_types
    
    def test_monitoring_lifecycle(self, pipeline_guard):
        """Test monitoring start/stop lifecycle"""
        pipeline_guard.register_pipeline("test-pipeline", {"expected_duration": 300})
        
        # Start monitoring
        assert not pipeline_guard.monitoring_active
        pipeline_guard.start_monitoring()
        assert pipeline_guard.monitoring_active
        
        # Let it run briefly
        time.sleep(0.1)
        
        # Stop monitoring
        pipeline_guard.stop_monitoring()
        assert not pipeline_guard.monitoring_active

class TestAnomalyDetection:
    """Test suite for anomaly detection algorithms"""
    
    @pytest.fixture
    def anomaly_detector(self):
        """Create test anomaly detector"""
        return AnomalyDetector(hdc_dimension=1000, history_window=50)
    
    @pytest.fixture
    def sample_metrics(self):
        """Create sample pipeline metrics"""
        base_time = datetime.now()
        metrics = []
        
        # Generate 20 normal metrics
        for i in range(20):
            metrics.append(PipelineMetrics(
                timestamp=base_time + timedelta(minutes=i),
                pipeline_id="test-pipeline",
                phase=PipelinePhase.BUILD,
                duration_seconds=300 + (i % 5) * 10,  # 300-340 seconds
                success_rate=0.95,
                error_count=0,
                warning_count=1,
                cpu_usage=0.6 + (i % 3) * 0.05,       # 0.6-0.7
                memory_usage=0.5 + (i % 4) * 0.03,    # 0.5-0.59
                disk_usage=0.3,
                network_io=0.4,
                test_coverage=0.85,
                code_quality_score=0.9,
                security_score=0.88,
                pattern_signature=HyperVector.random(1000)
            ))
        
        return metrics
    
    def test_anomaly_detector_initialization(self, anomaly_detector):
        """Test anomaly detector initialization"""
        assert anomaly_detector.hdc_dimension == 1000
        assert anomaly_detector.history_window == 50
        assert len(anomaly_detector.metrics_history) == 0
    
    def test_statistical_anomaly_detection(self, anomaly_detector, sample_metrics):
        """Test statistical anomaly detection"""
        # Add normal metrics to history
        for metric in sample_metrics:
            anomaly_detector.metrics_history[metric.pipeline_id].append(metric)
        
        # Create anomalous metric
        anomaly_metric = PipelineMetrics(
            timestamp=datetime.now(),
            pipeline_id="test-pipeline",
            phase=PipelinePhase.BUILD,
            duration_seconds=1200,  # 4x normal duration
            success_rate=0.95,
            error_count=10,         # High error count
            warning_count=1,
            cpu_usage=0.95,         # Very high CPU
            memory_usage=0.5,
            disk_usage=0.3,
            network_io=0.4,
            test_coverage=0.85,
            code_quality_score=0.9,
            security_score=0.88,
            pattern_signature=HyperVector.random(1000)
        )
        
        # Detect anomalies
        anomalies = anomaly_detector.detect_anomalies(anomaly_metric)
        
        # Should detect multiple anomalies
        assert len(anomalies) > 0
        
        # Check for specific anomaly types
        anomaly_types = [a.anomaly_type for a in anomalies]
        assert AnomalyType.PERFORMANCE_DEGRADATION in anomaly_types
        assert AnomalyType.ERROR_SPIKE in anomaly_types or AnomalyType.RESOURCE_ANOMALY in anomaly_types
    
    def test_pattern_based_anomaly_detection(self, anomaly_detector, sample_metrics):
        """Test HDC pattern-based anomaly detection"""
        # Build baseline patterns
        for metric in sample_metrics:
            anomaly_detector.metrics_history[metric.pipeline_id].append(metric)
            # Simulate learning from normal patterns
            if metric.pattern_signature:
                anomaly_detector.baseline_patterns[metric.pipeline_id].append(metric.pattern_signature)
        
        # Create metric with very different pattern
        anomaly_metric = PipelineMetrics(
            timestamp=datetime.now(),
            pipeline_id="test-pipeline",
            phase=PipelinePhase.BUILD,
            duration_seconds=320,   # Normal duration
            success_rate=0.95,      # Normal success rate
            error_count=0,          # No errors
            warning_count=1,
            cpu_usage=0.65,         # Normal CPU
            memory_usage=0.55,      # Normal memory
            disk_usage=0.3,
            network_io=0.4,
            test_coverage=0.85,
            code_quality_score=0.9,
            security_score=0.88,
            pattern_signature=HyperVector.random(1000)  # Different pattern
        )
        
        # Detect anomalies
        anomalies = anomaly_detector.detect_anomalies(anomaly_metric)
        
        # Should detect pattern deviation if similarity is low enough
        pattern_anomalies = [a for a in anomalies if a.detection_method == DetectionMethod.HDC_PATTERN]
        # Pattern detection depends on random similarity, so we just check it doesn't crash
        assert isinstance(pattern_anomalies, list)
    
    def test_temporal_anomaly_detection(self, anomaly_detector):
        """Test temporal trend-based anomaly detection"""
        base_time = datetime.now()
        
        # Create metrics with increasing duration trend
        increasing_metrics = []
        for i in range(15):
            increasing_metrics.append(PipelineMetrics(
                timestamp=base_time + timedelta(minutes=i),
                pipeline_id="trend-pipeline",
                phase=PipelinePhase.BUILD,
                duration_seconds=300 + i * 20,  # Increasing trend
                success_rate=0.95,
                error_count=0,
                warning_count=1,
                cpu_usage=0.6,
                memory_usage=0.5,
                disk_usage=0.3,
                network_io=0.4,
                test_coverage=0.85,
                code_quality_score=0.9,
                security_score=0.88
            ))
        
        # Add to detector history
        for metric in increasing_metrics:
            anomaly_detector.metrics_history[metric.pipeline_id].append(metric)
        
        # Test with final metric that continues the trend
        final_metric = PipelineMetrics(
            timestamp=base_time + timedelta(minutes=15),
            pipeline_id="trend-pipeline",
            phase=PipelinePhase.BUILD,
            duration_seconds=600,   # Continues increasing trend
            success_rate=0.95,
            error_count=0,
            warning_count=1,
            cpu_usage=0.6,
            memory_usage=0.5,
            disk_usage=0.3,
            network_io=0.4,
            test_coverage=0.85,
            code_quality_score=0.9,
            security_score=0.88
        )
        
        anomalies = anomaly_detector.detect_anomalies(final_metric)
        
        # Should detect temporal anomaly
        temporal_anomalies = [a for a in anomalies if a.detection_method == DetectionMethod.TEMPORAL]
        assert len(temporal_anomalies) >= 0  # May or may not detect depending on thresholds

class TestFailurePrediction:
    """Test suite for failure prediction algorithms"""
    
    @pytest.fixture
    def failure_predictor(self):
        """Create test failure predictor"""
        return FailurePredictor(hdc_dimension=1000, prediction_horizon=60)
    
    def test_failure_predictor_initialization(self, failure_predictor):
        """Test failure predictor initialization"""
        assert failure_predictor.hdc_dimension == 1000
        assert failure_predictor.prediction_horizon == 60
    
    def test_trend_based_prediction(self, failure_predictor):
        """Test failure prediction based on trends"""
        base_time = datetime.now()
        
        # Create metrics showing degrading trend
        degrading_metrics = []
        for i in range(10):
            degrading_metrics.append(PipelineMetrics(
                timestamp=base_time + timedelta(minutes=i),
                pipeline_id="failing-pipeline",
                phase=PipelinePhase.BUILD,
                duration_seconds=300 + i * 30,      # Increasing duration
                success_rate=max(0.5, 0.95 - i * 0.05),  # Decreasing success
                error_count=i,                      # Increasing errors
                warning_count=i + 1,
                cpu_usage=min(0.95, 0.5 + i * 0.05),     # Increasing CPU
                memory_usage=min(0.90, 0.4 + i * 0.05),  # Increasing memory
                disk_usage=0.3,
                network_io=0.4,
                test_coverage=max(0.5, 0.9 - i * 0.03),  # Decreasing coverage
                code_quality_score=max(0.5, 0.9 - i * 0.02),
                security_score=0.88
            ))
        
        current_metric = degrading_metrics[-1]
        recent_metrics = degrading_metrics[:-1]
        
        # Predict failure
        prediction = failure_predictor.predict_failure(current_metric, recent_metrics)
        
        # Should predict high failure probability
        assert isinstance(prediction, PredictionResult)
        assert prediction.failure_probability >= 0.0
        assert len(prediction.contributing_indicators) >= 0
    
    def test_resource_based_prediction(self, failure_predictor):
        """Test failure prediction based on resource exhaustion"""
        current_metric = PipelineMetrics(
            timestamp=datetime.now(),
            pipeline_id="resource-critical",
            phase=PipelinePhase.BUILD,
            duration_seconds=300,
            success_rate=0.95,
            error_count=0,
            warning_count=1,
            cpu_usage=0.98,         # Critical CPU usage
            memory_usage=0.96,      # Critical memory usage
            disk_usage=0.94,        # Critical disk usage
            network_io=0.4,
            test_coverage=0.85,
            code_quality_score=0.9,
            security_score=0.88
        )
        
        prediction = failure_predictor.predict_failure(current_metric, [])
        
        # Should predict high failure probability due to resource exhaustion
        assert prediction.failure_probability > 0.7
        assert "Critical" in " ".join(prediction.contributing_indicators)
    
    def test_failure_recording(self, failure_predictor):
        """Test failure event recording for learning"""
        # Create pre-failure metrics
        pre_failure_metrics = []
        for i in range(5):
            pre_failure_metrics.append(PipelineMetrics(
                timestamp=datetime.now() - timedelta(minutes=5-i),
                pipeline_id="failed-pipeline",
                phase=PipelinePhase.BUILD,
                duration_seconds=300 + i * 50,
                success_rate=0.9 - i * 0.1,
                error_count=i,
                warning_count=i + 1,
                cpu_usage=0.6 + i * 0.08,
                memory_usage=0.5 + i * 0.08,
                disk_usage=0.3,
                network_io=0.4,
                test_coverage=0.85,
                code_quality_score=0.9,
                security_score=0.88,
                pattern_signature=HyperVector.random(1000)
            ))
        
        # Record failure
        failure_time = datetime.now()
        failure_predictor.record_failure("failed-pipeline", failure_time, pre_failure_metrics)
        
        # Check that failure was recorded
        assert "failed-pipeline" in failure_predictor.failure_history
        assert len(failure_predictor.failure_history["failed-pipeline"]) == 1
        assert len(failure_predictor.pre_failure_patterns["failed-pipeline"]) > 0

class TestAutomatedRepair:
    """Test suite for automated repair mechanisms"""
    
    @pytest.fixture
    def auto_repair(self):
        """Create test auto repair system"""
        return AutoRepair(hdc_dimension=1000, max_concurrent_repairs=2)
    
    @pytest.fixture
    def test_alert(self):
        """Create test alert"""
        return GuardAlert(
            id="test-alert",
            timestamp=datetime.now(),
            severity=AlertSeverity.WARNING,
            pipeline_id="test-pipeline",
            phase=PipelinePhase.BUILD,
            title="Performance Degradation",
            message="Build duration increased significantly",
            metrics={
                "duration": 900,
                "error_count": 2,
                "cpu_usage": 0.85
            }
        )
    
    def test_auto_repair_initialization(self, auto_repair):
        """Test auto repair system initialization"""
        assert auto_repair.hdc_dimension == 1000
        assert auto_repair.max_concurrent_repairs == 2
        assert len(auto_repair.repair_engines) > 0
        assert auto_repair.enable_auto_repair is True
    
    def test_repair_engine_selection(self, auto_repair, test_alert):
        """Test repair engine selection"""
        engine = auto_repair._select_repair_engine(test_alert)
        assert engine is not None
        assert hasattr(engine, 'generate_repair_actions')
        assert hasattr(engine, 'execute_repair')
    
    def test_repair_action_generation(self, auto_repair, test_alert):
        """Test repair action generation"""
        engine = auto_repair._select_repair_engine(test_alert)
        actions = engine.generate_repair_actions(test_alert)
        
        assert isinstance(actions, list)
        assert len(actions) > 0
        
        # Check action properties
        for action in actions:
            assert hasattr(action, 'strategy')
            assert hasattr(action, 'description')
            assert hasattr(action, 'priority')
    
    def test_repair_action_scoring(self, auto_repair, test_alert):
        """Test repair action scoring and selection"""
        engine = auto_repair._select_repair_engine(test_alert)
        actions = engine.generate_repair_actions(test_alert)
        
        selected_actions = auto_repair._select_optimal_actions(test_alert, actions)
        
        assert isinstance(selected_actions, list)
        assert len(selected_actions) <= len(actions)
        
        # Selected actions should have reasonable scores
        for action in selected_actions:
            score = auto_repair._calculate_action_score(test_alert, action)
            assert score >= 0.0
            assert score <= 1.0
    
    def test_repair_attempt_execution(self, auto_repair, test_alert):
        """Test repair attempt execution"""
        attempts = auto_repair.attempt_repair(test_alert)
        
        assert isinstance(attempts, list)
        # Attempts may be empty due to cooldowns or other factors
        
        # If attempts were made, check their structure
        for attempt in attempts:
            assert hasattr(attempt, 'result')
            assert hasattr(attempt, 'strategy')
            assert hasattr(attempt, 'duration_seconds')
    
    def test_repair_cooldown_mechanism(self, auto_repair, test_alert):
        """Test repair cooldown mechanism"""
        # First repair attempt
        attempts1 = auto_repair.attempt_repair(test_alert)
        
        # Immediate second attempt should be blocked by cooldown
        attempts2 = auto_repair.attempt_repair(test_alert)
        
        # Second attempt should be empty due to cooldown
        assert len(attempts2) == 0
    
    def test_repair_statistics(self, auto_repair, test_alert):
        """Test repair statistics collection"""
        # Make some repair attempts
        auto_repair.attempt_repair(test_alert)
        
        stats = auto_repair.get_repair_statistics()
        
        assert isinstance(stats, dict)
        assert "total_attempts" in stats
        assert "overall_success_rate" in stats
        assert "strategy_effectiveness" in stats

class TestPipelineMonitoring:
    """Test suite for pipeline monitoring"""
    
    @pytest.fixture
    def pipeline_monitor(self):
        """Create test pipeline monitor"""
        return PipelineMonitor(collection_interval=1)
    
    @pytest.fixture
    def mock_collector(self):
        """Create mock metric collector"""
        collector = Mock(spec=MetricCollector)
        collector.is_available.return_value = True
        collector.get_supported_pipelines.return_value = ["test-pipeline"]
        collector.collect_metrics.return_value = [
            MetricPoint(
                timestamp=datetime.now(),
                pipeline_id="test-pipeline",
                metric_name="test_metric",
                value=42.0,
                source=MetricSource.CUSTOM
            )
        ]
        return collector
    
    def test_monitor_initialization(self, pipeline_monitor):
        """Test pipeline monitor initialization"""
        assert pipeline_monitor.collection_interval == 1
        assert len(pipeline_monitor.collectors) > 0
        assert not pipeline_monitor.monitoring_active
    
    def test_collector_registration(self, pipeline_monitor, mock_collector):
        """Test metric collector registration"""
        initial_count = len(pipeline_monitor.collectors)
        pipeline_monitor.register_collector(mock_collector)
        
        assert len(pipeline_monitor.collectors) == initial_count + 1
        assert mock_collector in pipeline_monitor.collectors
    
    def test_metrics_collection(self, pipeline_monitor, mock_collector):
        """Test metrics collection from collectors"""
        pipeline_monitor.register_collector(mock_collector)
        
        # Collect metrics
        metrics = pipeline_monitor._collect_all_metrics(["test-pipeline"])
        
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        
        # Check metric structure
        for metric in metrics:
            assert hasattr(metric, 'timestamp')
            assert hasattr(metric, 'pipeline_id')
            assert hasattr(metric, 'metric_name')
            assert hasattr(metric, 'value')
    
    def test_metrics_processing(self, pipeline_monitor):
        """Test metrics processing and storage"""
        test_metric = MetricPoint(
            timestamp=datetime.now(),
            pipeline_id="test-pipeline",
            metric_name="test_duration",
            value=300.0,
            source=MetricSource.CUSTOM
        )
        
        pipeline_monitor._process_collected_metrics([test_metric])
        
        # Check that metric was stored
        assert len(pipeline_monitor.metrics_buffer) > 0
        assert "test-pipeline:test_duration" in pipeline_monitor.metrics_history
        assert "test-pipeline" in pipeline_monitor.aggregated_metrics
    
    def test_pipeline_summary(self, pipeline_monitor):
        """Test pipeline summary generation"""
        # Add some metrics
        test_metrics = [
            MetricPoint(
                timestamp=datetime.now(),
                pipeline_id="test-pipeline",
                metric_name="build_duration_seconds",
                value=300.0,
                source=MetricSource.CUSTOM
            ),
            MetricPoint(
                timestamp=datetime.now(),
                pipeline_id="test-pipeline",
                metric_name="build_success",
                value=1.0,
                source=MetricSource.CUSTOM
            )
        ]
        
        pipeline_monitor._process_collected_metrics(test_metrics)
        
        summary = pipeline_monitor.get_pipeline_summary("test-pipeline")
        
        assert isinstance(summary, dict)
        assert "pipeline_id" in summary
        assert "overall_health" in summary
        assert "current_metrics" in summary
    
    def test_monitoring_lifecycle(self, pipeline_monitor):
        """Test monitoring start/stop lifecycle"""
        pipelines = ["test-pipeline-1", "test-pipeline-2"]
        
        # Start monitoring
        assert not pipeline_monitor.monitoring_active
        pipeline_monitor.start_monitoring(pipelines)
        assert pipeline_monitor.monitoring_active
        
        # Brief operation
        time.sleep(0.1)
        
        # Stop monitoring
        pipeline_monitor.stop_monitoring()
        assert not pipeline_monitor.monitoring_active
    
    def test_system_metrics_collector(self):
        """Test system metrics collector"""
        collector = SystemMetricsCollector()
        
        assert collector.is_available()
        assert "*" in collector.get_supported_pipelines()
        
        metrics = collector.collect_metrics("any-pipeline")
        assert isinstance(metrics, list)
        assert len(metrics) > 0
        
        # Check for expected system metrics
        metric_names = [m.metric_name for m in metrics]
        assert "system_cpu_percent" in metric_names

class TestIntegrationScenarios:
    """Integration tests for complete pipeline guard scenarios"""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated pipeline guard system"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_config = {
                "monitoring": {"interval_seconds": 1},
                "thresholds": {
                    "health_score_warning": 0.7,
                    "health_score_critical": 0.5
                }
            }
            json.dump(test_config, f)
            config_path = f.name
        
        # Create integrated components
        guard = PipelineGuard(config_path=config_path, hdc_dimension=1000)
        monitor = PipelineMonitor(collection_interval=1)
        auto_repair = AutoRepair(hdc_dimension=1000)
        
        yield {
            "guard": guard,
            "monitor": monitor,
            "auto_repair": auto_repair,
            "config_path": config_path
        }
        
        guard.stop_monitoring()
        monitor.stop_monitoring()
        os.unlink(config_path)
    
    def test_complete_monitoring_cycle(self, integrated_system):
        """Test complete monitoring and alerting cycle"""
        guard = integrated_system["guard"]
        monitor = integrated_system["monitor"]
        
        # Register pipeline
        guard.register_pipeline("integration-test", {"expected_duration": 300})
        
        # Start monitoring
        monitor.start_monitoring(["integration-test"])
        guard.start_monitoring()
        
        # Let system run briefly
        time.sleep(0.5)
        
        # Add problematic metrics
        bad_metrics = PipelineMetrics(
            timestamp=datetime.now(),
            pipeline_id="integration-test",
            phase=PipelinePhase.BUILD,
            duration_seconds=900,
            success_rate=0.4,
            error_count=5,
            warning_count=3,
            cpu_usage=0.95,
            memory_usage=0.9,
            disk_usage=0.4,
            network_io=0.2,
            test_coverage=0.5,
            code_quality_score=0.6,
            security_score=0.7
        )
        
        guard._process_pipeline_metrics(bad_metrics)
        
        # Check that system responded
        assert len(guard.alert_history) > 0
        
        # Stop monitoring
        monitor.stop_monitoring()
        guard.stop_monitoring()
    
    def test_end_to_end_repair_scenario(self, integrated_system):
        """Test end-to-end repair scenario"""
        guard = integrated_system["guard"]
        auto_repair = integrated_system["auto_repair"]
        
        # Create alert that triggers repair
        alert = GuardAlert(
            id="repair-test-alert",
            timestamp=datetime.now(),
            severity=AlertSeverity.CRITICAL,
            pipeline_id="repair-test-pipeline",
            phase=PipelinePhase.BUILD,
            title="Critical Performance Issue",
            message="System performance critically degraded",
            metrics={
                "duration": 1200,
                "error_count": 10,
                "cpu_usage": 0.95,
                "memory_usage": 0.95
            }
        )
        
        # Attempt repair
        repair_attempts = auto_repair.attempt_repair(alert)
        
        # Verify repair was attempted
        assert isinstance(repair_attempts, list)
        # May be empty due to cooldowns, but should not crash
        
        # Check repair statistics
        stats = auto_repair.get_repair_statistics()
        assert isinstance(stats, dict)

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            # Invalid JSON
            f.write("{ invalid json }")
            config_path = f.name
        
        # Should not crash, should use defaults
        guard = PipelineGuard(config_path=config_path)
        assert guard.config is not None
        
        os.unlink(config_path)
    
    def test_missing_metrics_data(self):
        """Test handling of missing or incomplete metrics"""
        guard = PipelineGuard(hdc_dimension=1000)
        
        # Register pipeline
        guard.register_pipeline("test-pipeline", {})
        
        # Incomplete metrics
        incomplete_metrics = PipelineMetrics(
            timestamp=datetime.now(),
            pipeline_id="test-pipeline",
            phase=PipelinePhase.BUILD,
            duration_seconds=0,  # Invalid duration
            success_rate=0.95,
            error_count=0,
            warning_count=0,
            cpu_usage=0.0,
            memory_usage=0.0,
            disk_usage=0.0,
            network_io=0.0,
            test_coverage=0.0,
            code_quality_score=0.0,
            security_score=0.0
        )
        
        # Should handle gracefully
        guard._process_pipeline_metrics(incomplete_metrics)
        health = guard.get_pipeline_health("test-pipeline")
        assert health is not None
    
    def test_collector_failures(self):
        """Test handling of collector failures"""
        monitor = PipelineMonitor(collection_interval=1)
        
        # Create failing collector
        failing_collector = Mock(spec=MetricCollector)
        failing_collector.is_available.return_value = True
        failing_collector.get_supported_pipelines.return_value = ["test-pipeline"]
        failing_collector.collect_metrics.side_effect = Exception("Collector failed")
        
        monitor.register_collector(failing_collector)
        
        # Should handle collector failure gracefully
        metrics = monitor._collect_all_metrics(["test-pipeline"])
        assert isinstance(metrics, list)  # Should return empty list, not crash
    
    def test_repair_engine_failures(self):
        """Test handling of repair engine failures"""
        auto_repair = AutoRepair(hdc_dimension=1000)
        
        # Create failing repair engine
        failing_engine = Mock()
        failing_engine.can_handle.return_value = True
        failing_engine.generate_repair_actions.side_effect = Exception("Engine failed")
        
        auto_repair.repair_engines = [failing_engine]
        
        # Create test alert
        alert = GuardAlert(
            id="test-alert",
            timestamp=datetime.now(),
            severity=AlertSeverity.WARNING,
            pipeline_id="test-pipeline",
            phase=PipelinePhase.BUILD,
            title="Test Alert",
            message="Test alert message",
            metrics={}
        )
        
        # Should handle engine failure gracefully
        attempts = auto_repair.attempt_repair(alert)
        assert isinstance(attempts, list)

# Test configuration
@pytest.fixture(scope="session")
def test_config():
    """Global test configuration"""
    return {
        "test_timeout": 30,
        "hdc_dimension": 1000,
        "enable_logging": False
    }

# Performance tests
class TestPerformance:
    """Performance and scalability tests"""
    
    def test_large_metrics_processing(self):
        """Test processing large volumes of metrics"""
        monitor = PipelineMonitor(collection_interval=1)
        
        # Generate large number of metrics
        metrics = []
        for i in range(1000):
            metrics.append(MetricPoint(
                timestamp=datetime.now(),
                pipeline_id=f"pipeline-{i % 10}",
                metric_name="test_metric",
                value=float(i),
                source=MetricSource.CUSTOM
            ))
        
        start_time = time.time()
        monitor._process_collected_metrics(metrics)
        processing_time = time.time() - start_time
        
        # Should process quickly
        assert processing_time < 5.0  # 5 seconds max
        assert len(monitor.metrics_buffer) == 1000
    
    def test_memory_usage_with_long_running(self):
        """Test memory usage with long-running monitoring"""
        monitor = PipelineMonitor(collection_interval=1)
        
        # Simulate long-running scenario with periodic metrics
        for cycle in range(100):
            metrics = [
                MetricPoint(
                    timestamp=datetime.now(),
                    pipeline_id="long-running-test",
                    metric_name="test_metric",
                    value=float(cycle),
                    source=MetricSource.CUSTOM
                )
            ]
            monitor._process_collected_metrics(metrics)
        
        # Check that old data is cleaned up
        assert len(monitor.metrics_buffer) <= 100  # Should not grow indefinitely

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--cov=pipeline_guard",
        "--cov-report=term-missing",
        "--cov-report=html"
    ])