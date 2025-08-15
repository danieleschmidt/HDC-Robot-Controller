#!/usr/bin/env python3
"""
Simple Test Runner for Pipeline Guard (without pytest dependency)
Basic functionality testing for the self-healing pipeline guard system
"""

import sys
import os
import time
import json
import tempfile
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(__file__))

# Import pipeline guard components
from pipeline_guard.core import (
    PipelineGuard, GuardStatus, GuardAlert, AlertSeverity, 
    PipelinePhase, PipelineMetrics
)
from pipeline_guard.detection import AnomalyDetector, FailurePredictor
from pipeline_guard.repair import AutoRepair
from pipeline_guard.monitoring import PipelineMonitor, SystemMetricsCollector
from hdc_robot_controller.core.hypervector import HyperVector

class TestRunner:
    """Simple test runner for pipeline guard functionality"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.total = 0
    
    def test(self, test_name, test_func):
        """Run a single test"""
        self.total += 1
        print(f"Running {test_name}...", end=" ")
        
        try:
            test_func()
            print("✅ PASSED")
            self.passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            self.failed += 1
    
    def summary(self):
        """Print test summary"""
        print(f"\n📊 Test Summary:")
        print(f"   Total: {self.total}")
        print(f"   Passed: {self.passed}")
        print(f"   Failed: {self.failed}")
        print(f"   Success Rate: {self.passed/self.total*100:.1f}%")
        return self.failed == 0

def test_pipeline_guard_initialization():
    """Test pipeline guard initialization"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        config = {"monitoring": {"interval_seconds": 1}}
        json.dump(config, f)
        config_path = f.name
    
    guard = PipelineGuard(config_path=config_path, hdc_dimension=1000)
    
    assert guard.status == GuardStatus.HEALTHY
    assert guard.hdc_dimension == 1000
    assert len(guard.active_pipelines) == 0
    
    os.unlink(config_path)

def test_pipeline_registration():
    """Test pipeline registration"""
    guard = PipelineGuard(hdc_dimension=1000)
    
    # Register pipeline
    pipeline_config = {"expected_duration": 300}
    guard.register_pipeline("test-pipeline", pipeline_config)
    
    assert "test-pipeline" in guard.active_pipelines
    assert guard.active_pipelines["test-pipeline"]["config"] == pipeline_config
    
    # Unregister pipeline
    guard.unregister_pipeline("test-pipeline")
    assert "test-pipeline" not in guard.active_pipelines

def test_metrics_processing():
    """Test pipeline metrics processing"""
    guard = PipelineGuard(hdc_dimension=1000)
    guard.register_pipeline("test-pipeline", {"expected_duration": 300})
    
    # Create test metrics
    metrics = PipelineMetrics(
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
    
    # Process metrics
    guard._process_pipeline_metrics(metrics)
    
    # Check pipeline health
    health = guard.get_pipeline_health("test-pipeline")
    assert health["status"] == "healthy"
    assert health["health_score"] > 0.8

def test_alert_generation():
    """Test alert generation for bad metrics"""
    guard = PipelineGuard(hdc_dimension=1000)
    guard.register_pipeline("test-pipeline", {"expected_duration": 300})
    
    # Create bad metrics
    bad_metrics = PipelineMetrics(
        timestamp=datetime.now(),
        pipeline_id="test-pipeline",
        phase=PipelinePhase.BUILD,
        duration_seconds=900,  # 3x expected
        success_rate=0.3,      # Low success
        error_count=5,         # High errors
        warning_count=3,
        cpu_usage=0.95,        # High CPU
        memory_usage=0.9,      # High memory
        disk_usage=0.4,
        network_io=0.2,
        test_coverage=0.5,     # Low coverage
        code_quality_score=0.6,
        security_score=0.7
    )
    
    # Process bad metrics
    guard._process_pipeline_metrics(bad_metrics)
    
    # Should generate alerts
    assert len(guard.alert_history) > 0

def test_anomaly_detection():
    """Test anomaly detection functionality"""
    detector = AnomalyDetector(hdc_dimension=1000, history_window=50)
    
    # Create normal metrics history
    for i in range(15):
        normal_metric = PipelineMetrics(
            timestamp=datetime.now() + timedelta(minutes=i),
            pipeline_id="test-pipeline",
            phase=PipelinePhase.BUILD,
            duration_seconds=300 + i * 5,  # Gradually increasing
            success_rate=0.95,
            error_count=0,
            warning_count=1,
            cpu_usage=0.6,
            memory_usage=0.5,
            disk_usage=0.3,
            network_io=0.4,
            test_coverage=0.85,
            code_quality_score=0.9,
            security_score=0.88,
            pattern_signature=HyperVector.random(1000)
        )
        detector.metrics_history[normal_metric.pipeline_id].append(normal_metric)
    
    # Create anomalous metric
    anomaly_metric = PipelineMetrics(
        timestamp=datetime.now(),
        pipeline_id="test-pipeline",
        phase=PipelinePhase.BUILD,
        duration_seconds=1200,  # Much higher than normal
        success_rate=0.95,
        error_count=10,         # High errors
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
    anomalies = detector.detect_anomalies(anomaly_metric)
    
    # Should detect anomalies
    assert len(anomalies) > 0

def test_failure_prediction():
    """Test failure prediction functionality"""
    predictor = FailurePredictor(hdc_dimension=1000)
    
    # Create metrics with degrading trend
    recent_metrics = []
    for i in range(5):
        metric = PipelineMetrics(
            timestamp=datetime.now() + timedelta(minutes=i),
            pipeline_id="failing-pipeline",
            phase=PipelinePhase.BUILD,
            duration_seconds=300 + i * 50,      # Increasing duration
            success_rate=max(0.5, 0.95 - i * 0.1),  # Decreasing success
            error_count=i,                      # Increasing errors
            warning_count=i + 1,
            cpu_usage=min(0.95, 0.5 + i * 0.1),     # Increasing CPU
            memory_usage=min(0.90, 0.4 + i * 0.1),  # Increasing memory
            disk_usage=0.3,
            network_io=0.4,
            test_coverage=0.85,
            code_quality_score=0.9,
            security_score=0.88
        )
        recent_metrics.append(metric)
    
    current_metric = recent_metrics[-1]
    
    # Predict failure
    prediction = predictor.predict_failure(current_metric, recent_metrics[:-1])
    
    # Should predict some failure probability
    assert prediction.failure_probability >= 0.0
    assert prediction.failure_probability <= 1.0

def test_automated_repair():
    """Test automated repair functionality"""
    auto_repair = AutoRepair(hdc_dimension=1000)
    
    # Create test alert
    alert = GuardAlert(
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
    
    # Attempt repair
    attempts = auto_repair.attempt_repair(alert)
    
    # Should return list of attempts (may be empty due to cooldowns)
    assert isinstance(attempts, list)

def test_pipeline_monitoring():
    """Test pipeline monitoring functionality"""
    monitor = PipelineMonitor(collection_interval=1)
    
    # Test system metrics collector
    system_collector = SystemMetricsCollector()
    assert system_collector.is_available()
    
    metrics = system_collector.collect_metrics("test-pipeline")
    assert isinstance(metrics, list)
    assert len(metrics) > 0
    
    # Test metrics processing
    monitor._process_collected_metrics(metrics)
    
    # Check aggregated metrics
    current_metrics = monitor.get_current_metrics("test-pipeline")
    assert isinstance(current_metrics, dict)

def test_monitoring_lifecycle():
    """Test monitoring start/stop lifecycle"""
    guard = PipelineGuard(hdc_dimension=1000)
    monitor = PipelineMonitor(collection_interval=1)
    
    guard.register_pipeline("test-pipeline", {"expected_duration": 300})
    
    # Start monitoring
    assert not guard.monitoring_active
    assert not monitor.monitoring_active
    
    guard.start_monitoring()
    monitor.start_monitoring(["test-pipeline"])
    
    assert guard.monitoring_active
    assert monitor.monitoring_active
    
    # Brief operation
    time.sleep(0.1)
    
    # Stop monitoring
    guard.stop_monitoring()
    monitor.stop_monitoring()
    
    assert not guard.monitoring_active
    assert not monitor.monitoring_active

def test_guard_summary():
    """Test guard summary generation"""
    guard = PipelineGuard(hdc_dimension=1000)
    
    # Register test pipelines
    guard.register_pipeline("pipeline-1", {"expected_duration": 300})
    guard.register_pipeline("pipeline-2", {"expected_duration": 400})
    
    summary = guard.get_guard_summary()
    
    assert summary["guard_status"] == GuardStatus.HEALTHY.value
    assert summary["monitored_pipelines"] == 2
    assert "timestamp" in summary

def test_repair_statistics():
    """Test repair statistics collection"""
    auto_repair = AutoRepair(hdc_dimension=1000)
    
    stats = auto_repair.get_repair_statistics()
    
    assert isinstance(stats, dict)
    assert "total_attempts" in stats
    assert stats["total_attempts"] == 0  # No attempts yet

def main():
    """Run all tests"""
    print("🛡️ Pipeline Guard Test Suite")
    print("=" * 50)
    
    runner = TestRunner()
    
    # Core functionality tests
    runner.test("Pipeline Guard Initialization", test_pipeline_guard_initialization)
    runner.test("Pipeline Registration", test_pipeline_registration)
    runner.test("Metrics Processing", test_metrics_processing)
    runner.test("Alert Generation", test_alert_generation)
    runner.test("Guard Summary", test_guard_summary)
    
    # Detection and prediction tests
    runner.test("Anomaly Detection", test_anomaly_detection)
    runner.test("Failure Prediction", test_failure_prediction)
    
    # Repair tests
    runner.test("Automated Repair", test_automated_repair)
    runner.test("Repair Statistics", test_repair_statistics)
    
    # Monitoring tests
    runner.test("Pipeline Monitoring", test_pipeline_monitoring)
    runner.test("Monitoring Lifecycle", test_monitoring_lifecycle)
    
    # Print summary
    success = runner.summary()
    
    if success:
        print("\n🎉 All tests passed! Pipeline Guard is working correctly.")
    else:
        print(f"\n⚠️ {runner.failed} test(s) failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)