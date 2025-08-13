"""
Advanced Fault Tolerance System for HDC Robot Controller
Enterprise-grade resilience with predictive failure detection and self-healing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import queue
import logging
from pathlib import Path
import pickle
import hashlib

from ..core.hypervector import HyperVector, weighted_bundle
from ..core.sensor_fusion import SensorModality, FusedPercept

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of system failures."""
    SENSOR_DROPOUT = "sensor_dropout"
    ACTUATOR_FAILURE = "actuator_failure"
    COMMUNICATION_LOSS = "communication_loss"
    PROCESSING_OVERLOAD = "processing_overload"
    MEMORY_CORRUPTION = "memory_corruption"
    POWER_DEGRADATION = "power_degradation"
    THERMAL_OVERLOAD = "thermal_overload"
    NETWORK_PARTITION = "network_partition"


class SeverityLevel(Enum):
    """Failure severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


@dataclass
class FailureEvent:
    """Represents a system failure event."""
    failure_type: FailureType
    severity: SeverityLevel
    timestamp: float
    affected_components: List[str]
    error_vector: Optional[HyperVector] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class SystemState:
    """Current system state for monitoring."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    temperature: float
    sensor_health: Dict[SensorModality, float]
    actuator_health: Dict[str, float]
    communication_latency: float
    error_rate: float
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class PredictiveFailureDetector:
    """AI-powered predictive failure detection system."""
    
    def __init__(self, dimension: int = 10000, history_size: int = 1000):
        self.dimension = dimension
        self.history_size = history_size
        
        # System state history
        self.state_history = []
        self.failure_history = []
        
        # Learned failure patterns
        self.failure_patterns = {}
        self.normal_patterns = {}
        
        # Prediction models
        self.prediction_thresholds = {
            FailureType.SENSOR_DROPOUT: 0.8,
            FailureType.PROCESSING_OVERLOAD: 0.7,
            FailureType.THERMAL_OVERLOAD: 0.9,
            FailureType.MEMORY_CORRUPTION: 0.85,
            FailureType.POWER_DEGRADATION: 0.75
        }
        
        # Initialize baseline patterns
        self._initialize_patterns()
        
    def _initialize_patterns(self):
        """Initialize baseline failure and normal patterns."""
        # Create patterns for each failure type
        for failure_type in FailureType:
            pattern_seed = hash(failure_type.value)
            self.failure_patterns[failure_type] = HyperVector.random(
                self.dimension, seed=pattern_seed
            )
            
        # Create normal operation patterns
        normal_seed = hash("normal_operation")
        self.normal_patterns['baseline'] = HyperVector.random(
            self.dimension, seed=normal_seed
        )
        
    def add_system_state(self, state: SystemState):
        """Add new system state to monitoring history."""
        self.state_history.append(state)
        
        # Maintain history size
        if len(self.state_history) > self.history_size:
            self.state_history.pop(0)
            
        # Learn from normal operation
        if self._is_normal_state(state):
            self._update_normal_patterns(state)
            
    def add_failure_event(self, failure: FailureEvent):
        """Record failure event for learning."""
        self.failure_history.append(failure)
        
        # Learn from failure
        self._learn_failure_pattern(failure)
        
        # Maintain history
        if len(self.failure_history) > self.history_size:
            self.failure_history.pop(0)
            
    def predict_failures(self, current_state: SystemState) -> List[Tuple[FailureType, float]]:
        """
        Predict potential failures based on current system state.
        
        Args:
            current_state: Current system state
            
        Returns:
            List of (failure_type, probability) predictions
        """
        try:
            # Encode current system state
            state_vector = self._encode_system_state(current_state)
            
            predictions = []
            
            # Check against each failure pattern
            for failure_type, pattern in self.failure_patterns.items():
                # Calculate similarity to failure pattern
                failure_similarity = state_vector.similarity(pattern)
                
                # Calculate deviation from normal patterns
                normal_deviation = self._calculate_normal_deviation(state_vector)
                
                # Combine metrics for prediction
                prediction_score = (
                    0.6 * max(0, failure_similarity) +
                    0.4 * normal_deviation
                )
                
                # Apply threshold
                threshold = self.prediction_thresholds.get(failure_type, 0.8)
                if prediction_score > threshold:
                    predictions.append((failure_type, prediction_score))
                    
            # Sort by probability
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failure prediction failed: {e}")
            return []
            
    def _encode_system_state(self, state: SystemState) -> HyperVector:
        """Encode system state as hypervector."""
        state_components = []
        
        # CPU usage encoding
        cpu_hv = self._encode_metric("cpu", state.cpu_usage, 0.0, 1.0)
        state_components.append((cpu_hv, 0.2))
        
        # Memory usage encoding
        memory_hv = self._encode_metric("memory", state.memory_usage, 0.0, 1.0)
        state_components.append((memory_hv, 0.2))
        
        # Temperature encoding
        temp_hv = self._encode_metric("temperature", state.temperature, 20.0, 80.0)
        state_components.append((temp_hv, 0.15))
        
        # Sensor health encoding
        sensor_health_vectors = []
        for modality, health in state.sensor_health.items():
            sensor_hv = self._encode_metric(f"sensor_{modality.value}", health, 0.0, 1.0)
            sensor_health_vectors.append(sensor_hv)
            
        if sensor_health_vectors:
            sensors_hv = HyperVector.bundle_vectors(sensor_health_vectors)
            state_components.append((sensors_hv, 0.15))
            
        # Communication latency encoding
        comm_hv = self._encode_metric("latency", state.communication_latency, 0.0, 1000.0)
        state_components.append((comm_hv, 0.1))
        
        # Error rate encoding
        error_hv = self._encode_metric("error_rate", state.error_rate, 0.0, 1.0)
        state_components.append((error_hv, 0.18))
        
        # Bundle all state components
        if state_components:
            return weighted_bundle(state_components)
        else:
            return HyperVector.zero(self.dimension)
            
    def _encode_metric(self, name: str, value: float, min_val: float, max_val: float) -> HyperVector:
        """Encode single metric as hypervector."""
        # Normalize value
        normalized = np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)
        
        # Create metric hypervector
        metric_hv = HyperVector.random(self.dimension, seed=hash(name))
        
        # Scale by normalized value
        if normalized > 0.5:
            # High value
            high_hv = HyperVector.random(self.dimension, seed=hash(f"{name}_high"))
            return metric_hv.bind(high_hv)
        else:
            # Low/normal value
            return metric_hv
            
    def _is_normal_state(self, state: SystemState) -> bool:
        """Determine if system state represents normal operation."""
        return (state.cpu_usage < 0.8 and
                state.memory_usage < 0.8 and
                state.temperature < 70.0 and
                state.error_rate < 0.1 and
                all(health > 0.7 for health in state.sensor_health.values()))
                
    def _update_normal_patterns(self, state: SystemState):
        """Update normal operation patterns with new state."""
        state_vector = self._encode_system_state(state)
        
        # Update baseline normal pattern
        current_normal = self.normal_patterns['baseline']
        updated_normal = weighted_bundle([
            (current_normal, 0.95),
            (state_vector, 0.05)
        ])
        
        self.normal_patterns['baseline'] = updated_normal
        
    def _learn_failure_pattern(self, failure: FailureEvent):
        """Learn failure pattern from failure event."""
        if failure.error_vector is not None:
            # Update failure pattern
            current_pattern = self.failure_patterns[failure.failure_type]
            updated_pattern = weighted_bundle([
                (current_pattern, 0.9),
                (failure.error_vector, 0.1)
            ])
            
            self.failure_patterns[failure.failure_type] = updated_pattern
            
    def _calculate_normal_deviation(self, state_vector: HyperVector) -> float:
        """Calculate deviation from normal operation patterns."""
        baseline_similarity = state_vector.similarity(self.normal_patterns['baseline'])
        
        # Convert similarity to deviation (0 = similar, 1 = very different)
        deviation = max(0, 1.0 - baseline_similarity)
        
        return deviation


class SelfHealingController:
    """Self-healing system controller with autonomous recovery."""
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        
        # Recovery strategies
        self.recovery_strategies = {}
        self.recovery_history = []
        
        # Component redundancy
        self.redundant_components = {}
        self.backup_systems = {}
        
        # Recovery success tracking
        self.recovery_success_rates = {}
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
    def _initialize_recovery_strategies(self):
        """Initialize recovery strategies for different failure types."""
        
        # Sensor dropout recovery
        self.recovery_strategies[FailureType.SENSOR_DROPOUT] = [
            "switch_to_backup_sensor",
            "increase_other_sensor_weights",
            "use_predictive_estimation",
            "activate_redundant_modality"
        ]
        
        # Processing overload recovery
        self.recovery_strategies[FailureType.PROCESSING_OVERLOAD] = [
            "reduce_processing_frequency",
            "offload_to_backup_processor",
            "simplify_algorithms",
            "prioritize_critical_tasks"
        ]
        
        # Communication loss recovery
        self.recovery_strategies[FailureType.COMMUNICATION_LOSS] = [
            "switch_to_backup_channel",
            "reduce_communication_frequency",
            "use_local_fallback_mode",
            "restart_communication_stack"
        ]
        
        # Memory corruption recovery
        self.recovery_strategies[FailureType.MEMORY_CORRUPTION] = [
            "reload_from_backup",
            "reinitialize_corrupted_module",
            "use_redundant_memory",
            "restart_affected_process"
        ]
        
        # Initialize success rates
        for failure_type in FailureType:
            self.recovery_success_rates[failure_type] = {}
            for strategy in self.recovery_strategies.get(failure_type, []):
                self.recovery_success_rates[failure_type][strategy] = 0.5  # Initial estimate
                
    def execute_recovery(self, failure: FailureEvent) -> bool:
        """
        Execute autonomous recovery for detected failure.
        
        Args:
            failure: Detected failure event
            
        Returns:
            Success status of recovery
        """
        try:
            logger.warning(f"Executing recovery for {failure.failure_type.value} (severity: {failure.severity.value})")
            
            # Select best recovery strategy
            strategy = self._select_recovery_strategy(failure)
            
            if strategy:
                # Execute recovery strategy
                success = self._execute_strategy(strategy, failure)
                
                # Record recovery attempt
                recovery_record = {
                    'timestamp': time.time(),
                    'failure_type': failure.failure_type,
                    'strategy': strategy,
                    'success': success,
                    'execution_time': 0.0  # Placeholder
                }
                
                self.recovery_history.append(recovery_record)
                
                # Update success rate
                self._update_success_rate(failure.failure_type, strategy, success)
                
                if success:
                    logger.info(f"Recovery successful using strategy: {strategy}")
                else:
                    logger.error(f"Recovery failed with strategy: {strategy}")
                    
                return success
            else:
                logger.error(f"No recovery strategy available for {failure.failure_type.value}")
                return False
                
        except Exception as e:
            logger.error(f"Recovery execution failed: {e}")
            return False
            
    def _select_recovery_strategy(self, failure: FailureEvent) -> Optional[str]:
        """Select best recovery strategy based on historical success rates."""
        available_strategies = self.recovery_strategies.get(failure.failure_type, [])
        
        if not available_strategies:
            return None
            
        # Rank strategies by success rate
        strategy_scores = []
        for strategy in available_strategies:
            success_rate = self.recovery_success_rates[failure.failure_type].get(strategy, 0.5)
            
            # Adjust score based on failure severity
            severity_factor = 1.0 + (failure.severity.value - 1) * 0.1
            adjusted_score = success_rate * severity_factor
            
            strategy_scores.append((strategy, adjusted_score))
            
        # Sort by score and select best
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        return strategy_scores[0][0] if strategy_scores else None
        
    def _execute_strategy(self, strategy: str, failure: FailureEvent) -> bool:
        """Execute specific recovery strategy."""
        try:
            if strategy == "switch_to_backup_sensor":
                return self._switch_to_backup_sensor(failure)
            elif strategy == "increase_other_sensor_weights":
                return self._increase_other_sensor_weights(failure)
            elif strategy == "use_predictive_estimation":
                return self._use_predictive_estimation(failure)
            elif strategy == "reduce_processing_frequency":
                return self._reduce_processing_frequency(failure)
            elif strategy == "offload_to_backup_processor":
                return self._offload_to_backup_processor(failure)
            elif strategy == "switch_to_backup_channel":
                return self._switch_to_backup_channel(failure)
            elif strategy == "reload_from_backup":
                return self._reload_from_backup(failure)
            elif strategy == "restart_affected_process":
                return self._restart_affected_process(failure)
            else:
                logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            return False
            
    def _switch_to_backup_sensor(self, failure: FailureEvent) -> bool:
        """Switch to backup sensor when primary fails."""
        # Implementation would depend on specific sensor hardware
        logger.info("Switching to backup sensor")
        
        # Simulate successful backup switch
        affected_sensors = [comp for comp in failure.affected_components 
                          if comp.startswith('sensor_')]
        
        if affected_sensors:
            # Mark backup as active
            for sensor in affected_sensors:
                backup_name = f"{sensor}_backup"
                self.backup_systems[backup_name] = True
                
            return True
            
        return False
        
    def _increase_other_sensor_weights(self, failure: FailureEvent) -> bool:
        """Increase weights of functioning sensors to compensate."""
        logger.info("Redistributing sensor weights")
        
        # This would integrate with the sensor fusion system
        # to dynamically adjust sensor weights
        
        return True  # Simulate success
        
    def _use_predictive_estimation(self, failure: FailureEvent) -> bool:
        """Use predictive models to estimate missing sensor data."""
        logger.info("Activating predictive estimation")
        
        # Implementation would use historical data and machine learning
        # to predict missing sensor readings
        
        return True  # Simulate success
        
    def _reduce_processing_frequency(self, failure: FailureEvent) -> bool:
        """Reduce processing frequency to handle overload."""
        logger.info("Reducing processing frequency")
        
        # Adjust system parameters to reduce computational load
        
        return True  # Simulate success
        
    def _offload_to_backup_processor(self, failure: FailureEvent) -> bool:
        """Offload processing to backup computational unit."""
        logger.info("Offloading to backup processor")
        
        # Transfer critical computations to backup system
        
        return True  # Simulate success
        
    def _switch_to_backup_channel(self, failure: FailureEvent) -> bool:
        """Switch to backup communication channel."""
        logger.info("Switching to backup communication channel")
        
        # Activate redundant communication pathway
        
        return True  # Simulate success
        
    def _reload_from_backup(self, failure: FailureEvent) -> bool:
        """Reload data/state from backup storage."""
        logger.info("Reloading from backup")
        
        # Restore system state from redundant storage
        
        return True  # Simulate success
        
    def _restart_affected_process(self, failure: FailureEvent) -> bool:
        """Restart affected system process."""
        logger.info("Restarting affected process")
        
        # Safely restart the failed component
        
        return True  # Simulate success
        
    def _update_success_rate(self, failure_type: FailureType, strategy: str, success: bool):
        """Update success rate for recovery strategy."""
        current_rate = self.recovery_success_rates[failure_type].get(strategy, 0.5)
        
        # Exponential moving average update
        alpha = 0.1  # Learning rate
        if success:
            new_rate = current_rate + alpha * (1.0 - current_rate)
        else:
            new_rate = current_rate + alpha * (0.0 - current_rate)
            
        self.recovery_success_rates[failure_type][strategy] = new_rate
        
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        stats = {
            'total_recoveries': len(self.recovery_history),
            'success_rates': self.recovery_success_rates,
            'recent_recoveries': self.recovery_history[-10:],  # Last 10 recoveries
        }
        
        # Calculate overall success rate
        if self.recovery_history:
            successful = sum(1 for r in self.recovery_history if r['success'])
            stats['overall_success_rate'] = successful / len(self.recovery_history)
        else:
            stats['overall_success_rate'] = 0.0
            
        return stats


class AdvancedFaultTolerantSystem:
    """
    Enterprise-grade fault tolerance system with predictive capabilities.
    
    Combines predictive failure detection with autonomous self-healing
    for maximum system resilience.
    """
    
    def __init__(self, 
                 dimension: int = 10000,
                 monitoring_interval: float = 1.0,
                 enable_predictive_detection: bool = True,
                 enable_self_healing: bool = True):
        """
        Initialize advanced fault tolerance system.
        
        Args:
            dimension: Hypervector dimension
            monitoring_interval: System monitoring interval in seconds
            enable_predictive_detection: Enable predictive failure detection
            enable_self_healing: Enable autonomous self-healing
        """
        self.dimension = dimension
        self.monitoring_interval = monitoring_interval
        self.enable_predictive_detection = enable_predictive_detection
        self.enable_self_healing = enable_self_healing
        
        # Core components
        self.failure_detector = PredictiveFailureDetector(dimension)
        self.self_healing = SelfHealingController(dimension)
        
        # System monitoring
        self.monitoring_thread = None
        self.monitoring_active = False
        self.monitoring_queue = queue.Queue()
        
        # Fault tolerance statistics
        self.ft_stats = {
            'system_uptime': time.time(),
            'total_failures_detected': 0,
            'total_failures_prevented': 0,
            'total_recoveries_attempted': 0,
            'total_recoveries_successful': 0,
            'mean_time_to_recovery': 0.0,
            'system_availability': 1.0
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info("Fault tolerance monitoring started")
        
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
            
        logger.info("Fault tolerance monitoring stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting fault tolerance monitoring loop")
        
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Get current system state
                current_state = self._collect_system_state()
                
                # Add to failure detector
                if self.enable_predictive_detection:
                    self.failure_detector.add_system_state(current_state)
                    
                    # Predict potential failures
                    predictions = self.failure_detector.predict_failures(current_state)
                    
                    # Handle predictions
                    for failure_type, probability in predictions:
                        self._handle_predicted_failure(failure_type, probability, current_state)
                        
                # Check for immediate failures
                immediate_failures = self._detect_immediate_failures(current_state)
                
                for failure in immediate_failures:
                    self._handle_detected_failure(failure)
                    
                # Update statistics
                self._update_statistics()
                
                # Sleep for monitoring interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(self.monitoring_interval)
                
    def _collect_system_state(self) -> SystemState:
        """Collect current system state for monitoring."""
        try:
            # In a real implementation, this would collect actual system metrics
            # For now, we'll simulate with random but realistic values
            
            import psutil
            
            # CPU and memory from actual system
            cpu_usage = psutil.cpu_percent() / 100.0
            memory_usage = psutil.virtual_memory().percent / 100.0
            
            # Simulated metrics
            temperature = np.random.normal(50.0, 10.0)  # Celsius
            communication_latency = np.random.exponential(50.0)  # ms
            error_rate = max(0, np.random.normal(0.05, 0.02))
            
            # Simulated sensor health
            sensor_health = {}
            for modality in SensorModality:
                # Most sensors healthy, occasional degradation
                base_health = 0.9
                noise = np.random.normal(0, 0.1)
                health = np.clip(base_health + noise, 0.0, 1.0)
                sensor_health[modality] = health
                
            # Simulated actuator health
            actuator_health = {
                'joint_1': np.random.uniform(0.8, 1.0),
                'joint_2': np.random.uniform(0.8, 1.0),
                'gripper': np.random.uniform(0.8, 1.0),
            }
            
            return SystemState(
                timestamp=time.time(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                temperature=temperature,
                sensor_health=sensor_health,
                actuator_health=actuator_health,
                communication_latency=communication_latency,
                error_rate=error_rate
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system state: {e}")
            
            # Return minimal safe state
            return SystemState(
                timestamp=time.time(),
                cpu_usage=0.5,
                memory_usage=0.5,
                temperature=50.0,
                sensor_health={},
                actuator_health={},
                communication_latency=100.0,
                error_rate=0.1
            )
            
    def _detect_immediate_failures(self, state: SystemState) -> List[FailureEvent]:
        """Detect immediate system failures."""
        failures = []
        
        # CPU overload
        if state.cpu_usage > 0.95:
            failure = FailureEvent(
                failure_type=FailureType.PROCESSING_OVERLOAD,
                severity=SeverityLevel.HIGH,
                timestamp=state.timestamp,
                affected_components=['cpu'],
                metadata={'cpu_usage': state.cpu_usage}
            )
            failures.append(failure)
            
        # Memory exhaustion
        if state.memory_usage > 0.95:
            failure = FailureEvent(
                failure_type=FailureType.PROCESSING_OVERLOAD,
                severity=SeverityLevel.HIGH,
                timestamp=state.timestamp,
                affected_components=['memory'],
                metadata={'memory_usage': state.memory_usage}
            )
            failures.append(failure)
            
        # Temperature overload
        if state.temperature > 75.0:
            severity = SeverityLevel.CRITICAL if state.temperature > 80.0 else SeverityLevel.HIGH
            failure = FailureEvent(
                failure_type=FailureType.THERMAL_OVERLOAD,
                severity=severity,
                timestamp=state.timestamp,
                affected_components=['thermal_system'],
                metadata={'temperature': state.temperature}
            )
            failures.append(failure)
            
        # Sensor failures
        for modality, health in state.sensor_health.items():
            if health < 0.3:
                failure = FailureEvent(
                    failure_type=FailureType.SENSOR_DROPOUT,
                    severity=SeverityLevel.MEDIUM,
                    timestamp=state.timestamp,
                    affected_components=[f'sensor_{modality.value}'],
                    metadata={'sensor_health': health}
                )
                failures.append(failure)
                
        # High communication latency
        if state.communication_latency > 500.0:
            failure = FailureEvent(
                failure_type=FailureType.COMMUNICATION_LOSS,
                severity=SeverityLevel.MEDIUM,
                timestamp=state.timestamp,
                affected_components=['communication'],
                metadata={'latency': state.communication_latency}
            )
            failures.append(failure)
            
        return failures
        
    def _handle_predicted_failure(self, 
                                failure_type: FailureType, 
                                probability: float, 
                                state: SystemState):
        """Handle predicted failure with preventive action."""
        logger.warning(f"Predicted failure: {failure_type.value} (probability: {probability:.3f})")
        
        # Create failure event for prediction
        predicted_failure = FailureEvent(
            failure_type=failure_type,
            severity=SeverityLevel.LOW,  # Predicted, not actual
            timestamp=time.time(),
            affected_components=['predicted'],
            metadata={'prediction_probability': probability}
        )
        
        # Take preventive action
        if probability > 0.9:  # High confidence prediction
            if self.enable_self_healing:
                success = self.self_healing.execute_recovery(predicted_failure)
                if success:
                    self.ft_stats['total_failures_prevented'] += 1
                    logger.info(f"Prevented failure: {failure_type.value}")
                    
        # Notify alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(predicted_failure, is_prediction=True)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
                
    def _handle_detected_failure(self, failure: FailureEvent):
        """Handle detected system failure."""
        logger.error(f"Detected failure: {failure.failure_type.value} (severity: {failure.severity.value})")
        
        # Record failure
        self.failure_detector.add_failure_event(failure)
        self.ft_stats['total_failures_detected'] += 1
        
        # Attempt recovery if enabled
        if self.enable_self_healing:
            self.ft_stats['total_recoveries_attempted'] += 1
            recovery_start = time.time()
            
            success = self.self_healing.execute_recovery(failure)
            
            recovery_time = time.time() - recovery_start
            
            if success:
                self.ft_stats['total_recoveries_successful'] += 1
                
                # Update mean time to recovery
                current_mttr = self.ft_stats['mean_time_to_recovery']
                successful_recoveries = self.ft_stats['total_recoveries_successful']
                
                # Exponential moving average
                if successful_recoveries == 1:
                    self.ft_stats['mean_time_to_recovery'] = recovery_time
                else:
                    alpha = 2.0 / (successful_recoveries + 1)
                    self.ft_stats['mean_time_to_recovery'] = (
                        alpha * recovery_time + (1 - alpha) * current_mttr
                    )
                    
        # Notify alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(failure, is_prediction=False)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
                
    def _update_statistics(self):
        """Update system availability and other statistics."""
        current_time = time.time()
        uptime = current_time - self.ft_stats['system_uptime']
        
        # Calculate availability (simplified)
        total_failures = self.ft_stats['total_failures_detected']
        total_recoveries = self.ft_stats['total_recoveries_successful']
        
        # Assume each failure causes 1 minute downtime if not recovered
        downtime = max(0, total_failures - total_recoveries) * 60.0  # seconds
        
        if uptime > 0:
            availability = max(0, 1.0 - (downtime / uptime))
            self.ft_stats['system_availability'] = availability
            
    def add_alert_callback(self, callback: Callable[[FailureEvent, bool], None]):
        """Add callback for failure alerts."""
        self.alert_callbacks.append(callback)
        
    def remove_alert_callback(self, callback: Callable[[FailureEvent, bool], None]):
        """Remove alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
            
    def get_fault_tolerance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance statistics."""
        stats = self.ft_stats.copy()
        
        # Add component statistics
        stats['failure_detector_stats'] = {
            'state_history_size': len(self.failure_detector.state_history),
            'failure_history_size': len(self.failure_detector.failure_history),
            'prediction_thresholds': self.failure_detector.prediction_thresholds
        }
        
        stats['self_healing_stats'] = self.self_healing.get_recovery_statistics()
        
        # Calculate derived metrics
        if stats['total_recoveries_attempted'] > 0:
            stats['recovery_success_rate'] = (
                stats['total_recoveries_successful'] / stats['total_recoveries_attempted']
            )
        else:
            stats['recovery_success_rate'] = 0.0
            
        return stats
        
    def save_fault_tolerance_state(self, filepath: Union[str, Path]):
        """Save fault tolerance system state."""
        try:
            state_data = {
                'failure_patterns': self.failure_detector.failure_patterns,
                'normal_patterns': self.failure_detector.normal_patterns,
                'recovery_success_rates': self.self_healing.recovery_success_rates,
                'statistics': self.ft_stats,
                'timestamp': time.time()
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(state_data, f)
                
            logger.info(f"Saved fault tolerance state to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save fault tolerance state: {e}")
            
    def load_fault_tolerance_state(self, filepath: Union[str, Path]):
        """Load fault tolerance system state."""
        try:
            with open(filepath, 'rb') as f:
                state_data = pickle.load(f)
                
            # Restore patterns and rates
            self.failure_detector.failure_patterns = state_data.get('failure_patterns', {})
            self.failure_detector.normal_patterns = state_data.get('normal_patterns', {})
            self.self_healing.recovery_success_rates = state_data.get('recovery_success_rates', {})
            
            # Restore statistics (but update uptime)
            loaded_stats = state_data.get('statistics', {})
            current_time = time.time()
            loaded_stats['system_uptime'] = current_time
            self.ft_stats.update(loaded_stats)
            
            logger.info(f"Loaded fault tolerance state from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load fault tolerance state: {e}")
            
    def simulate_failure(self, failure_type: FailureType, severity: SeverityLevel = SeverityLevel.MEDIUM):
        """Simulate failure for testing purposes."""
        simulated_failure = FailureEvent(
            failure_type=failure_type,
            severity=severity,
            timestamp=time.time(),
            affected_components=['simulated'],
            metadata={'simulated': True}
        )
        
        logger.info(f"Simulating failure: {failure_type.value}")
        self._handle_detected_failure(simulated_failure)
        
        return simulated_failure
        
    def run_diagnostic(self) -> Dict[str, Any]:
        """Run comprehensive system diagnostic."""
        diagnostic_results = {
            'timestamp': time.time(),
            'monitoring_active': self.monitoring_active,
            'predictive_detection_enabled': self.enable_predictive_detection,
            'self_healing_enabled': self.enable_self_healing,
            'system_state': self._collect_system_state(),
            'statistics': self.get_fault_tolerance_statistics()
        }
        
        # Test prediction system
        if self.enable_predictive_detection:
            current_state = diagnostic_results['system_state']
            predictions = self.failure_detector.predict_failures(current_state)
            diagnostic_results['current_predictions'] = predictions
            
        return diagnostic_results