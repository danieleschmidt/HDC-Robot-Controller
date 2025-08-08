"""
Advanced Fault Tolerance System for HDC Robotics

Implements comprehensive fault tolerance mechanisms including graceful degradation,
automatic recovery, circuit breakers, and adaptive redundancy for production robotics.

Fault Tolerance Features:
1. Sensor Dropout Compensation: Maintain performance with missing sensors
2. Circuit Breaker Pattern: Prevent cascade failures
3. Adaptive Redundancy: Dynamic backup system activation
4. Self-Healing: Automatic recovery from transient failures
5. Graceful Degradation: Performance scaling under failures
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import time
import threading
import logging
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import psutil

from ..core.hypervector import HyperVector
from ..core.operations import HDCOperations
from ..core.memory import HDCAssociativeMemory


class ComponentState(Enum):
    """Component operational states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    RECOVERING = "recovering"
    MAINTENANCE = "maintenance"


class FailureType(Enum):
    """Types of system failures."""
    SENSOR_DROPOUT = "sensor_dropout"
    MEMORY_CORRUPTION = "memory_corruption"
    COMPUTATION_ERROR = "computation_error"
    COMMUNICATION_FAILURE = "communication_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    HARDWARE_MALFUNCTION = "hardware_malfunction"


@dataclass
class FailureEvent:
    """Failure event record."""
    timestamp: float
    component_id: str
    failure_type: FailureType
    severity: float  # 0.0 (minor) to 1.0 (critical)
    details: Dict[str, Any] = field(default_factory=dict)
    recovery_actions: List[str] = field(default_factory=list)


@dataclass
class ComponentHealth:
    """Component health status."""
    component_id: str
    state: ComponentState
    health_score: float  # 0.0 to 1.0
    last_heartbeat: float
    failure_count: int = 0
    recovery_attempts: int = 0
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class CircuitBreaker:
    """Circuit breaker for preventing cascade failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0,
                 success_threshold: int = 2):
        """
        Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Time to wait before attempting to close circuit (seconds)
            success_threshold: Consecutive successes needed to close circuit
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.state = ComponentState.HEALTHY
        
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Tuple[Any, bool]:
        """
        Execute function through circuit breaker.
        
        Args:
            func: Function to execute
            *args, **kwargs: Function arguments
            
        Returns:
            (result, success) tuple
        """
        with self._lock:
            # Check if circuit is open
            if self.state == ComponentState.FAILED:
                current_time = time.time()
                if current_time - self.last_failure_time < self.timeout:
                    return None, False  # Circuit open
                else:
                    self.state = ComponentState.RECOVERING  # Half-open state
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result, True
                
            except Exception as e:
                self._on_failure()
                logging.error(f"Circuit breaker caught exception: {e}")
                return None, False
    
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        
        if self.state == ComponentState.RECOVERING:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = ComponentState.HEALTHY
                self.success_count = 0
        elif self.state == ComponentState.DEGRADED:
            self.state = ComponentState.HEALTHY
    
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.success_count = 0
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = ComponentState.FAILED
        else:
            self.state = ComponentState.DEGRADED


class SensorDropoutCompensator:
    """Compensates for sensor dropouts using HDC correlation patterns."""
    
    def __init__(self, dimension: int = 10000):
        """
        Initialize sensor dropout compensator.
        
        Args:
            dimension: HDC vector dimension
        """
        self.dimension = dimension
        
        # Sensor correlation patterns learned during normal operation
        self.sensor_correlations = {}
        self.sensor_history = defaultdict(lambda: deque(maxlen=100))
        
        # Compensation models for each sensor
        self.compensation_models = {}
        
        # Performance tracking
        self.compensation_accuracy = defaultdict(list)
        
        self.logger = logging.getLogger(__name__)
    
    def learn_sensor_correlations(self, sensor_data: Dict[str, HyperVector]):
        """
        Learn correlations between sensors during normal operation.
        
        Args:
            sensor_data: Dictionary of sensor hypervectors
        """
        sensor_names = list(sensor_data.keys())
        
        # Update sensor history
        for sensor_name, sensor_vector in sensor_data.items():
            self.sensor_history[sensor_name].append(sensor_vector)
        
        # Learn pairwise correlations
        for i, sensor_a in enumerate(sensor_names):
            for j, sensor_b in enumerate(sensor_names):
                if i != j:
                    correlation_key = f"{sensor_a}_{sensor_b}"
                    
                    if correlation_key not in self.sensor_correlations:
                        self.sensor_correlations[correlation_key] = {
                            'correlation_vector': HyperVector.zero(self.dimension),
                            'sample_count': 0
                        }
                    
                    # Update correlation using exponential moving average
                    current_correlation = sensor_data[sensor_a].bind(sensor_data[sensor_b])
                    
                    correlation_info = self.sensor_correlations[correlation_key]
                    alpha = 0.1  # Learning rate
                    
                    if correlation_info['sample_count'] == 0:
                        correlation_info['correlation_vector'] = current_correlation
                    else:
                        # Weighted bundle for moving average
                        old_weight = 1 - alpha
                        new_weight = alpha
                        
                        old_contrib = correlation_info['correlation_vector'].add_noise(1 - old_weight)
                        new_contrib = current_correlation.add_noise(1 - new_weight)
                        
                        correlation_info['correlation_vector'] = old_contrib.bundle(new_contrib)
                    
                    correlation_info['sample_count'] += 1
    
    def compensate_sensor_dropout(self, available_sensors: Dict[str, HyperVector], 
                                 missing_sensors: List[str]) -> Dict[str, HyperVector]:
        """
        Compensate for missing sensors using learned correlations.
        
        Args:
            available_sensors: Dictionary of available sensor data
            missing_sensors: List of missing sensor names
            
        Returns:
            Dictionary including compensated sensor data
        """
        compensated_data = available_sensors.copy()
        
        for missing_sensor in missing_sensors:
            self.logger.info(f"Compensating for missing sensor: {missing_sensor}")
            
            # Find best correlated available sensor
            best_correlation = -1.0
            best_source_sensor = None
            best_correlation_vector = None
            
            for available_sensor in available_sensors:
                correlation_key = f"{available_sensor}_{missing_sensor}"
                reverse_key = f"{missing_sensor}_{available_sensor}"
                
                if correlation_key in self.sensor_correlations:
                    correlation_info = self.sensor_correlations[correlation_key]
                elif reverse_key in self.sensor_correlations:
                    correlation_info = self.sensor_correlations[reverse_key]
                else:
                    continue
                
                # Estimate correlation strength
                if len(self.sensor_history[available_sensor]) > 0:
                    recent_vector = self.sensor_history[available_sensor][-1]
                    correlation_strength = correlation_info['correlation_vector'].similarity(recent_vector)
                    
                    if correlation_strength > best_correlation:
                        best_correlation = correlation_strength
                        best_source_sensor = available_sensor
                        best_correlation_vector = correlation_info['correlation_vector']
            
            # Generate compensated sensor data
            if best_source_sensor and best_correlation_vector:
                source_data = available_sensors[best_source_sensor]
                
                # Use correlation pattern to generate missing sensor data
                compensated_vector = source_data.bind(best_correlation_vector)
                
                # Add learned historical patterns if available
                if len(self.sensor_history[missing_sensor]) > 0:
                    historical_pattern = self._extract_historical_pattern(missing_sensor)
                    compensated_vector = compensated_vector.bundle(historical_pattern)
                
                compensated_data[missing_sensor] = compensated_vector
                
                # Track compensation quality
                self.compensation_accuracy[missing_sensor].append(best_correlation)
                
                self.logger.info(f"Compensated {missing_sensor} using {best_source_sensor} "
                               f"(correlation: {best_correlation:.3f})")
            else:
                # Fallback: use historical average or default pattern
                if len(self.sensor_history[missing_sensor]) > 0:
                    compensated_data[missing_sensor] = self._extract_historical_pattern(missing_sensor)
                    self.logger.warning(f"Using historical pattern for {missing_sensor}")
                else:
                    compensated_data[missing_sensor] = HyperVector.random(self.dimension)
                    self.logger.warning(f"Using random pattern for {missing_sensor}")
        
        return compensated_data
    
    def _extract_historical_pattern(self, sensor_name: str) -> HyperVector:
        """Extract representative pattern from sensor history."""
        history = list(self.sensor_history[sensor_name])
        if not history:
            return HyperVector.random(self.dimension)
        
        # Bundle recent history with exponential weighting
        weighted_vectors = []
        weights = []
        
        for i, vector in enumerate(reversed(history[-20:])):  # Last 20 samples
            weight = np.exp(-i * 0.1)  # Exponential decay
            weighted_vectors.append(vector)
            weights.append(weight)
        
        # Weighted bundle
        if len(weighted_vectors) == 1:
            return weighted_vectors[0]
        
        # Create weighted representation
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        result = HyperVector.zero(self.dimension)
        for vector, weight in zip(weighted_vectors, normalized_weights):
            # Add weighted contribution
            weighted_contribution = vector.add_noise(1 - weight)
            result = result.bundle(weighted_contribution)
        
        return result
    
    def get_compensation_statistics(self) -> Dict[str, Any]:
        """Get compensation performance statistics."""
        stats = {}
        
        for sensor_name, accuracies in self.compensation_accuracy.items():
            if accuracies:
                stats[sensor_name] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'num_compensations': len(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies)
                }
        
        return stats


class AdaptiveRedundancyManager:
    """Manages adaptive redundancy for critical system components."""
    
    def __init__(self, dimension: int = 10000):
        """
        Initialize adaptive redundancy manager.
        
        Args:
            dimension: HDC vector dimension
        """
        self.dimension = dimension
        
        # Redundancy configurations
        self.redundancy_configs = {}
        self.active_backups = {}
        
        # Performance monitoring
        self.component_performance = defaultdict(lambda: deque(maxlen=100))
        
        # Resource management
        self.resource_limits = {
            'cpu_threshold': 0.8,
            'memory_threshold': 0.8,
            'response_time_threshold': 1.0  # seconds
        }
        
        self.logger = logging.getLogger(__name__)
    
    def register_component(self, component_id: str, primary_instance: Any, 
                          backup_instances: List[Any], criticality: float = 1.0):
        """
        Register component with redundancy configuration.
        
        Args:
            component_id: Unique component identifier
            primary_instance: Primary component instance
            backup_instances: List of backup instances
            criticality: Component criticality (0.0 to 1.0)
        """
        self.redundancy_configs[component_id] = {
            'primary': primary_instance,
            'backups': backup_instances,
            'criticality': criticality,
            'active_backup_index': -1,
            'failover_count': 0,
            'last_failover': 0.0
        }
        
        self.logger.info(f"Registered component {component_id} with "
                        f"{len(backup_instances)} backups")
    
    def monitor_component_health(self, component_id: str, 
                               performance_metrics: Dict[str, float]):
        """
        Monitor component health and trigger failover if needed.
        
        Args:
            component_id: Component identifier
            performance_metrics: Performance metrics dictionary
        """
        if component_id not in self.redundancy_configs:
            return
        
        # Update performance history
        self.component_performance[component_id].append({
            'timestamp': time.time(),
            'metrics': performance_metrics.copy()
        })
        
        # Check if failover is needed
        if self._should_failover(component_id, performance_metrics):
            self.trigger_failover(component_id)
    
    def _should_failover(self, component_id: str, metrics: Dict[str, float]) -> bool:
        """Determine if failover should be triggered."""
        # Check resource thresholds
        cpu_usage = metrics.get('cpu_usage', 0.0)
        memory_usage = metrics.get('memory_usage', 0.0)
        response_time = metrics.get('response_time', 0.0)
        error_rate = metrics.get('error_rate', 0.0)
        
        # Immediate failover conditions
        if error_rate > 0.5:  # 50% error rate
            self.logger.warning(f"High error rate for {component_id}: {error_rate}")
            return True
        
        if response_time > self.resource_limits['response_time_threshold']:
            self.logger.warning(f"High response time for {component_id}: {response_time}s")
            return True
        
        # Resource exhaustion conditions
        resource_stress = (cpu_usage > self.resource_limits['cpu_threshold'] or
                          memory_usage > self.resource_limits['memory_threshold'])
        
        if resource_stress:
            # Check trend over recent history
            recent_performance = list(self.component_performance[component_id])[-10:]
            if len(recent_performance) >= 5:
                recent_stress_count = sum(1 for p in recent_performance
                                        if (p['metrics'].get('cpu_usage', 0) > self.resource_limits['cpu_threshold'] or
                                            p['metrics'].get('memory_usage', 0) > self.resource_limits['memory_threshold']))
                
                if recent_stress_count >= 3:  # 3 out of last 10 samples
                    self.logger.warning(f"Persistent resource stress for {component_id}")
                    return True
        
        return False
    
    def trigger_failover(self, component_id: str):
        """
        Trigger failover to backup instance.
        
        Args:
            component_id: Component to failover
        """
        if component_id not in self.redundancy_configs:
            self.logger.error(f"No redundancy config for {component_id}")
            return
        
        config = self.redundancy_configs[component_id]
        
        # Find next available backup
        next_backup_index = (config['active_backup_index'] + 1) % len(config['backups'])
        next_backup = config['backups'][next_backup_index]
        
        try:
            # Activate backup instance
            self._activate_backup(component_id, next_backup, next_backup_index)
            
            config['active_backup_index'] = next_backup_index
            config['failover_count'] += 1
            config['last_failover'] = time.time()
            
            self.logger.info(f"Failover successful for {component_id} to backup {next_backup_index}")
            
        except Exception as e:
            self.logger.error(f"Failover failed for {component_id}: {e}")
            self._try_next_backup(component_id)
    
    def _activate_backup(self, component_id: str, backup_instance: Any, backup_index: int):
        """Activate backup instance."""
        # Implementation depends on specific component type
        # This is a placeholder for component-specific activation logic
        
        if hasattr(backup_instance, 'activate'):
            backup_instance.activate()
        
        if hasattr(backup_instance, 'initialize'):
            backup_instance.initialize()
        
        # Store active backup reference
        self.active_backups[component_id] = backup_instance
    
    def _try_next_backup(self, component_id: str):
        """Try the next backup if current failover failed."""
        config = self.redundancy_configs[component_id]
        
        # Try remaining backups
        for i in range(len(config['backups']) - 1):
            next_index = (config['active_backup_index'] + i + 2) % len(config['backups'])
            next_backup = config['backups'][next_index]
            
            try:
                self._activate_backup(component_id, next_backup, next_index)
                config['active_backup_index'] = next_index
                config['failover_count'] += 1
                config['last_failover'] = time.time()
                
                self.logger.info(f"Fallback successful for {component_id} to backup {next_index}")
                return
                
            except Exception as e:
                self.logger.error(f"Fallback attempt {i+1} failed for {component_id}: {e}")
        
        self.logger.critical(f"All backup instances failed for {component_id}")
    
    def get_redundancy_status(self) -> Dict[str, Any]:
        """Get current redundancy status."""
        status = {}
        
        for component_id, config in self.redundancy_configs.items():
            status[component_id] = {
                'criticality': config['criticality'],
                'active_backup_index': config['active_backup_index'],
                'failover_count': config['failover_count'],
                'last_failover': config['last_failover'],
                'num_backups': len(config['backups']),
                'is_on_backup': config['active_backup_index'] >= 0
            }
        
        return status


class SelfHealingSystem:
    """Self-healing system for automatic recovery from failures."""
    
    def __init__(self, dimension: int = 10000):
        """
        Initialize self-healing system.
        
        Args:
            dimension: HDC vector dimension
        """
        self.dimension = dimension
        
        # Component health monitoring
        self.component_health = {}
        self.health_monitors = {}
        
        # Recovery strategies
        self.recovery_strategies = {}
        self.recovery_history = defaultdict(list)
        
        # System state
        self.system_health_score = 1.0
        self.critical_failures = deque(maxlen=100)
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitoring_thread = None
        
        self.logger = logging.getLogger(__name__)
    
    def register_component(self, component_id: str, health_check: Callable,
                          recovery_strategies: List[Callable]):
        """
        Register component for self-healing monitoring.
        
        Args:
            component_id: Component identifier
            health_check: Function that returns health status
            recovery_strategies: List of recovery functions to try
        """
        self.component_health[component_id] = ComponentHealth(
            component_id=component_id,
            state=ComponentState.HEALTHY,
            health_score=1.0,
            last_heartbeat=time.time()
        )
        
        self.health_monitors[component_id] = health_check
        self.recovery_strategies[component_id] = recovery_strategies
        
        self.logger.info(f"Registered component {component_id} for self-healing")
    
    def start_monitoring(self, check_interval: float = 1.0):
        """
        Start continuous health monitoring.
        
        Args:
            check_interval: Time between health checks (seconds)
        """
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Self-healing monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Self-healing monitoring stopped")
    
    def _monitoring_loop(self, check_interval: float):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                self._check_all_components()
                self._update_system_health()
                time.sleep(check_interval)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(check_interval)
    
    def _check_all_components(self):
        """Check health of all registered components."""
        for component_id in self.component_health:
            try:
                self._check_component_health(component_id)
            except Exception as e:
                self.logger.error(f"Health check failed for {component_id}: {e}")
                self._handle_health_check_failure(component_id)
    
    def _check_component_health(self, component_id: str):
        """Check health of specific component."""
        if component_id not in self.health_monitors:
            return
        
        health_check = self.health_monitors[component_id]
        health_info = self.component_health[component_id]
        
        # Perform health check
        health_result = health_check()
        
        # Update component health
        if isinstance(health_result, dict):
            health_score = health_result.get('health_score', 1.0)
            performance_metrics = health_result.get('metrics', {})
        else:
            health_score = float(health_result) if health_result else 0.0
            performance_metrics = {}
        
        health_info.health_score = health_score
        health_info.last_heartbeat = time.time()
        health_info.performance_metrics = performance_metrics
        
        # Update component state
        previous_state = health_info.state
        
        if health_score >= 0.8:
            health_info.state = ComponentState.HEALTHY
        elif health_score >= 0.5:
            health_info.state = ComponentState.DEGRADED
        else:
            health_info.state = ComponentState.FAILED
        
        # Trigger recovery if component failed
        if (previous_state != ComponentState.FAILED and 
            health_info.state == ComponentState.FAILED):
            self._trigger_recovery(component_id)
    
    def _handle_health_check_failure(self, component_id: str):
        """Handle failure of health check itself."""
        health_info = self.component_health[component_id]
        health_info.failure_count += 1
        
        if health_info.failure_count >= 3:
            health_info.state = ComponentState.FAILED
            self._trigger_recovery(component_id)
    
    def _trigger_recovery(self, component_id: str):
        """
        Trigger recovery sequence for failed component.
        
        Args:
            component_id: Component to recover
        """
        self.logger.warning(f"Triggering recovery for component {component_id}")
        
        if component_id not in self.recovery_strategies:
            self.logger.error(f"No recovery strategies for {component_id}")
            return
        
        health_info = self.component_health[component_id]
        health_info.state = ComponentState.RECOVERING
        health_info.recovery_attempts += 1
        
        # Try recovery strategies in order
        strategies = self.recovery_strategies[component_id]
        
        for i, recovery_func in enumerate(strategies):
            try:
                self.logger.info(f"Attempting recovery strategy {i+1} for {component_id}")
                
                recovery_result = recovery_func()
                
                if recovery_result:
                    self.logger.info(f"Recovery successful for {component_id} using strategy {i+1}")
                    
                    # Record successful recovery
                    self.recovery_history[component_id].append({
                        'timestamp': time.time(),
                        'strategy_index': i,
                        'success': True,
                        'attempts': health_info.recovery_attempts
                    })
                    
                    # Reset failure counts
                    health_info.failure_count = 0
                    health_info.state = ComponentState.HEALTHY
                    return
                
            except Exception as e:
                self.logger.error(f"Recovery strategy {i+1} failed for {component_id}: {e}")
        
        # All recovery strategies failed
        self.logger.critical(f"All recovery strategies failed for {component_id}")
        
        failure_event = FailureEvent(
            timestamp=time.time(),
            component_id=component_id,
            failure_type=FailureType.HARDWARE_MALFUNCTION,
            severity=1.0,
            details={'recovery_attempts': health_info.recovery_attempts}
        )
        self.critical_failures.append(failure_event)
        
        # Record failed recovery
        self.recovery_history[component_id].append({
            'timestamp': time.time(),
            'strategy_index': -1,
            'success': False,
            'attempts': health_info.recovery_attempts
        })
    
    def _update_system_health(self):
        """Update overall system health score."""
        if not self.component_health:
            self.system_health_score = 1.0
            return
        
        # Weighted average of component health scores
        total_weight = 0.0
        weighted_health = 0.0
        
        for component_id, health_info in self.component_health.items():
            # Weight by criticality (assume 1.0 if not specified)
            weight = 1.0  # Could be made configurable per component
            
            total_weight += weight
            weighted_health += weight * health_info.health_score
        
        self.system_health_score = weighted_health / total_weight if total_weight > 0 else 1.0
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        component_status = {}
        
        for component_id, health_info in self.component_health.items():
            component_status[component_id] = {
                'state': health_info.state.value,
                'health_score': health_info.health_score,
                'failure_count': health_info.failure_count,
                'recovery_attempts': health_info.recovery_attempts,
                'last_heartbeat': health_info.last_heartbeat,
                'performance_metrics': health_info.performance_metrics
            }
        
        return {
            'system_health_score': self.system_health_score,
            'components': component_status,
            'critical_failures': len(self.critical_failures),
            'monitoring_active': self.monitoring_active,
            'recovery_history_size': sum(len(history) for history in self.recovery_history.values())
        }


class GracefulDegradationController:
    """Controls graceful degradation of system performance under failures."""
    
    def __init__(self, dimension: int = 10000):
        """
        Initialize graceful degradation controller.
        
        Args:
            dimension: HDC vector dimension
        """
        self.dimension = dimension
        
        # Degradation policies
        self.degradation_policies = {}
        
        # Performance scaling factors
        self.performance_scaling = {
            'accuracy': 1.0,
            'speed': 1.0,
            'memory': 1.0,
            'energy': 1.0
        }
        
        # Minimum performance thresholds
        self.min_thresholds = {
            'accuracy': 0.6,
            'speed': 0.3,
            'memory': 0.2,
            'energy': 0.1
        }
        
        self.logger = logging.getLogger(__name__)
    
    def register_degradation_policy(self, failure_type: FailureType, 
                                  degradation_steps: List[Dict[str, float]]):
        """
        Register degradation policy for specific failure type.
        
        Args:
            failure_type: Type of failure
            degradation_steps: List of performance scaling factors for each step
        """
        self.degradation_policies[failure_type] = degradation_steps
        self.logger.info(f"Registered degradation policy for {failure_type.value}")
    
    def apply_degradation(self, failure_events: List[FailureEvent]) -> Dict[str, float]:
        """
        Apply graceful degradation based on active failures.
        
        Args:
            failure_events: List of current failure events
            
        Returns:
            Updated performance scaling factors
        """
        # Start with full performance
        new_scaling = {
            'accuracy': 1.0,
            'speed': 1.0,
            'memory': 1.0,
            'energy': 1.0
        }
        
        # Apply degradation for each failure
        for failure in failure_events:
            if failure.failure_type in self.degradation_policies:
                policy = self.degradation_policies[failure.failure_type]
                
                # Select degradation step based on failure severity
                step_index = min(int(failure.severity * len(policy)), len(policy) - 1)
                degradation_step = policy[step_index]
                
                # Apply multiplicative degradation
                for metric, factor in degradation_step.items():
                    if metric in new_scaling:
                        new_scaling[metric] *= factor
                
                self.logger.info(f"Applied degradation for {failure.failure_type.value} "
                               f"(severity: {failure.severity:.2f})")
        
        # Enforce minimum thresholds
        for metric, min_threshold in self.min_thresholds.items():
            if new_scaling[metric] < min_threshold:
                new_scaling[metric] = min_threshold
                self.logger.warning(f"Clamped {metric} scaling to minimum: {min_threshold}")
        
        self.performance_scaling = new_scaling
        return new_scaling
    
    def adapt_hdc_operations(self, original_vector: HyperVector) -> HyperVector:
        """
        Adapt HDC operations based on current performance scaling.
        
        Args:
            original_vector: Original HDC vector
            
        Returns:
            Adapted HDC vector with reduced precision if needed
        """
        # Apply accuracy scaling by adding controlled noise
        accuracy_scaling = self.performance_scaling['accuracy']
        
        if accuracy_scaling < 1.0:
            # Reduce accuracy by adding noise
            noise_level = (1.0 - accuracy_scaling) * 0.5  # Max 50% noise
            adapted_vector = original_vector.add_noise(noise_level)
            
            self.logger.debug(f"Applied accuracy degradation: noise_level={noise_level:.3f}")
            return adapted_vector
        
        return original_vector
    
    def get_degradation_status(self) -> Dict[str, Any]:
        """Get current degradation status."""
        return {
            'performance_scaling': self.performance_scaling.copy(),
            'min_thresholds': self.min_thresholds.copy(),
            'registered_policies': list(self.degradation_policies.keys()),
            'degradation_active': any(scale < 1.0 for scale in self.performance_scaling.values())
        }


class FaultToleranceOrchestrator:
    """Main orchestrator for all fault tolerance mechanisms."""
    
    def __init__(self, dimension: int = 10000):
        """
        Initialize fault tolerance orchestrator.
        
        Args:
            dimension: HDC vector dimension
        """
        self.dimension = dimension
        
        # Initialize subsystems
        self.sensor_compensator = SensorDropoutCompensator(dimension)
        self.redundancy_manager = AdaptiveRedundancyManager(dimension)
        self.self_healing_system = SelfHealingSystem(dimension)
        self.degradation_controller = GracefulDegradationController(dimension)
        
        # Circuit breakers for critical operations
        self.circuit_breakers = {}
        
        # Active failure events
        self.active_failures = []
        
        # Performance monitoring
        self.performance_history = deque(maxlen=1000)
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_fault_tolerance(self):
        """Initialize all fault tolerance mechanisms."""
        # Start self-healing monitoring
        self.self_healing_system.start_monitoring()
        
        # Register default degradation policies
        self._register_default_policies()
        
        # Initialize circuit breakers for critical operations
        self._initialize_circuit_breakers()
        
        self.logger.info("Fault tolerance system initialized")
    
    def _register_default_policies(self):
        """Register default degradation policies."""
        # Sensor dropout policy
        sensor_dropout_steps = [
            {'accuracy': 0.95, 'speed': 1.0},     # Minor dropout
            {'accuracy': 0.85, 'speed': 0.9},     # Moderate dropout
            {'accuracy': 0.7, 'speed': 0.8},      # Severe dropout
            {'accuracy': 0.6, 'speed': 0.7}       # Critical dropout
        ]
        self.degradation_controller.register_degradation_policy(
            FailureType.SENSOR_DROPOUT, sensor_dropout_steps
        )
        
        # Memory corruption policy
        memory_corruption_steps = [
            {'accuracy': 0.9, 'memory': 1.2},     # Minor corruption
            {'accuracy': 0.8, 'memory': 1.5},     # Moderate corruption
            {'accuracy': 0.65, 'memory': 2.0},    # Severe corruption
            {'accuracy': 0.5, 'memory': 3.0}      # Critical corruption
        ]
        self.degradation_controller.register_degradation_policy(
            FailureType.MEMORY_CORRUPTION, memory_corruption_steps
        )
        
        # Resource exhaustion policy
        resource_exhaustion_steps = [
            {'speed': 0.9, 'energy': 1.1},        # Minor exhaustion
            {'speed': 0.7, 'energy': 1.3},        # Moderate exhaustion
            {'speed': 0.5, 'energy': 1.6},        # Severe exhaustion
            {'speed': 0.3, 'energy': 2.0}         # Critical exhaustion
        ]
        self.degradation_controller.register_degradation_policy(
            FailureType.RESOURCE_EXHAUSTION, resource_exhaustion_steps
        )
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for critical operations."""
        # HDC computation circuit breaker
        self.circuit_breakers['hdc_computation'] = CircuitBreaker(
            failure_threshold=3,
            timeout=30.0,
            success_threshold=2
        )
        
        # Sensor processing circuit breaker
        self.circuit_breakers['sensor_processing'] = CircuitBreaker(
            failure_threshold=5,
            timeout=10.0,
            success_threshold=3
        )
        
        # Memory operations circuit breaker
        self.circuit_breakers['memory_operations'] = CircuitBreaker(
            failure_threshold=2,
            timeout=60.0,
            success_threshold=1
        )
    
    def process_with_fault_tolerance(self, operation_name: str, operation_func: Callable,
                                   sensor_data: Optional[Dict[str, HyperVector]] = None,
                                   *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute operation with full fault tolerance protection.
        
        Args:
            operation_name: Name of operation for monitoring
            operation_func: Function to execute
            sensor_data: Optional sensor data for dropout compensation
            *args, **kwargs: Arguments for operation function
            
        Returns:
            (result, fault_tolerance_info) tuple
        """
        start_time = time.time()
        fault_tolerance_info = {
            'circuit_breaker_used': False,
            'sensor_compensation_applied': False,
            'degradation_applied': False,
            'execution_time': 0.0,
            'success': False
        }
        
        try:
            # Check if sensor compensation is needed
            if sensor_data:
                available_sensors = {k: v for k, v in sensor_data.items() if v is not None}
                missing_sensors = [k for k, v in sensor_data.items() if v is None]
                
                if missing_sensors:
                    compensated_data = self.sensor_compensator.compensate_sensor_dropout(
                        available_sensors, missing_sensors
                    )
                    sensor_data = compensated_data
                    fault_tolerance_info['sensor_compensation_applied'] = True
                    
                    # Learn correlations during normal operation
                    if len(missing_sensors) == 0:
                        self.sensor_compensator.learn_sensor_correlations(sensor_data)
            
            # Apply graceful degradation if needed
            if self.active_failures:
                performance_scaling = self.degradation_controller.apply_degradation(self.active_failures)
                fault_tolerance_info['degradation_applied'] = True
                fault_tolerance_info['performance_scaling'] = performance_scaling
            
            # Execute operation through circuit breaker
            circuit_breaker_name = self._get_circuit_breaker_name(operation_name)
            if circuit_breaker_name in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[circuit_breaker_name]
                result, success = circuit_breaker.call(operation_func, *args, **kwargs)
                fault_tolerance_info['circuit_breaker_used'] = True
            else:
                result = operation_func(*args, **kwargs)
                success = True
            
            if success:
                fault_tolerance_info['success'] = True
                execution_time = time.time() - start_time
                fault_tolerance_info['execution_time'] = execution_time
                
                # Record performance metrics
                self.performance_history.append({
                    'timestamp': time.time(),
                    'operation': operation_name,
                    'execution_time': execution_time,
                    'success': True
                })
                
                return result, fault_tolerance_info
            else:
                raise Exception("Operation failed through circuit breaker")
                
        except Exception as e:
            self.logger.error(f"Fault-tolerant operation failed: {operation_name}: {e}")
            
            # Record failure
            failure_event = FailureEvent(
                timestamp=time.time(),
                component_id=operation_name,
                failure_type=FailureType.COMPUTATION_ERROR,
                severity=0.8,
                details={'error': str(e)}
            )
            self.active_failures.append(failure_event)
            
            # Update performance history
            execution_time = time.time() - start_time
            self.performance_history.append({
                'timestamp': time.time(),
                'operation': operation_name,
                'execution_time': execution_time,
                'success': False
            })
            
            fault_tolerance_info['execution_time'] = execution_time
            raise
    
    def _get_circuit_breaker_name(self, operation_name: str) -> str:
        """Map operation name to circuit breaker name."""
        if 'hdc' in operation_name.lower() or 'compute' in operation_name.lower():
            return 'hdc_computation'
        elif 'sensor' in operation_name.lower():
            return 'sensor_processing'
        elif 'memory' in operation_name.lower():
            return 'memory_operations'
        else:
            return 'hdc_computation'  # Default
    
    def shutdown_fault_tolerance(self):
        """Shutdown fault tolerance system."""
        self.self_healing_system.stop_monitoring()
        self.logger.info("Fault tolerance system shutdown")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive fault tolerance status."""
        return {
            'sensor_compensator': self.sensor_compensator.get_compensation_statistics(),
            'redundancy_manager': self.redundancy_manager.get_redundancy_status(),
            'self_healing': self.self_healing_system.get_system_status(),
            'degradation_controller': self.degradation_controller.get_degradation_status(),
            'circuit_breakers': {name: {
                'state': breaker.state.value,
                'failure_count': breaker.failure_count,
                'success_count': breaker.success_count
            } for name, breaker in self.circuit_breakers.items()},
            'active_failures': len(self.active_failures),
            'performance_history_size': len(self.performance_history)
        }