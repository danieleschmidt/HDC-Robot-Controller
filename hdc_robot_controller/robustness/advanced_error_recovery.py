"""
Advanced Error Recovery and Fault Tolerance for HDC Robot Controller.

Implements comprehensive error recovery strategies, graceful degradation,
and autonomous system repair capabilities.
"""

import time
import threading
import traceback
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict, deque
from enum import Enum
import numpy as np

from ..core.hypervector import HyperVector
from ..core.memory import HierarchicalMemory
from ..core.logging_system import get_logger


class FaultSeverity(Enum):
    """Fault severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    DEGRADE = "degrade"
    ISOLATE = "isolate"
    RESTART = "restart"
    MANUAL = "manual"


class SystemState(Enum):
    """System operational states."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    RECOVERY = "recovery"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class FaultRecord:
    """Record of a system fault."""
    
    def __init__(self, fault_id: str, component: str, severity: FaultSeverity,
                 description: str, context: Optional[Dict[str, Any]] = None):
        self.fault_id = fault_id
        self.component = component
        self.severity = severity
        self.description = description
        self.context = context or {}
        self.timestamp = time.time()
        self.recovery_attempts = []
        self.resolved = False
        self.resolution_timestamp = None


class RecoveryAction:
    """Represents a recovery action."""
    
    def __init__(self, action_id: str, strategy: RecoveryStrategy,
                 handler: Callable[[], bool], timeout: float = 30.0,
                 max_attempts: int = 3):
        self.action_id = action_id
        self.strategy = strategy
        self.handler = handler
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.attempts = 0
        self.last_attempt = None
        self.success_history = deque(maxlen=10)


class AdvancedErrorRecovery:
    """Advanced error recovery and fault tolerance system."""
    
    def __init__(self, dimension: int = 10000, enable_learning: bool = True):
        """Initialize advanced error recovery system.
        
        Args:
            dimension: Hypervector dimension for fault encoding
            enable_learning: Whether to enable learning from recovery patterns
        """
        self.dimension = dimension
        self.enable_learning = enable_learning
        self.logger = get_logger()
        
        # System state
        self.current_state = SystemState.NORMAL
        self.state_history = deque(maxlen=100)
        
        # Fault tracking
        self.active_faults = {}  # fault_id -> FaultRecord
        self.fault_history = deque(maxlen=1000)
        self.component_health = defaultdict(lambda: 1.0)  # component -> health score (0-1)
        
        # Recovery strategies
        self.recovery_actions = {}  # component -> List[RecoveryAction]
        self.fallback_handlers = {}  # component -> fallback function
        self.degradation_levels = {}  # component -> current degradation level
        
        # Learning system
        if enable_learning:
            self.fault_memory = HierarchicalMemory(dimension)
            self.pattern_encoder = FaultPatternEncoder(dimension)
        
        # Statistics
        self.recovery_stats = {
            'total_faults': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'manual_interventions': 0,
            'system_restarts': 0
        }
        
        # Threading
        self._lock = threading.RLock()
        self._monitoring_active = False
        self._monitor_thread = None
        
        # Watchdog
        self.watchdog_enabled = False
        self.watchdog_timeout = 300.0  # 5 minutes default
        self.last_heartbeat = time.time()
    
    def start_monitoring(self, monitoring_interval: float = 5.0):
        """Start background fault monitoring and recovery."""
        with self._lock:
            if self._monitoring_active:
                return
            
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                args=(monitoring_interval,),
                daemon=True
            )
            self._monitor_thread.start()
            
            self.logger.info("Advanced error recovery monitoring started",
                           monitoring_interval=monitoring_interval)
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        with self._lock:
            self._monitoring_active = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=10.0)
            
            self.logger.info("Advanced error recovery monitoring stopped")
    
    def register_recovery_action(self, component: str, action: RecoveryAction):
        """Register a recovery action for a component.
        
        Args:
            component: Component name
            action: Recovery action to register
        """
        with self._lock:
            if component not in self.recovery_actions:
                self.recovery_actions[component] = []
            
            self.recovery_actions[component].append(action)
            
            self.logger.info(f"Registered recovery action for {component}",
                           component=component,
                           action_id=action.action_id,
                           strategy=action.strategy.value)
    
    def register_fallback_handler(self, component: str, handler: Callable[[], bool]):
        """Register a fallback handler for a component.
        
        Args:
            component: Component name
            handler: Fallback function that returns success/failure
        """
        with self._lock:
            self.fallback_handlers[component] = handler
            self.logger.info(f"Registered fallback handler for {component}")
    
    def report_fault(self, component: str, severity: FaultSeverity, 
                    description: str, context: Optional[Dict[str, Any]] = None,
                    auto_recover: bool = True) -> str:
        """Report a system fault.
        
        Args:
            component: Component where fault occurred
            severity: Fault severity level
            description: Human-readable fault description
            context: Additional context data
            auto_recover: Whether to attempt automatic recovery
            
        Returns:
            Fault ID for tracking
        """
        fault_id = f"{component}_{int(time.time() * 1000)}"
        
        with self._lock:
            # Create fault record
            fault = FaultRecord(fault_id, component, severity, description, context)
            self.active_faults[fault_id] = fault
            self.fault_history.append(fault)
            
            # Update component health
            self._update_component_health(component, severity)
            
            # Update system state
            self._update_system_state()
            
            # Update statistics
            self.recovery_stats['total_faults'] += 1
            
            # Log fault
            self.logger.error(f"Fault reported: {description}",
                            fault_id=fault_id,
                            component=component,
                            severity=severity.name,
                            context=context or {})
            
            # Learn from fault pattern
            if self.enable_learning:
                self._learn_fault_pattern(fault)
            
            # Attempt automatic recovery if enabled
            if auto_recover:
                threading.Thread(
                    target=self._attempt_recovery,
                    args=(fault_id,),
                    daemon=True
                ).start()
        
        return fault_id
    
    def resolve_fault(self, fault_id: str, resolution_notes: str = ""):
        """Mark a fault as resolved.
        
        Args:
            fault_id: Fault identifier
            resolution_notes: Optional resolution notes
        """
        with self._lock:
            if fault_id not in self.active_faults:
                self.logger.warning(f"Cannot resolve unknown fault: {fault_id}")
                return
            
            fault = self.active_faults[fault_id]
            fault.resolved = True
            fault.resolution_timestamp = time.time()
            
            # Remove from active faults
            del self.active_faults[fault_id]
            
            # Update component health (improve)
            self._improve_component_health(fault.component)
            
            # Update system state
            self._update_system_state()
            
            self.logger.info(f"Fault resolved: {fault.description}",
                           fault_id=fault_id,
                           component=fault.component,
                           resolution_time=fault.resolution_timestamp - fault.timestamp,
                           resolution_notes=resolution_notes)
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report.
        
        Returns:
            Dictionary with system health information
        """
        with self._lock:
            # Active faults by severity
            fault_summary = defaultdict(int)
            for fault in self.active_faults.values():
                fault_summary[fault.severity.name] += 1
            
            # Component health summary
            unhealthy_components = {
                comp: health for comp, health in self.component_health.items()
                if health < 0.8
            }
            
            # Recent fault trends
            recent_faults = [f for f in self.fault_history 
                           if time.time() - f.timestamp < 3600]  # Last hour
            
            return {
                'system_state': self.current_state.value,
                'overall_health': self._calculate_overall_health(),
                'active_faults_count': len(self.active_faults),
                'active_faults_by_severity': dict(fault_summary),
                'component_health': dict(self.component_health),
                'unhealthy_components': unhealthy_components,
                'recent_faults_count': len(recent_faults),
                'recovery_stats': self.recovery_stats.copy(),
                'last_heartbeat': self.last_heartbeat,
                'watchdog_enabled': self.watchdog_enabled,
                'timestamp': time.time()
            }
    
    def enable_watchdog(self, timeout: float = 300.0):
        """Enable system watchdog for deadlock detection.
        
        Args:
            timeout: Watchdog timeout in seconds
        """
        self.watchdog_enabled = True
        self.watchdog_timeout = timeout
        self.heartbeat()
        
        self.logger.info("System watchdog enabled",
                        timeout=timeout)
    
    def heartbeat(self):
        """Update system heartbeat (call periodically to prevent watchdog timeout)."""
        self.last_heartbeat = time.time()
    
    def force_degradation(self, component: str, level: float):
        """Force degradation of a component.
        
        Args:
            component: Component to degrade
            level: Degradation level (0.0 = fully degraded, 1.0 = normal)
        """
        with self._lock:
            level = max(0.0, min(1.0, level))
            self.degradation_levels[component] = level
            self.component_health[component] = min(self.component_health[component], level)
            
            self.logger.warning(f"Forced degradation of {component}",
                              component=component,
                              degradation_level=level)
    
    def emergency_shutdown(self, reason: str):
        """Initiate emergency shutdown sequence.
        
        Args:
            reason: Reason for emergency shutdown
        """
        with self._lock:
            self.current_state = SystemState.EMERGENCY
            
            self.logger.critical("EMERGENCY SHUTDOWN INITIATED",
                               reason=reason,
                               timestamp=time.time())
            
            # Stop all monitoring
            self.stop_monitoring()
            
            # Execute emergency procedures
            self._execute_emergency_procedures()
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                # Check watchdog
                if self.watchdog_enabled:
                    if time.time() - self.last_heartbeat > self.watchdog_timeout:
                        self.report_fault(
                            "system_watchdog",
                            FaultSeverity.CRITICAL,
                            f"Watchdog timeout: no heartbeat for {self.watchdog_timeout}s"
                        )
                
                # Monitor component health
                self._monitor_component_health()
                
                # Check for automatic recovery opportunities
                self._check_recovery_opportunities()
                
                # Update degradation levels
                self._update_degradation_levels()
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.log_exception(e, "error recovery monitoring loop")
                time.sleep(interval)
    
    def _attempt_recovery(self, fault_id: str):
        """Attempt automatic recovery for a fault.
        
        Args:
            fault_id: Fault identifier
        """
        with self._lock:
            if fault_id not in self.active_faults:
                return
            
            fault = self.active_faults[fault_id]
            component = fault.component
            
            self.logger.info(f"Attempting recovery for fault: {fault.description}",
                           fault_id=fault_id,
                           component=component)
            
            # Try registered recovery actions
            if component in self.recovery_actions:
                for action in self.recovery_actions[component]:
                    if action.attempts >= action.max_attempts:
                        continue
                    
                    try:
                        self.logger.info(f"Executing recovery action: {action.action_id}",
                                       fault_id=fault_id,
                                       component=component,
                                       strategy=action.strategy.value,
                                       attempt=action.attempts + 1)
                        
                        # Execute recovery action
                        action.attempts += 1
                        action.last_attempt = time.time()
                        
                        success = action.handler()
                        action.success_history.append(success)
                        
                        if success:
                            self.logger.info(f"Recovery action successful: {action.action_id}",
                                           fault_id=fault_id)
                            
                            self.recovery_stats['successful_recoveries'] += 1
                            self.resolve_fault(fault_id, f"Auto-recovered via {action.action_id}")
                            return
                        else:
                            self.logger.warning(f"Recovery action failed: {action.action_id}",
                                              fault_id=fault_id)
                    
                    except Exception as e:
                        self.logger.log_exception(e, f"recovery action {action.action_id}")
                        action.success_history.append(False)
            
            # Try fallback handler
            if component in self.fallback_handlers:
                try:
                    self.logger.info(f"Attempting fallback recovery for {component}",
                                   fault_id=fault_id)
                    
                    if self.fallback_handlers[component]():
                        self.logger.info(f"Fallback recovery successful for {component}",
                                       fault_id=fault_id)
                        
                        self.recovery_stats['successful_recoveries'] += 1
                        self.resolve_fault(fault_id, "Auto-recovered via fallback")
                        return
                
                except Exception as e:
                    self.logger.log_exception(e, f"fallback recovery for {component}")
            
            # Recovery failed
            self.recovery_stats['failed_recoveries'] += 1
            
            # Escalate based on severity
            self._escalate_fault(fault)
    
    def _escalate_fault(self, fault: FaultRecord):
        """Escalate unresolved fault.
        
        Args:
            fault: Fault record
        """
        if fault.severity == FaultSeverity.CRITICAL:
            self.logger.critical(f"Critical fault escalation: {fault.description}",
                               fault_id=fault.fault_id,
                               component=fault.component)
            
            # Consider emergency shutdown
            if len([f for f in self.active_faults.values() 
                   if f.severity == FaultSeverity.CRITICAL]) >= 3:
                self.emergency_shutdown("Multiple critical faults")
        
        elif fault.severity == FaultSeverity.HIGH:
            # Force degradation of component
            self.force_degradation(fault.component, 0.5)
        
        # Request manual intervention for unresolved high/critical faults
        if fault.severity.value >= FaultSeverity.HIGH.value:
            self.recovery_stats['manual_interventions'] += 1
            self.logger.error(f"Manual intervention required: {fault.description}",
                            fault_id=fault.fault_id,
                            component=fault.component,
                            severity=fault.severity.name)
    
    def _update_component_health(self, component: str, severity: FaultSeverity):
        """Update component health based on fault severity."""
        health_impact = {
            FaultSeverity.LOW: 0.05,
            FaultSeverity.MEDIUM: 0.15,
            FaultSeverity.HIGH: 0.3,
            FaultSeverity.CRITICAL: 0.5
        }
        
        current_health = self.component_health[component]
        impact = health_impact[severity]
        
        self.component_health[component] = max(0.0, current_health - impact)
    
    def _improve_component_health(self, component: str):
        """Improve component health after successful resolution."""
        current_health = self.component_health[component]
        improvement = 0.1  # 10% improvement
        
        self.component_health[component] = min(1.0, current_health + improvement)
    
    def _update_system_state(self):
        """Update overall system state based on current conditions."""
        critical_faults = sum(1 for f in self.active_faults.values() 
                            if f.severity == FaultSeverity.CRITICAL)
        high_faults = sum(1 for f in self.active_faults.values() 
                        if f.severity == FaultSeverity.HIGH)
        
        overall_health = self._calculate_overall_health()
        
        if critical_faults >= 2:
            new_state = SystemState.EMERGENCY
        elif critical_faults >= 1 or high_faults >= 3:
            new_state = SystemState.RECOVERY
        elif overall_health < 0.7 or high_faults >= 1:
            new_state = SystemState.DEGRADED
        else:
            new_state = SystemState.NORMAL
        
        if new_state != self.current_state:
            old_state = self.current_state
            self.current_state = new_state
            self.state_history.append((time.time(), old_state, new_state))
            
            self.logger.info(f"System state changed: {old_state.value} -> {new_state.value}",
                           old_state=old_state.value,
                           new_state=new_state.value,
                           overall_health=overall_health)
    
    def _calculate_overall_health(self) -> float:
        """Calculate overall system health score."""
        if not self.component_health:
            return 1.0
        
        return sum(self.component_health.values()) / len(self.component_health)
    
    def _monitor_component_health(self):
        """Monitor component health and detect degradation."""
        for component, health in self.component_health.items():
            if health < 0.3:  # Very unhealthy
                if component not in [f.component for f in self.active_faults.values()]:
                    self.report_fault(
                        component,
                        FaultSeverity.HIGH,
                        f"Component health critically low: {health:.2f}"
                    )
    
    def _check_recovery_opportunities(self):
        """Check for automatic recovery opportunities."""
        # Gradually improve component health over time if no new faults
        for component in list(self.component_health.keys()):
            if component not in [f.component for f in self.active_faults.values()]:
                # No active faults for this component, gradually improve health
                current_health = self.component_health[component]
                if current_health < 1.0:
                    self.component_health[component] = min(1.0, current_health + 0.01)
    
    def _update_degradation_levels(self):
        """Update component degradation levels."""
        # Remove degradation for components that have recovered
        for component in list(self.degradation_levels.keys()):
            if self.component_health[component] > 0.8:
                del self.degradation_levels[component]
    
    def _learn_fault_pattern(self, fault: FaultRecord):
        """Learn from fault patterns for future prediction."""
        if not self.enable_learning:
            return
        
        try:
            # Encode fault as hypervector
            fault_hv = self.pattern_encoder.encode_fault(fault)
            
            # Store in memory
            self.fault_memory.store_experience(
                fault_hv,
                HyperVector.zero(self.dimension),  # No action for now
                f"fault_{fault.component}_{fault.severity.name}",
                confidence=0.8
            )
        
        except Exception as e:
            self.logger.log_exception(e, "fault pattern learning")
    
    def _execute_emergency_procedures(self):
        """Execute emergency shutdown procedures."""
        try:
            # Save critical data
            self.logger.info("Executing emergency procedures")
            
            # Attempt graceful shutdown of all components
            for component in self.component_health.keys():
                try:
                    if component in self.fallback_handlers:
                        self.fallback_handlers[component]()
                except:
                    pass  # Best effort
            
            self.logger.info("Emergency procedures completed")
            
        except Exception as e:
            self.logger.log_exception(e, "emergency procedures")


class FaultPatternEncoder:
    """Encode fault patterns as hypervectors for learning."""
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        from ..core.operations import BasisVectors
        self.basis_vectors = BasisVectors(dimension)
    
    def encode_fault(self, fault: FaultRecord) -> HyperVector:
        """Encode fault as hypervector.
        
        Args:
            fault: Fault record to encode
            
        Returns:
            HyperVector representing the fault
        """
        components = []
        
        # Encode component
        comp_hv = self.basis_vectors.encode_category(fault.component)
        components.append(comp_hv)
        
        # Encode severity
        severity_hv = self.basis_vectors.encode_category(f"severity_{fault.severity.name}")
        components.append(severity_hv)
        
        # Encode time of day (for temporal patterns)
        hour = int((fault.timestamp % (24 * 3600)) // 3600)
        time_hv = self.basis_vectors.encode_integer(hour, 0, 23)
        components.append(time_hv)
        
        # Encode context if available
        if fault.context:
            for key, value in fault.context.items():
                key_hv = self.basis_vectors.encode_category(f"context_{key}")
                if isinstance(value, (int, float)):
                    value_hv = self.basis_vectors.encode_float(float(value), -100, 100, 100)
                else:
                    value_hv = self.basis_vectors.encode_category(str(value))
                components.append(key_hv.bind(value_hv))
        
        return HyperVector.bundle_vectors(components) if components else HyperVector.zero(self.dimension)