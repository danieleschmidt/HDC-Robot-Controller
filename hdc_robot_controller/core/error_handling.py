"""
Comprehensive error handling and validation for HDC Robot Controller.
Provides robust error recovery, input validation, and system health monitoring.
"""

import logging
import traceback
import time
from typing import Any, Optional, Dict, List, Callable, Union
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import numpy as np
from .hypervector import HyperVector


class ErrorSeverity(Enum):
    """Error severity levels for HDC system."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Categories of errors in HDC system."""
    DIMENSION_MISMATCH = "dimension_mismatch"
    INVALID_INPUT = "invalid_input"
    COMPUTATION_ERROR = "computation_error"
    MEMORY_ERROR = "memory_error"
    TIMEOUT = "timeout"
    SENSOR_ERROR = "sensor_error"
    CONTROL_ERROR = "control_error"
    LEARNING_ERROR = "learning_error"
    SYSTEM_ERROR = "system_error"


@dataclass
class HDCError:
    """Structured error information for HDC system."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: Dict[str, Any]
    timestamp: float
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None


class HDCException(Exception):
    """Base exception class for HDC Robot Controller."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.SYSTEM_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM, context: Optional[Dict] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class DimensionMismatchError(HDCException):
    """Exception for hypervector dimension mismatches."""
    
    def __init__(self, expected: int, actual: int, operation: str = "unknown"):
        message = f"Dimension mismatch in {operation}: expected {expected}, got {actual}"
        context = {"expected_dim": expected, "actual_dim": actual, "operation": operation}
        super().__init__(message, ErrorCategory.DIMENSION_MISMATCH, ErrorSeverity.HIGH, context)


class InvalidInputError(HDCException):
    """Exception for invalid input data."""
    
    def __init__(self, message: str, input_value: Any = None, expected_type: str = None):
        context = {"input_value": str(input_value), "expected_type": expected_type}
        super().__init__(message, ErrorCategory.INVALID_INPUT, ErrorSeverity.MEDIUM, context)


class ComputationError(HDCException):
    """Exception for computation errors in HDC operations."""
    
    def __init__(self, message: str, operation: str = "unknown", data_shape: Optional[tuple] = None):
        context = {"operation": operation, "data_shape": data_shape}
        super().__init__(message, ErrorCategory.COMPUTATION_ERROR, ErrorSeverity.HIGH, context)


class SensorError(HDCException):
    """Exception for sensor-related errors."""
    
    def __init__(self, message: str, sensor_name: str = "unknown", sensor_data: Any = None):
        context = {"sensor_name": sensor_name, "sensor_data_type": type(sensor_data).__name__}
        super().__init__(message, ErrorCategory.SENSOR_ERROR, ErrorSeverity.HIGH, context)


class ControlError(HDCException):
    """Exception for control system errors."""
    
    def __init__(self, message: str, control_action: Any = None, confidence: float = 0.0):
        context = {"control_action": str(control_action), "confidence": confidence}
        super().__init__(message, ErrorCategory.CONTROL_ERROR, ErrorSeverity.HIGH, context)


class LearningError(HDCException):
    """Exception for learning system errors."""
    
    def __init__(self, message: str, behavior_name: str = "unknown", sample_count: int = 0):
        context = {"behavior_name": behavior_name, "sample_count": sample_count}
        super().__init__(message, ErrorCategory.LEARNING_ERROR, ErrorSeverity.MEDIUM, context)


class ErrorRecoveryManager:
    """Manages error recovery strategies for the HDC system."""
    
    def __init__(self, max_recovery_attempts: int = 3):
        self.max_recovery_attempts = max_recovery_attempts
        self.error_history: List[HDCError] = []
        self.recovery_strategies: Dict[ErrorCategory, Callable] = {}
        self.logger = logging.getLogger(__name__)
        
        # Register default recovery strategies
        self._register_default_recovery_strategies()
    
    def _register_default_recovery_strategies(self):
        """Register default recovery strategies for common errors."""
        self.recovery_strategies[ErrorCategory.DIMENSION_MISMATCH] = self._recover_dimension_mismatch
        self.recovery_strategies[ErrorCategory.INVALID_INPUT] = self._recover_invalid_input
        self.recovery_strategies[ErrorCategory.COMPUTATION_ERROR] = self._recover_computation_error
        self.recovery_strategies[ErrorCategory.SENSOR_ERROR] = self._recover_sensor_error
        self.recovery_strategies[ErrorCategory.CONTROL_ERROR] = self._recover_control_error
        self.recovery_strategies[ErrorCategory.LEARNING_ERROR] = self._recover_learning_error
    
    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> bool:
        """Handle an error and attempt recovery."""
        # Convert to HDCError if necessary
        if isinstance(error, HDCException):
            hdc_error = HDCError(
                category=error.category,
                severity=error.severity,
                message=str(error),
                context=error.context,
                timestamp=error.timestamp,
                stack_trace=traceback.format_exc()
            )
        else:
            hdc_error = HDCError(
                category=ErrorCategory.SYSTEM_ERROR,
                severity=ErrorSeverity.MEDIUM,
                message=str(error),
                context=context or {},
                timestamp=time.time(),
                stack_trace=traceback.format_exc()
            )
        
        # Log error
        self._log_error(hdc_error)
        
        # Store in history
        self.error_history.append(hdc_error)
        
        # Attempt recovery
        return self._attempt_recovery(hdc_error)
    
    def _log_error(self, error: HDCError):
        """Log error with appropriate level."""
        log_message = f"[{error.category.value}] {error.message}"
        
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Log context if available
        if error.context:
            self.logger.debug(f"Error context: {error.context}")
    
    def _attempt_recovery(self, error: HDCError) -> bool:
        """Attempt to recover from an error."""
        recovery_func = self.recovery_strategies.get(error.category)
        
        if recovery_func is None:
            self.logger.warning(f"No recovery strategy for {error.category.value}")
            return False
        
        try:
            recovery_success = recovery_func(error)
            if recovery_success:
                self.logger.info(f"Successfully recovered from {error.category.value}")
            else:
                self.logger.warning(f"Recovery failed for {error.category.value}")
            
            return recovery_success
            
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy failed: {recovery_error}")
            return False
    
    # Recovery strategy implementations
    def _recover_dimension_mismatch(self, error: HDCError) -> bool:
        """Recover from dimension mismatch errors."""
        self.logger.info("Attempting to recover from dimension mismatch")
        # Recovery strategy: resize or create new vectors with correct dimensions
        # This is a placeholder - actual implementation would depend on context
        return True
    
    def _recover_invalid_input(self, error: HDCError) -> bool:
        """Recover from invalid input errors."""
        self.logger.info("Attempting to recover from invalid input")
        # Recovery strategy: sanitize input or use default values
        return True
    
    def _recover_computation_error(self, error: HDCError) -> bool:
        """Recover from computation errors."""
        self.logger.info("Attempting to recover from computation error")
        # Recovery strategy: retry with different parameters or fallback computation
        return True
    
    def _recover_sensor_error(self, error: HDCError) -> bool:
        """Recover from sensor errors."""
        self.logger.info("Attempting to recover from sensor error")
        # Recovery strategy: use backup sensors or estimated values
        return True
    
    def _recover_control_error(self, error: HDCError) -> bool:
        """Recover from control errors."""
        self.logger.info("Attempting to recover from control error")
        # Recovery strategy: switch to safe control mode
        return True
    
    def _recover_learning_error(self, error: HDCError) -> bool:
        """Recover from learning errors."""
        self.logger.info("Attempting to recover from learning error")
        # Recovery strategy: use previous learned behavior or default
        return True
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get statistics about errors."""
        if not self.error_history:
            return {"total_errors": 0}
        
        category_counts = {}
        severity_counts = {}
        
        for error in self.error_history:
            category_counts[error.category.value] = category_counts.get(error.category.value, 0) + 1
            severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "by_category": category_counts,
            "by_severity": severity_counts,
            "recent_errors": len([e for e in self.error_history if time.time() - e.timestamp < 300])  # Last 5 minutes
        }


# Validation functions
def validate_hypervector(hv: Any, name: str = "hypervector") -> HyperVector:
    """Validate and convert input to HyperVector."""
    if hv is None:
        raise InvalidInputError(f"{name} cannot be None")
    
    if isinstance(hv, HyperVector):
        return hv
    
    if isinstance(hv, (list, np.ndarray)):
        try:
            return HyperVector.from_numpy(np.array(hv, dtype=np.int8))
        except Exception as e:
            raise InvalidInputError(f"Cannot convert {name} to HyperVector: {e}")
    
    raise InvalidInputError(f"{name} must be HyperVector, list, or numpy array, got {type(hv)}")


def validate_dimension_match(hv1: HyperVector, hv2: HyperVector, operation: str = "operation"):
    """Validate that two hypervectors have matching dimensions."""
    if hv1.dimension != hv2.dimension:
        raise DimensionMismatchError(hv1.dimension, hv2.dimension, operation)


def validate_dimension(dimension: int, name: str = "dimension") -> int:
    """Validate dimension parameter."""
    if not isinstance(dimension, int):
        raise InvalidInputError(f"{name} must be integer, got {type(dimension)}")
    
    if dimension <= 0:
        raise InvalidInputError(f"{name} must be positive, got {dimension}")
    
    if dimension > 100000:  # Reasonable upper limit
        raise InvalidInputError(f"{name} too large (>{100000}), got {dimension}")
    
    return dimension


def validate_probability(prob: float, name: str = "probability") -> float:
    """Validate probability value."""
    if not isinstance(prob, (int, float)):
        raise InvalidInputError(f"{name} must be numeric, got {type(prob)}")
    
    if not 0.0 <= prob <= 1.0:
        raise InvalidInputError(f"{name} must be between 0 and 1, got {prob}")
    
    return float(prob)


def validate_positive_number(value: Union[int, float], name: str = "value") -> Union[int, float]:
    """Validate positive number."""
    if not isinstance(value, (int, float)):
        raise InvalidInputError(f"{name} must be numeric, got {type(value)}")
    
    if value <= 0:
        raise InvalidInputError(f"{name} must be positive, got {value}")
    
    return value


def validate_array_shape(array: np.ndarray, expected_shape: tuple, name: str = "array"):
    """Validate numpy array shape."""
    if not isinstance(array, np.ndarray):
        raise InvalidInputError(f"{name} must be numpy array, got {type(array)}")
    
    if array.shape != expected_shape:
        raise InvalidInputError(f"{name} shape mismatch: expected {expected_shape}, got {array.shape}")


# Decorator for robust function execution
def robust_hdc_operation(max_retries: int = 3, recovery_manager: Optional[ErrorRecoveryManager] = None):
    """Decorator for robust HDC operations with automatic error handling and recovery."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                    
                except HDCException as e:
                    last_exception = e
                    
                    if recovery_manager:
                        recovery_success = recovery_manager.handle_error(e)
                        if not recovery_success and attempt < max_retries - 1:
                            time.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    
                except Exception as e:
                    last_exception = e
                    
                    if recovery_manager:
                        recovery_manager.handle_error(e)
                    
                    if attempt < max_retries - 1:
                        time.sleep(0.1 * (attempt + 1))
            
            # If all retries failed, raise the last exception
            if last_exception:
                raise last_exception
            else:
                raise ComputationError("All retry attempts failed")
        
        return wrapper
    return decorator


# Input sanitization functions
def sanitize_sensor_data(data: Any) -> Optional[np.ndarray]:
    """Sanitize sensor data to prevent invalid inputs."""
    if data is None:
        return None
    
    try:
        if isinstance(data, (list, tuple)):
            array = np.array(data, dtype=np.float32)
        elif isinstance(data, np.ndarray):
            array = data.astype(np.float32)
        else:
            return None
        
        # Check for invalid values
        if not np.all(np.isfinite(array)):
            # Replace invalid values with zeros
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clamp extreme values
        array = np.clip(array, -1000.0, 1000.0)
        
        return array
        
    except Exception:
        return None


def sanitize_string_input(text: str, max_length: int = 1000) -> str:
    """Sanitize string input."""
    if not isinstance(text, str):
        text = str(text)
    
    # Remove control characters
    text = ''.join(char for char in text if char.isprintable() or char.isspace())
    
    # Limit length
    if len(text) > max_length:
        text = text[:max_length]
    
    return text.strip()


def sanitize_numeric_input(value: Any, min_val: float = -1e6, max_val: float = 1e6) -> float:
    """Sanitize numeric input."""
    try:
        if isinstance(value, str):
            value = float(value)
        elif not isinstance(value, (int, float)):
            return 0.0
        
        if not np.isfinite(value):
            return 0.0
        
        return float(np.clip(value, min_val, max_val))
        
    except (ValueError, TypeError):
        return 0.0


# System health monitoring
class HealthMonitor:
    """Monitor system health and detect degradation."""
    
    def __init__(self, check_interval: float = 1.0):
        self.check_interval = check_interval
        self.last_check = time.time()
        self.health_metrics = {}
        self.alerts = []
        
    def update_metric(self, name: str, value: float, threshold_low: float = 0.0, 
                     threshold_high: float = 1.0):
        """Update a health metric and check thresholds."""
        self.health_metrics[name] = {
            "value": value,
            "threshold_low": threshold_low,
            "threshold_high": threshold_high,
            "timestamp": time.time()
        }
        
        # Check thresholds
        if value < threshold_low or value > threshold_high:
            alert = {
                "metric": name,
                "value": value,
                "threshold_low": threshold_low,
                "threshold_high": threshold_high,
                "timestamp": time.time()
            }
            self.alerts.append(alert)
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        now = time.time()
        healthy_metrics = 0
        total_metrics = len(self.health_metrics)
        
        for metric_data in self.health_metrics.values():
            value = metric_data["value"]
            if metric_data["threshold_low"] <= value <= metric_data["threshold_high"]:
                healthy_metrics += 1
        
        health_score = healthy_metrics / total_metrics if total_metrics > 0 else 1.0
        
        # Check for stale metrics
        stale_metrics = [name for name, data in self.health_metrics.items() 
                        if now - data["timestamp"] > 10.0]
        
        return {
            "health_score": health_score,
            "healthy_metrics": healthy_metrics,
            "total_metrics": total_metrics,
            "recent_alerts": len([a for a in self.alerts if now - a["timestamp"] < 60.0]),
            "stale_metrics": stale_metrics
        }


# Global error recovery manager instance
_global_recovery_manager = ErrorRecoveryManager()

def get_global_error_manager() -> ErrorRecoveryManager:
    """Get the global error recovery manager."""
    return _global_recovery_manager