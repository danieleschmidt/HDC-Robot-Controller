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


class AdvancedSecurityValidator:
    """Advanced security validation and threat detection for HDC systems."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".security")
        self.suspicious_patterns = {
            'injection_attempts': [
                r'.*;.*DROP.*TABLE.*;.*',
                r'.*<script.*>.*</script>.*',
                r'.*\{\{.*\}\}.*',  # Template injection
                r'.*\$\{.*\}.*',    # Variable substitution
            ],
            'buffer_overflow': [
                r'.*A{100,}.*',  # Excessive repeated characters
                r'.*\x00.*',     # Null byte injection
            ],
            'path_traversal': [
                r'.*\.\./.*',
                r'.*%2e%2e%2f.*',
                r'.*\.\.\\.*',
            ]
        }
        self.threat_scores = {}
        self.blocked_patterns = []
        
    def validate_input_security(self, 
                              data: Any, 
                              input_type: str = "general",
                              max_size: int = 10000,
                              allow_binary: bool = False) -> Dict[str, Any]:
        """
        Comprehensive security validation of input data.
        
        Args:
            data: Input data to validate
            input_type: Type of input for context-specific validation
            max_size: Maximum allowed size for input
            allow_binary: Whether to allow binary data
            
        Returns:
            Dict with validation results and security assessment
        """
        validation_result = {
            'is_safe': True,
            'threats_detected': [],
            'sanitized_data': data,
            'risk_score': 0.0,
            'validation_time': time.time()
        }
        
        try:
            # Size validation
            if hasattr(data, '__len__') and len(data) > max_size:
                validation_result['threats_detected'].append({
                    'type': 'size_limit_exceeded',
                    'severity': 'high',
                    'details': f'Input size {len(data)} exceeds limit {max_size}'
                })
                validation_result['risk_score'] += 0.4
                
            # Type-specific validation
            if isinstance(data, str):
                validation_result.update(self._validate_string_security(data))
            elif isinstance(data, (list, np.ndarray)):
                validation_result.update(self._validate_array_security(data, allow_binary))
            elif isinstance(data, dict):
                validation_result.update(self._validate_dict_security(data))
                
            # Update overall safety assessment
            validation_result['is_safe'] = (
                validation_result['risk_score'] < 0.5 and
                len(validation_result['threats_detected']) == 0
            )
            
            # Log security events
            if not validation_result['is_safe']:
                self.logger.warning(f"Security threat detected in {input_type}: "
                                  f"Risk score {validation_result['risk_score']:.3f}")
                
        except Exception as e:
            self.logger.error(f"Security validation error: {e}")
            validation_result['is_safe'] = False
            validation_result['threats_detected'].append({
                'type': 'validation_error',
                'severity': 'critical',
                'details': str(e)
            })
            
        return validation_result
        
    def _validate_string_security(self, text: str) -> Dict[str, Any]:
        """Validate string input for security threats."""
        threats = []
        risk_score = 0.0
        sanitized = text
        
        # Check for suspicious patterns
        for category, patterns in self.suspicious_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    threats.append({
                        'type': f'pattern_match_{category}',
                        'severity': 'high',
                        'pattern': pattern,
                        'details': f'Suspicious pattern detected: {category}'
                    })
                    risk_score += 0.3
                    
        # Check encoding issues
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            threats.append({
                'type': 'encoding_issue',
                'severity': 'medium',
                'details': 'Text contains invalid Unicode characters'
            })
            risk_score += 0.2
            
        # Sanitize common threats
        import html
        import re
        
        # HTML escape
        sanitized = html.escape(sanitized)
        
        # Remove control characters except whitespace
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', sanitized)
        
        return {
            'threats_detected': threats,
            'risk_score': risk_score,
            'sanitized_data': sanitized
        }
        
    def _validate_array_security(self, array: Union[list, np.ndarray], allow_binary: bool) -> Dict[str, Any]:
        """Validate array input for security threats."""
        threats = []
        risk_score = 0.0
        sanitized = array
        
        try:
            # Convert to numpy array for validation
            if isinstance(array, list):
                np_array = np.array(array)
            else:
                np_array = array
                
            # Check for suspicious data patterns
            if np_array.dtype.kind in ['U', 'S']:  # String arrays
                # Check each string element
                for i, item in enumerate(np_array.flat):
                    if isinstance(item, str):
                        string_validation = self._validate_string_security(item)
                        if string_validation['threats_detected']:
                            threats.extend(string_validation['threats_detected'])
                            risk_score += 0.1
                            
            elif np_array.dtype.kind in ['i', 'u', 'f']:  # Numeric arrays
                # Check for suspicious numeric patterns
                if np.any(np.isnan(np_array)) or np.any(np.isinf(np_array)):
                    threats.append({
                        'type': 'invalid_numeric_values',
                        'severity': 'medium',
                        'details': 'Array contains NaN or Inf values'
                    })
                    risk_score += 0.2
                    
                # Check for extreme values that might cause overflow
                max_val = np.max(np.abs(np_array))
                if max_val > 1e10:
                    threats.append({
                        'type': 'extreme_values',
                        'severity': 'medium',
                        'details': f'Array contains extreme values (max: {max_val})'
                    })
                    risk_score += 0.2
                    
            # Sanitize array
            if isinstance(array, list):
                sanitized = [self._sanitize_array_element(item) for item in array]
            else:
                sanitized = self._sanitize_numpy_array(np_array)
                
        except Exception as e:
            threats.append({
                'type': 'array_processing_error',
                'severity': 'high',
                'details': f'Error processing array: {e}'
            })
            risk_score += 0.4
            
        return {
            'threats_detected': threats,
            'risk_score': risk_score,
            'sanitized_data': sanitized
        }
        
    def _validate_dict_security(self, data_dict: Dict) -> Dict[str, Any]:
        """Validate dictionary input for security threats."""
        threats = []
        risk_score = 0.0
        sanitized = {}
        
        try:
            for key, value in data_dict.items():
                # Validate key
                if isinstance(key, str):
                    key_validation = self._validate_string_security(key)
                    if key_validation['threats_detected']:
                        threats.extend(key_validation['threats_detected'])
                        risk_score += 0.1
                    sanitized_key = key_validation['sanitized_data']
                else:
                    sanitized_key = str(key)
                    
                # Validate value recursively
                if isinstance(value, str):
                    value_validation = self._validate_string_security(value)
                    sanitized[sanitized_key] = value_validation['sanitized_data']
                    threats.extend(value_validation['threats_detected'])
                    risk_score += value_validation['risk_score'] * 0.5
                elif isinstance(value, (list, np.ndarray)):
                    value_validation = self._validate_array_security(value, allow_binary=False)
                    sanitized[sanitized_key] = value_validation['sanitized_data']
                    threats.extend(value_validation['threats_detected'])
                    risk_score += value_validation['risk_score'] * 0.5
                elif isinstance(value, dict):
                    value_validation = self._validate_dict_security(value)
                    sanitized[sanitized_key] = value_validation['sanitized_data']
                    threats.extend(value_validation['threats_detected'])
                    risk_score += value_validation['risk_score'] * 0.5
                else:
                    sanitized[sanitized_key] = value
                    
        except Exception as e:
            threats.append({
                'type': 'dict_processing_error',
                'severity': 'high',
                'details': f'Error processing dictionary: {e}'
            })
            risk_score += 0.4
            
        return {
            'threats_detected': threats,
            'risk_score': risk_score,
            'sanitized_data': sanitized
        }
        
    def _sanitize_array_element(self, element):
        """Sanitize individual array element."""
        if isinstance(element, str):
            return sanitize_string_input(element, max_length=1000)
        elif isinstance(element, (int, float)):
            return sanitize_numeric_input(element)
        else:
            return element
            
    def _sanitize_numpy_array(self, array: np.ndarray) -> np.ndarray:
        """Sanitize numpy array."""
        if array.dtype.kind in ['i', 'u', 'f']:  # Numeric
            # Replace invalid values
            sanitized = np.nan_to_num(array, nan=0.0, posinf=1e6, neginf=-1e6)
            # Clip extreme values
            sanitized = np.clip(sanitized, -1e10, 1e10)
            return sanitized
        else:
            return array
            
    def get_security_summary(self) -> Dict[str, Any]:
        """Get summary of security validation results."""
        return {
            'total_validations': len(self.threat_scores),
            'average_risk_score': np.mean(list(self.threat_scores.values())) if self.threat_scores else 0.0,
            'high_risk_inputs': sum(1 for score in self.threat_scores.values() if score > 0.7),
            'blocked_patterns': len(self.blocked_patterns),
            'last_validation': max(self.threat_scores.keys()) if self.threat_scores else None
        }


class CircuitBreakerPattern:
    """Circuit breaker pattern for system protection."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 30.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = logging.getLogger(__name__ + ".circuit_breaker")
        
    def call(self, func, *args, **kwargs):
        """Call function through circuit breaker."""
        if self.state == 'OPEN':
            if self._should_attempt_reset():
                self.state = 'HALF_OPEN'
                self.logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise ComputationError("Circuit breaker is OPEN - operation blocked")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            raise e
            
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
        
    def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.logger.info("Circuit breaker reset to CLOSED state")
            
    def _on_failure(self):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'
            self.logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
            
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'time_until_retry': max(0, self.recovery_timeout - (time.time() - (self.last_failure_time or 0)))
        }


class EnhancedHealthMonitor(HealthMonitor):
    """Enhanced health monitoring with predictive capabilities."""
    
    def __init__(self, check_interval: float = 1.0):
        super().__init__(check_interval)
        self.metric_history = {}
        self.trend_analyzer = {}
        self.predictive_alerts = []
        
    def update_metric_with_history(self, name: str, value: float, 
                                 threshold_low: float = 0.0, 
                                 threshold_high: float = 1.0,
                                 history_size: int = 100):
        """Update metric with historical tracking."""
        # Update current metric
        self.update_metric(name, value, threshold_low, threshold_high)
        
        # Maintain history
        if name not in self.metric_history:
            self.metric_history[name] = deque(maxlen=history_size)
        self.metric_history[name].append({
            'value': value,
            'timestamp': time.time()
        })
        
        # Analyze trends
        self._analyze_trends(name)
        
    def _analyze_trends(self, metric_name: str):
        """Analyze metric trends for predictive alerts."""
        if metric_name not in self.metric_history:
            return
            
        history = list(self.metric_history[metric_name])
        if len(history) < 10:  # Need minimum history
            return
            
        values = [h['value'] for h in history]
        timestamps = [h['timestamp'] for h in history]
        
        # Simple linear trend analysis
        if len(values) > 1:
            time_diffs = np.diff(timestamps)
            value_diffs = np.diff(values)
            
            if len(time_diffs) > 0:
                trend_slope = np.mean(value_diffs / time_diffs)
                
                # Predict future value
                current_time = timestamps[-1]
                predicted_value = values[-1] + trend_slope * 10.0  # 10 seconds ahead
                
                # Check if predicted value will exceed thresholds
                metric_config = self.health_metrics.get(metric_name)
                if metric_config:
                    threshold_low = metric_config['threshold_low']
                    threshold_high = metric_config['threshold_high']
                    
                    if predicted_value < threshold_low or predicted_value > threshold_high:
                        alert = {
                            'metric': metric_name,
                            'current_value': values[-1],
                            'predicted_value': predicted_value,
                            'trend_slope': trend_slope,
                            'time_to_threshold': 10.0,
                            'timestamp': current_time
                        }
                        self.predictive_alerts.append(alert)
                        
    def get_predictive_health_analysis(self) -> Dict[str, Any]:
        """Get predictive health analysis."""
        current_health = self.get_system_health()
        
        # Analyze trends
        trend_analysis = {}
        for metric_name, history in self.metric_history.items():
            if len(history) >= 5:
                values = [h['value'] for h in history]
                trend_analysis[metric_name] = {
                    'current_value': values[-1],
                    'trend': 'increasing' if values[-1] > values[-5] else 'decreasing',
                    'volatility': np.std(values[-10:]) if len(values) >= 10 else 0.0,
                    'stability_score': 1.0 / (1.0 + np.std(values[-10:])) if len(values) >= 10 else 1.0
                }
                
        return {
            'current_health': current_health,
            'trend_analysis': trend_analysis,
            'predictive_alerts': len(self.predictive_alerts),
            'overall_stability': (np.mean([ta['stability_score'] for ta in trend_analysis.values()]) 
                                 if trend_analysis else 1.0)
        }


# Global instances
_global_security_validator = AdvancedSecurityValidator()
_global_health_monitor = EnhancedHealthMonitor()

def get_security_validator() -> AdvancedSecurityValidator:
    """Get global security validator instance."""
    return _global_security_validator
    
def get_health_monitor() -> EnhancedHealthMonitor:
    """Get global health monitor instance."""
    return _global_health_monitor

import re