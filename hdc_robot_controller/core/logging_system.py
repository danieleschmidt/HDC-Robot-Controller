"""
Advanced Logging and Monitoring System for HDC Robot Controller.

Provides structured logging, performance monitoring, health checks,
and observability for production robotics applications.
"""

import logging
import json
import time
import threading
import psutil
import os
import sys
from typing import Dict, List, Optional, Any, Callable
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
import traceback


class HDCLogger:
    """Advanced structured logger for HDC operations."""
    
    def __init__(self, name: str = "HDCRobotController", 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 structured_format: bool = True):
        """Initialize HDC logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            structured_format: Whether to use structured JSON logging
        """
        self.name = name
        self.structured_format = structured_format
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create formatters
        if structured_format:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Performance metrics
        self.metrics = defaultdict(list)
        self.counters = defaultdict(int)
        self.timers = {}
        
        # Thread safety
        self._lock = threading.RLock()
    
    def debug(self, message: str, **kwargs):
        """Log debug message with structured data."""
        self._log(logging.DEBUG, message, kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with structured data."""
        self._log(logging.INFO, message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with structured data."""
        self._log(logging.WARNING, message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with structured data."""
        self._log(logging.ERROR, message, kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with structured data."""
        self._log(logging.CRITICAL, message, kwargs)
    
    def _log(self, level: int, message: str, extra_data: Dict[str, Any]):
        """Internal logging method with structured data."""
        with self._lock:
            if self.structured_format:
                log_data = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'logger': self.name,
                    'level': logging.getLevelName(level),
                    'message': message,
                    'thread_id': threading.get_ident(),
                    'process_id': os.getpid(),
                    **extra_data
                }
                self.logger.log(level, json.dumps(log_data))
            else:
                extra_str = f" | {json.dumps(extra_data)}" if extra_data else ""
                self.logger.log(level, f"{message}{extra_str}")
    
    def log_performance(self, operation: str, duration: float, **metadata):
        """Log performance metrics."""
        with self._lock:
            self.metrics[operation].append(duration)
            self.info(f"Performance: {operation}", 
                     duration=duration, 
                     operation=operation,
                     **metadata)
    
    def increment_counter(self, counter_name: str, value: int = 1):
        """Increment a named counter."""
        with self._lock:
            self.counters[counter_name] += value
    
    def start_timer(self, timer_name: str):
        """Start a named timer."""
        with self._lock:
            self.timers[timer_name] = time.time()
    
    def end_timer(self, timer_name: str, **metadata) -> Optional[float]:
        """End a named timer and log performance."""
        with self._lock:
            if timer_name not in self.timers:
                self.warning(f"Timer '{timer_name}' not found")
                return None
            
            duration = time.time() - self.timers[timer_name]
            del self.timers[timer_name]
            
            self.log_performance(timer_name, duration, **metadata)
            return duration
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        with self._lock:
            summary = {}
            
            # Performance metrics
            for operation, durations in self.metrics.items():
                if durations:
                    summary[f"{operation}_avg_duration"] = sum(durations) / len(durations)
                    summary[f"{operation}_max_duration"] = max(durations)
                    summary[f"{operation}_min_duration"] = min(durations)
                    summary[f"{operation}_count"] = len(durations)
            
            # Counters
            summary.update(self.counters)
            
            return summary
    
    def log_exception(self, exception: Exception, context: str = "", **kwargs):
        """Log exception with full traceback and context."""
        self.error(
            f"Exception in {context}: {str(exception)}",
            exception_type=type(exception).__name__,
            traceback=traceback.format_exc(),
            context=context,
            **kwargs
        )


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        # If record.msg is already a JSON string, return it
        if isinstance(record.msg, str) and record.msg.strip().startswith('{'):
            return record.msg
        
        # Otherwise, create structured log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread_id': record.thread,
            'process_id': record.process
        }
        
        return json.dumps(log_entry)


class SystemMonitor:
    """System health and performance monitoring."""
    
    def __init__(self, logger: HDCLogger, monitoring_interval: float = 10.0):
        """Initialize system monitor.
        
        Args:
            logger: HDC logger instance
            monitoring_interval: How often to collect metrics (seconds)
        """
        self.logger = logger
        self.monitoring_interval = monitoring_interval
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Metrics history
        self.metrics_history = deque(maxlen=1000)  # Keep last 1000 measurements
        
        # Health status
        self.health_status = {
            'status': 'unknown',
            'last_check': None,
            'issues': []
        }
        
        # Thresholds
        self.thresholds = {
            'cpu_warning': 80.0,      # CPU usage %
            'cpu_critical': 95.0,
            'memory_warning': 80.0,   # Memory usage %
            'memory_critical': 95.0,
            'disk_warning': 85.0,     # Disk usage %
            'disk_critical': 95.0,
            'temperature_warning': 70.0,  # CPU temperature °C
            'temperature_critical': 85.0
        }
    
    def start_monitoring(self):
        """Start background system monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("System monitoring started", 
                        interval=self.monitoring_interval)
    
    def stop_monitoring(self):
        """Stop background system monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # Check health
                self._check_system_health(metrics)
                
                # Log metrics periodically
                if len(self.metrics_history) % 6 == 0:  # Every 6th measurement (1 minute if 10s interval)
                    self.logger.info("System metrics", **metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.log_exception(e, "system monitoring")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics."""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_bytes_sent': psutil.net_io_counters().bytes_sent,
            'network_bytes_recv': psutil.net_io_counters().bytes_recv,
            'process_count': len(psutil.pids()),
            'boot_time': psutil.boot_time()
        }
        
        # CPU temperature (if available)
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get CPU temperature
                for name, entries in temps.items():
                    if 'cpu' in name.lower() or 'coretemp' in name.lower():
                        if entries:
                            metrics['cpu_temperature'] = entries[0].current
                        break
        except:
            pass  # Temperature not available
        
        # Process-specific metrics
        try:
            current_process = psutil.Process()
            metrics['process_memory_percent'] = current_process.memory_percent()
            metrics['process_cpu_percent'] = current_process.cpu_percent()
            metrics['process_num_threads'] = current_process.num_threads()
            metrics['process_num_fds'] = current_process.num_fds() if hasattr(current_process, 'num_fds') else 0
        except:
            pass
        
        return metrics
    
    def _check_system_health(self, metrics: Dict[str, Any]):
        """Check system health against thresholds."""
        issues = []
        
        # CPU check
        cpu_percent = metrics.get('cpu_percent', 0)
        if cpu_percent >= self.thresholds['cpu_critical']:
            issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
        elif cpu_percent >= self.thresholds['cpu_warning']:
            issues.append(f"CPU usage high: {cpu_percent:.1f}%")
        
        # Memory check
        memory_percent = metrics.get('memory_percent', 0)
        if memory_percent >= self.thresholds['memory_critical']:
            issues.append(f"Memory usage critical: {memory_percent:.1f}%")
        elif memory_percent >= self.thresholds['memory_warning']:
            issues.append(f"Memory usage high: {memory_percent:.1f}%")
        
        # Disk check
        disk_percent = metrics.get('disk_percent', 0)
        if disk_percent >= self.thresholds['disk_critical']:
            issues.append(f"Disk usage critical: {disk_percent:.1f}%")
        elif disk_percent >= self.thresholds['disk_warning']:
            issues.append(f"Disk usage high: {disk_percent:.1f}%")
        
        # Temperature check
        temp = metrics.get('cpu_temperature')
        if temp:
            if temp >= self.thresholds['temperature_critical']:
                issues.append(f"CPU temperature critical: {temp:.1f}°C")
            elif temp >= self.thresholds['temperature_warning']:
                issues.append(f"CPU temperature high: {temp:.1f}°C")
        
        # Update health status
        if issues:
            status = 'critical' if any('critical' in issue for issue in issues) else 'warning'
            self.health_status = {
                'status': status,
                'last_check': time.time(),
                'issues': issues
            }
            
            # Log health issues
            if status == 'critical':
                self.logger.critical("System health critical", issues=issues)
            else:
                self.logger.warning("System health warning", issues=issues)
        else:
            self.health_status = {
                'status': 'healthy',
                'last_check': time.time(),
                'issues': []
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
        return self.health_status.copy()
    
    def get_metrics_summary(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """Get summary of recent metrics.
        
        Args:
            duration_minutes: How many minutes of history to summarize
            
        Returns:
            Dictionary with metric summaries
        """
        if not self.metrics_history:
            return {}
        
        # Filter recent metrics
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [m for m in self.metrics_history if m['timestamp'] >= cutoff_time]
        
        if not recent_metrics:
            return {}
        
        # Calculate summaries
        summary = {}
        metric_keys = ['cpu_percent', 'memory_percent', 'disk_percent', 
                      'cpu_temperature', 'process_memory_percent', 'process_cpu_percent']
        
        for key in metric_keys:
            values = [m.get(key) for m in recent_metrics if key in m and m[key] is not None]
            if values:
                summary[f"{key}_avg"] = sum(values) / len(values)
                summary[f"{key}_max"] = max(values)
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_current"] = recent_metrics[-1].get(key)
        
        summary['metrics_count'] = len(recent_metrics)
        summary['time_span_minutes'] = duration_minutes
        
        return summary


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self, logger: HDCLogger):
        """Initialize health checker."""
        self.logger = logger
        self.health_checks = {}
        self.check_results = {}
        self._lock = threading.RLock()
    
    def register_check(self, name: str, check_function: Callable[[], Dict[str, Any]],
                      interval_seconds: float = 60.0, timeout_seconds: float = 10.0):
        """Register a health check function.
        
        Args:
            name: Check name
            check_function: Function that returns health status dict
            interval_seconds: How often to run the check
            timeout_seconds: Max time to wait for check completion
        """
        with self._lock:
            self.health_checks[name] = {
                'function': check_function,
                'interval': interval_seconds,
                'timeout': timeout_seconds,
                'last_run': 0,
                'enabled': True
            }
            
            self.logger.info(f"Registered health check: {name}",
                           interval=interval_seconds,
                           timeout=timeout_seconds)
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        current_time = time.time()
        
        with self._lock:
            for name, check_config in self.health_checks.items():
                if not check_config['enabled']:
                    continue
                
                # Check if it's time to run this check
                if (current_time - check_config['last_run']) < check_config['interval']:
                    # Use cached result
                    if name in self.check_results:
                        results[name] = self.check_results[name]
                    continue
                
                # Run the check
                try:
                    self.logger.debug(f"Running health check: {name}")
                    
                    # Run with timeout
                    result = self._run_check_with_timeout(
                        check_config['function'],
                        check_config['timeout']
                    )
                    
                    result['timestamp'] = current_time
                    result['check_name'] = name
                    
                    self.check_results[name] = result
                    results[name] = result
                    check_config['last_run'] = current_time
                    
                    # Log result
                    if result.get('status') == 'healthy':
                        self.logger.debug(f"Health check '{name}' passed", 
                                        check_name=name, **{k: v for k, v in result.items() if k != 'message'})
                    else:
                        self.logger.warning(f"Health check '{name}' failed", 
                                          check_name=name, **{k: v for k, v in result.items() if k != 'message'})
                
                except Exception as e:
                    error_result = {
                        'status': 'error',
                        'message': f"Health check failed: {str(e)}",
                        'error': str(e),
                        'timestamp': current_time,
                        'check_name': name
                    }
                    
                    self.check_results[name] = error_result
                    results[name] = error_result
                    
                    self.logger.log_exception(e, f"health check '{name}'")
        
        return results
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        check_results = self.run_all_checks()
        
        if not check_results:
            return {
                'status': 'unknown',
                'message': 'No health checks registered',
                'timestamp': time.time()
            }
        
        # Analyze results
        healthy_checks = 0
        warning_checks = 0
        critical_checks = 0
        error_checks = 0
        
        issues = []
        
        for name, result in check_results.items():
            status = result.get('status', 'unknown')
            
            if status == 'healthy':
                healthy_checks += 1
            elif status == 'warning':
                warning_checks += 1
                issues.append(f"{name}: {result.get('message', 'Warning')}")
            elif status == 'critical':
                critical_checks += 1
                issues.append(f"{name}: {result.get('message', 'Critical issue')}")
            else:  # error or unknown
                error_checks += 1
                issues.append(f"{name}: {result.get('message', 'Check error')}")
        
        # Determine overall status
        if critical_checks > 0:
            overall_status = 'critical'
        elif error_checks > 0:
            overall_status = 'error'
        elif warning_checks > 0:
            overall_status = 'warning'
        else:
            overall_status = 'healthy'
        
        return {
            'status': overall_status,
            'message': f"{healthy_checks} healthy, {warning_checks} warnings, {critical_checks} critical, {error_checks} errors",
            'timestamp': time.time(),
            'details': {
                'total_checks': len(check_results),
                'healthy': healthy_checks,
                'warnings': warning_checks,
                'critical': critical_checks,
                'errors': error_checks
            },
            'issues': issues,
            'check_results': check_results
        }
    
    def _run_check_with_timeout(self, check_function: Callable, timeout: float) -> Dict[str, Any]:
        """Run health check with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Health check timed out")
        
        # Set timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = check_function()
            signal.alarm(0)  # Cancel timeout
            return result
        except TimeoutError:
            return {
                'status': 'error',
                'message': f'Health check timed out after {timeout} seconds'
            }
        finally:
            signal.signal(signal.SIGALRM, old_handler)


# Global logger instance
_global_logger = None


def get_logger(name: str = "HDCRobotController") -> HDCLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = HDCLogger(name)
    return _global_logger


def setup_production_logging(log_file: str = "/tmp/hdc_robot_controller.log",
                           log_level: str = "INFO",
                           enable_monitoring: bool = True) -> HDCLogger:
    """Setup production-ready logging configuration."""
    global _global_logger
    
    # Create logs directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    _global_logger = HDCLogger(
        name="HDCRobotController",
        log_level=log_level,
        log_file=log_file,
        structured_format=True
    )
    
    # Setup system monitoring
    if enable_monitoring:
        monitor = SystemMonitor(_global_logger)
        monitor.start_monitoring()
        
        # Register basic health checks
        health_checker = HealthChecker(_global_logger)
        
        def basic_health_check():
            return {'status': 'healthy', 'message': 'Basic health check passed'}
        
        health_checker.register_check("basic", basic_health_check, 60.0)
        
        _global_logger.info("Production logging setup complete",
                          log_file=log_file,
                          log_level=log_level,
                          monitoring_enabled=enable_monitoring)
    
    return _global_logger