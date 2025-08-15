#!/usr/bin/env python3
"""
Automated Repair Engine for Self-Healing Pipeline Guard
Intelligent repair and recovery mechanisms for CI/CD pipeline issues

Features:
- Multi-strategy automated repair system
- Intelligent repair selection based on failure patterns
- Safe rollback and recovery mechanisms  
- Learning-based repair optimization
- Integration with CI/CD systems (Jenkins, GitLab, GitHub Actions)
- Escalation and human intervention protocols
"""

import time
import json
import asyncio
import subprocess
import logging
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# HDC imports for pattern-based repair selection
from hdc_robot_controller.core.hypervector import HyperVector
from hdc_robot_controller.core.memory import HierarchicalMemory
from hdc_robot_controller.robustness.advanced_error_recovery import AdvancedErrorRecovery

# Import pipeline guard types
from .core import PipelineMetrics, GuardAlert, AlertSeverity, PipelinePhase

class RepairStrategy(Enum):
    RESTART_SERVICE = "restart_service"
    CLEAR_CACHE = "clear_cache"
    SCALE_RESOURCES = "scale_resources"
    ROLLBACK_DEPLOYMENT = "rollback_deployment"
    RETRY_FAILED_JOBS = "retry_failed_jobs"
    UPDATE_DEPENDENCIES = "update_dependencies"
    RESET_ENVIRONMENT = "reset_environment"
    OPTIMIZE_CONFIGURATION = "optimize_configuration"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    CUSTOM_SCRIPT = "custom_script"

class RepairResult(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ESCALATED = "escalated"

class RepairPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class RepairAction:
    """Individual repair action definition"""
    id: str
    strategy: RepairStrategy
    description: str
    command: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    priority: RepairPriority = RepairPriority.MEDIUM
    prerequisites: List[str] = field(default_factory=list)
    rollback_action: Optional['RepairAction'] = None
    estimated_duration: int = 60  # seconds
    success_criteria: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RepairAttempt:
    """Record of a repair attempt"""
    id: str
    timestamp: datetime
    alert_id: str
    pipeline_id: str
    strategy: RepairStrategy
    action: RepairAction
    result: RepairResult
    duration_seconds: float
    error_message: Optional[str] = None
    metrics_before: Optional[PipelineMetrics] = None
    metrics_after: Optional[PipelineMetrics] = None
    success_score: float = 0.0

class RepairEngine(ABC):
    """Abstract base class for repair engines"""
    
    @abstractmethod
    def can_handle(self, alert: GuardAlert) -> bool:
        """Check if this engine can handle the given alert"""
        pass
    
    @abstractmethod
    def generate_repair_actions(self, alert: GuardAlert) -> List[RepairAction]:
        """Generate appropriate repair actions for the alert"""
        pass
    
    @abstractmethod
    def execute_repair(self, action: RepairAction, alert: GuardAlert) -> RepairAttempt:
        """Execute a repair action"""
        pass

class GenericRepairEngine(RepairEngine):
    """Generic repair engine for common CI/CD issues"""
    
    def __init__(self):
        self.logger = logging.getLogger('generic_repair')
    
    def can_handle(self, alert: GuardAlert) -> bool:
        """Can handle most common pipeline issues"""
        return True
    
    def generate_repair_actions(self, alert: GuardAlert) -> List[RepairAction]:
        """Generate repair actions based on alert type and severity"""
        
        actions = []
        
        # Performance issues
        if "performance" in alert.title.lower() or "duration" in alert.message.lower():
            actions.extend([
                RepairAction(
                    id=str(uuid.uuid4()),
                    strategy=RepairStrategy.CLEAR_CACHE,
                    description="Clear build cache to resolve performance issues",
                    command="rm -rf .cache/* && rm -rf node_modules/.cache/*",
                    timeout_seconds=120,
                    priority=RepairPriority.MEDIUM
                ),
                RepairAction(
                    id=str(uuid.uuid4()),
                    strategy=RepairStrategy.SCALE_RESOURCES,
                    description="Scale up computational resources",
                    parameters={"cpu_scale": 1.5, "memory_scale": 1.3},
                    timeout_seconds=180,
                    priority=RepairPriority.HIGH
                )
            ])
        
        # Error-related issues
        if "error" in alert.title.lower() or alert.metrics.get('error_count', 0) > 0:
            actions.extend([
                RepairAction(
                    id=str(uuid.uuid4()),
                    strategy=RepairStrategy.RETRY_FAILED_JOBS,
                    description="Retry failed pipeline jobs",
                    timeout_seconds=600,
                    priority=RepairPriority.HIGH
                ),
                RepairAction(
                    id=str(uuid.uuid4()),
                    strategy=RepairStrategy.RESET_ENVIRONMENT,
                    description="Reset pipeline environment to clean state",
                    timeout_seconds=300,
                    priority=RepairPriority.MEDIUM
                )
            ])
        
        # Resource issues
        cpu_usage = alert.metrics.get('cpu_usage', 0)
        memory_usage = alert.metrics.get('memory_usage', 0)
        
        if cpu_usage > 0.9 or memory_usage > 0.9:
            actions.append(
                RepairAction(
                    id=str(uuid.uuid4()),
                    strategy=RepairStrategy.RESTART_SERVICE,
                    description="Restart services to free up resources",
                    timeout_seconds=180,
                    priority=RepairPriority.HIGH
                )
            )
        
        # Critical issues
        if alert.severity == AlertSeverity.CRITICAL:
            actions.append(
                RepairAction(
                    id=str(uuid.uuid4()),
                    strategy=RepairStrategy.ROLLBACK_DEPLOYMENT,
                    description="Rollback to last known good deployment",
                    timeout_seconds=900,
                    priority=RepairPriority.CRITICAL,
                    success_criteria={"health_score_threshold": 0.8}
                )
            )
        
        return actions
    
    def execute_repair(self, action: RepairAction, alert: GuardAlert) -> RepairAttempt:
        """Execute generic repair action"""
        
        attempt = RepairAttempt(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            alert_id=alert.id,
            pipeline_id=alert.pipeline_id,
            strategy=action.strategy,
            action=action,
            result=RepairResult.FAILED,
            duration_seconds=0.0
        )
        
        start_time = time.time()
        
        try:
            self.logger.info(f"Executing repair: {action.description}")
            
            # Simulate repair execution based on strategy
            if action.strategy == RepairStrategy.CLEAR_CACHE:
                success = self._clear_cache(action)
            elif action.strategy == RepairStrategy.SCALE_RESOURCES:
                success = self._scale_resources(action)
            elif action.strategy == RepairStrategy.RETRY_FAILED_JOBS:
                success = self._retry_failed_jobs(action, alert)
            elif action.strategy == RepairStrategy.RESET_ENVIRONMENT:
                success = self._reset_environment(action)
            elif action.strategy == RepairStrategy.RESTART_SERVICE:
                success = self._restart_service(action)
            elif action.strategy == RepairStrategy.ROLLBACK_DEPLOYMENT:
                success = self._rollback_deployment(action)
            else:
                self.logger.warning(f"Unknown repair strategy: {action.strategy}")
                success = False
            
            # Calculate duration
            duration = time.time() - start_time
            attempt.duration_seconds = duration
            
            # Determine result
            if success:
                attempt.result = RepairResult.SUCCESS
                attempt.success_score = 1.0
            else:
                attempt.result = RepairResult.FAILED
                attempt.success_score = 0.0
            
        except Exception as e:
            attempt.result = RepairResult.FAILED
            attempt.error_message = str(e)
            attempt.duration_seconds = time.time() - start_time
            self.logger.error(f"Repair execution failed: {e}")
        
        return attempt
    
    def _clear_cache(self, action: RepairAction) -> bool:
        """Simulate cache clearing"""
        time.sleep(2)  # Simulate operation
        return True
    
    def _scale_resources(self, action: RepairAction) -> bool:
        """Simulate resource scaling"""
        time.sleep(3)  # Simulate scaling operation
        return True
    
    def _retry_failed_jobs(self, action: RepairAction, alert: GuardAlert) -> bool:
        """Simulate job retry"""
        time.sleep(5)  # Simulate job restart
        # Success rate depends on the type of error
        import random
        return random.random() > 0.3  # 70% success rate
    
    def _reset_environment(self, action: RepairAction) -> bool:
        """Simulate environment reset"""
        time.sleep(4)  # Simulate environment reset
        return True
    
    def _restart_service(self, action: RepairAction) -> bool:
        """Simulate service restart"""
        time.sleep(3)  # Simulate restart
        return True
    
    def _rollback_deployment(self, action: RepairAction) -> bool:
        """Simulate deployment rollback"""
        time.sleep(6)  # Simulate rollback
        return True

class JenkinsRepairEngine(RepairEngine):
    """Specialized repair engine for Jenkins pipelines"""
    
    def __init__(self, jenkins_url: str, username: str, token: str):
        self.jenkins_url = jenkins_url
        self.username = username
        self.token = token
        self.logger = logging.getLogger('jenkins_repair')
    
    def can_handle(self, alert: GuardAlert) -> bool:
        """Check if alert is from Jenkins pipeline"""
        return "jenkins" in alert.pipeline_id.lower()
    
    def generate_repair_actions(self, alert: GuardAlert) -> List[RepairAction]:
        """Generate Jenkins-specific repair actions"""
        
        actions = []
        
        # Jenkins build retry
        actions.append(
            RepairAction(
                id=str(uuid.uuid4()),
                strategy=RepairStrategy.RETRY_FAILED_JOBS,
                description="Retry failed Jenkins build",
                command=f"curl -X POST {self.jenkins_url}/job/{alert.pipeline_id}/build --user {self.username}:{self.token}",
                timeout_seconds=60,
                priority=RepairPriority.HIGH
            )
        )
        
        # Jenkins workspace cleanup
        actions.append(
            RepairAction(
                id=str(uuid.uuid4()),
                strategy=RepairStrategy.CLEAR_CACHE,
                description="Clean Jenkins workspace",
                command=f"curl -X POST {self.jenkins_url}/job/{alert.pipeline_id}/doWipeOutWorkspace --user {self.username}:{self.token}",
                timeout_seconds=120,
                priority=RepairPriority.MEDIUM
            )
        )
        
        return actions
    
    def execute_repair(self, action: RepairAction, alert: GuardAlert) -> RepairAttempt:
        """Execute Jenkins-specific repair action"""
        
        attempt = RepairAttempt(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            alert_id=alert.id,
            pipeline_id=alert.pipeline_id,
            strategy=action.strategy,
            action=action,
            result=RepairResult.FAILED,
            duration_seconds=0.0
        )
        
        start_time = time.time()
        
        try:
            if action.command:
                # Execute Jenkins API call
                result = subprocess.run(
                    action.command,
                    shell=True,
                    timeout=action.timeout_seconds,
                    capture_output=True,
                    text=True
                )
                
                success = result.returncode == 0
                if not success:
                    attempt.error_message = result.stderr
            else:
                success = False
            
            attempt.duration_seconds = time.time() - start_time
            attempt.result = RepairResult.SUCCESS if success else RepairResult.FAILED
            attempt.success_score = 1.0 if success else 0.0
            
        except subprocess.TimeoutExpired:
            attempt.result = RepairResult.TIMEOUT
            attempt.duration_seconds = action.timeout_seconds
            attempt.error_message = "Command timeout"
        except Exception as e:
            attempt.result = RepairResult.FAILED
            attempt.error_message = str(e)
            attempt.duration_seconds = time.time() - start_time
        
        return attempt

class GitLabRepairEngine(RepairEngine):
    """Specialized repair engine for GitLab CI/CD"""
    
    def __init__(self, gitlab_url: str, token: str):
        self.gitlab_url = gitlab_url
        self.token = token
        self.logger = logging.getLogger('gitlab_repair')
    
    def can_handle(self, alert: GuardAlert) -> bool:
        """Check if alert is from GitLab pipeline"""
        return "gitlab" in alert.pipeline_id.lower() or "ci" in alert.pipeline_id.lower()
    
    def generate_repair_actions(self, alert: GuardAlert) -> List[RepairAction]:
        """Generate GitLab-specific repair actions"""
        
        actions = []
        
        # Retry pipeline
        actions.append(
            RepairAction(
                id=str(uuid.uuid4()),
                strategy=RepairStrategy.RETRY_FAILED_JOBS,
                description="Retry failed GitLab pipeline",
                parameters={"project_id": alert.pipeline_id, "pipeline_id": "latest"},
                timeout_seconds=300,
                priority=RepairPriority.HIGH
            )
        )
        
        # Clear runner cache
        actions.append(
            RepairAction(
                id=str(uuid.uuid4()),
                strategy=RepairStrategy.CLEAR_CACHE,
                description="Clear GitLab runner cache",
                timeout_seconds=180,
                priority=RepairPriority.MEDIUM
            )
        )
        
        return actions
    
    def execute_repair(self, action: RepairAction, alert: GuardAlert) -> RepairAttempt:
        """Execute GitLab-specific repair action"""
        
        # Implementation would integrate with GitLab API
        # For now, simulate the repair
        
        attempt = RepairAttempt(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            alert_id=alert.id,
            pipeline_id=alert.pipeline_id,
            strategy=action.strategy,
            action=action,
            result=RepairResult.SUCCESS,
            duration_seconds=2.0,
            success_score=0.8
        )
        
        time.sleep(2)  # Simulate API call
        
        return attempt

class AutoRepair:
    """
    Main automated repair orchestrator
    
    Coordinates multiple repair engines and implements intelligent repair selection,
    learning, and escalation protocols.
    """
    
    def __init__(self, hdc_dimension: int = 10000, max_concurrent_repairs: int = 3):
        self.hdc_dimension = hdc_dimension
        self.max_concurrent_repairs = max_concurrent_repairs
        
        # HDC components for pattern-based repair selection
        self.memory = HierarchicalMemory(hdc_dimension)
        self.error_recovery = AdvancedErrorRecovery(hdc_dimension)
        
        # Repair engines
        self.repair_engines: List[RepairEngine] = []
        self.register_default_engines()
        
        # Repair history and learning
        self.repair_attempts: List[RepairAttempt] = []
        self.repair_patterns: Dict[str, List[HyperVector]] = {}
        self.success_patterns: Dict[RepairStrategy, List[HyperVector]] = {}
        
        # Configuration
        self.config = {
            "max_repair_attempts": 3,
            "repair_cooldown_minutes": 10,
            "escalation_threshold": 0.3,  # Success rate threshold for escalation
            "parallel_repairs": False,
            "require_confirmation": False
        }
        
        # State management
        self.active_repairs: Dict[str, RepairAttempt] = {}
        self.repair_cooldowns: Dict[str, datetime] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_repairs)
        
        self.logger = logging.getLogger('auto_repair')
        self.logger.info("Auto repair system initialized")
    
    def register_default_engines(self):
        """Register default repair engines"""
        self.repair_engines.append(GenericRepairEngine())
        
        # Additional engines can be registered based on configuration
        # Example: self.repair_engines.append(JenkinsRepairEngine(...))
    
    def register_engine(self, engine: RepairEngine):
        """Register a custom repair engine"""
        self.repair_engines.append(engine)
        self.logger.info(f"Registered repair engine: {type(engine).__name__}")
    
    def attempt_repair(self, alert: GuardAlert) -> List[RepairAttempt]:
        """
        Attempt automated repair for the given alert
        Returns list of repair attempts made
        """
        
        pipeline_id = alert.pipeline_id
        
        # Check repair cooldown
        if self._is_in_cooldown(pipeline_id):
            self.logger.info(f"Pipeline {pipeline_id} in repair cooldown, skipping")
            return []
        
        # Check if we've exceeded max attempts for this type of issue
        recent_attempts = self._get_recent_attempts(pipeline_id, alert.severity)
        if len(recent_attempts) >= self.config["max_repair_attempts"]:
            self.logger.warning(f"Max repair attempts reached for {pipeline_id}, escalating")
            self._escalate_issue(alert, recent_attempts)
            return []
        
        # Select appropriate repair engine
        repair_engine = self._select_repair_engine(alert)
        if not repair_engine:
            self.logger.warning(f"No suitable repair engine found for alert {alert.id}")
            return []
        
        # Generate repair actions
        repair_actions = repair_engine.generate_repair_actions(alert)
        if not repair_actions:
            self.logger.warning(f"No repair actions generated for alert {alert.id}")
            return []
        
        # Select best repair actions using pattern matching
        selected_actions = self._select_optimal_actions(alert, repair_actions)
        
        # Execute repairs
        repair_attempts = []
        
        if self.config["parallel_repairs"] and len(selected_actions) > 1:
            # Execute repairs in parallel
            repair_attempts = self._execute_parallel_repairs(alert, selected_actions, repair_engine)
        else:
            # Execute repairs sequentially
            repair_attempts = self._execute_sequential_repairs(alert, selected_actions, repair_engine)
        
        # Learn from repair attempts
        self._learn_from_repairs(alert, repair_attempts)
        
        # Update cooldown
        if repair_attempts:
            self.repair_cooldowns[pipeline_id] = datetime.now()
        
        return repair_attempts
    
    def _select_repair_engine(self, alert: GuardAlert) -> Optional[RepairEngine]:
        """Select the most appropriate repair engine for the alert"""
        
        # Find engines that can handle this alert
        capable_engines = [engine for engine in self.repair_engines if engine.can_handle(alert)]
        
        if not capable_engines:
            return None
        
        # For now, use first capable engine
        # In advanced implementation, this could use pattern matching to select best engine
        return capable_engines[0]
    
    def _select_optimal_actions(self, alert: GuardAlert, 
                              available_actions: List[RepairAction]) -> List[RepairAction]:
        """Select optimal repair actions using HDC pattern matching and historical success"""
        
        if not available_actions:
            return []
        
        # Score actions based on historical success and pattern matching
        scored_actions = []
        
        for action in available_actions:
            score = self._calculate_action_score(alert, action)
            scored_actions.append((action, score))
        
        # Sort by score (highest first)
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        
        # Select top actions based on severity
        if alert.severity == AlertSeverity.CRITICAL:
            # Try multiple strategies for critical issues
            return [action for action, score in scored_actions[:3] if score > 0.3]
        else:
            # Single best action for non-critical issues
            return [scored_actions[0][0]] if scored_actions[0][1] > 0.5 else []
    
    def _calculate_action_score(self, alert: GuardAlert, action: RepairAction) -> float:
        """Calculate score for repair action based on historical success and patterns"""
        
        # Base score from priority
        priority_scores = {
            RepairPriority.LOW: 0.2,
            RepairPriority.MEDIUM: 0.4,
            RepairPriority.HIGH: 0.6,
            RepairPriority.CRITICAL: 0.8,
            RepairPriority.EMERGENCY: 1.0
        }
        base_score = priority_scores.get(action.priority, 0.5)
        
        # Historical success rate for this strategy
        strategy_attempts = [
            attempt for attempt in self.repair_attempts
            if attempt.strategy == action.strategy
        ]
        
        if strategy_attempts:
            success_rate = len([a for a in strategy_attempts if a.result == RepairResult.SUCCESS]) / len(strategy_attempts)
            historical_score = success_rate
        else:
            historical_score = 0.5  # Neutral score for untested strategies
        
        # Pattern matching score (if available)
        pattern_score = 0.5  # Default neutral score
        
        # Combine scores
        final_score = (base_score * 0.4 + historical_score * 0.4 + pattern_score * 0.2)
        
        return final_score
    
    def _execute_sequential_repairs(self, alert: GuardAlert, actions: List[RepairAction], 
                                  engine: RepairEngine) -> List[RepairAttempt]:
        """Execute repair actions sequentially"""
        
        attempts = []
        
        for action in actions:
            self.logger.info(f"Executing repair action: {action.description}")
            
            # Execute repair
            attempt = engine.execute_repair(action, alert)
            attempts.append(attempt)
            
            # Store attempt
            self.repair_attempts.append(attempt)
            
            # If successful, stop trying additional repairs
            if attempt.result == RepairResult.SUCCESS:
                self.logger.info(f"Repair successful: {action.description}")
                break
            else:
                self.logger.warning(f"Repair failed: {action.description} - {attempt.error_message}")
        
        return attempts
    
    def _execute_parallel_repairs(self, alert: GuardAlert, actions: List[RepairAction],
                                engine: RepairEngine) -> List[RepairAttempt]:
        """Execute repair actions in parallel"""
        
        attempts = []
        
        # Submit all repair actions to thread pool
        future_to_action = {
            self.executor.submit(engine.execute_repair, action, alert): action
            for action in actions
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_action):
            action = future_to_action[future]
            try:
                attempt = future.result()
                attempts.append(attempt)
                self.repair_attempts.append(attempt)
                
                if attempt.result == RepairResult.SUCCESS:
                    self.logger.info(f"Parallel repair successful: {action.description}")
                else:
                    self.logger.warning(f"Parallel repair failed: {action.description}")
                    
            except Exception as e:
                self.logger.error(f"Parallel repair exception for {action.description}: {e}")
        
        return attempts
    
    def _learn_from_repairs(self, alert: GuardAlert, attempts: List[RepairAttempt]):
        """Learn from repair attempts to improve future selection"""
        
        for attempt in attempts:
            # Store successful patterns
            if attempt.result == RepairResult.SUCCESS and attempt.metrics_before:
                if attempt.strategy not in self.success_patterns:
                    self.success_patterns[attempt.strategy] = []
                
                # Create pattern from alert characteristics
                pattern = self._create_repair_pattern(alert, attempt)
                if pattern:
                    self.success_patterns[attempt.strategy].append(pattern)
                    
                    # Limit pattern storage
                    if len(self.success_patterns[attempt.strategy]) > 20:
                        self.success_patterns[attempt.strategy] = self.success_patterns[attempt.strategy][-20:]
    
    def _create_repair_pattern(self, alert: GuardAlert, attempt: RepairAttempt) -> Optional[HyperVector]:
        """Create HDC pattern from repair context"""
        
        # This would encode the alert characteristics and repair context into an HDC pattern
        # For demonstration, return a random pattern
        # In real implementation, this would encode:
        # - Alert type and severity
        # - Pipeline metrics
        # - Time of day, day of week
        # - Recent changes or deployments
        
        return HyperVector.random(self.hdc_dimension)
    
    def _is_in_cooldown(self, pipeline_id: str) -> bool:
        """Check if pipeline is in repair cooldown"""
        
        if pipeline_id not in self.repair_cooldowns:
            return False
        
        cooldown_end = self.repair_cooldowns[pipeline_id] + timedelta(
            minutes=self.config["repair_cooldown_minutes"]
        )
        
        return datetime.now() < cooldown_end
    
    def _get_recent_attempts(self, pipeline_id: str, severity: AlertSeverity) -> List[RepairAttempt]:
        """Get recent repair attempts for pipeline"""
        
        cutoff_time = datetime.now() - timedelta(hours=1)  # Last hour
        
        return [
            attempt for attempt in self.repair_attempts
            if (attempt.pipeline_id == pipeline_id and 
                attempt.timestamp > cutoff_time)
        ]
    
    def _escalate_issue(self, alert: GuardAlert, failed_attempts: List[RepairAttempt]):
        """Escalate issue when automated repair fails"""
        
        self.logger.critical(f"Escalating issue for pipeline {alert.pipeline_id}")
        
        # Create escalation record
        escalation_data = {
            "alert_id": alert.id,
            "pipeline_id": alert.pipeline_id,
            "escalation_time": datetime.now(),
            "failed_attempts": len(failed_attempts),
            "alert_severity": alert.severity.value,
            "repair_attempts": [
                {
                    "strategy": attempt.strategy.value,
                    "result": attempt.result.value,
                    "error": attempt.error_message
                }
                for attempt in failed_attempts
            ]
        }
        
        # In real implementation, this would:
        # - Send notifications to on-call engineers
        # - Create tickets in incident management system
        # - Trigger emergency procedures if needed
        
        self.logger.info(f"Escalation data: {json.dumps(escalation_data, indent=2, default=str)}")
    
    def get_repair_statistics(self) -> Dict[str, Any]:
        """Get comprehensive repair statistics"""
        
        if not self.repair_attempts:
            return {"total_attempts": 0}
        
        # Overall statistics
        total_attempts = len(self.repair_attempts)
        successful_attempts = len([a for a in self.repair_attempts if a.result == RepairResult.SUCCESS])
        success_rate = successful_attempts / total_attempts
        
        # Strategy effectiveness
        strategy_stats = {}
        for strategy in RepairStrategy:
            strategy_attempts = [a for a in self.repair_attempts if a.strategy == strategy]
            if strategy_attempts:
                strategy_success = len([a for a in strategy_attempts if a.result == RepairResult.SUCCESS])
                strategy_stats[strategy.value] = {
                    "attempts": len(strategy_attempts),
                    "successes": strategy_success,
                    "success_rate": strategy_success / len(strategy_attempts),
                    "avg_duration": sum(a.duration_seconds for a in strategy_attempts) / len(strategy_attempts)
                }
        
        # Recent performance
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_attempts = [a for a in self.repair_attempts if a.timestamp > recent_cutoff]
        recent_success_rate = 0.0
        if recent_attempts:
            recent_successes = len([a for a in recent_attempts if a.result == RepairResult.SUCCESS])
            recent_success_rate = recent_successes / len(recent_attempts)
        
        # Pipeline-specific stats
        pipeline_stats = {}
        pipelines = set(a.pipeline_id for a in self.repair_attempts)
        for pipeline_id in pipelines:
            pipeline_attempts = [a for a in self.repair_attempts if a.pipeline_id == pipeline_id]
            pipeline_successes = len([a for a in pipeline_attempts if a.result == RepairResult.SUCCESS])
            pipeline_stats[pipeline_id] = {
                "attempts": len(pipeline_attempts),
                "successes": pipeline_successes,
                "success_rate": pipeline_successes / len(pipeline_attempts),
                "last_attempt": max(a.timestamp for a in pipeline_attempts)
            }
        
        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "overall_success_rate": success_rate,
            "recent_success_rate_7d": recent_success_rate,
            "strategy_effectiveness": strategy_stats,
            "pipeline_statistics": pipeline_stats,
            "active_cooldowns": len(self.repair_cooldowns),
            "patterns_learned": sum(len(patterns) for patterns in self.success_patterns.values())
        }
    
    def export_repair_data(self, filepath: str):
        """Export repair data for analysis"""
        
        export_data = {
            "export_time": datetime.now(),
            "repair_attempts": [
                {
                    "id": attempt.id,
                    "timestamp": attempt.timestamp,
                    "pipeline_id": attempt.pipeline_id,
                    "strategy": attempt.strategy.value,
                    "result": attempt.result.value,
                    "duration_seconds": attempt.duration_seconds,
                    "success_score": attempt.success_score,
                    "error_message": attempt.error_message
                }
                for attempt in self.repair_attempts
            ],
            "statistics": self.get_repair_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        self.logger.info(f"Repair data exported to {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize auto repair system
    auto_repair = AutoRepair(hdc_dimension=10000, max_concurrent_repairs=2)
    
    # Create example alert
    from .core import GuardAlert, PipelinePhase
    
    test_alert = GuardAlert(
        id="test-alert-1",
        timestamp=datetime.now(),
        severity=AlertSeverity.WARNING,
        pipeline_id="web-app-build",
        phase=PipelinePhase.BUILD,
        title="Performance Degradation",
        message="Build duration increased by 50%",
        metrics={
            "duration": 900,
            "error_count": 2,
            "cpu_usage": 0.85,
            "memory_usage": 0.75
        }
    )
    
    # Attempt repair
    print("🔧 Testing automated repair system...")
    repair_attempts = auto_repair.attempt_repair(test_alert)
    
    print(f"Repair attempts made: {len(repair_attempts)}")
    for attempt in repair_attempts:
        print(f"  - {attempt.strategy.value}: {attempt.result.value}")
        if attempt.error_message:
            print(f"    Error: {attempt.error_message}")
    
    # Show statistics
    stats = auto_repair.get_repair_statistics()
    print(f"\nRepair Statistics:")
    print(f"  Total attempts: {stats['total_attempts']}")
    print(f"  Success rate: {stats['overall_success_rate']:.2%}")
    
    print("Automated repair system test completed.")