#!/usr/bin/env python3
"""
Self-Improving Adaptive Optimizer for HDC Robot Controller
Autonomous system optimization using machine learning and HDC principles

Features:
- Real-time performance monitoring and optimization
- Adaptive algorithm selection (CPU/GPU/JIT)
- Self-healing and auto-recovery mechanisms
- Predictive performance tuning
- Learning from operational patterns
- Resource allocation optimization

Author: Terry - Terragon Labs Self-Improving Systems
"""

import time
import threading
import logging
import json
import numpy as np
import statistics
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import pickle

# Optimizer logging setup
logging.basicConfig(level=logging.INFO)
optimizer_logger = logging.getLogger('hdc_optimizer')

class OptimizationStrategy(Enum):
    CPU_INTENSIVE = "cpu_intensive"
    GPU_ACCELERATED = "gpu_accelerated"  
    JIT_COMPILED = "jit_compiled"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"

class ResourceType(Enum):
    CPU = "cpu"
    GPU = "gpu"
    MEMORY = "memory"
    NETWORK = "network"
    DISK = "disk"

class OptimizationGoal(Enum):
    LATENCY = "minimize_latency"
    THROUGHPUT = "maximize_throughput"
    ACCURACY = "maximize_accuracy"
    EFFICIENCY = "maximize_efficiency"
    POWER = "minimize_power"

@dataclass 
class PerformanceMetrics:
    """Comprehensive performance metrics for optimization decisions"""
    timestamp: float
    latency_ms: float
    throughput_ops_per_sec: float
    accuracy: float
    cpu_utilization: float
    gpu_utilization: float
    memory_usage_mb: float
    power_consumption_watts: float
    error_rate: float
    
    # HDC-specific metrics
    similarity_computation_time: float = 0.0
    bundle_operation_time: float = 0.0
    hypervector_memory_efficiency: float = 1.0
    adaptation_success_rate: float = 1.0
    
    def get_composite_score(self, goal: OptimizationGoal) -> float:
        """Calculate composite performance score based on optimization goal"""
        
        if goal == OptimizationGoal.LATENCY:
            # Lower latency is better
            return 1.0 / (1.0 + self.latency_ms / 100.0)
        
        elif goal == OptimizationGoal.THROUGHPUT:
            # Higher throughput is better
            return min(1.0, self.throughput_ops_per_sec / 1000.0)
        
        elif goal == OptimizationGoal.ACCURACY:
            # Higher accuracy is better
            return self.accuracy
        
        elif goal == OptimizationGoal.EFFICIENCY:
            # Balance of performance and resource usage
            performance = (self.accuracy + (1.0 / (1.0 + self.latency_ms / 100.0))) / 2
            resource_efficiency = 1.0 - ((self.cpu_utilization + self.gpu_utilization) / 2)
            return (performance + resource_efficiency) / 2
        
        elif goal == OptimizationGoal.POWER:
            # Lower power consumption is better
            return 1.0 / (1.0 + self.power_consumption_watts / 100.0)
        
        else:
            return 0.5  # Default neutral score

@dataclass
class OptimizationConfiguration:
    """Configuration parameters for optimization strategies"""
    strategy: OptimizationStrategy
    parameters: Dict[str, Any] = field(default_factory=dict)
    resource_allocation: Dict[ResourceType, float] = field(default_factory=dict)
    expected_performance: Optional[PerformanceMetrics] = None
    confidence: float = 0.5
    
    def get_config_hash(self) -> str:
        """Generate unique hash for this configuration"""
        import hashlib
        config_str = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

class SelfImprovingOptimizer:
    """
    Self-improving adaptive optimizer that learns optimal configurations
    
    Core Capabilities:
    1. Performance Pattern Learning
    2. Predictive Configuration Selection  
    3. Real-time Adaptation
    4. Resource-aware Optimization
    5. Multi-objective Optimization
    6. Self-healing and Recovery
    """
    
    def __init__(self, 
                 optimization_goal: OptimizationGoal = OptimizationGoal.EFFICIENCY,
                 learning_rate: float = 0.1,
                 adaptation_threshold: float = 0.05):
        
        self.optimization_goal = optimization_goal
        self.learning_rate = learning_rate
        self.adaptation_threshold = adaptation_threshold
        
        # Performance tracking
        self.performance_history = deque(maxlen=10000)
        self.configuration_performance = defaultdict(list)  # config_hash -> [metrics]
        
        # Current optimization state
        self.current_configuration = self._get_default_configuration()
        self.optimization_active = False
        self.optimization_thread = None
        
        # Learning components
        self.strategy_performance = defaultdict(list)  # strategy -> [scores]
        self.resource_efficiency_model = {}
        self.pattern_recognition_model = {}
        
        # Available optimization strategies
        self.optimization_strategies = self._initialize_optimization_strategies()
        
        # Self-improvement tracking
        self.improvement_history = deque(maxlen=1000)
        self.adaptation_count = 0
        self.total_improvement = 0.0
        
        optimizer_logger.info(f"Self-improving optimizer initialized")
        optimizer_logger.info(f"Optimization goal: {optimization_goal.value}")
        optimizer_logger.info(f"Learning rate: {learning_rate}")
        optimizer_logger.info(f"Available strategies: {len(self.optimization_strategies)}")
    
    def _initialize_optimization_strategies(self) -> Dict[OptimizationStrategy, Dict[str, Any]]:
        """Initialize available optimization strategies with their configurations"""
        
        strategies = {
            OptimizationStrategy.CPU_INTENSIVE: {
                "description": "CPU-optimized processing with multi-threading",
                "resource_profile": {
                    ResourceType.CPU: 0.9,
                    ResourceType.GPU: 0.1,
                    ResourceType.MEMORY: 0.6
                },
                "parameters": {
                    "thread_count": "auto",
                    "cpu_affinity": True,
                    "memory_pool_size": "dynamic"
                },
                "best_for": ["small_datasets", "low_gpu_availability", "power_constrained"]
            },
            
            OptimizationStrategy.GPU_ACCELERATED: {
                "description": "GPU-accelerated HDC operations with CUDA",
                "resource_profile": {
                    ResourceType.CPU: 0.3,
                    ResourceType.GPU: 0.95,
                    ResourceType.MEMORY: 0.8
                },
                "parameters": {
                    "cuda_streams": 4,
                    "gpu_memory_fraction": 0.8,
                    "batch_size": "auto"
                },
                "best_for": ["large_datasets", "high_throughput", "parallel_processing"]
            },
            
            OptimizationStrategy.JIT_COMPILED: {
                "description": "Just-in-time compiled operations for optimization",
                "resource_profile": {
                    ResourceType.CPU: 0.7,
                    ResourceType.GPU: 0.2,
                    ResourceType.MEMORY: 0.5
                },
                "parameters": {
                    "compilation_cache": True,
                    "optimization_level": "aggressive",
                    "inline_functions": True
                },
                "best_for": ["repeated_operations", "predictable_patterns", "warm_up_acceptable"]
            },
            
            OptimizationStrategy.HYBRID: {
                "description": "Hybrid CPU/GPU approach with intelligent load balancing",
                "resource_profile": {
                    ResourceType.CPU: 0.6,
                    ResourceType.GPU: 0.7,
                    ResourceType.MEMORY: 0.7
                },
                "parameters": {
                    "cpu_gpu_ratio": "dynamic",
                    "load_balancing": "adaptive",
                    "fallback_strategy": "cpu"
                },
                "best_for": ["variable_workloads", "mixed_operations", "fault_tolerance"]
            },
            
            OptimizationStrategy.DISTRIBUTED: {
                "description": "Distributed processing across multiple nodes",
                "resource_profile": {
                    ResourceType.CPU: 0.5,
                    ResourceType.GPU: 0.5,
                    ResourceType.NETWORK: 0.8
                },
                "parameters": {
                    "node_count": "auto",
                    "communication_protocol": "optimized",
                    "load_distribution": "intelligent"
                },
                "best_for": ["massive_datasets", "horizontal_scaling", "high_availability"]
            }
        }
        
        return strategies
    
    def _get_default_configuration(self) -> OptimizationConfiguration:
        """Get default optimization configuration"""
        
        return OptimizationConfiguration(
            strategy=OptimizationStrategy.HYBRID,
            parameters={"mode": "balanced"},
            resource_allocation={
                ResourceType.CPU: 0.6,
                ResourceType.GPU: 0.4,
                ResourceType.MEMORY: 0.7
            }
        )
    
    def start_optimization(self):
        """Start continuous optimization process"""
        
        if self.optimization_active:
            optimizer_logger.warning("Optimization already active")
            return
        
        self.optimization_active = True
        self.optimization_thread = threading.Thread(target=self._optimization_loop, daemon=True)
        self.optimization_thread.start()
        
        optimizer_logger.info("Started continuous optimization")
    
    def stop_optimization(self):
        """Stop optimization process"""
        
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5)
        
        optimizer_logger.info("Stopped optimization")
    
    def _optimization_loop(self):
        """Main optimization loop with self-improvement"""
        
        optimizer_logger.info("Optimization loop started")
        
        while self.optimization_active:
            try:
                # Collect current performance metrics
                current_metrics = self._collect_performance_metrics()
                self.performance_history.append(current_metrics)
                
                # Evaluate current configuration performance
                current_score = current_metrics.get_composite_score(self.optimization_goal)
                config_hash = self.current_configuration.get_config_hash()
                self.configuration_performance[config_hash].append(current_score)
                
                # Determine if optimization is needed
                if self._should_optimize(current_metrics):
                    new_config = self._select_optimal_configuration(current_metrics)
                    
                    if new_config.get_config_hash() != config_hash:
                        optimizer_logger.info(f"Adapting configuration: {self.current_configuration.strategy.value} -> {new_config.strategy.value}")
                        
                        # Apply new configuration
                        improvement = self._apply_configuration(new_config)
                        
                        # Track improvement
                        if improvement > 0:
                            self.adaptation_count += 1
                            self.total_improvement += improvement
                            self.improvement_history.append({
                                'timestamp': time.time(),
                                'improvement': improvement,
                                'old_config': config_hash,
                                'new_config': new_config.get_config_hash()
                            })
                            
                            optimizer_logger.info(f"Configuration improvement: {improvement:.3f}")
                
                # Learn from performance patterns
                self._update_learning_models(current_metrics)
                
                # Self-healing check
                if current_metrics.error_rate > 0.1:  # 10% error rate threshold
                    self._perform_self_healing(current_metrics)
                
                # Wait before next optimization cycle
                time.sleep(5.0)  # 5-second optimization cycle
                
            except Exception as e:
                optimizer_logger.error(f"Error in optimization loop: {e}")
                time.sleep(5.0)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics"""
        
        # Simulate performance metric collection
        # In real implementation, this would gather actual system metrics
        
        base_latency = 50.0  # ms
        base_throughput = 500.0  # ops/sec
        base_accuracy = 0.90
        
        # Add some realistic variation based on current configuration
        strategy_modifiers = {
            OptimizationStrategy.CPU_INTENSIVE: {"latency": 1.2, "throughput": 0.8, "accuracy": 0.98},
            OptimizationStrategy.GPU_ACCELERATED: {"latency": 0.3, "throughput": 3.0, "accuracy": 1.02},
            OptimizationStrategy.JIT_COMPILED: {"latency": 0.6, "throughput": 1.5, "accuracy": 1.0},
            OptimizationStrategy.HYBRID: {"latency": 0.7, "throughput": 1.8, "accuracy": 1.01},
            OptimizationStrategy.DISTRIBUTED: {"latency": 0.9, "throughput": 2.5, "accuracy": 0.99}
        }
        
        modifier = strategy_modifiers[self.current_configuration.strategy]
        
        # Apply strategy-specific performance characteristics
        latency = base_latency * modifier["latency"] * (0.8 + np.random.random() * 0.4)
        throughput = base_throughput * modifier["throughput"] * (0.8 + np.random.random() * 0.4)
        accuracy = min(1.0, base_accuracy * modifier["accuracy"] * (0.95 + np.random.random() * 0.1))
        
        # Resource utilization simulation
        cpu_util = min(0.95, 0.3 + np.random.random() * 0.4)
        gpu_util = 0.8 if self.current_configuration.strategy == OptimizationStrategy.GPU_ACCELERATED else 0.2 + np.random.random() * 0.3
        memory_usage = 500 + np.random.random() * 1000  # MB
        power_consumption = 50 + cpu_util * 100 + gpu_util * 150  # Watts
        
        # HDC-specific metrics
        similarity_time = latency * 0.3
        bundle_time = latency * 0.2
        memory_efficiency = 0.8 + np.random.random() * 0.2
        adaptation_success = 0.9 + np.random.random() * 0.1
        
        return PerformanceMetrics(
            timestamp=time.time(),
            latency_ms=latency,
            throughput_ops_per_sec=throughput,
            accuracy=accuracy,
            cpu_utilization=cpu_util,
            gpu_utilization=gpu_util,
            memory_usage_mb=memory_usage,
            power_consumption_watts=power_consumption,
            error_rate=max(0.0, 0.01 + np.random.normal(0, 0.02)),
            similarity_computation_time=similarity_time,
            bundle_operation_time=bundle_time,
            hypervector_memory_efficiency=memory_efficiency,
            adaptation_success_rate=adaptation_success
        )
    
    def _should_optimize(self, current_metrics: PerformanceMetrics) -> bool:
        """Determine if optimization should be performed"""
        
        # Don't optimize too frequently
        if len(self.performance_history) < 10:
            return False
        
        # Check if performance has degraded
        recent_scores = [m.get_composite_score(self.optimization_goal) for m in list(self.performance_history)[-10:]]
        current_score = current_metrics.get_composite_score(self.optimization_goal)
        
        if len(recent_scores) > 0:
            recent_avg = statistics.mean(recent_scores)
            
            # Optimize if performance has degraded beyond threshold
            if recent_avg - current_score > self.adaptation_threshold:
                return True
        
        # Optimize if error rate is high
        if current_metrics.error_rate > 0.05:  # 5% error rate threshold
            return True
        
        # Optimize if resource utilization is unbalanced
        cpu_gpu_imbalance = abs(current_metrics.cpu_utilization - current_metrics.gpu_utilization)
        if cpu_gpu_imbalance > 0.6:  # Significant resource imbalance
            return True
        
        # Periodic optimization check (every 100 cycles)
        if len(self.performance_history) % 100 == 0:
            return True
        
        return False
    
    def _select_optimal_configuration(self, current_metrics: PerformanceMetrics) -> OptimizationConfiguration:
        """Select optimal configuration based on current conditions and learned patterns"""
        
        best_strategy = self.current_configuration.strategy
        best_score = -1.0
        
        # Evaluate each available strategy
        for strategy in OptimizationStrategy:
            predicted_score = self._predict_strategy_performance(strategy, current_metrics)
            
            if predicted_score > best_score:
                best_score = predicted_score
                best_strategy = strategy
        
        # Create optimized configuration
        strategy_config = self.optimization_strategies[best_strategy]
        
        # Adaptive parameter tuning based on current conditions
        optimized_params = self._optimize_strategy_parameters(best_strategy, current_metrics)
        
        # Resource allocation optimization
        optimal_resources = self._optimize_resource_allocation(best_strategy, current_metrics)
        
        return OptimizationConfiguration(
            strategy=best_strategy,
            parameters=optimized_params,
            resource_allocation=optimal_resources,
            confidence=min(1.0, best_score)
        )
    
    def _predict_strategy_performance(self, strategy: OptimizationStrategy, 
                                    current_metrics: PerformanceMetrics) -> float:
        """Predict performance score for a given strategy"""
        
        # Use historical performance data if available
        if strategy in self.strategy_performance and self.strategy_performance[strategy]:
            historical_scores = self.strategy_performance[strategy][-20:]  # Last 20 data points
            base_score = statistics.mean(historical_scores)
        else:
            # Use strategy characteristics for initial prediction
            strategy_config = self.optimization_strategies[strategy]
            base_score = 0.7  # Default base score
        
        # Adjust prediction based on current conditions
        adjustment_factors = []
        
        # CPU availability factor
        if current_metrics.cpu_utilization < 0.5 and strategy == OptimizationStrategy.CPU_INTENSIVE:
            adjustment_factors.append(0.2)  # Favor CPU strategy when CPU is available
        elif current_metrics.cpu_utilization > 0.9 and strategy == OptimizationStrategy.CPU_INTENSIVE:
            adjustment_factors.append(-0.3)  # Penalize CPU strategy when CPU is saturated
        
        # GPU availability factor  
        if current_metrics.gpu_utilization < 0.3 and strategy == OptimizationStrategy.GPU_ACCELERATED:
            adjustment_factors.append(0.3)  # Favor GPU strategy when GPU is available
        elif current_metrics.gpu_utilization > 0.95 and strategy == OptimizationStrategy.GPU_ACCELERATED:
            adjustment_factors.append(-0.4)  # Penalize GPU strategy when GPU is saturated
        
        # Error rate factor
        if current_metrics.error_rate > 0.05 and strategy == OptimizationStrategy.HYBRID:
            adjustment_factors.append(0.15)  # Favor hybrid strategy for fault tolerance
        
        # Memory efficiency factor
        if current_metrics.hypervector_memory_efficiency < 0.7 and strategy == OptimizationStrategy.JIT_COMPILED:
            adjustment_factors.append(0.1)  # JIT can improve memory efficiency
        
        # Apply adjustments
        total_adjustment = sum(adjustment_factors)
        predicted_score = max(0.0, min(1.0, base_score + total_adjustment))
        
        return predicted_score
    
    def _optimize_strategy_parameters(self, strategy: OptimizationStrategy, 
                                    current_metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Optimize parameters for the selected strategy"""
        
        base_params = self.optimization_strategies[strategy]["parameters"].copy()
        
        # Strategy-specific parameter optimization
        if strategy == OptimizationStrategy.CPU_INTENSIVE:
            # Optimize thread count based on CPU utilization
            if current_metrics.cpu_utilization < 0.5:
                base_params["thread_count"] = "max_cores"
            elif current_metrics.cpu_utilization > 0.8:
                base_params["thread_count"] = "conservative"
            
        elif strategy == OptimizationStrategy.GPU_ACCELERATED:
            # Optimize GPU memory usage and batch size
            if current_metrics.memory_usage_mb > 1500:
                base_params["gpu_memory_fraction"] = 0.6
                base_params["batch_size"] = "small"
            elif current_metrics.memory_usage_mb < 800:
                base_params["gpu_memory_fraction"] = 0.9
                base_params["batch_size"] = "large"
        
        elif strategy == OptimizationStrategy.HYBRID:
            # Optimize CPU/GPU ratio based on current utilization
            cpu_ratio = 1.0 - current_metrics.cpu_utilization
            gpu_ratio = 1.0 - current_metrics.gpu_utilization
            
            if cpu_ratio > gpu_ratio:
                base_params["cpu_gpu_ratio"] = "cpu_heavy"
            elif gpu_ratio > cpu_ratio:
                base_params["cpu_gpu_ratio"] = "gpu_heavy"
            else:
                base_params["cpu_gpu_ratio"] = "balanced"
        
        return base_params
    
    def _optimize_resource_allocation(self, strategy: OptimizationStrategy,
                                    current_metrics: PerformanceMetrics) -> Dict[ResourceType, float]:
        """Optimize resource allocation for the strategy"""
        
        base_allocation = self.optimization_strategies[strategy]["resource_profile"].copy()
        
        # Adjust based on current resource utilization
        adjustments = {}
        
        # CPU adjustment
        if current_metrics.cpu_utilization > 0.9:
            adjustments[ResourceType.CPU] = -0.1  # Reduce CPU pressure
        elif current_metrics.cpu_utilization < 0.3:
            adjustments[ResourceType.CPU] = 0.1   # Utilize available CPU
        
        # GPU adjustment
        if current_metrics.gpu_utilization > 0.95:
            adjustments[ResourceType.GPU] = -0.1  # Reduce GPU pressure
        elif current_metrics.gpu_utilization < 0.2:
            adjustments[ResourceType.GPU] = 0.1   # Utilize available GPU
        
        # Memory adjustment based on usage
        if current_metrics.memory_usage_mb > 2000:
            adjustments[ResourceType.MEMORY] = -0.1  # Reduce memory usage
        elif current_metrics.memory_usage_mb < 500:
            adjustments[ResourceType.MEMORY] = 0.1   # Can use more memory
        
        # Apply adjustments
        optimized_allocation = {}
        for resource, base_value in base_allocation.items():
            adjustment = adjustments.get(resource, 0.0)
            optimized_allocation[resource] = max(0.1, min(1.0, base_value + adjustment))
        
        return optimized_allocation
    
    def _apply_configuration(self, new_config: OptimizationConfiguration) -> float:
        """Apply new optimization configuration and measure improvement"""
        
        # Store previous performance for comparison
        if self.performance_history:
            previous_score = self.performance_history[-1].get_composite_score(self.optimization_goal)
        else:
            previous_score = 0.5
        
        # Apply configuration (simulation)
        old_config = self.current_configuration
        self.current_configuration = new_config
        
        # Simulate configuration application time
        time.sleep(0.1)  # Brief delay for configuration change
        
        # Measure new performance after brief settling period
        time.sleep(1.0)  # Allow system to settle
        new_metrics = self._collect_performance_metrics()
        new_score = new_metrics.get_composite_score(self.optimization_goal)
        
        improvement = new_score - previous_score
        
        # Update strategy performance tracking
        self.strategy_performance[new_config.strategy].append(new_score)
        
        optimizer_logger.info(f"Applied configuration: {new_config.strategy.value}")
        optimizer_logger.info(f"Performance change: {previous_score:.3f} -> {new_score:.3f} ({improvement:+.3f})")
        
        return improvement
    
    def _update_learning_models(self, metrics: PerformanceMetrics):
        """Update internal learning models with new performance data"""
        
        current_strategy = self.current_configuration.strategy
        
        # Update strategy performance model
        score = metrics.get_composite_score(self.optimization_goal)
        self.strategy_performance[current_strategy].append(score)
        
        # Resource efficiency learning
        resource_key = (
            round(metrics.cpu_utilization, 1),
            round(metrics.gpu_utilization, 1),
            round(metrics.memory_usage_mb / 100) * 100  # Rounded to nearest 100MB
        )
        
        if resource_key not in self.resource_efficiency_model:
            self.resource_efficiency_model[resource_key] = []
        
        self.resource_efficiency_model[resource_key].append({
            'strategy': current_strategy,
            'score': score,
            'timestamp': metrics.timestamp
        })
        
        # Pattern recognition learning (simplified)
        pattern_key = (
            current_strategy,
            round(metrics.latency_ms / 10) * 10,  # Rounded to nearest 10ms
            round(metrics.throughput_ops_per_sec / 100) * 100  # Rounded to nearest 100 ops/sec
        )
        
        if pattern_key not in self.pattern_recognition_model:
            self.pattern_recognition_model[pattern_key] = []
        
        self.pattern_recognition_model[pattern_key].append({
            'outcome_score': score,
            'timestamp': metrics.timestamp
        })
    
    def _perform_self_healing(self, metrics: PerformanceMetrics):
        """Perform self-healing actions when errors are detected"""
        
        optimizer_logger.warning(f"Self-healing triggered: error rate {metrics.error_rate:.2%}")
        
        # Attempt self-healing strategies
        healing_actions = []
        
        # High error rate - switch to more reliable strategy
        if metrics.error_rate > 0.1:
            healing_actions.append("switch_to_hybrid_strategy")
            
        # High resource utilization - reduce load
        if metrics.cpu_utilization > 0.95 or metrics.gpu_utilization > 0.95:
            healing_actions.append("reduce_resource_allocation")
            
        # Memory pressure - optimize memory usage
        if metrics.memory_usage_mb > 3000:
            healing_actions.append("optimize_memory_usage")
        
        # Apply healing actions
        for action in healing_actions:
            if action == "switch_to_hybrid_strategy":
                # Switch to hybrid strategy for better fault tolerance
                healing_config = OptimizationConfiguration(
                    strategy=OptimizationStrategy.HYBRID,
                    parameters={"mode": "conservative", "fallback_enabled": True},
                    resource_allocation={
                        ResourceType.CPU: 0.5,
                        ResourceType.GPU: 0.5,
                        ResourceType.MEMORY: 0.6
                    }
                )
                self.current_configuration = healing_config
                optimizer_logger.info("Self-healing: Switched to hybrid strategy")
                
            elif action == "reduce_resource_allocation":
                # Reduce resource pressure
                for resource in self.current_configuration.resource_allocation:
                    current_allocation = self.current_configuration.resource_allocation[resource]
                    self.current_configuration.resource_allocation[resource] = current_allocation * 0.8
                optimizer_logger.info("Self-healing: Reduced resource allocation")
                
            elif action == "optimize_memory_usage":
                # Enable memory optimization parameters
                self.current_configuration.parameters.update({
                    "memory_optimization": True,
                    "garbage_collection": "aggressive",
                    "cache_size_limit": "conservative"
                })
                optimizer_logger.info("Self-healing: Enabled memory optimization")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        
        # Calculate performance trends
        if len(self.performance_history) > 10:
            recent_scores = [m.get_composite_score(self.optimization_goal) 
                           for m in list(self.performance_history)[-50:]]
            avg_recent_performance = statistics.mean(recent_scores)
            performance_trend = statistics.mean(recent_scores[-10:]) - statistics.mean(recent_scores[:10])
        else:
            avg_recent_performance = 0.0
            performance_trend = 0.0
        
        # Strategy effectiveness analysis
        strategy_effectiveness = {}
        for strategy, scores in self.strategy_performance.items():
            if scores:
                strategy_effectiveness[strategy.value] = {
                    'average_score': statistics.mean(scores),
                    'best_score': max(scores),
                    'usage_count': len(scores),
                    'recent_trend': statistics.mean(scores[-5:]) - statistics.mean(scores[:5]) if len(scores) >= 10 else 0
                }
        
        report = {
            'timestamp': time.time(),
            'optimization_goal': self.optimization_goal.value,
            'current_configuration': {
                'strategy': self.current_configuration.strategy.value,
                'parameters': self.current_configuration.parameters,
                'resource_allocation': {k.value: v for k, v in self.current_configuration.resource_allocation.items()}
            },
            'performance_summary': {
                'average_recent_performance': avg_recent_performance,
                'performance_trend': performance_trend,
                'total_adaptations': self.adaptation_count,
                'total_improvement': self.total_improvement,
                'data_points_collected': len(self.performance_history)
            },
            'strategy_effectiveness': strategy_effectiveness,
            'learning_model_size': {
                'resource_efficiency_patterns': len(self.resource_efficiency_model),
                'performance_patterns': len(self.pattern_recognition_model)
            },
            'recent_improvements': [
                {
                    'improvement': imp['improvement'],
                    'timestamp': imp['timestamp']
                }
                for imp in list(self.improvement_history)[-5:]  # Last 5 improvements
            ],
            'recommendations': self._generate_optimization_recommendations()
        }
        
        return report
    
    def _generate_optimization_recommendations(self) -> List[str]:
        """Generate actionable optimization recommendations"""
        
        recommendations = []
        
        # Analyze recent performance trends
        if len(self.performance_history) > 20:
            recent_metrics = list(self.performance_history)[-20:]
            avg_cpu = statistics.mean([m.cpu_utilization for m in recent_metrics])
            avg_gpu = statistics.mean([m.gpu_utilization for m in recent_metrics])
            avg_error_rate = statistics.mean([m.error_rate for m in recent_metrics])
            
            # CPU utilization recommendations
            if avg_cpu > 0.9:
                recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
            elif avg_cpu < 0.3:
                recommendations.append("CPU resources are underutilized - consider CPU-intensive optimizations")
            
            # GPU utilization recommendations
            if avg_gpu > 0.95:
                recommendations.append("GPU is saturated - consider batch optimization or additional GPU resources")
            elif avg_gpu < 0.2 and self.current_configuration.strategy != OptimizationStrategy.GPU_ACCELERATED:
                recommendations.append("GPU resources available - consider GPU-accelerated processing")
            
            # Error rate recommendations
            if avg_error_rate > 0.05:
                recommendations.append("High error rate detected - consider more robust processing strategies")
        
        # Strategy-specific recommendations
        current_strategy = self.current_configuration.strategy
        if current_strategy in self.strategy_performance:
            strategy_scores = self.strategy_performance[current_strategy]
            if len(strategy_scores) > 5:
                recent_performance = statistics.mean(strategy_scores[-5:])
                if recent_performance < 0.6:
                    recommendations.append(f"Current strategy ({current_strategy.value}) showing poor performance - consider switching")
        
        # Learning model recommendations
        if len(self.resource_efficiency_model) > 100:
            recommendations.append("Sufficient learning data collected - optimization accuracy should improve")
        elif len(self.resource_efficiency_model) < 10:
            recommendations.append("Limited learning data - optimization may be less accurate initially")
        
        # Adaptation frequency recommendations
        if self.adaptation_count > 50 and len(self.performance_history) > 100:
            adaptation_rate = self.adaptation_count / len(self.performance_history) * 100
            if adaptation_rate > 20:
                recommendations.append("High adaptation rate - consider tuning adaptation threshold")
            elif adaptation_rate < 2:
                recommendations.append("Low adaptation rate - system may not be responding to performance changes")
        
        if not recommendations:
            recommendations.append("System is operating optimally with current configuration")
        
        return recommendations
    
    def save_optimization_state(self, filepath: str):
        """Save current optimization state for persistence"""
        
        state = {
            'configuration_performance': dict(self.configuration_performance),
            'strategy_performance': {k.value: v for k, v in self.strategy_performance.items()},
            'resource_efficiency_model': self.resource_efficiency_model,
            'pattern_recognition_model': self.pattern_recognition_model,
            'adaptation_count': self.adaptation_count,
            'total_improvement': self.total_improvement,
            'optimization_goal': self.optimization_goal.value,
            'current_configuration': asdict(self.current_configuration)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        optimizer_logger.info(f"Optimization state saved to {filepath}")
    
    def load_optimization_state(self, filepath: str):
        """Load optimization state from file"""
        
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            self.configuration_performance.update(state['configuration_performance'])
            
            # Convert strategy keys back to enums
            for strategy_name, scores in state['strategy_performance'].items():
                strategy_enum = OptimizationStrategy(strategy_name)
                self.strategy_performance[strategy_enum] = scores
            
            self.resource_efficiency_model = state['resource_efficiency_model']
            self.pattern_recognition_model = state['pattern_recognition_model']
            self.adaptation_count = state['adaptation_count']
            self.total_improvement = state['total_improvement']
            
            optimizer_logger.info(f"Optimization state loaded from {filepath}")
            optimizer_logger.info(f"Loaded {len(self.configuration_performance)} configuration records")
            optimizer_logger.info(f"Loaded {sum(len(scores) for scores in self.strategy_performance.values())} performance records")
            
        except Exception as e:
            optimizer_logger.error(f"Failed to load optimization state: {e}")

def main():
    """Demonstrate self-improving adaptive optimizer"""
    optimizer_logger.info("HDC Self-Improving Adaptive Optimizer Demo")
    optimizer_logger.info("=" * 60)
    
    # Initialize optimizer
    optimizer = SelfImprovingOptimizer(
        optimization_goal=OptimizationGoal.EFFICIENCY,
        learning_rate=0.15,
        adaptation_threshold=0.03
    )
    
    # Start optimization
    optimizer.start_optimization()
    
    try:
        optimizer_logger.info("Running self-improving optimization... (Press Ctrl+C to stop)")
        
        # Simulate different workload conditions
        for cycle in range(60):  # 5-minute demo (60 * 5 seconds)
            time.sleep(5)
            
            # Simulate workload changes every 20 cycles
            if cycle % 20 == 10:
                optimizer_logger.info("💥 Simulating high CPU workload...")
                # This would trigger CPU-intensive optimization
                
            elif cycle % 20 == 15:
                optimizer_logger.info("🚀 Simulating high throughput demand...")  
                # This would trigger GPU-accelerated optimization
            
            # Generate periodic reports
            if cycle % 15 == 0:
                report = optimizer.get_optimization_report()
                
                optimizer_logger.info(f"📊 Optimization Report (Cycle {cycle}):")
                optimizer_logger.info(f"  Current Strategy: {report['current_configuration']['strategy']}")
                optimizer_logger.info(f"  Performance Score: {report['performance_summary']['average_recent_performance']:.3f}")
                optimizer_logger.info(f"  Total Adaptations: {report['performance_summary']['total_adaptations']}")
                optimizer_logger.info(f"  Total Improvement: {report['performance_summary']['total_improvement']:+.3f}")
                
                # Show top recommendation
                if report['recommendations']:
                    optimizer_logger.info(f"  Top Recommendation: {report['recommendations'][0]}")
    
    except KeyboardInterrupt:
        optimizer_logger.info("Stopping optimization demo...")
    
    finally:
        # Stop optimization
        optimizer.stop_optimization()
        
        # Generate final report
        final_report = optimizer.get_optimization_report()
        
        print(f"\n🎯 FINAL OPTIMIZATION RESULTS:")
        print("=" * 50)
        print(f"Total Runtime Adaptations: {final_report['performance_summary']['total_adaptations']}")
        print(f"Cumulative Improvement: {final_report['performance_summary']['total_improvement']:+.3f}")
        print(f"Final Strategy: {final_report['current_configuration']['strategy']}")
        print(f"Learning Patterns Collected: {final_report['learning_model_size']['resource_efficiency_patterns']}")
        
        # Strategy effectiveness summary
        if final_report['strategy_effectiveness']:
            print(f"\n📈 Strategy Effectiveness:")
            for strategy, stats in final_report['strategy_effectiveness'].items():
                print(f"  {strategy}: {stats['average_score']:.3f} avg ({stats['usage_count']} uses)")
        
        print(f"\n💡 Final Recommendations:")
        for i, rec in enumerate(final_report['recommendations'][:3], 1):
            print(f"  {i}. {rec}")
        
        # Save optimization state
        os.makedirs('/root/repo/self_improving/states', exist_ok=True)
        state_file = f"/root/repo/self_improving/states/optimizer_state_{int(time.time())}.pkl"
        optimizer.save_optimization_state(state_file)
        
        # Save report
        report_file = f"/root/repo/self_improving/reports/optimization_report_{int(time.time())}.json"
        os.makedirs('/root/repo/self_improving/reports', exist_ok=True)
        with open(report_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        optimizer_logger.info(f"Optimization state saved to {state_file}")
        optimizer_logger.info(f"Final report saved to {report_file}")
        optimizer_logger.info("Self-improving optimization demonstration completed!")
    
    return optimizer

if __name__ == "__main__":
    optimizer = main()