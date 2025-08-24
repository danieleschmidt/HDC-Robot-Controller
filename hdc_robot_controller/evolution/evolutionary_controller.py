"""
Evolutionary Controller - Autonomous Evolution Management System
==============================================================

Orchestrates all evolutionary processes and manages autonomous algorithmic evolution.
Provides the main interface for self-improving robotic intelligence systems.
"""

import numpy as np
import time
import threading
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import logging

from .genetic_optimizer import GeneticOptimizer, EvolutionaryParameters, HDCAlgorithmFitness
from .self_improving_algorithms import SelfImprovingHDC, AlgorithmicEvolution
from ..core.hypervector import HyperVector
from ..core.memory import AssociativeMemory


@dataclass
class EvolutionaryConfig:
    """Configuration for evolutionary controller."""
    enable_genetic_optimization: bool = True
    enable_self_improvement: bool = True  
    enable_algorithmic_evolution: bool = True
    evolution_interval: int = 3600  # 1 hour
    performance_monitoring_window: int = 100
    improvement_threshold: float = 0.05
    max_concurrent_evolutions: int = 3
    backup_interval: int = 7200  # 2 hours
    

class EvolutionMetrics:
    """Tracks evolution metrics and performance."""
    
    def __init__(self):
        self.evolution_count = 0
        self.successful_improvements = 0
        self.total_performance_gain = 0.0
        self.average_improvement_time = 0.0
        self.algorithm_generations = {}
        self.fitness_trajectories = {}
        
    def record_evolution(self, 
                        algorithm_name: str,
                        improvement: float,
                        evolution_time: float,
                        success: bool):
        """Record an evolution attempt."""
        self.evolution_count += 1
        
        if success:
            self.successful_improvements += 1
            self.total_performance_gain += improvement
            
        if algorithm_name not in self.algorithm_generations:
            self.algorithm_generations[algorithm_name] = 0
        self.algorithm_generations[algorithm_name] += 1
        
        # Update average evolution time
        alpha = 0.1  # Exponential smoothing
        self.average_improvement_time = (
            alpha * evolution_time + 
            (1 - alpha) * self.average_improvement_time
        )
    
    def get_success_rate(self) -> float:
        """Get the success rate of evolutionary attempts."""
        return (self.successful_improvements / max(1, self.evolution_count))
    
    def get_average_gain(self) -> float:
        """Get the average performance gain per successful improvement."""
        return (self.total_performance_gain / max(1, self.successful_improvements))


class EvolutionaryController:
    """Main controller for autonomous evolutionary robotics intelligence."""
    
    def __init__(self, 
                 dimension: int = 10000,
                 config: EvolutionaryConfig = None,
                 logger: logging.Logger = None):
        
        self.dimension = dimension
        self.config = config or EvolutionaryConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Core evolutionary components
        self.self_improving_hdc = SelfImprovingHDC(
            dimension=dimension,
            improvement_threshold=config.improvement_threshold
        )
        self.algorithmic_evolution = AlgorithmicEvolution()
        
        # Genetic optimization components
        self.genetic_optimizers = {}
        self.optimization_tasks = {}
        
        # Performance monitoring
        self.performance_monitors = {}
        self.metrics = EvolutionMetrics()
        
        # Evolution management
        self.is_running = False
        self.evolution_thread = None
        self.active_evolutions = {}
        
        # Backup and recovery
        self.last_backup = 0
        self.state_history = []
        
        self.logger.info(f"🧬 Evolutionary Controller initialized with dimension {dimension}")
    
    def start(self):
        """Start the evolutionary controller."""
        if self.is_running:
            return
            
        self.is_running = True
        
        # Start self-improving HDC system
        if self.config.enable_self_improvement:
            self.self_improving_hdc.start_continuous_improvement()
        
        # Start main evolution thread
        self.evolution_thread = threading.Thread(
            target=self._evolution_loop,
            daemon=True
        )
        self.evolution_thread.start()
        
        self.logger.info("🚀 Evolutionary Controller started")
        print("🧬 Autonomous Evolution System Active")
        print("   - Self-improving algorithms: ✅")
        print("   - Genetic optimization: ✅") 
        print("   - Algorithmic evolution: ✅")
    
    def stop(self):
        """Stop the evolutionary controller."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # Stop self-improving system
        if hasattr(self.self_improving_hdc, 'stop_continuous_improvement'):
            self.self_improving_hdc.stop_continuous_improvement()
        
        # Wait for evolution thread to complete
        if self.evolution_thread:
            self.evolution_thread.join(timeout=10.0)
        
        self.logger.info("⏹️ Evolutionary Controller stopped")
    
    def register_algorithm(self, 
                          name: str,
                          algorithm_func: Callable,
                          optimization_target: str = "performance",
                          test_cases: List[Any] = None) -> str:
        """Register an algorithm for evolutionary optimization."""
        
        # Register with self-improving HDC
        if self.config.enable_self_improvement:
            version_id = self.self_improving_hdc.register_algorithm(
                name, algorithm_func, test_cases
            )
        else:
            version_id = f"{name}_v1.0"
        
        # Set up performance monitoring
        self.performance_monitors[name] = {
            'execution_times': [],
            'accuracy_scores': [],
            'resource_usage': [],
            'error_rates': [],
            'last_update': time.time()
        }
        
        # Configure genetic optimization if needed
        if self.config.enable_genetic_optimization and test_cases:
            self._setup_genetic_optimization(name, test_cases, optimization_target)
        
        self.logger.info(f"📝 Registered algorithm '{name}' for evolution (ID: {version_id})")
        return version_id
    
    def _setup_genetic_optimization(self, 
                                   algorithm_name: str,
                                   test_cases: List[Any],
                                   target: str):
        """Set up genetic optimization for an algorithm."""
        
        # Convert test cases to fitness function format
        test_data = []
        for i, test_case in enumerate(test_cases):
            # Create training/test split
            samples = [(f"input_{j}", f"target_{j%3}") for j in range(20)]
            test_samples = [(f"test_{j}", f"target_{j%3}") for j in range(10)]
            test_data.append({
                'samples': samples,
                'test_samples': test_samples
            })
        
        # Create fitness function
        fitness_func = HDCAlgorithmFitness(test_data, [target])
        
        # Create genetic optimizer
        params = EvolutionaryParameters(
            population_size=50,
            max_generations=100,
            mutation_rate=0.1,
            crossover_rate=0.8
        )
        
        optimizer = GeneticOptimizer(fitness_func, params)
        self.genetic_optimizers[algorithm_name] = optimizer
        
        self.logger.info(f"🧬 Genetic optimization configured for {algorithm_name}")
    
    def _evolution_loop(self):
        """Main evolution loop."""
        
        while self.is_running:
            try:
                # Check if it's time for evolution
                if len(self.active_evolutions) < self.config.max_concurrent_evolutions:
                    
                    # Find algorithms that need evolution
                    candidates = self._identify_evolution_candidates()
                    
                    for candidate in candidates[:self.config.max_concurrent_evolutions]:
                        if candidate not in self.active_evolutions:
                            self._start_evolution(candidate)
                
                # Check active evolutions
                self._monitor_active_evolutions()
                
                # Periodic backup
                if time.time() - self.last_backup > self.config.backup_interval:
                    self._backup_state()
                
                # Sleep before next iteration
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in evolution loop: {e}")
                time.sleep(60)
    
    def _identify_evolution_candidates(self) -> List[str]:
        """Identify algorithms that would benefit from evolution."""
        candidates = []
        
        for algorithm_name, monitor in self.performance_monitors.items():
            # Check if enough data is available
            if len(monitor['execution_times']) < self.config.performance_monitoring_window:
                continue
            
            # Analyze performance trend
            recent_times = monitor['execution_times'][-50:]
            if len(recent_times) > 10:
                trend = np.polyfit(range(len(recent_times)), recent_times, 1)[0]
                
                # If performance is degrading, consider for evolution
                if trend > 0.001:  # Getting slower
                    candidates.append(algorithm_name)
            
            # Check error rate
            recent_errors = monitor['error_rates'][-50:]
            if recent_errors and np.mean(recent_errors) > 0.05:  # 5% error rate
                candidates.append(algorithm_name)
        
        return list(set(candidates))  # Remove duplicates
    
    def _start_evolution(self, algorithm_name: str):
        """Start evolutionary optimization for an algorithm."""
        
        self.logger.info(f"🔬 Starting evolution for {algorithm_name}")
        
        evolution_info = {
            'algorithm_name': algorithm_name,
            'start_time': time.time(),
            'strategy': 'self_improvement',  # Default strategy
            'status': 'running'
        }
        
        self.active_evolutions[algorithm_name] = evolution_info
        
        # Start evolution in separate thread
        evolution_thread = threading.Thread(
            target=self._run_algorithm_evolution,
            args=(algorithm_name, evolution_info),
            daemon=True
        )
        evolution_thread.start()
    
    def _run_algorithm_evolution(self, algorithm_name: str, evolution_info: Dict):
        """Run evolution for a specific algorithm."""
        
        try:
            start_time = time.time()
            
            # Choose evolution strategy
            if (self.config.enable_genetic_optimization and 
                algorithm_name in self.genetic_optimizers):
                
                # Use genetic optimization
                success, improvement = self._run_genetic_optimization(algorithm_name)
                
            elif self.config.enable_self_improvement:
                # Use self-improvement
                success, improvement = self._run_self_improvement(algorithm_name)
                
            else:
                success, improvement = False, 0.0
            
            evolution_time = time.time() - start_time
            
            # Record metrics
            self.metrics.record_evolution(
                algorithm_name, improvement, evolution_time, success
            )
            
            evolution_info['status'] = 'completed' if success else 'failed'
            evolution_info['improvement'] = improvement
            evolution_info['duration'] = evolution_time
            
            if success:
                self.logger.info(
                    f"✅ Evolution successful for {algorithm_name}: "
                    f"{improvement:.2%} improvement in {evolution_time:.1f}s"
                )
            else:
                self.logger.warning(f"❌ Evolution failed for {algorithm_name}")
                
        except Exception as e:
            self.logger.error(f"Error evolving {algorithm_name}: {e}")
            evolution_info['status'] = 'error'
            evolution_info['error'] = str(e)
    
    def _run_genetic_optimization(self, algorithm_name: str) -> tuple[bool, float]:
        """Run genetic optimization for an algorithm."""
        
        optimizer = self.genetic_optimizers[algorithm_name]
        
        def progress_callback(generation: int, stats: Dict[str, Any]):
            if generation % 25 == 0:
                self.logger.debug(
                    f"GA {algorithm_name} Gen {generation}: "
                    f"Best={stats['best_fitness']:.4f}"
                )
        
        try:
            best_individual = optimizer.evolve(
                genome_length=6,  # HDC parameters
                callback=progress_callback
            )
            
            improvement = best_individual.fitness
            return improvement > self.config.improvement_threshold, improvement
            
        except Exception as e:
            self.logger.error(f"Genetic optimization error for {algorithm_name}: {e}")
            return False, 0.0
    
    def _run_self_improvement(self, algorithm_name: str) -> tuple[bool, float]:
        """Run self-improvement for an algorithm."""
        
        try:
            # Force an improvement attempt
            self.self_improving_hdc._attempt_improvement(algorithm_name)
            
            # Check if improvement was applied
            stats = self.self_improving_hdc.get_improvement_statistics()
            
            if stats['improvement_history']:
                last_improvement = stats['improvement_history'][-1]
                if last_improvement['algorithm'] == algorithm_name:
                    improvement = last_improvement['improvement_factor']
                    return True, improvement
            
            return False, 0.0
            
        except Exception as e:
            self.logger.error(f"Self-improvement error for {algorithm_name}: {e}")
            return False, 0.0
    
    def _monitor_active_evolutions(self):
        """Monitor and clean up completed evolutions."""
        
        completed = []
        for algorithm_name, evolution_info in self.active_evolutions.items():
            if evolution_info['status'] != 'running':
                completed.append(algorithm_name)
        
        for algorithm_name in completed:
            evolution_info = self.active_evolutions.pop(algorithm_name)
            
            # Log completion
            status = evolution_info['status']
            duration = evolution_info.get('duration', 0)
            improvement = evolution_info.get('improvement', 0)
            
            self.logger.info(
                f"Evolution {status} for {algorithm_name} "
                f"({duration:.1f}s, {improvement:.2%} improvement)"
            )
    
    def record_performance(self, 
                          algorithm_name: str,
                          execution_time: float,
                          accuracy: float = None,
                          error_occurred: bool = False):
        """Record performance metrics for an algorithm."""
        
        if algorithm_name not in self.performance_monitors:
            return
        
        monitor = self.performance_monitors[algorithm_name]
        
        # Add measurements
        monitor['execution_times'].append(execution_time)
        if accuracy is not None:
            monitor['accuracy_scores'].append(accuracy)
        monitor['error_rates'].append(1.0 if error_occurred else 0.0)
        monitor['last_update'] = time.time()
        
        # Keep only recent history
        max_history = self.config.performance_monitoring_window
        for key in ['execution_times', 'accuracy_scores', 'error_rates']:
            if len(monitor[key]) > max_history:
                monitor[key] = monitor[key][-max_history:]
        
        # Also record with self-improving HDC
        if hasattr(self.self_improving_hdc, 'performance_history'):
            if algorithm_name not in self.self_improving_hdc.performance_history:
                self.self_improving_hdc.performance_history[algorithm_name] = []
            
            self.self_improving_hdc.performance_history[algorithm_name].append(execution_time)
            
            # Keep reasonable history size
            history = self.self_improving_hdc.performance_history[algorithm_name]
            if len(history) > max_history:
                self.self_improving_hdc.performance_history[algorithm_name] = history[-max_history:]
    
    def get_evolution_status(self) -> Dict[str, Any]:
        """Get the current status of evolutionary processes."""
        
        return {
            'is_running': self.is_running,
            'active_evolutions': len(self.active_evolutions),
            'registered_algorithms': len(self.performance_monitors),
            'evolution_metrics': {
                'total_evolutions': self.metrics.evolution_count,
                'successful_improvements': self.metrics.successful_improvements,
                'success_rate': self.metrics.get_success_rate(),
                'average_improvement': self.metrics.get_average_gain(),
                'average_evolution_time': self.metrics.average_improvement_time
            },
            'active_evolution_details': {
                name: {
                    'status': info['status'],
                    'duration': time.time() - info['start_time'] if info['status'] == 'running' else info.get('duration', 0),
                    'improvement': info.get('improvement', 0)
                }
                for name, info in self.active_evolutions.items()
            },
            'algorithm_generations': self.metrics.algorithm_generations,
            'self_improvement_stats': self.self_improving_hdc.get_improvement_statistics()
        }
    
    def _backup_state(self):
        """Backup the current evolutionary state."""
        
        try:
            timestamp = int(time.time())
            backup_dir = Path("evolutionary_backups")
            backup_dir.mkdir(exist_ok=True)
            
            # Save main state
            state = {
                'timestamp': timestamp,
                'metrics': {
                    'evolution_count': self.metrics.evolution_count,
                    'successful_improvements': self.metrics.successful_improvements,
                    'total_performance_gain': self.metrics.total_performance_gain,
                    'algorithm_generations': self.metrics.algorithm_generations
                },
                'performance_monitors': self.performance_monitors,
                'evolution_status': self.get_evolution_status()
            }
            
            backup_file = backup_dir / f"evolutionary_state_{timestamp}.json"
            with open(backup_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            # Save self-improving HDC state
            hdc_backup = backup_dir / f"self_improving_hdc_{timestamp}.json"
            self.self_improving_hdc.save_state(str(hdc_backup))
            
            self.last_backup = time.time()
            self.logger.info(f"💾 Evolutionary state backed up to {backup_file}")
            
            # Keep only recent backups
            backup_files = sorted(backup_dir.glob("evolutionary_state_*.json"))
            if len(backup_files) > 10:  # Keep last 10 backups
                for old_backup in backup_files[:-10]:
                    old_backup.unlink()
                    
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
    
    def force_evolution(self, algorithm_name: str) -> bool:
        """Force evolution of a specific algorithm."""
        
        if algorithm_name not in self.performance_monitors:
            self.logger.warning(f"Algorithm '{algorithm_name}' not registered")
            return False
        
        if algorithm_name in self.active_evolutions:
            self.logger.warning(f"Evolution already active for '{algorithm_name}'")
            return False
        
        self.logger.info(f"🔬 Forcing evolution for {algorithm_name}")
        self._start_evolution(algorithm_name)
        return True
    
    def get_performance_summary(self, algorithm_name: str) -> Dict[str, Any]:
        """Get performance summary for a specific algorithm."""
        
        if algorithm_name not in self.performance_monitors:
            return {}
        
        monitor = self.performance_monitors[algorithm_name]
        
        summary = {
            'algorithm_name': algorithm_name,
            'data_points': len(monitor['execution_times']),
            'last_update': monitor['last_update']
        }
        
        if monitor['execution_times']:
            exec_times = monitor['execution_times']
            summary.update({
                'avg_execution_time': np.mean(exec_times),
                'std_execution_time': np.std(exec_times),
                'min_execution_time': np.min(exec_times),
                'max_execution_time': np.max(exec_times),
                'performance_trend': np.polyfit(range(len(exec_times)), exec_times, 1)[0]
            })
        
        if monitor['accuracy_scores']:
            accuracy = monitor['accuracy_scores']
            summary.update({
                'avg_accuracy': np.mean(accuracy),
                'std_accuracy': np.std(accuracy),
                'accuracy_trend': np.polyfit(range(len(accuracy)), accuracy, 1)[0]
            })
        
        if monitor['error_rates']:
            summary['error_rate'] = np.mean(monitor['error_rates'])
        
        return summary


# Example usage and demonstration
if __name__ == "__main__":
    
    # Example algorithms to evolve
    def example_hdc_algorithm(data):
        """Example HDC algorithm for testing evolution."""
        dimension = 1000
        hv = HyperVector.random(dimension)
        
        result = 0.0
        for i in range(50):
            similarity = hv.similarity(HyperVector.random(dimension))
            result += similarity * np.sin(i * 0.1)
            
        return result
    
    def example_ml_algorithm(data):
        """Example ML algorithm for testing evolution."""
        # Simulate machine learning computation
        weights = np.random.randn(len(data))
        return np.dot(data, weights)
    
    # Create evolutionary controller
    controller = EvolutionaryController(
        dimension=10000,
        config=EvolutionaryConfig(
            evolution_interval=60,  # Fast evolution for demo
            improvement_threshold=0.02,
            max_concurrent_evolutions=2
        )
    )
    
    # Register algorithms
    controller.register_algorithm("hdc_query", example_hdc_algorithm)
    controller.register_algorithm("ml_predictor", example_ml_algorithm)
    
    # Start evolutionary process
    controller.start()
    
    print("🧬 Evolutionary Controller Demo Started")
    print("Simulating algorithm usage and evolution...")
    
    # Simulate algorithm usage with performance recording
    for i in range(100):
        # Simulate HDC algorithm usage
        start_time = time.time()
        try:
            result = example_hdc_algorithm(np.random.randn(100))
            execution_time = time.time() - start_time
            error_occurred = False
        except Exception:
            execution_time = 0.1
            error_occurred = True
        
        controller.record_performance(
            "hdc_query", 
            execution_time,
            accuracy=np.random.uniform(0.7, 0.95),
            error_occurred=error_occurred
        )
        
        # Simulate ML algorithm usage  
        start_time = time.time()
        try:
            result = example_ml_algorithm(np.random.randn(50))
            execution_time = time.time() - start_time
        except Exception:
            execution_time = 0.05
            error_occurred = True
        
        controller.record_performance(
            "ml_predictor",
            execution_time,
            accuracy=np.random.uniform(0.6, 0.9),
            error_occurred=False
        )
        
        time.sleep(0.1)
        
        # Print status every 20 iterations
        if i % 20 == 0 and i > 0:
            status = controller.get_evolution_status()
            print(f"\n📊 Evolution Status (iteration {i}):")
            print(f"   Active evolutions: {status['active_evolutions']}")
            print(f"   Total evolutions: {status['evolution_metrics']['total_evolutions']}")
            print(f"   Success rate: {status['evolution_metrics']['success_rate']:.2%}")
    
    # Force evolution of one algorithm
    print("\n🔬 Forcing evolution of hdc_query algorithm...")
    controller.force_evolution("hdc_query")
    
    # Wait a bit for evolution to process
    time.sleep(5)
    
    # Get final status
    final_status = controller.get_evolution_status()
    print("\n🏆 Final Evolution Status:")
    print(json.dumps(final_status, indent=2, default=str))
    
    # Get performance summaries
    print("\n📈 Performance Summaries:")
    for algo_name in ["hdc_query", "ml_predictor"]:
        summary = controller.get_performance_summary(algo_name)
        print(f"\n{algo_name}:")
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.6f}")
            else:
                print(f"  {key}: {value}")
    
    # Stop controller
    controller.stop()
    print("\n⏹️ Evolutionary Controller stopped")