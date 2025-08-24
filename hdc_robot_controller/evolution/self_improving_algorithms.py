"""
Self-Improving Algorithms for Autonomous HDC Evolution
=====================================================

Implements algorithms that can modify and improve themselves during runtime,
enabling autonomous algorithmic evolution and performance optimization.
"""

import numpy as np
import time
import json
import threading
import concurrent.futures
from typing import Dict, List, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect
import ast
import textwrap
from pathlib import Path

from ..core.hypervector import HyperVector
from ..core.operations import HDCOperations
from ..core.memory import AssociativeMemory


@dataclass
class AlgorithmVersion:
    """Represents a version of an algorithm with performance metrics."""
    version_id: str
    algorithm_code: str
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    creation_time: float = field(default_factory=time.time)
    parent_version: Optional[str] = None
    improvement_factor: float = 0.0
    validation_results: Dict[str, Any] = field(default_factory=dict)
    usage_count: int = 0
    success_rate: float = 0.0


@dataclass
class ImprovementCandidate:
    """Candidate algorithm improvement."""
    candidate_id: str
    modification_type: str  # 'parameter', 'structure', 'logic', 'hybrid'
    modified_code: str
    expected_improvement: float
    confidence: float
    test_results: Dict[str, Any] = field(default_factory=dict)
    

class AlgorithmMutator:
    """Generates algorithmic mutations and improvements."""
    
    def __init__(self, mutation_strategies: List[str] = None):
        self.mutation_strategies = mutation_strategies or [
            'parameter_optimization',
            'loop_optimization', 
            'data_structure_optimization',
            'parallel_optimization',
            'memory_optimization',
            'numerical_optimization'
        ]
        
    def generate_candidates(self, 
                          algorithm_code: str,
                          performance_data: Dict[str, float],
                          num_candidates: int = 5) -> List[ImprovementCandidate]:
        """Generate improvement candidates for an algorithm."""
        candidates = []
        
        for i in range(num_candidates):
            strategy = np.random.choice(self.mutation_strategies)
            candidate = self._apply_mutation_strategy(
                algorithm_code, strategy, performance_data, i
            )
            candidates.append(candidate)
            
        return candidates
    
    def _apply_mutation_strategy(self, 
                                code: str, 
                                strategy: str,
                                performance_data: Dict[str, float],
                                candidate_id: int) -> ImprovementCandidate:
        """Apply a specific mutation strategy."""
        
        if strategy == 'parameter_optimization':
            return self._optimize_parameters(code, performance_data, candidate_id)
        elif strategy == 'loop_optimization':
            return self._optimize_loops(code, performance_data, candidate_id)
        elif strategy == 'data_structure_optimization':
            return self._optimize_data_structures(code, performance_data, candidate_id)
        elif strategy == 'parallel_optimization':
            return self._add_parallelization(code, performance_data, candidate_id)
        elif strategy == 'memory_optimization':
            return self._optimize_memory_usage(code, performance_data, candidate_id)
        else:  # numerical_optimization
            return self._optimize_numerical_methods(code, performance_data, candidate_id)
    
    def _optimize_parameters(self, code: str, perf_data: Dict[str, float], cid: int) -> ImprovementCandidate:
        """Optimize numerical parameters in the algorithm."""
        # Parse and identify parameters
        tree = ast.parse(code)
        
        # Find numerical constants and modify them
        class ParameterOptimizer(ast.NodeTransformer):
            def visit_Num(self, node):
                if isinstance(node.n, float) and 0.01 < node.n < 100:
                    # Apply small random changes
                    factor = np.random.uniform(0.8, 1.2)
                    node.n = node.n * factor
                return node
                
            def visit_Constant(self, node):
                if isinstance(node.value, float) and 0.01 < node.value < 100:
                    factor = np.random.uniform(0.8, 1.2)
                    node.value = node.value * factor
                return node
        
        optimizer = ParameterOptimizer()
        new_tree = optimizer.visit(tree)
        
        modified_code = ast.unparse(new_tree)
        
        return ImprovementCandidate(
            candidate_id=f"param_opt_{cid}",
            modification_type='parameter',
            modified_code=modified_code,
            expected_improvement=0.1,
            confidence=0.7
        )
    
    def _optimize_loops(self, code: str, perf_data: Dict[str, float], cid: int) -> ImprovementCandidate:
        """Optimize loop structures."""
        
        # Add vectorization hints
        optimized_code = code.replace(
            "for i in range(len(",
            "# Vectorized operation\nfor i in range(len("
        )
        
        # Add loop unrolling suggestions
        if "for i in range(" in code:
            optimized_code = optimized_code.replace(
                "for i in range(",
                "# Consider loop unrolling for small ranges\nfor i in range("
            )
            
        return ImprovementCandidate(
            candidate_id=f"loop_opt_{cid}",
            modification_type='structure',
            modified_code=optimized_code,
            expected_improvement=0.15,
            confidence=0.6
        )
    
    def _optimize_data_structures(self, code: str, perf_data: Dict[str, float], cid: int) -> ImprovementCandidate:
        """Optimize data structure usage."""
        
        optimized_code = code
        
        # Replace lists with numpy arrays where appropriate
        optimized_code = optimized_code.replace(
            "result = []",
            "result = np.zeros(expected_size, dtype=np.float32)"
        )
        
        # Use sets for membership testing
        optimized_code = optimized_code.replace(
            "item in list_name",
            "item in set_name"
        )
        
        return ImprovementCandidate(
            candidate_id=f"data_opt_{cid}",
            modification_type='structure',
            modified_code=optimized_code,
            expected_improvement=0.2,
            confidence=0.5
        )
    
    def _add_parallelization(self, code: str, perf_data: Dict[str, float], cid: int) -> ImprovementCandidate:
        """Add parallelization to CPU-bound operations."""
        
        # Add parallel processing imports
        parallel_imports = """
import concurrent.futures
from multiprocessing import Pool
"""
        
        # Identify parallelizable loops
        if "for " in code and "range(" in code:
            optimized_code = parallel_imports + "\n" + code
            
            # Add parallel execution suggestion
            optimized_code += """

# Parallel version of main computation
def parallel_version(data_chunks):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_chunk, data_chunks))
    return combine_results(results)
"""
        else:
            optimized_code = code
            
        return ImprovementCandidate(
            candidate_id=f"parallel_{cid}",
            modification_type='parallel',
            modified_code=optimized_code,
            expected_improvement=0.3,
            confidence=0.4
        )
    
    def _optimize_memory_usage(self, code: str, perf_data: Dict[str, float], cid: int) -> ImprovementCandidate:
        """Optimize memory usage patterns."""
        
        optimized_code = code
        
        # Add memory pooling
        if "= []" in code:
            optimized_code = optimized_code.replace(
                "= []",
                "= []  # Consider using memory pool"
            )
            
        # Suggest generator usage
        optimized_code = optimized_code.replace(
            "return [",
            "return (x for x in ["
        ).replace("]", "] if using generator else]")
        
        return ImprovementCandidate(
            candidate_id=f"memory_opt_{cid}",
            modification_type='memory',
            modified_code=optimized_code,
            expected_improvement=0.1,
            confidence=0.6
        )
    
    def _optimize_numerical_methods(self, code: str, perf_data: Dict[str, float], cid: int) -> ImprovementCandidate:
        """Optimize numerical computation methods."""
        
        optimized_code = code
        
        # Replace basic math with optimized versions
        optimizations = {
            "math.sqrt(": "np.sqrt(",
            "math.sin(": "np.sin(",
            "math.cos(": "np.cos(",
            "**2": "np.square(",
            "sum(": "np.sum(",
            "max(": "np.max(",
            "min(": "np.min("
        }
        
        for old, new in optimizations.items():
            optimized_code = optimized_code.replace(old, new)
            
        return ImprovementCandidate(
            candidate_id=f"numerical_opt_{cid}",
            modification_type='numerical',
            modified_code=optimized_code,
            expected_improvement=0.15,
            confidence=0.7
        )


class AlgorithmTester:
    """Tests algorithm improvements for correctness and performance."""
    
    def __init__(self, test_suite: List[Callable] = None):
        self.test_suite = test_suite or []
        self.benchmark_data = []
        
    def validate_candidate(self, 
                          candidate: ImprovementCandidate,
                          original_algorithm: Callable,
                          test_cases: List[Any]) -> Dict[str, Any]:
        """Validate an improvement candidate."""
        
        try:
            # Compile the candidate code
            exec_globals = {'np': np, 'time': time}
            exec(candidate.modified_code, exec_globals)
            
            # Extract the main function (assuming it has the same name)
            func_name = original_algorithm.__name__
            if func_name in exec_globals:
                candidate_func = exec_globals[func_name]
            else:
                # Try to find any function in the namespace
                functions = [v for v in exec_globals.values() if callable(v)]
                if functions:
                    candidate_func = functions[0]
                else:
                    return {'valid': False, 'error': 'No callable function found'}
            
            # Run correctness tests
            correctness_results = self._test_correctness(
                original_algorithm, candidate_func, test_cases
            )
            
            if not correctness_results['passed']:
                return {
                    'valid': False,
                    'correctness': correctness_results,
                    'performance': {}
                }
            
            # Run performance tests
            performance_results = self._test_performance(
                original_algorithm, candidate_func, test_cases
            )
            
            return {
                'valid': True,
                'correctness': correctness_results,
                'performance': performance_results
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _test_correctness(self, 
                         original_func: Callable,
                         candidate_func: Callable,
                         test_cases: List[Any]) -> Dict[str, Any]:
        """Test correctness by comparing outputs."""
        
        passed_tests = 0
        total_tests = len(test_cases)
        errors = []
        
        for i, test_case in enumerate(test_cases):
            try:
                original_result = original_func(test_case)
                candidate_result = candidate_func(test_case)
                
                # Compare results (with tolerance for floating point)
                if isinstance(original_result, (int, float)):
                    if abs(original_result - candidate_result) < 1e-6:
                        passed_tests += 1
                    else:
                        errors.append(f"Test {i}: Expected {original_result}, got {candidate_result}")
                elif np.allclose(original_result, candidate_result, rtol=1e-5):
                    passed_tests += 1
                else:
                    errors.append(f"Test {i}: Results differ significantly")
                    
            except Exception as e:
                errors.append(f"Test {i}: Exception - {str(e)}")
        
        return {
            'passed': passed_tests == total_tests,
            'pass_rate': passed_tests / total_tests,
            'errors': errors[:5]  # Keep only first 5 errors
        }
    
    def _test_performance(self, 
                         original_func: Callable,
                         candidate_func: Callable,
                         test_cases: List[Any]) -> Dict[str, float]:
        """Test performance improvements."""
        
        # Warm up
        if test_cases:
            original_func(test_cases[0])
            candidate_func(test_cases[0])
        
        # Measure original performance
        original_times = []
        for test_case in test_cases:
            start_time = time.perf_counter()
            original_func(test_case)
            original_times.append(time.perf_counter() - start_time)
        
        # Measure candidate performance
        candidate_times = []
        for test_case in test_cases:
            start_time = time.perf_counter()
            candidate_func(test_case)
            candidate_times.append(time.perf_counter() - start_time)
        
        original_avg = np.mean(original_times)
        candidate_avg = np.mean(candidate_times)
        
        speedup = original_avg / candidate_avg if candidate_avg > 0 else 0.0
        improvement = (original_avg - candidate_avg) / original_avg if original_avg > 0 else 0.0
        
        return {
            'original_time': original_avg,
            'candidate_time': candidate_avg,
            'speedup': speedup,
            'improvement': improvement,
            'original_std': np.std(original_times),
            'candidate_std': np.std(candidate_times)
        }


class SelfImprovingHDC:
    """Self-improving HDC system that evolves its own algorithms."""
    
    def __init__(self, 
                 dimension: int = 10000,
                 improvement_threshold: float = 0.05,
                 max_versions: int = 100):
        self.dimension = dimension
        self.improvement_threshold = improvement_threshold
        self.max_versions = max_versions
        
        # Algorithm version management
        self.algorithm_versions: Dict[str, List[AlgorithmVersion]] = {}
        self.current_versions: Dict[str, str] = {}
        
        # Improvement infrastructure
        self.mutator = AlgorithmMutator()
        self.tester = AlgorithmTester()
        
        # Performance monitoring
        self.performance_history: Dict[str, List[float]] = {}
        self.improvement_history: List[Dict[str, Any]] = []
        
        # Core HDC components
        self.memory = AssociativeMemory(dimension)
        self.operations = HDCOperations(dimension)
        
        # Self-improvement thread
        self.improvement_thread = None
        self.is_improving = False
        self.improvement_interval = 300  # 5 minutes
        
    def register_algorithm(self, 
                          name: str,
                          algorithm_func: Callable,
                          test_cases: List[Any] = None) -> str:
        """Register an algorithm for self-improvement."""
        
        # Get source code
        source_code = inspect.getsource(algorithm_func)
        source_code = textwrap.dedent(source_code)
        
        # Create initial version
        version = AlgorithmVersion(
            version_id=f"{name}_v1.0",
            algorithm_code=source_code,
            parent_version=None
        )
        
        # Initialize tracking
        if name not in self.algorithm_versions:
            self.algorithm_versions[name] = []
        
        self.algorithm_versions[name].append(version)
        self.current_versions[name] = version.version_id
        
        if name not in self.performance_history:
            self.performance_history[name] = []
        
        print(f"📝 Registered algorithm '{name}' for self-improvement")
        return version.version_id
    
    def start_continuous_improvement(self):
        """Start continuous self-improvement process."""
        if self.is_improving:
            return
            
        self.is_improving = True
        self.improvement_thread = threading.Thread(
            target=self._improvement_loop,
            daemon=True
        )
        self.improvement_thread.start()
        print("🔄 Started continuous self-improvement")
    
    def stop_continuous_improvement(self):
        """Stop continuous self-improvement process."""
        self.is_improving = False
        if self.improvement_thread:
            self.improvement_thread.join(timeout=5.0)
        print("⏹️ Stopped continuous self-improvement")
    
    def _improvement_loop(self):
        """Main self-improvement loop."""
        while self.is_improving:
            try:
                for algorithm_name in self.algorithm_versions.keys():
                    self._attempt_improvement(algorithm_name)
                
                time.sleep(self.improvement_interval)
                
            except Exception as e:
                print(f"❌ Error in improvement loop: {e}")
                time.sleep(60)  # Wait before retrying
    
    def _attempt_improvement(self, algorithm_name: str):
        """Attempt to improve a specific algorithm."""
        
        current_version_id = self.current_versions.get(algorithm_name)
        if not current_version_id:
            return
            
        # Get current version
        current_version = None
        for version in self.algorithm_versions[algorithm_name]:
            if version.version_id == current_version_id:
                current_version = version
                break
        
        if not current_version:
            return
        
        # Check if improvement is needed
        recent_performance = self.performance_history.get(algorithm_name, [])
        if len(recent_performance) < 10:
            return  # Not enough data
        
        performance_trend = np.polyfit(range(len(recent_performance)), recent_performance, 1)[0]
        if performance_trend > -0.001:  # Performance not degrading
            return
        
        print(f"🔬 Attempting improvement for {algorithm_name}")
        
        # Generate improvement candidates
        performance_data = {
            'avg_time': np.mean(recent_performance),
            'std_time': np.std(recent_performance),
            'trend': performance_trend
        }
        
        candidates = self.mutator.generate_candidates(
            current_version.algorithm_code,
            performance_data,
            num_candidates=3
        )
        
        # Test candidates
        best_candidate = None
        best_improvement = 0.0
        
        for candidate in candidates:
            # Compile and test candidate
            test_results = self._test_candidate(algorithm_name, candidate)
            
            if test_results.get('valid', False):
                improvement = test_results['performance'].get('improvement', 0.0)
                if improvement > best_improvement and improvement > self.improvement_threshold:
                    best_candidate = candidate
                    best_improvement = improvement
        
        # Apply best improvement
        if best_candidate:
            self._apply_improvement(algorithm_name, best_candidate, best_improvement)
    
    def _test_candidate(self, 
                       algorithm_name: str,
                       candidate: ImprovementCandidate) -> Dict[str, Any]:
        """Test an improvement candidate."""
        
        try:
            # Create test function from current version
            current_version_id = self.current_versions[algorithm_name]
            current_version = None
            
            for version in self.algorithm_versions[algorithm_name]:
                if version.version_id == current_version_id:
                    current_version = version
                    break
            
            if not current_version:
                return {'valid': False, 'error': 'Current version not found'}
            
            # Compile original algorithm
            exec_globals = {'np': np, 'time': time, 'HyperVector': HyperVector}
            exec(current_version.algorithm_code, exec_globals)
            
            # Find the main function
            original_func = None
            for name, obj in exec_globals.items():
                if callable(obj) and hasattr(obj, '__code__'):
                    original_func = obj
                    break
            
            if not original_func:
                return {'valid': False, 'error': 'Original function not found'}
            
            # Generate test cases
            test_cases = self._generate_test_cases(algorithm_name)
            
            # Test the candidate
            return self.tester.validate_candidate(candidate, original_func, test_cases)
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _generate_test_cases(self, algorithm_name: str) -> List[Any]:
        """Generate test cases for an algorithm."""
        # This is a simplified version - in practice, would be more sophisticated
        
        test_cases = []
        
        # Generate various input scenarios
        for _ in range(10):
            if 'hdc' in algorithm_name.lower():
                # HDC-specific test case
                test_case = {
                    'hypervector': HyperVector.random(self.dimension),
                    'query': HyperVector.random(self.dimension),
                    'threshold': np.random.uniform(0.1, 0.9)
                }
            else:
                # Generic numerical test case
                test_case = np.random.randn(np.random.randint(10, 1000))
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _apply_improvement(self, 
                          algorithm_name: str,
                          candidate: ImprovementCandidate,
                          improvement: float):
        """Apply a validated improvement."""
        
        current_version_id = self.current_versions[algorithm_name]
        
        # Create new version
        version_number = len(self.algorithm_versions[algorithm_name]) + 1
        new_version = AlgorithmVersion(
            version_id=f"{algorithm_name}_v{version_number/10:.1f}",
            algorithm_code=candidate.modified_code,
            parent_version=current_version_id,
            improvement_factor=improvement
        )
        
        self.algorithm_versions[algorithm_name].append(new_version)
        self.current_versions[algorithm_name] = new_version.version_id
        
        # Log improvement
        improvement_record = {
            'timestamp': time.time(),
            'algorithm': algorithm_name,
            'old_version': current_version_id,
            'new_version': new_version.version_id,
            'improvement_factor': improvement,
            'modification_type': candidate.modification_type
        }
        
        self.improvement_history.append(improvement_record)
        
        print(f"✅ Applied improvement to {algorithm_name}: "
              f"{improvement:.2%} performance gain")
        
        # Clean up old versions if needed
        if len(self.algorithm_versions[algorithm_name]) > self.max_versions:
            self.algorithm_versions[algorithm_name] = \
                self.algorithm_versions[algorithm_name][-self.max_versions:]
    
    def get_algorithm(self, name: str) -> Optional[str]:
        """Get the current version of an algorithm."""
        version_id = self.current_versions.get(name)
        if not version_id:
            return None
            
        for version in self.algorithm_versions[name]:
            if version.version_id == version_id:
                return version.algorithm_code
        
        return None
    
    def get_improvement_statistics(self) -> Dict[str, Any]:
        """Get statistics about self-improvement process."""
        
        total_improvements = len(self.improvement_history)
        
        if total_improvements == 0:
            return {
                'total_improvements': 0,
                'algorithms_improved': 0,
                'average_improvement': 0.0,
                'improvement_rate': 0.0
            }
        
        # Calculate statistics
        algorithms_improved = len(set(record['algorithm'] for record in self.improvement_history))
        
        improvements = [record['improvement_factor'] for record in self.improvement_history]
        average_improvement = np.mean(improvements)
        
        # Improvement rate (improvements per day)
        if len(self.improvement_history) > 1:
            time_span = (self.improvement_history[-1]['timestamp'] - 
                        self.improvement_history[0]['timestamp'])
            improvement_rate = total_improvements / (time_span / 86400)  # per day
        else:
            improvement_rate = 0.0
        
        return {
            'total_improvements': total_improvements,
            'algorithms_improved': algorithms_improved,
            'average_improvement': average_improvement,
            'improvement_rate': improvement_rate,
            'improvement_history': self.improvement_history[-10:],  # Last 10
            'algorithm_versions': {
                name: len(versions) 
                for name, versions in self.algorithm_versions.items()
            }
        }
    
    def save_state(self, filepath: str):
        """Save the complete state of the self-improving system."""
        
        state = {
            'dimension': self.dimension,
            'improvement_threshold': self.improvement_threshold,
            'max_versions': self.max_versions,
            'algorithm_versions': {
                name: [
                    {
                        'version_id': v.version_id,
                        'algorithm_code': v.algorithm_code,
                        'performance_metrics': v.performance_metrics,
                        'creation_time': v.creation_time,
                        'parent_version': v.parent_version,
                        'improvement_factor': v.improvement_factor
                    }
                    for v in versions
                ]
                for name, versions in self.algorithm_versions.items()
            },
            'current_versions': self.current_versions,
            'improvement_history': self.improvement_history,
            'performance_history': self.performance_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
            
        print(f"💾 Saved self-improving HDC state to {filepath}")


class AlgorithmicEvolution:
    """Meta-algorithm for evolving algorithmic structures."""
    
    def __init__(self):
        self.algorithm_templates = {}
        self.evolution_strategies = [
            'combine_algorithms',
            'decompose_algorithm', 
            'add_meta_layer',
            'optimize_flow',
            'hybridize_approaches'
        ]
    
    def evolve_algorithm_structure(self, 
                                  base_algorithms: List[str],
                                  target_performance: Dict[str, float]) -> str:
        """Evolve new algorithmic structures from existing ones."""
        
        # This is a conceptual implementation
        # In practice, would use more sophisticated program synthesis
        
        evolved_code = f"""
# Evolved algorithm combining: {', '.join(base_algorithms)}
# Target performance: {target_performance}

def evolved_algorithm(input_data):
    # Multi-stage processing pipeline
    stage1_result = base_algorithm_1(input_data)
    stage2_result = base_algorithm_2(stage1_result)
    
    # Adaptive selection based on input characteristics
    if should_use_parallel(input_data):
        return parallel_process(stage2_result)
    else:
        return sequential_process(stage2_result)

def should_use_parallel(data):
    # Learned decision boundary
    return len(data) > optimal_threshold

def parallel_process(data):
    # Generated parallel implementation
    pass

def sequential_process(data):
    # Generated sequential implementation
    pass
"""
        
        return evolved_code


# Example usage and demonstration
if __name__ == "__main__":
    
    def example_hdc_query(query_data):
        """Example HDC algorithm to be improved."""
        dimension = 10000
        hv = HyperVector.random(dimension)
        
        # Simulate some computation
        result = 0.0
        for i in range(100):
            similarity = hv.similarity(HyperVector.random(dimension))
            result += similarity * np.sin(i * 0.1)
            
        return result
    
    # Create self-improving system
    self_improving = SelfImprovingHDC(dimension=10000)
    
    # Register algorithm for improvement
    self_improving.register_algorithm("hdc_query", example_hdc_query)
    
    # Start continuous improvement
    self_improving.start_continuous_improvement()
    
    print("🧠 Self-improving HDC system started")
    print("System will continuously monitor and improve algorithms...")
    
    # Simulate some usage to generate performance data
    for i in range(50):
        example_hdc_query({"test": i})
        self_improving.performance_history["hdc_query"].append(
            np.random.uniform(0.1, 0.3)  # Simulate execution time
        )
        time.sleep(0.1)
    
    # Get improvement statistics
    stats = self_improving.get_improvement_statistics()
    print("\n📊 Improvement Statistics:")
    for key, value in stats.items():
        if key != 'improvement_history':
            print(f"  {key}: {value}")
    
    # Save state
    self_improving.save_state("self_improving_hdc_state.json")
    
    # Stop improvement
    self_improving.stop_continuous_improvement()