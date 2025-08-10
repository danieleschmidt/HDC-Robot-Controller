"""
Self-Modifying Code Engine

Enables autonomous code generation, modification, and optimization based on
runtime performance analysis and evolutionary programming principles.
"""

import ast
import inspect
import textwrap
import importlib
import sys
import os
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import hashlib
import subprocess

from ..core.hypervector import HyperVector


@dataclass
class CodeFragment:
    """Represents a modifiable code fragment."""
    fragment_id: str
    source_code: str
    ast_node: ast.AST
    performance_metrics: Dict[str, float]
    modification_history: List[Dict[str, Any]]
    fitness_score: float = 0.0
    generation: int = 0
    
    def __post_init__(self):
        if not self.modification_history:
            self.modification_history = []


@dataclass
class CodeMutation:
    """Represents a code mutation operation."""
    mutation_type: str  # "replace", "insert", "delete", "transform"
    target_node: str   # AST node type
    mutation_code: str
    expected_improvement: float
    risk_level: float  # 0.0 (safe) to 1.0 (risky)


class ASTAnalyzer:
    """Analyzes and manipulates Abstract Syntax Trees."""
    
    def __init__(self):
        self.node_patterns = {
            'loops': [ast.For, ast.While],
            'conditionals': [ast.If],
            'functions': [ast.FunctionDef, ast.AsyncFunctionDef],
            'classes': [ast.ClassDef],
            'operations': [ast.BinOp, ast.UnaryOp, ast.Compare],
            'assignments': [ast.Assign, ast.AugAssign]
        }
        
    def analyze_complexity(self, code: str) -> Dict[str, int]:
        """Analyze code complexity metrics."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {'error': 1, 'total_nodes': 0}
        
        metrics = {
            'total_nodes': 0,
            'functions': 0,
            'classes': 0,
            'loops': 0,
            'conditionals': 0,
            'depth': 0
        }
        
        def visit_node(node, depth=0):
            metrics['total_nodes'] += 1
            metrics['depth'] = max(metrics['depth'], depth)
            
            for pattern_name, node_types in self.node_patterns.items():
                if any(isinstance(node, node_type) for node_type in node_types):
                    if pattern_name in metrics:
                        metrics[pattern_name] += 1
            
            for child in ast.iter_child_nodes(node):
                visit_node(child, depth + 1)
        
        visit_node(tree)
        return metrics
    
    def find_optimization_candidates(self, code: str) -> List[Dict[str, Any]]:
        """Find code sections that could be optimized."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        
        candidates = []
        
        class OptimizationFinder(ast.NodeVisitor):
            def visit_For(self, node):
                # Look for nested loops
                nested_loops = []
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)) and child != node:
                        nested_loops.append(child)
                
                if nested_loops:
                    candidates.append({
                        'type': 'nested_loops',
                        'line': node.lineno,
                        'optimization': 'vectorization or loop fusion',
                        'risk': 0.3
                    })
                
                self.generic_visit(node)
            
            def visit_If(self, node):
                # Look for deeply nested conditions
                depth = self._get_nesting_depth(node)
                if depth > 3:
                    candidates.append({
                        'type': 'deep_nesting',
                        'line': node.lineno,
                        'optimization': 'guard clauses or lookup tables',
                        'risk': 0.2
                    })
                
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                # Analyze function complexity
                complexity = len(list(ast.walk(node)))
                if complexity > 50:  # Arbitrary threshold
                    candidates.append({
                        'type': 'complex_function',
                        'name': node.name,
                        'line': node.lineno,
                        'optimization': 'function decomposition',
                        'risk': 0.4
                    })
                
                self.generic_visit(node)
            
            def _get_nesting_depth(self, node, depth=0):
                max_depth = depth
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.If):
                        child_depth = self._get_nesting_depth(child, depth + 1)
                        max_depth = max(max_depth, child_depth)
                return max_depth
        
        finder = OptimizationFinder()
        finder.visit(tree)
        
        return candidates
    
    def generate_mutations(self, code: str) -> List[CodeMutation]:
        """Generate possible code mutations."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return []
        
        mutations = []
        
        # Pattern-based mutations
        mutations.extend(self._generate_loop_mutations(tree))
        mutations.extend(self._generate_function_mutations(tree))
        mutations.extend(self._generate_optimization_mutations(tree))
        
        return mutations
    
    def _generate_loop_mutations(self, tree: ast.AST) -> List[CodeMutation]:
        """Generate loop optimization mutations."""
        mutations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Suggest list comprehension
                mutations.append(CodeMutation(
                    mutation_type="transform",
                    target_node="For",
                    mutation_code="# Convert to list comprehension for better performance",
                    expected_improvement=0.2,
                    risk_level=0.1
                ))
                
                # Suggest vectorization
                mutations.append(CodeMutation(
                    mutation_type="replace",
                    target_node="For",
                    mutation_code="# Use numpy vectorization instead of explicit loop",
                    expected_improvement=0.5,
                    risk_level=0.3
                ))
        
        return mutations
    
    def _generate_function_mutations(self, tree: ast.AST) -> List[CodeMutation]:
        """Generate function optimization mutations."""
        mutations = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Suggest memoization
                mutations.append(CodeMutation(
                    mutation_type="insert",
                    target_node="FunctionDef",
                    mutation_code="@functools.lru_cache(maxsize=128)",
                    expected_improvement=0.3,
                    risk_level=0.1
                ))
                
                # Suggest parallel execution
                mutations.append(CodeMutation(
                    mutation_type="transform",
                    target_node="FunctionDef",
                    mutation_code="# Add multiprocessing.Pool for parallel execution",
                    expected_improvement=0.6,
                    risk_level=0.4
                ))
        
        return mutations
    
    def _generate_optimization_mutations(self, tree: ast.AST) -> List[CodeMutation]:
        """Generate general optimization mutations."""
        mutations = []
        
        # Data structure optimizations
        mutations.append(CodeMutation(
            mutation_type="replace",
            target_node="List",
            mutation_code="# Use numpy arrays for numerical computations",
            expected_improvement=0.4,
            risk_level=0.2
        ))
        
        # Algorithm optimizations
        mutations.append(CodeMutation(
            mutation_type="transform",
            target_node="BinOp",
            mutation_code="# Use bitwise operations for power-of-2 operations",
            expected_improvement=0.1,
            risk_level=0.1
        ))
        
        return mutations
    
    def apply_mutation(self, code: str, mutation: CodeMutation) -> str:
        """Apply a mutation to code."""
        lines = code.split('\n')
        
        if mutation.mutation_type == "insert":
            # Simple insertion at the beginning
            return mutation.mutation_code + '\n' + code
        
        elif mutation.mutation_type == "replace":
            # Add mutation as comment for now (safe approach)
            return f"# MUTATION: {mutation.mutation_code}\n" + code
        
        elif mutation.mutation_type == "transform":
            # Add transformation suggestion
            return f"# OPTIMIZATION SUGGESTION: {mutation.mutation_code}\n" + code
        
        return code


class PerformanceProfiler:
    """Profiles code performance for optimization decisions."""
    
    def __init__(self):
        self.profile_history = {}
        self.benchmark_results = {}
        
    def profile_function(self, func: Callable, *args, **kwargs) -> Dict[str, float]:
        """Profile a function's performance."""
        import cProfile
        import pstats
        import io
        from contextlib import redirect_stdout
        
        # Time-based profiling
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Basic metrics
            metrics = {
                'execution_time': end_time - start_time,
                'memory_delta': end_memory - start_memory,
                'success': True
            }
            
            # Detailed profiling with cProfile
            pr = cProfile.Profile()
            pr.enable()
            
            # Re-run for detailed profiling
            func(*args, **kwargs)
            
            pr.disable()
            
            # Capture profile stats
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s)
            ps.sort_stats('cumulative')
            
            metrics.update({
                'function_calls': ps.total_calls,
                'primitive_calls': ps.prim_calls,
                'profile_data': s.getvalue()
            })
            
        except Exception as e:
            metrics = {
                'execution_time': float('inf'),
                'memory_delta': 0,
                'success': False,
                'error': str(e)
            }
        
        return metrics
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def benchmark_code_variants(self, code_variants: List[str], 
                               test_inputs: List[Tuple]) -> Dict[str, Dict[str, float]]:
        """Benchmark multiple code variants."""
        results = {}
        
        for i, code in enumerate(code_variants):
            variant_id = f"variant_{i}"
            results[variant_id] = []
            
            # Try to execute each variant
            try:
                # Create temporary function from code
                exec_globals = {}
                exec(code, exec_globals)
                
                # Find the main function (assume it's the last defined function)
                main_func = None
                for name, obj in exec_globals.items():
                    if callable(obj) and not name.startswith('_'):
                        main_func = obj
                
                if main_func:
                    # Benchmark with different inputs
                    for test_input in test_inputs:
                        metrics = self.profile_function(main_func, *test_input)
                        results[variant_id].append(metrics)
                
            except Exception as e:
                results[variant_id] = [{'error': str(e), 'success': False}]
        
        return results


class SelfModifyingCodeEngine:
    """
    Self-Modifying Code Engine.
    
    Enables autonomous code generation, optimization, and evolution based on
    runtime performance analysis and genetic programming principles.
    """
    
    def __init__(self, 
                 hdc_dimension: int = 10000,
                 safety_level: float = 0.8,
                 max_generations: int = 50):
        """
        Initialize self-modifying code engine.
        
        Args:
            hdc_dimension: Dimension for HDC encoding of code
            safety_level: Safety threshold for accepting mutations (0-1)
            max_generations: Maximum evolution generations
        """
        self.hdc_dimension = hdc_dimension
        self.safety_level = safety_level
        self.max_generations = max_generations
        
        # Core components
        self.ast_analyzer = ASTAnalyzer()
        self.profiler = PerformanceProfiler()
        
        # Code management
        self.code_fragments = {}  # fragment_id -> CodeFragment
        self.active_functions = {}  # function_name -> compiled function
        self.performance_history = {}
        
        # Evolution state
        self.generation_counter = 0
        self.mutation_success_rate = 0.0
        self.evolutionary_pressure = 1.0
        
        # Safety mechanisms
        self.backup_code = {}  # Store original code for rollback
        self.quarantine = set()  # Quarantined unsafe mutations
        
        # Performance tracking
        self.modification_metrics = {
            'total_modifications': 0,
            'successful_optimizations': 0,
            'failed_mutations': 0,
            'rollbacks_performed': 0,
            'average_improvement': 0.0,
            'safety_violations': 0
        }
        
    def register_function(self, func: Callable, 
                         optimization_target: str = "speed") -> str:
        """
        Register a function for self-modification.
        
        Args:
            func: Function to register
            optimization_target: "speed", "memory", or "accuracy"
            
        Returns:
            Fragment ID for tracking
        """
        # Get source code
        try:
            source = inspect.getsource(func)
            source = textwrap.dedent(source)
        except OSError:
            raise ValueError(f"Cannot get source code for function {func.__name__}")
        
        # Create fragment ID
        fragment_id = f"{func.__name__}_{hashlib.md5(source.encode()).hexdigest()[:8]}"
        
        # Analyze the code
        complexity_metrics = self.ast_analyzer.analyze_complexity(source)
        
        # Create code fragment
        fragment = CodeFragment(
            fragment_id=fragment_id,
            source_code=source,
            ast_node=ast.parse(source),
            performance_metrics={
                'complexity': complexity_metrics,
                'optimization_target': optimization_target
            },
            modification_history=[]
        )
        
        self.code_fragments[fragment_id] = fragment
        self.backup_code[fragment_id] = source
        
        # Initial performance baseline
        baseline_metrics = self.profiler.profile_function(func)
        fragment.performance_metrics.update(baseline_metrics)
        
        return fragment_id
    
    def evolve_code(self, fragment_id: str, 
                   test_inputs: List[Tuple],
                   generations: Optional[int] = None) -> Dict[str, Any]:
        """
        Evolve code through multiple generations of mutations.
        
        Args:
            fragment_id: ID of code fragment to evolve
            test_inputs: Test inputs for benchmarking
            generations: Number of generations (uses default if None)
            
        Returns:
            Evolution results and statistics
        """
        if fragment_id not in self.code_fragments:
            raise ValueError(f"Fragment {fragment_id} not found")
        
        generations = generations or self.max_generations
        fragment = self.code_fragments[fragment_id]
        
        evolution_stats = {
            'generation_results': [],
            'best_fitness_history': [],
            'mutation_history': [],
            'final_improvement': 0.0
        }
        
        # Initial population
        population = [fragment.source_code]
        fitness_scores = [self._evaluate_fitness(fragment.source_code, test_inputs)]
        
        best_code = fragment.source_code
        best_fitness = fitness_scores[0]
        
        for generation in range(generations):
            generation_start = time.time()
            
            # Generate mutations
            new_population = []
            new_fitness_scores = []
            
            for code in population:
                mutations = self.ast_analyzer.generate_mutations(code)
                
                # Apply mutations with safety checks
                for mutation in mutations:
                    if mutation.risk_level <= (1.0 - self.safety_level):
                        try:
                            mutated_code = self.ast_analyzer.apply_mutation(code, mutation)
                            
                            # Safety check: can we parse it?
                            ast.parse(mutated_code)
                            
                            fitness = self._evaluate_fitness(mutated_code, test_inputs)
                            
                            if fitness > best_fitness * 0.9:  # Keep if reasonably good
                                new_population.append(mutated_code)
                                new_fitness_scores.append(fitness)
                                
                                # Track mutation
                                evolution_stats['mutation_history'].append({
                                    'generation': generation,
                                    'mutation_type': mutation.mutation_type,
                                    'fitness_change': fitness - best_fitness,
                                    'accepted': fitness > best_fitness
                                })
                                
                                if fitness > best_fitness:
                                    best_code = mutated_code
                                    best_fitness = fitness
                            
                        except (SyntaxError, Exception) as e:
                            # Mutation failed safety check
                            self.quarantine.add(mutation.mutation_code)
                            self.modification_metrics['safety_violations'] += 1
            
            # Selection: keep best performing variants
            if new_population:
                # Combine old and new population
                all_codes = population + new_population
                all_fitness = fitness_scores + new_fitness_scores
                
                # Sort by fitness and select top performers
                sorted_population = sorted(zip(all_codes, all_fitness), 
                                         key=lambda x: x[1], reverse=True)
                
                population_size = min(10, len(sorted_population))  # Keep top 10
                population = [code for code, _ in sorted_population[:population_size]]
                fitness_scores = [fitness for _, fitness in sorted_population[:population_size]]
            
            # Track generation statistics
            generation_time = time.time() - generation_start
            evolution_stats['generation_results'].append({
                'generation': generation,
                'population_size': len(population),
                'best_fitness': max(fitness_scores) if fitness_scores else 0,
                'average_fitness': np.mean(fitness_scores) if fitness_scores else 0,
                'generation_time': generation_time
            })
            
            evolution_stats['best_fitness_history'].append(best_fitness)
            
            # Early stopping if no improvement
            if len(evolution_stats['best_fitness_history']) > 10:
                recent_improvements = np.diff(evolution_stats['best_fitness_history'][-10:])
                if np.all(recent_improvements <= 0.01):  # Minimal improvement
                    break
        
        # Update the fragment with best code
        if best_fitness > fragment.fitness_score:
            old_fitness = fragment.fitness_score
            fragment.source_code = best_code
            fragment.fitness_score = best_fitness
            fragment.generation += 1
            
            # Add to modification history
            fragment.modification_history.append({
                'generation': self.generation_counter,
                'old_fitness': old_fitness,
                'new_fitness': best_fitness,
                'improvement': best_fitness - old_fitness,
                'timestamp': time.time()
            })
            
            # Update metrics
            self.modification_metrics['successful_optimizations'] += 1
            self.modification_metrics['total_modifications'] += 1
            
            improvement = best_fitness - old_fitness
            evolution_stats['final_improvement'] = improvement
            
            # Update running average
            current_avg = self.modification_metrics['average_improvement']
            total_mods = self.modification_metrics['total_modifications']
            self.modification_metrics['average_improvement'] = (
                (current_avg * (total_mods - 1) + improvement) / total_mods
            )
        
        self.generation_counter += 1
        
        return evolution_stats
    
    def _evaluate_fitness(self, code: str, test_inputs: List[Tuple]) -> float:
        """Evaluate fitness of code based on performance metrics."""
        try:
            # Try to execute the code
            exec_globals = {}
            exec(code, exec_globals)
            
            # Find the main function
            main_func = None
            for name, obj in exec_globals.items():
                if callable(obj) and not name.startswith('_'):
                    main_func = obj
                    break
            
            if not main_func:
                return 0.0
            
            # Profile performance
            total_time = 0.0
            total_memory = 0.0
            success_count = 0
            
            for test_input in test_inputs[:5]:  # Limit to 5 tests for speed
                try:
                    metrics = self.profiler.profile_function(main_func, *test_input)
                    
                    if metrics.get('success', False):
                        total_time += metrics.get('execution_time', 0)
                        total_memory += abs(metrics.get('memory_delta', 0))
                        success_count += 1
                    
                except Exception:
                    continue
            
            if success_count == 0:
                return 0.0
            
            # Calculate fitness (higher is better)
            avg_time = total_time / success_count
            avg_memory = total_memory / success_count
            
            # Fitness function: prioritize speed, penalize memory usage
            time_score = max(0, 10.0 / (avg_time + 0.001))  # Higher for faster
            memory_score = max(0, 5.0 / (avg_memory + 1.0))  # Higher for lower memory
            success_score = success_count / len(test_inputs) * 10  # Success rate
            
            fitness = time_score + memory_score + success_score
            
            return fitness
            
        except Exception:
            return 0.0
    
    def deploy_optimized_code(self, fragment_id: str) -> bool:
        """Deploy optimized code to replace the original function."""
        if fragment_id not in self.code_fragments:
            return False
        
        fragment = self.code_fragments[fragment_id]
        
        try:
            # Compile the optimized code
            exec_globals = {}
            exec(fragment.source_code, exec_globals)
            
            # Find the function
            for name, obj in exec_globals.items():
                if callable(obj) and not name.startswith('_'):
                    self.active_functions[name] = obj
                    return True
            
            return False
            
        except Exception as e:
            # Deployment failed, rollback
            self._rollback_fragment(fragment_id)
            self.modification_metrics['rollbacks_performed'] += 1
            return False
    
    def _rollback_fragment(self, fragment_id: str):
        """Rollback a fragment to its backup code."""
        if fragment_id in self.backup_code:
            original_code = self.backup_code[fragment_id]
            self.code_fragments[fragment_id].source_code = original_code
    
    def generate_new_function(self, 
                             specification: str,
                             example_inputs: List[Tuple],
                             example_outputs: List[Any]) -> str:
        """
        Generate a new function based on specifications and examples.
        
        Args:
            specification: Natural language description
            example_inputs: Example input tuples
            example_outputs: Expected outputs
            
        Returns:
            Generated function code
        """
        # This is a simplified version - in practice, would use more advanced
        # techniques like program synthesis or neural code generation
        
        # Analyze patterns in examples
        input_patterns = self._analyze_input_patterns(example_inputs)
        output_patterns = self._analyze_output_patterns(example_outputs)
        
        # Generate basic function template
        function_template = self._generate_function_template(
            specification, input_patterns, output_patterns
        )
        
        # Evolve the function
        fragment_id = f"generated_{int(time.time())}"
        
        # Create a temporary fragment
        fragment = CodeFragment(
            fragment_id=fragment_id,
            source_code=function_template,
            ast_node=ast.parse(function_template),
            performance_metrics={},
            modification_history=[]
        )
        
        self.code_fragments[fragment_id] = fragment
        
        # Evolve it
        test_inputs = [(inp,) for inp in example_inputs]
        evolution_results = self.evolve_code(fragment_id, test_inputs, generations=20)
        
        return self.code_fragments[fragment_id].source_code
    
    def _analyze_input_patterns(self, inputs: List[Tuple]) -> Dict[str, Any]:
        """Analyze patterns in input data."""
        if not inputs:
            return {}
        
        patterns = {
            'input_count': len(inputs[0]) if inputs else 0,
            'types': [],
            'ranges': {}
        }
        
        # Analyze each input position
        for i in range(patterns['input_count']):
            values = [inp[i] for inp in inputs if len(inp) > i]
            
            if values:
                value_type = type(values[0]).__name__
                patterns['types'].append(value_type)
                
                if all(isinstance(v, (int, float)) for v in values):
                    patterns['ranges'][i] = {
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values)
                    }
        
        return patterns
    
    def _analyze_output_patterns(self, outputs: List[Any]) -> Dict[str, Any]:
        """Analyze patterns in output data."""
        if not outputs:
            return {}
        
        patterns = {
            'output_type': type(outputs[0]).__name__ if outputs else 'unknown',
            'unique_values': len(set(str(o) for o in outputs)),
            'is_sequence': isinstance(outputs[0], (list, tuple)) if outputs else False
        }
        
        if all(isinstance(o, (int, float)) for o in outputs):
            patterns['numeric_range'] = {
                'min': min(outputs),
                'max': max(outputs),
                'avg': sum(outputs) / len(outputs)
            }
        
        return patterns
    
    def _generate_function_template(self, 
                                   specification: str,
                                   input_patterns: Dict[str, Any],
                                   output_patterns: Dict[str, Any]) -> str:
        """Generate a basic function template."""
        
        # Create parameter list
        param_count = input_patterns.get('input_count', 1)
        param_names = [f"param_{i}" for i in range(param_count)]
        param_list = ", ".join(param_names)
        
        # Determine return type hint
        output_type = output_patterns.get('output_type', 'Any')
        type_hint = f" -> {output_type}" if output_type != 'unknown' else ""
        
        # Generate basic function body
        if 'sum' in specification.lower() or 'add' in specification.lower():
            body = f"    return sum([{', '.join(param_names)}])"
        elif 'multiply' in specification.lower() or 'product' in specification.lower():
            body = f"    result = 1\\n    for val in [{', '.join(param_names)}]:\\n        result *= val\\n    return result"
        elif 'max' in specification.lower() or 'maximum' in specification.lower():
            body = f"    return max([{', '.join(param_names)}])"
        elif 'min' in specification.lower() or 'minimum' in specification.lower():
            body = f"    return min([{', '.join(param_names)}])"
        else:
            # Generic template
            body = f"    # TODO: Implement based on specification: {specification}\\n    return {param_names[0]} if param_names else None"
        
        template = f"""def generated_function({param_list}){type_hint}:
    \"\"\"
    Generated function: {specification}
    \"\"\"
{body}
"""
        
        return template
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            'modification_metrics': self.modification_metrics.copy(),
            'fragment_summary': {},
            'performance_improvements': [],
            'safety_analysis': {
                'quarantined_mutations': len(self.quarantine),
                'safety_violations': self.modification_metrics['safety_violations'],
                'rollback_rate': 0.0
            },
            'evolution_statistics': {
                'total_generations': self.generation_counter,
                'mutation_success_rate': self.mutation_success_rate,
                'evolutionary_pressure': self.evolutionary_pressure
            }
        }
        
        # Fragment analysis
        for frag_id, fragment in self.code_fragments.items():
            report['fragment_summary'][frag_id] = {
                'fitness_score': fragment.fitness_score,
                'generation': fragment.generation,
                'modification_count': len(fragment.modification_history),
                'total_improvement': sum(
                    mod['improvement'] for mod in fragment.modification_history
                )
            }
            
            # Track individual improvements
            for mod in fragment.modification_history:
                report['performance_improvements'].append({
                    'fragment_id': frag_id,
                    'improvement': mod['improvement'],
                    'generation': mod['generation']
                })
        
        # Safety analysis
        total_mods = self.modification_metrics['total_modifications']
        rollbacks = self.modification_metrics['rollbacks_performed']
        if total_mods > 0:
            report['safety_analysis']['rollback_rate'] = rollbacks / total_mods
        
        return report
    
    def save_evolved_code(self, output_directory: str):
        """Save all evolved code to files."""
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save each fragment
        for frag_id, fragment in self.code_fragments.items():
            file_path = output_path / f"{frag_id}.py"
            
            with open(file_path, 'w') as f:
                f.write(f"# Evolved code fragment: {frag_id}\\n")
                f.write(f"# Generation: {fragment.generation}\\n")
                f.write(f"# Fitness score: {fragment.fitness_score}\\n")
                f.write(f"# Modifications: {len(fragment.modification_history)}\\n\\n")
                f.write(fragment.source_code)
        
        # Save optimization report
        import json
        report = self.get_optimization_report()
        
        with open(output_path / "optimization_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save backup code
        backup_path = output_path / "backups"
        backup_path.mkdir(exist_ok=True)
        
        for frag_id, code in self.backup_code.items():
            with open(backup_path / f"{frag_id}_original.py", 'w') as f:
                f.write(code)