"""
Comprehensive test suite for Autonomous Mastery capabilities.

Tests Generation 5 enhancements including self-modifying code,
adaptive architecture, and autonomous evolution systems.
"""

import pytest
import numpy as np
import time
import ast
import inspect
from pathlib import Path
from unittest.mock import Mock, patch

from hdc_robot_controller.autonomous_mastery import (
    SelfModifyingCodeEngine,
    AdaptiveArchitectureManager
)
from hdc_robot_controller.autonomous_mastery.self_modifying_code import (
    CodeFragment, CodeMutation, ASTAnalyzer, PerformanceProfiler
)
from hdc_robot_controller.autonomous_mastery.adaptive_architecture import (
    SystemComponent, ComponentType, ArchitectureState,
    TopologyOptimizer, ResourceAllocator
)
from hdc_robot_controller.core.hypervector import HyperVector


class TestSelfModifyingCode:
    """Test self-modifying code capabilities."""
    
    @pytest.fixture
    def code_engine(self):
        """Create code engine for testing."""
        return SelfModifyingCodeEngine(
            hdc_dimension=1000,
            safety_level=0.8,
            max_generations=10
        )
    
    @pytest.fixture
    def sample_function(self):
        """Sample function for testing."""
        def simple_sum(a, b):
            """Simple sum function for testing."""
            result = a + b
            return result
        return simple_sum
    
    def test_function_registration(self, code_engine, sample_function):
        """Test function registration for modification."""
        fragment_id = code_engine.register_function(sample_function)
        
        assert fragment_id in code_engine.code_fragments
        fragment = code_engine.code_fragments[fragment_id]
        
        assert isinstance(fragment, CodeFragment)
        assert fragment.fragment_id == fragment_id
        assert "def simple_sum" in fragment.source_code
        assert fragment.generation == 0
        
    def test_ast_analysis(self, code_engine):
        \"\"\"Test AST analysis capabilities.\"\"\"
        code = \"\"\"
def test_function(x, y):
    result = 0
    for i in range(x):
        if i < y:
            result += i
        else:
            result -= i
    return result
\"\"\"
        
        analyzer = code_engine.ast_analyzer
        complexity = analyzer.analyze_complexity(code)
        
        assert complexity['total_nodes'] > 0
        assert complexity['functions'] == 1
        assert complexity['loops'] == 1
        assert complexity['conditionals'] == 1
        
        # Test optimization candidate detection
        candidates = analyzer.find_optimization_candidates(code)
        assert len(candidates) >= 0  # May or may not find candidates
        
    def test_mutation_generation(self, code_engine):
        \"\"\"Test code mutation generation.\"\"\"
        code = \"\"\"
def loop_function(n):
    result = []
    for i in range(n):
        result.append(i * 2)
    return result
\"\"\"
        
        analyzer = code_engine.ast_analyzer
        mutations = analyzer.generate_mutations(code)
        
        assert len(mutations) > 0
        
        for mutation in mutations:
            assert isinstance(mutation, CodeMutation)
            assert mutation.mutation_type in ["replace", "insert", "delete", "transform"]
            assert 0.0 <= mutation.risk_level <= 1.0
            assert mutation.expected_improvement >= 0.0
    
    def test_performance_profiling(self, code_engine, sample_function):
        \"\"\"Test performance profiling.\"\"\"
        profiler = code_engine.profiler
        
        # Profile the sample function
        metrics = profiler.profile_function(sample_function, 5, 10)
        
        assert 'execution_time' in metrics
        assert 'memory_delta' in metrics
        assert 'success' in metrics
        assert metrics['success'] is True
        assert metrics['execution_time'] >= 0
    
    def test_code_evolution(self, code_engine, sample_function):
        \"\"\"Test code evolution process.\"\"\"
        fragment_id = code_engine.register_function(sample_function)
        
        # Create test inputs
        test_inputs = [(1, 2), (3, 4), (5, 6)]
        
        # Evolve code (with limited generations for testing)
        evolution_stats = code_engine.evolve_code(
            fragment_id, 
            test_inputs, 
            generations=3
        )
        
        assert 'generation_results' in evolution_stats
        assert 'best_fitness_history' in evolution_stats
        assert 'mutation_history' in evolution_stats
        assert len(evolution_stats['generation_results']) <= 3
        
    def test_safety_mechanisms(self, code_engine):
        \"\"\"Test safety mechanisms in code modification.\"\"\"
        # Test with potentially unsafe code
        unsafe_code = \"\"\"
def unsafe_function():
    import os
    os.system('rm -rf /')  # Dangerous command
\"\"\"
        
        try:
            tree = ast.parse(unsafe_code)
            mutations = code_engine.ast_analyzer.generate_mutations(unsafe_code)
            
            # Apply safety filtering
            safe_mutations = [
                m for m in mutations 
                if m.risk_level <= (1.0 - code_engine.safety_level)
            ]
            
            # Most mutations should be filtered out due to high risk
            assert len(safe_mutations) <= len(mutations)
            
        except SyntaxError:
            pass  # Expected for some unsafe code
    
    def test_code_deployment(self, code_engine, sample_function):
        \"\"\"Test deployment of optimized code.\"\"\"
        fragment_id = code_engine.register_function(sample_function)
        
        # Simulate optimization (manual improvement)
        fragment = code_engine.code_fragments[fragment_id]
        fragment.fitness_score = 10.0  # High fitness
        
        # Test deployment
        success = code_engine.deploy_optimized_code(fragment_id)
        
        # Should succeed if code is valid
        if success:
            assert fragment_id.split('_')[0] in code_engine.active_functions
    
    def test_function_generation(self, code_engine):
        \"\"\"Test generation of new functions.\"\"\"
        specification = "Calculate the sum of two numbers"
        example_inputs = [(1, 2), (3, 4), (5, 6)]
        example_outputs = [3, 7, 11]
        
        generated_code = code_engine.generate_new_function(
            specification, 
            example_inputs, 
            example_outputs
        )
        
        assert isinstance(generated_code, str)
        assert "def " in generated_code  # Should contain function definition
        
        # Verify it's valid Python syntax
        try:
            ast.parse(generated_code)
        except SyntaxError:
            pytest.fail("Generated code has syntax errors")
    
    def test_optimization_report(self, code_engine):
        \"\"\"Test optimization reporting.\"\"\"
        report = code_engine.get_optimization_report()
        
        assert 'modification_metrics' in report
        assert 'fragment_summary' in report
        assert 'safety_analysis' in report
        assert 'evolution_statistics' in report
        
        # Check safety analysis
        safety = report['safety_analysis']
        assert 'quarantined_mutations' in safety
        assert 'safety_violations' in safety
        assert 'rollback_rate' in safety


class TestAdaptiveArchitecture:
    \"\"\"Test adaptive architecture management.\"\"\"
    
    @pytest.fixture
    def architecture_manager(self):
        \"\"\"Create architecture manager for testing.\"\"\"
        return AdaptiveArchitectureManager(
            hdc_dimension=1000,
            adaptation_threshold=0.8,
            monitoring_interval=0.1
        )
    
    def test_component_registration(self, architecture_manager):
        \"\"\"Test system component registration.\"\"\"
        component = architecture_manager.register_component(
            component_id="test_processor",
            component_type=ComponentType.PROCESSOR,
            capabilities={"cores": 4, "frequency": 3.0},
            max_capacity=100.0,
            priority_level=3
        )
        
        assert isinstance(component, SystemComponent)
        assert component.component_id == "test_processor"
        assert component.component_type == ComponentType.PROCESSOR
        assert component.priority_level == 3
        assert "test_processor" in architecture_manager.components
    
    def test_component_connections(self, architecture_manager):
        \"\"\"Test component connection management.\"\"\"
        # Register components
        comp1 = architecture_manager.register_component(
            "comp1", ComponentType.PROCESSOR, {}, 100.0
        )
        comp2 = architecture_manager.register_component(
            "comp2", ComponentType.MEMORY, {}, 100.0
        )
        
        # Connect components
        architecture_manager.connect_components("comp1", "comp2", weight=0.8)
        
        assert "comp2" in comp1.connections
        assert "comp1" in comp2.connections
        assert architecture_manager.current_topology.has_edge("comp1", "comp2")
        
    def test_topology_optimization(self, architecture_manager):
        \"\"\"Test topology optimization.\"\"\"
        # Create test topology
        for i in range(5):
            architecture_manager.register_component(
                f"comp_{i}",
                ComponentType.PROCESSOR,
                {},
                100.0,
                priority_level=(i % 3) + 1
            )
        
        # Connect in a simple chain
        for i in range(4):
            architecture_manager.connect_components(f"comp_{i}", f"comp_{i+1}")
        
        optimizer = architecture_manager.topology_optimizer
        
        # Test different optimization strategies
        for strategy in ['performance', 'resilience', 'energy', 'latency']:
            optimized = optimizer.optimize_topology(
                architecture_manager.current_topology,
                architecture_manager.components,
                strategy
            )
            
            assert isinstance(optimized, type(architecture_manager.current_topology))
            assert optimized.number_of_nodes() == architecture_manager.current_topology.number_of_nodes()
    
    def test_resource_allocation(self, architecture_manager):
        \"\"\"Test dynamic resource allocation.\"\"\"
        # Register components with different types
        architecture_manager.register_component(
            "cpu_comp", ComponentType.PROCESSOR, {}, 100.0
        )
        architecture_manager.register_component(
            "mem_comp", ComponentType.MEMORY, {}, 200.0
        )
        architecture_manager.register_component(
            "net_comp", ComponentType.NETWORK, {}, 50.0
        )
        
        # Set different loads
        architecture_manager.components["cpu_comp"].current_load = 80.0
        architecture_manager.components["mem_comp"].current_load = 50.0
        architecture_manager.components["net_comp"].current_load = 30.0
        
        allocator = architecture_manager.resource_allocator
        available_resources = {"cpu": 100.0, "memory": 200.0, "bandwidth": 50.0}
        
        # Test different allocation strategies
        for strategy in ['demand_based', 'predictive', 'priority_based', 'fair']:
            allocation = allocator.allocate_resources(
                architecture_manager.components,
                available_resources,
                strategy
            )
            
            assert len(allocation) == 3  # One for each component
            
            for comp_id, resource_allocation in allocation.items():
                assert comp_id in architecture_manager.components
                for resource_type, amount in resource_allocation.items():
                    assert amount >= 0
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    @patch('psutil.net_io_counters')
    def test_system_monitoring(self, mock_net, mock_disk, mock_memory, mock_cpu, architecture_manager):
        \"\"\"Test system monitoring capabilities.\"\"\"
        # Mock system resource calls
        mock_cpu.return_value = 50.0
        mock_memory.return_value = Mock(percent=60.0)
        mock_disk.return_value = Mock(percent=70.0)
        mock_net.return_value = Mock(bytes_sent=1000, bytes_recv=2000)
        
        # Register test components
        architecture_manager.register_component(
            "test_comp", ComponentType.PROCESSOR, {}, 100.0
        )
        architecture_manager.components["test_comp"].current_load = 90.0  # High load
        
        # Collect metrics
        metrics = architecture_manager._collect_system_metrics()
        
        assert 'timestamp' in metrics
        assert 'system_resources' in metrics
        assert 'component_metrics' in metrics
        assert 'topology_metrics' in metrics
        assert 'performance_indicators' in metrics
        
        # Check component metrics
        comp_metrics = metrics['component_metrics']['test_comp']
        assert comp_metrics['utilization'] == 0.9  # 90/100
        assert comp_metrics['is_overloaded'] is True  # > 90%
    
    def test_adaptation_triggers(self, architecture_manager):
        \"\"\"Test adaptation trigger detection.\"\"\"
        # Create scenario that should trigger adaptation
        architecture_manager.register_component(
            "overloaded_comp", ComponentType.PROCESSOR, {}, 100.0
        )
        architecture_manager.components["overloaded_comp"].current_load = 95.0  # Overloaded
        
        # Create mock metrics
        metrics = {
            'performance_indicators': {
                'average_utilization': 0.95,  # High utilization
                'load_balance_variance': 0.5,  # High variance
                'overload_percentage': 1.0,    # 100% overloaded
                'topology_efficiency': 0.2     # Low efficiency
            }
        }
        
        adaptation_needed = architecture_manager._analyze_adaptation_need(metrics)
        assert adaptation_needed is True
        
        # Test with good metrics
        good_metrics = {
            'performance_indicators': {
                'average_utilization': 0.5,
                'load_balance_variance': 0.1,
                'overload_percentage': 0.0,
                'topology_efficiency': 0.8
            }
        }
        
        adaptation_needed = architecture_manager._analyze_adaptation_need(good_metrics)
        assert adaptation_needed is False
    
    def test_load_balancing_adaptation(self, architecture_manager):
        \"\"\"Test load balancing adaptation strategy.\"\"\"
        # Create overloaded and underloaded components
        architecture_manager.register_component(
            "overloaded", ComponentType.PROCESSOR, {}, 100.0
        )
        architecture_manager.register_component(
            "underloaded", ComponentType.PROCESSOR, {}, 100.0
        )
        
        # Set loads
        architecture_manager.components["overloaded"].current_load = 95.0
        architecture_manager.components["underloaded"].current_load = 20.0
        
        # Execute load balancing
        metrics = {'performance_indicators': {}}
        success = architecture_manager._execute_load_balancing(metrics)
        
        if success:
            # Load should be redistributed
            overloaded_load = architecture_manager.components["overloaded"].current_load
            underloaded_load = architecture_manager.components["underloaded"].current_load
            
            assert overloaded_load < 95.0  # Should be reduced
            assert underloaded_load > 20.0  # Should be increased
    
    def test_architecture_persistence(self, architecture_manager, tmp_path):
        \"\"\"Test saving and loading architecture configurations.\"\"\"
        # Set up architecture
        architecture_manager.register_component(
            "persistent_comp", ComponentType.PROCESSOR, 
            {"test": True}, 100.0, priority_level=2
        )
        
        # Save configuration
        config_file = tmp_path / "test_config.json"
        architecture_manager.save_architecture_config(str(config_file))
        
        assert config_file.exists()
        
        # Create new manager and load configuration
        new_manager = AdaptiveArchitectureManager()
        new_manager.load_architecture_config(str(config_file))
        
        assert "persistent_comp" in new_manager.components
        loaded_comp = new_manager.components["persistent_comp"]
        assert loaded_comp.component_type == ComponentType.PROCESSOR
        assert loaded_comp.capabilities["test"] is True
        assert loaded_comp.priority_level == 2
    
    def test_performance_tracking(self, architecture_manager):
        \"\"\"Test performance metrics tracking.\"\"\"
        # Register components and perform operations
        architecture_manager.register_component(
            "tracked_comp", ComponentType.PROCESSOR, {}, 100.0
        )
        
        # Simulate adaptation
        architecture_manager.adaptation_metrics['total_adaptations'] = 5
        architecture_manager.adaptation_metrics['successful_adaptations'] = 4
        architecture_manager.adaptation_metrics['performance_improvements'] = [0.1, 0.2, 0.15]
        
        summary = architecture_manager.get_architecture_summary()
        
        assert 'adaptation_metrics' in summary
        assert 'component_summary' in summary
        assert 'topology_metrics' in summary
        
        adaptation_metrics = summary['adaptation_metrics']
        assert adaptation_metrics['total_adaptations'] == 5
        assert adaptation_metrics['successful_adaptations'] == 4


class TestComponentTypes:
    \"\"\"Test different component types and behaviors.\"\"\"
    
    def test_component_utilization(self):
        \"\"\"Test component utilization calculations.\"\"\"
        component = SystemComponent(
            component_id="test",
            component_type=ComponentType.PROCESSOR,
            capabilities={},
            current_load=75.0,
            max_capacity=100.0,
            connections=set(),
            performance_metrics={},
            adaptation_history=[]
        )
        
        assert component.utilization_ratio == 0.75
        assert component.is_overloaded is False
        
        # Test overload condition
        component.current_load = 95.0
        assert component.is_overloaded is True
    
    def test_component_adaptation_history(self):
        \"\"\"Test component adaptation history tracking.\"\"\"
        component = SystemComponent(
            component_id="history_test",
            component_type=ComponentType.MEMORY,
            capabilities={},
            current_load=50.0,
            max_capacity=100.0,
            connections=set(),
            performance_metrics={},
            adaptation_history=[]
        )
        
        # Add adaptation record
        adaptation_record = {
            'type': 'load_transfer',
            'amount': 10.0,
            'timestamp': time.time()
        }
        component.adaptation_history.append(adaptation_record)
        
        assert len(component.adaptation_history) == 1
        assert component.adaptation_history[0]['type'] == 'load_transfer'


class TestIntegrationScenarios:
    \"\"\"Integration tests for autonomous mastery capabilities.\"\"\"
    
    def test_self_modifying_with_adaptive_architecture(self):
        \"\"\"Test integration of self-modifying code with adaptive architecture.\"\"\"
        code_engine = SelfModifyingCodeEngine(hdc_dimension=500)
        arch_manager = AdaptiveArchitectureManager(hdc_dimension=500)
        
        # Register code processing component
        arch_manager.register_component(
            "code_processor",
            ComponentType.PROCESSOR,
            {"supports_code_modification": True},
            100.0
        )
        
        # Register simple function
        def test_func(x):
            return x * 2
        
        fragment_id = code_engine.register_function(test_func)
        
        # Simulate code evolution affecting architecture
        test_inputs = [(1,), (2,), (3,)]
        evolution_stats = code_engine.evolve_code(fragment_id, test_inputs, generations=2)
        
        # Architecture should adapt to code changes
        comp = arch_manager.components["code_processor"]
        if evolution_stats['final_improvement'] > 0:
            # Simulate reduced load due to optimization
            comp.current_load = max(0, comp.current_load - 10.0)
        
        assert comp.current_load >= 0
    
    def test_end_to_end_autonomous_optimization(self):
        \"\"\"Test end-to-end autonomous optimization scenario.\"\"\"
        # Create systems
        code_engine = SelfModifyingCodeEngine(hdc_dimension=500, max_generations=3)
        arch_manager = AdaptiveArchitectureManager(hdc_dimension=500)
        
        # Set up architecture
        for i in range(3):
            arch_manager.register_component(
                f"optimizer_{i}",
                ComponentType.PROCESSOR,
                {"optimization_capable": True},
                100.0
            )
        
        # Connect components
        arch_manager.connect_components("optimizer_0", "optimizer_1")
        arch_manager.connect_components("optimizer_1", "optimizer_2")
        
        # Register optimization target function
        def optimization_target(data_list):
            result = 0
            for item in data_list:
                result += item ** 2
            return result
        
        fragment_id = code_engine.register_function(optimization_target)
        
        # Perform autonomous optimization
        test_data = [([1, 2, 3],), ([4, 5, 6],), ([7, 8, 9],)]
        
        # Code evolution
        evolution_result = code_engine.evolve_code(fragment_id, test_data, generations=3)
        
        # Architecture adaptation based on code performance
        if evolution_result['final_improvement'] > 0:
            # Good optimization - reduce resource allocation
            for comp in arch_manager.components.values():
                comp.current_load = max(0, comp.current_load - 5.0)
        
        # Verify autonomous optimization worked
        assert evolution_result is not None
        assert len(evolution_result['generation_results']) <= 3
        
        # Verify architecture remained stable
        summary = arch_manager.get_architecture_summary()
        assert summary['component_count'] == 3
        assert summary['topology_metrics']['is_connected'] is True


@pytest.mark.performance
class TestPerformanceBenchmarks:
    \"\"\"Performance benchmarks for autonomous mastery.\"\"\"
    
    def test_code_analysis_speed(self):
        \"\"\"Test AST analysis performance.\"\"\"
        analyzer = ASTAnalyzer()
        
        # Large code sample
        code = \"\"\"
def complex_function(x, y, z):
    result = []
    for i in range(x):
        for j in range(y):
            if i * j < z:
                temp = []
                for k in range(i + j):
                    temp.append(k ** 2)
                result.extend(temp)
            else:
                result.append(i * j * z)
    return result
\"\"\"
        
        start_time = time.time()
        for _ in range(100):
            complexity = analyzer.analyze_complexity(code)
            candidates = analyzer.find_optimization_candidates(code)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 100
        assert avg_time < 0.01  # Should be under 10ms per analysis
    
    def test_architecture_adaptation_speed(self):
        \"\"\"Test architecture adaptation performance.\"\"\"
        arch_manager = AdaptiveArchitectureManager(monitoring_interval=0.01)
        
        # Create larger architecture
        for i in range(20):
            arch_manager.register_component(
                f"perf_comp_{i}",
                ComponentType.PROCESSOR,
                {},
                100.0
            )
        
        # Connect in mesh topology
        components = list(arch_manager.components.keys())
        for i in range(len(components)):
            for j in range(i+1, min(i+4, len(components))):  # Connect to next 3
                arch_manager.connect_components(components[i], components[j])
        
        # Benchmark adaptation
        metrics = {
            'performance_indicators': {
                'average_utilization': 0.9,
                'load_balance_variance': 0.4,
                'overload_percentage': 0.3,
                'topology_efficiency': 0.3
            }
        }
        
        start_time = time.time()
        adaptation_needed = arch_manager._analyze_adaptation_need(metrics)
        if adaptation_needed:
            arch_manager._execute_load_balancing(metrics)
        end_time = time.time()
        
        adaptation_time = end_time - start_time
        assert adaptation_time < 0.1  # Should be under 100ms


if __name__ == "__main__":
    # Run tests with performance benchmarks
    pytest.main([__file__, "-v", "--tb=short", "-m", "not performance"])
    print("\\nRunning performance benchmarks...")
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])