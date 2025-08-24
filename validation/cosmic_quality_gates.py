"""
Cosmic Quality Gates - Advanced Validation System
================================================

Comprehensive validation system for cosmic-scale robotic intelligence,
ensuring quality across all generations of implementation.
"""

import numpy as np
import time
import threading
import json
import traceback
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from enum import Enum
from pathlib import Path
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hdc_robot_controller.core.hypervector import HyperVector
from hdc_robot_controller.core.memory import AssociativeMemory
from hdc_robot_controller.evolution.genetic_optimizer import GeneticOptimizer
from hdc_robot_controller.evolution.self_improving_algorithms import SelfImprovingHDC


class ValidationLevel(Enum):
    """Validation complexity levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    COSMIC = "cosmic"
    TRANSCENDENT = "transcendent"


class ValidationResult(Enum):
    """Validation results."""
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    CRITICAL = "critical"


@dataclass
class QualityMetric:
    """Quality metric with validation thresholds."""
    name: str
    current_value: float
    target_value: float
    minimum_threshold: float
    critical_threshold: float
    unit: str = ""
    description: str = ""
    
    def evaluate(self) -> ValidationResult:
        """Evaluate quality metric against thresholds."""
        if self.current_value < self.critical_threshold:
            return ValidationResult.CRITICAL
        elif self.current_value < self.minimum_threshold:
            return ValidationResult.FAIL
        elif self.current_value < self.target_value:
            return ValidationResult.WARN
        else:
            return ValidationResult.PASS


@dataclass
class ValidationReport:
    """Complete validation report."""
    test_suite_name: str
    validation_level: ValidationLevel
    start_time: float
    end_time: float
    total_tests: int
    passed_tests: int
    warning_tests: int
    failed_tests: int
    critical_tests: int
    quality_metrics: Dict[str, QualityMetric]
    detailed_results: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate overall success rate."""
        if self.total_tests == 0:
            return 0.0
        return self.passed_tests / self.total_tests
    
    @property
    def overall_status(self) -> ValidationResult:
        """Determine overall validation status."""
        if self.critical_tests > 0:
            return ValidationResult.CRITICAL
        elif self.failed_tests > 0:
            return ValidationResult.FAIL
        elif self.warning_tests > 0:
            return ValidationResult.WARN
        else:
            return ValidationResult.PASS


class QualityValidator(ABC):
    """Abstract base class for quality validators."""
    
    @abstractmethod
    def validate(self, component: Any, context: Dict[str, Any]) -> ValidationReport:
        """Validate component and return report."""
        pass
    
    @abstractmethod
    def get_validation_level(self) -> ValidationLevel:
        """Get validation complexity level."""
        pass


class CoreHDCValidator(QualityValidator):
    """Validates core HDC functionality."""
    
    def __init__(self):
        self.validation_level = ValidationLevel.BASIC
        
    def validate(self, component: Any, context: Dict[str, Any]) -> ValidationReport:
        """Validate core HDC components."""
        
        start_time = time.time()
        results = []
        quality_metrics = {}
        
        # Test 1: HyperVector Operations
        test_result = self._test_hypervector_operations()
        results.append(test_result)
        
        # Test 2: Memory Operations
        test_result = self._test_memory_operations()
        results.append(test_result)
        
        # Test 3: Performance Benchmarks
        performance_metrics = self._benchmark_performance()
        quality_metrics.update(performance_metrics)
        
        # Aggregate results
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['status'] == 'pass')
        warning_tests = sum(1 for r in results if r['status'] == 'warn')
        failed_tests = sum(1 for r in results if r['status'] == 'fail')
        critical_tests = sum(1 for r in results if r['status'] == 'critical')
        
        end_time = time.time()
        
        return ValidationReport(
            test_suite_name="Core HDC Validation",
            validation_level=self.validation_level,
            start_time=start_time,
            end_time=end_time,
            total_tests=total_tests,
            passed_tests=passed_tests,
            warning_tests=warning_tests,
            failed_tests=failed_tests,
            critical_tests=critical_tests,
            quality_metrics=quality_metrics,
            detailed_results=results
        )
    
    def _test_hypervector_operations(self) -> Dict[str, Any]:
        """Test hypervector operations."""
        try:
            dimension = 10000
            
            # Create hypervectors
            hv1 = HyperVector.random(dimension)
            hv2 = HyperVector.random(dimension)
            
            # Test bundling
            bundled = hv1.bundle(hv2)
            assert bundled.dimension() == dimension
            
            # Test binding
            bound = hv1.bind(hv2)
            assert bound.dimension() == dimension
            
            # Test similarity
            similarity = hv1.similarity(hv2)
            assert -1.0 <= similarity <= 1.0
            
            # Test self-similarity
            self_sim = hv1.similarity(hv1)
            assert abs(self_sim - 1.0) < 0.1  # Should be close to 1.0
            
            return {
                'test_name': 'HyperVector Operations',
                'status': 'pass',
                'execution_time': time.time(),
                'details': f'All operations successful (dimension: {dimension})'
            }
            
        except Exception as e:
            return {
                'test_name': 'HyperVector Operations',
                'status': 'critical',
                'execution_time': time.time(),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_memory_operations(self) -> Dict[str, Any]:
        """Test memory operations."""
        try:
            dimension = 5000  # Smaller for testing
            memory = AssociativeMemory(dimension)
            
            # Test storage and retrieval
            test_vectors = []
            for i in range(10):
                hv = HyperVector.random(dimension, seed=i)
                key = f"test_vector_{i}"
                memory.store(key, hv)
                test_vectors.append((key, hv))
            
            # Test retrieval
            success_count = 0
            for key, original_hv in test_vectors:
                if memory.contains(key):
                    retrieved = memory.recall(key)
                    if retrieved and retrieved.similarity(original_hv) > 0.9:
                        success_count += 1
            
            success_rate = success_count / len(test_vectors)
            
            if success_rate >= 0.9:
                status = 'pass'
            elif success_rate >= 0.7:
                status = 'warn'
            else:
                status = 'fail'
            
            return {
                'test_name': 'Memory Operations',
                'status': status,
                'execution_time': time.time(),
                'success_rate': success_rate,
                'details': f'Retrieved {success_count}/{len(test_vectors)} vectors successfully'
            }
            
        except Exception as e:
            return {
                'test_name': 'Memory Operations',
                'status': 'critical',
                'execution_time': time.time(),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _benchmark_performance(self) -> Dict[str, QualityMetric]:
        """Benchmark performance metrics."""
        metrics = {}
        
        try:
            dimension = 10000
            iterations = 100
            
            # Benchmark hypervector creation
            start_time = time.perf_counter()
            for i in range(iterations):
                hv = HyperVector.random(dimension)
            creation_time = (time.perf_counter() - start_time) / iterations * 1000  # ms
            
            metrics['hv_creation_time'] = QualityMetric(
                name="HyperVector Creation Time",
                current_value=creation_time,
                target_value=1.0,  # 1ms target
                minimum_threshold=5.0,  # 5ms acceptable
                critical_threshold=20.0,  # 20ms critical
                unit="ms",
                description="Time to create random hypervector"
            )
            
            # Benchmark similarity computation
            hv1 = HyperVector.random(dimension)
            hv2 = HyperVector.random(dimension)
            
            start_time = time.perf_counter()
            for i in range(iterations):
                similarity = hv1.similarity(hv2)
            similarity_time = (time.perf_counter() - start_time) / iterations * 1000  # ms
            
            metrics['similarity_time'] = QualityMetric(
                name="Similarity Computation Time",
                current_value=similarity_time,
                target_value=0.1,  # 0.1ms target
                minimum_threshold=1.0,  # 1ms acceptable
                critical_threshold=5.0,  # 5ms critical
                unit="ms",
                description="Time to compute similarity between hypervectors"
            )
            
            # Memory throughput benchmark
            memory = AssociativeMemory(dimension)
            test_data = [(f"key_{i}", HyperVector.random(dimension)) for i in range(50)]
            
            start_time = time.perf_counter()
            for key, hv in test_data:
                memory.store(key, hv)
            storage_time = time.perf_counter() - start_time
            
            storage_throughput = len(test_data) / storage_time  # items/second
            
            metrics['storage_throughput'] = QualityMetric(
                name="Memory Storage Throughput",
                current_value=storage_throughput,
                target_value=1000.0,  # 1000 items/sec target
                minimum_threshold=100.0,  # 100 items/sec minimum
                critical_threshold=10.0,  # 10 items/sec critical
                unit="items/sec",
                description="Memory storage throughput"
            )
            
        except Exception as e:
            # Add error metric
            metrics['benchmark_error'] = QualityMetric(
                name="Benchmark Error",
                current_value=1.0,  # Error occurred
                target_value=0.0,
                minimum_threshold=0.0,
                critical_threshold=0.5,
                unit="boolean",
                description=f"Benchmark error: {str(e)}"
            )
        
        return metrics
    
    def get_validation_level(self) -> ValidationLevel:
        return self.validation_level


class EvolutionaryValidator(QualityValidator):
    """Validates evolutionary systems."""
    
    def __init__(self):
        self.validation_level = ValidationLevel.ADVANCED
        
    def validate(self, component: Any, context: Dict[str, Any]) -> ValidationReport:
        """Validate evolutionary components."""
        
        start_time = time.time()
        results = []
        quality_metrics = {}
        
        # Test 1: Genetic Optimizer
        test_result = self._test_genetic_optimizer()
        results.append(test_result)
        
        # Test 2: Self-Improving Algorithms
        test_result = self._test_self_improving_system()
        results.append(test_result)
        
        # Test 3: Evolution Performance
        evolution_metrics = self._benchmark_evolution_performance()
        quality_metrics.update(evolution_metrics)
        
        # Aggregate results
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['status'] == 'pass')
        warning_tests = sum(1 for r in results if r['status'] == 'warn')
        failed_tests = sum(1 for r in results if r['status'] == 'fail')
        critical_tests = sum(1 for r in results if r['status'] == 'critical')
        
        end_time = time.time()
        
        return ValidationReport(
            test_suite_name="Evolutionary Systems Validation",
            validation_level=self.validation_level,
            start_time=start_time,
            end_time=end_time,
            total_tests=total_tests,
            passed_tests=passed_tests,
            warning_tests=warning_tests,
            failed_tests=failed_tests,
            critical_tests=critical_tests,
            quality_metrics=quality_metrics,
            detailed_results=results
        )
    
    def _test_genetic_optimizer(self) -> Dict[str, Any]:
        """Test genetic optimization."""
        try:
            # Simple fitness function for testing
            class TestFitnessFunction:
                def evaluate(self, individual, context):
                    # Maximize sum of genome values
                    return sum(individual.genome)
                
                def get_optimal_fitness(self):
                    return 10.0  # Max possible for 10-element genome
            
            from hdc_robot_controller.evolution.genetic_optimizer import (
                GeneticOptimizer, EvolutionaryParameters
            )
            
            fitness_func = TestFitnessFunction()
            params = EvolutionaryParameters(
                population_size=20,
                max_generations=10,
                mutation_rate=0.1
            )
            
            optimizer = GeneticOptimizer(fitness_func, params, seed=42)
            
            # Run optimization
            best_individual = optimizer.evolve(
                genome_length=10,
                context={}
            )
            
            # Validate results
            if best_individual.fitness > 7.0:  # Should achieve good fitness
                status = 'pass'
            elif best_individual.fitness > 5.0:
                status = 'warn'
            else:
                status = 'fail'
            
            return {
                'test_name': 'Genetic Optimizer',
                'status': status,
                'execution_time': time.time(),
                'best_fitness': best_individual.fitness,
                'details': f'Achieved fitness: {best_individual.fitness:.3f}'
            }
            
        except Exception as e:
            return {
                'test_name': 'Genetic Optimizer',
                'status': 'critical',
                'execution_time': time.time(),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_self_improving_system(self) -> Dict[str, Any]:
        """Test self-improving algorithms."""
        try:
            # Create simple test algorithm
            def test_algorithm(data):
                return sum(x**2 for x in data) / len(data)
            
            # Initialize self-improving system
            self_improving = SelfImprovingHDC(dimension=1000)
            
            # Register test algorithm
            version_id = self_improving.register_algorithm("test_algo", test_algorithm)
            
            # Simulate some performance data
            for i in range(20):
                self_improving.performance_history["test_algo"].append(
                    np.random.uniform(0.05, 0.15)  # Execution time
                )
            
            # Get algorithm back
            current_algo = self_improving.get_algorithm("test_algo")
            
            if current_algo and version_id:
                status = 'pass'
                details = f"Successfully registered algorithm: {version_id}"
            else:
                status = 'fail'
                details = "Failed to register or retrieve algorithm"
            
            return {
                'test_name': 'Self-Improving System',
                'status': status,
                'execution_time': time.time(),
                'version_id': version_id,
                'details': details
            }
            
        except Exception as e:
            return {
                'test_name': 'Self-Improving System',
                'status': 'critical',
                'execution_time': time.time(),
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _benchmark_evolution_performance(self) -> Dict[str, QualityMetric]:
        """Benchmark evolution performance."""
        metrics = {}
        
        try:
            # Measure genetic algorithm convergence
            class SimpleFunction:
                def evaluate(self, individual, context):
                    return sum(individual.genome)
                def get_optimal_fitness(self):
                    return 5.0
            
            from hdc_robot_controller.evolution.genetic_optimizer import (
                GeneticOptimizer, EvolutionaryParameters
            )
            
            params = EvolutionaryParameters(
                population_size=10,
                max_generations=5
            )
            
            start_time = time.perf_counter()
            optimizer = GeneticOptimizer(SimpleFunction(), params)
            best_individual = optimizer.evolve(genome_length=5)
            convergence_time = time.perf_counter() - start_time
            
            metrics['convergence_time'] = QualityMetric(
                name="Evolution Convergence Time",
                current_value=convergence_time,
                target_value=1.0,  # 1 second target
                minimum_threshold=10.0,  # 10 seconds acceptable
                critical_threshold=60.0,  # 60 seconds critical
                unit="seconds",
                description="Time for genetic algorithm to converge"
            )
            
            metrics['convergence_fitness'] = QualityMetric(
                name="Evolution Convergence Fitness",
                current_value=best_individual.fitness,
                target_value=4.0,  # Good fitness target
                minimum_threshold=2.0,  # Minimum acceptable
                critical_threshold=1.0,  # Critical threshold
                unit="fitness",
                description="Fitness achieved by genetic algorithm"
            )
            
        except Exception as e:
            metrics['evolution_error'] = QualityMetric(
                name="Evolution Error",
                current_value=1.0,
                target_value=0.0,
                minimum_threshold=0.0,
                critical_threshold=0.5,
                unit="boolean",
                description=f"Evolution error: {str(e)}"
            )
        
        return metrics
    
    def get_validation_level(self) -> ValidationLevel:
        return self.validation_level


class CosmicQualityGateSystem:
    """Main cosmic quality gate validation system."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Register validators
        self.validators = {
            'core_hdc': CoreHDCValidator(),
            'evolutionary': EvolutionaryValidator()
        }
        
        # Validation history
        self.validation_history = []
        
        # Quality thresholds
        self.quality_thresholds = {
            ValidationLevel.BASIC: 0.8,      # 80% pass rate
            ValidationLevel.INTERMEDIATE: 0.85,  # 85% pass rate
            ValidationLevel.ADVANCED: 0.9,      # 90% pass rate
            ValidationLevel.COSMIC: 0.95,       # 95% pass rate
            ValidationLevel.TRANSCENDENT: 0.99  # 99% pass rate
        }
        
        self.logger.info("🛡️ Cosmic Quality Gate System initialized")
    
    def run_validation_suite(self, 
                           validation_level: ValidationLevel = ValidationLevel.ADVANCED,
                           component_filter: List[str] = None) -> Dict[str, ValidationReport]:
        """Run complete validation suite."""
        
        self.logger.info(f"🔍 Starting validation suite (level: {validation_level.value})")
        
        reports = {}
        start_time = time.time()
        
        # Filter validators based on level and component filter
        active_validators = {}
        for name, validator in self.validators.items():
            if component_filter and name not in component_filter:
                continue
            
            # Only run validators at or below requested level
            validator_level_order = {
                ValidationLevel.BASIC: 0,
                ValidationLevel.INTERMEDIATE: 1, 
                ValidationLevel.ADVANCED: 2,
                ValidationLevel.COSMIC: 3,
                ValidationLevel.TRANSCENDENT: 4
            }
            
            requested_order = validator_level_order[validation_level]
            validator_order = validator_level_order[validator.get_validation_level()]
            
            if validator_order <= requested_order:
                active_validators[name] = validator
        
        # Run validations
        for name, validator in active_validators.items():
            try:
                self.logger.info(f"  Running {name} validation...")
                report = validator.validate(None, {})
                reports[name] = report
                
                # Log results
                self.logger.info(f"    ✅ {report.passed_tests}/{report.total_tests} tests passed "
                               f"(success rate: {report.success_rate:.1%})")
                
                if report.failed_tests > 0:
                    self.logger.warning(f"    ⚠️  {report.failed_tests} tests failed")
                if report.critical_tests > 0:
                    self.logger.error(f"    🚨 {report.critical_tests} critical failures")
                    
            except Exception as e:
                self.logger.error(f"    💥 Validator {name} crashed: {e}")
                # Create error report
                reports[name] = ValidationReport(
                    test_suite_name=f"{name} (CRASHED)",
                    validation_level=validation_level,
                    start_time=time.time(),
                    end_time=time.time(),
                    total_tests=1,
                    passed_tests=0,
                    warning_tests=0,
                    failed_tests=0,
                    critical_tests=1,
                    quality_metrics={}
                )
        
        total_time = time.time() - start_time
        
        # Calculate overall statistics
        total_tests = sum(r.total_tests for r in reports.values())
        total_passed = sum(r.passed_tests for r in reports.values())
        total_failed = sum(r.failed_tests for r in reports.values())
        total_critical = sum(r.critical_tests for r in reports.values())
        
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0.0
        
        # Log summary
        self.logger.info(f"\n🏁 Validation Suite Complete ({total_time:.1f}s)")
        self.logger.info(f"   Overall Success Rate: {overall_success_rate:.1%}")
        self.logger.info(f"   Total Tests: {total_tests}")
        self.logger.info(f"   Passed: {total_passed}")
        self.logger.info(f"   Failed: {total_failed}")
        self.logger.info(f"   Critical: {total_critical}")
        
        # Check quality gate
        required_success_rate = self.quality_thresholds[validation_level]
        if overall_success_rate >= required_success_rate:
            self.logger.info(f"✅ QUALITY GATE PASSED (required: {required_success_rate:.1%})")
        else:
            self.logger.error(f"❌ QUALITY GATE FAILED (required: {required_success_rate:.1%})")
        
        # Store validation history
        validation_record = {
            'timestamp': start_time,
            'validation_level': validation_level.value,
            'duration': total_time,
            'overall_success_rate': overall_success_rate,
            'quality_gate_passed': overall_success_rate >= required_success_rate,
            'validator_count': len(reports),
            'total_tests': total_tests
        }
        self.validation_history.append(validation_record)
        
        return reports
    
    def generate_quality_report(self, reports: Dict[str, ValidationReport]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        
        # Aggregate metrics
        all_metrics = {}
        for report_name, report in reports.items():
            for metric_name, metric in report.quality_metrics.items():
                full_name = f"{report_name}_{metric_name}"
                all_metrics[full_name] = metric
        
        # Calculate quality scores
        quality_scores = {}
        for metric_name, metric in all_metrics.items():
            result = metric.evaluate()
            if result == ValidationResult.PASS:
                score = 100
            elif result == ValidationResult.WARN:
                score = 75
            elif result == ValidationResult.FAIL:
                score = 50
            else:  # CRITICAL
                score = 0
            quality_scores[metric_name] = score
        
        # Overall quality score
        overall_quality = np.mean(list(quality_scores.values())) if quality_scores else 0
        
        # Generate recommendations
        recommendations = []
        for report in reports.values():
            if report.failed_tests > 0:
                recommendations.append(f"Fix {report.failed_tests} failed tests in {report.test_suite_name}")
            if report.critical_tests > 0:
                recommendations.append(f"URGENT: Fix {report.critical_tests} critical issues in {report.test_suite_name}")
        
        # Performance analysis
        performance_metrics = {
            name: metric for name, metric in all_metrics.items() 
            if 'time' in metric.name.lower() or 'throughput' in metric.name.lower()
        }
        
        quality_report = {
            'generation_timestamp': time.time(),
            'overall_quality_score': overall_quality,
            'total_validators': len(reports),
            'total_tests': sum(r.total_tests for r in reports.values()),
            'success_rate': sum(r.passed_tests for r in reports.values()) / sum(r.total_tests for r in reports.values()),
            'quality_scores': quality_scores,
            'performance_metrics': {
                name: {
                    'value': metric.current_value,
                    'unit': metric.unit,
                    'status': metric.evaluate().value
                }
                for name, metric in performance_metrics.items()
            },
            'recommendations': recommendations,
            'validation_reports': {
                name: {
                    'test_suite_name': report.test_suite_name,
                    'validation_level': report.validation_level.value,
                    'success_rate': report.success_rate,
                    'status': report.overall_status.value,
                    'duration': report.end_time - report.start_time
                }
                for name, report in reports.items()
            }
        }
        
        return quality_report
    
    def save_validation_results(self, 
                              reports: Dict[str, ValidationReport],
                              filepath: str = "cosmic_quality_validation.json"):
        """Save validation results to file."""
        
        quality_report = self.generate_quality_report(reports)
        
        # Add detailed results
        detailed_data = {
            'quality_report': quality_report,
            'detailed_reports': {
                name: {
                    'test_suite_name': report.test_suite_name,
                    'validation_level': report.validation_level.value,
                    'start_time': report.start_time,
                    'end_time': report.end_time,
                    'total_tests': report.total_tests,
                    'passed_tests': report.passed_tests,
                    'warning_tests': report.warning_tests,
                    'failed_tests': report.failed_tests,
                    'critical_tests': report.critical_tests,
                    'success_rate': report.success_rate,
                    'overall_status': report.overall_status.value,
                    'quality_metrics': {
                        mname: {
                            'name': m.name,
                            'current_value': m.current_value,
                            'target_value': m.target_value,
                            'status': m.evaluate().value,
                            'unit': m.unit,
                            'description': m.description
                        }
                        for mname, m in report.quality_metrics.items()
                    },
                    'detailed_results': report.detailed_results,
                    'recommendations': report.recommendations
                }
                for name, report in reports.items()
            },
            'validation_history': self.validation_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(detailed_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Validation results saved to {filepath}")
        return filepath


def run_cosmic_quality_gates():
    """Main function to run cosmic quality gates."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🛡️ COSMIC QUALITY GATES - ADVANCED VALIDATION SYSTEM")
    print("=" * 60)
    
    # Initialize quality gate system
    quality_gates = CosmicQualityGateSystem()
    
    # Run validation at different levels
    validation_levels = [
        ValidationLevel.BASIC,
        ValidationLevel.ADVANCED
    ]
    
    all_results = {}
    
    for level in validation_levels:
        print(f"\n🔍 Running {level.value.upper()} validation...")
        
        reports = quality_gates.run_validation_suite(
            validation_level=level
        )
        
        all_results[level.value] = reports
        
        # Generate quality report for this level
        quality_report = quality_gates.generate_quality_report(reports)
        
        print(f"\n📊 Quality Report ({level.value}):")
        print(f"   Overall Quality Score: {quality_report['overall_quality_score']:.1f}/100")
        print(f"   Success Rate: {quality_report['success_rate']:.1%}")
        print(f"   Total Tests: {quality_report['total_tests']}")
        
        if quality_report['recommendations']:
            print(f"   Recommendations: {len(quality_report['recommendations'])}")
            for i, rec in enumerate(quality_report['recommendations'][:3]):
                print(f"     {i+1}. {rec}")
    
    # Save comprehensive results
    print(f"\n💾 Saving validation results...")
    
    final_reports = all_results.get('advanced', all_results.get('basic', {}))
    filepath = quality_gates.save_validation_results(final_reports)
    
    # Final summary
    print(f"\n🏁 COSMIC QUALITY GATES COMPLETE")
    print(f"   Results saved to: {filepath}")
    
    # Check if all quality gates passed
    final_quality_report = quality_gates.generate_quality_report(final_reports)
    overall_score = final_quality_report['overall_quality_score']
    
    if overall_score >= 80:
        print(f"   ✅ QUALITY GATES PASSED (Score: {overall_score:.1f}/100)")
        exit_code = 0
    elif overall_score >= 60:
        print(f"   ⚠️  QUALITY GATES WARNING (Score: {overall_score:.1f}/100)")
        exit_code = 1
    else:
        print(f"   ❌ QUALITY GATES FAILED (Score: {overall_score:.1f}/100)")
        exit_code = 2
    
    print("=" * 60)
    return exit_code


if __name__ == "__main__":
    exit_code = run_cosmic_quality_gates()
    exit(exit_code)