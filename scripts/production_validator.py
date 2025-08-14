#!/usr/bin/env python3
"""
Production Validation Script for HDC Robot Controller.

Comprehensive validation of all system components before production deployment.
Includes performance benchmarks, security checks, and integration tests.
"""

import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hdc_robot_controller.core.hypervector import HyperVector
from hdc_robot_controller.core.memory import HierarchicalMemory
from hdc_robot_controller.core.sensor_encoder import SensorEncoder
from hdc_robot_controller.core.behavior_learner import BehaviorLearner
from hdc_robot_controller.core.logging_system import setup_production_logging
from hdc_robot_controller.robustness.advanced_error_recovery import AdvancedErrorRecovery, FaultSeverity
from hdc_robot_controller.optimization.gpu_accelerator import get_gpu_accelerator
from hdc_robot_controller.scaling.distributed_coordinator import DistributedCoordinator


class ProductionValidator:
    """Comprehensive production validation system."""
    
    def __init__(self):
        """Initialize production validator."""
        self.logger = setup_production_logging("/tmp/hdc_validation.log", "INFO", False)
        self.validation_results = {}
        self.performance_metrics = {}
        self.start_time = time.time()
        
        print("🚀 HDC Robot Controller - Production Validation")
        print("=" * 60)
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run all production validation checks."""
        validations = [
            ("Core Functionality", self._validate_core_functionality),
            ("Memory Systems", self._validate_memory_systems),
            ("Sensor Encoding", self._validate_sensor_encoding),
            ("Behavior Learning", self._validate_behavior_learning),
            ("Error Recovery", self._validate_error_recovery),
            ("Performance", self._validate_performance),
            ("Scalability", self._validate_scalability),
            ("Security", self._validate_security),
            ("Integration", self._validate_integration)
        ]
        
        for validation_name, validation_func in validations:
            print(f"\n🔍 Running {validation_name} validation...")
            
            try:
                start_time = time.time()
                result = validation_func()
                duration = time.time() - start_time
                
                self.validation_results[validation_name] = {
                    'status': 'PASS' if result else 'FAIL',
                    'duration': duration,
                    'details': result if isinstance(result, dict) else {}
                }
                
                status_emoji = "✅" if result else "❌"
                print(f"{status_emoji} {validation_name}: {self.validation_results[validation_name]['status']} ({duration:.2f}s)")
                
            except Exception as e:
                self.validation_results[validation_name] = {
                    'status': 'ERROR',
                    'duration': 0.0,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                print(f"❌ {validation_name}: ERROR - {str(e)}")
                self.logger.error(f"Validation error in {validation_name}", error=str(e))
        
        return self._generate_final_report()
    
    def _validate_core_functionality(self) -> Dict[str, Any]:
        """Validate core HDC operations."""
        results = {}
        
        # Test HyperVector operations
        hv1 = HyperVector.random(1000, seed=42)
        hv2 = HyperVector.random(1000, seed=43)
        
        # Test similarity calculation
        self_similarity = hv1.similarity(hv1)
        assert abs(self_similarity - 1.0) < 0.001, f"Self similarity should be 1.0, got {self_similarity}"
        
        inverse_similarity = hv1.similarity(hv1.invert())
        assert abs(inverse_similarity - (-1.0)) < 0.001, f"Inverse similarity should be -1.0, got {inverse_similarity}"
        
        # Test bundling
        bundled = hv1.bundle(hv2)
        assert bundled.dimension == 1000, "Bundle dimension mismatch"
        
        # Test binding
        bound = hv1.bind(hv2)
        assert bound.dimension == 1000, "Bind dimension mismatch"
        
        # Test multiple vector operations
        vectors = [HyperVector.random(1000, seed=i) for i in range(10)]
        multi_bundled = HyperVector.bundle_vectors(vectors)
        assert multi_bundled.dimension == 1000, "Multi-bundle dimension mismatch"
        
        results['hypervector_operations'] = True
        results['similarity_accuracy'] = True
        results['bundling_operations'] = True
        results['binding_operations'] = True
        
        return results
    
    def _validate_memory_systems(self) -> Dict[str, Any]:
        """Validate memory systems functionality."""
        results = {}
        
        # Test hierarchical memory
        memory = HierarchicalMemory(1000)
        
        # Test associative memory
        assoc_mem = memory.get_associative_memory()
        test_vector = HyperVector.random(1000, seed=100)
        
        assoc_mem.store("test_pattern", test_vector, 0.9)
        retrieved = assoc_mem.retrieve("test_pattern")
        
        assert retrieved.similarity(test_vector) > 0.99, "Memory retrieval accuracy issue"
        
        # Test query functionality
        query_results = assoc_mem.query(test_vector, max_results=1)
        assert len(query_results) > 0, "Query returned no results"
        assert query_results[0]['similarity'] > 0.99, "Query similarity too low"
        
        # Test episodic memory
        episodic_mem = memory.get_episodic_memory()
        episode_vectors = [HyperVector.random(1000, seed=i) for i in range(5)]
        episodic_mem.store_episode(episode_vectors, importance=0.8)
        
        assert episodic_mem.size() == 1, "Episodic memory storage failed"
        
        # Test working memory
        working_mem = memory.get_working_memory()
        working_mem.push(test_vector)
        assert working_mem.size() == 1, "Working memory push failed"
        
        retrieved_working = working_mem.peek()
        assert retrieved_working.similarity(test_vector) > 0.99, "Working memory accuracy issue"
        
        results['associative_memory'] = True
        results['episodic_memory'] = True
        results['working_memory'] = True
        results['memory_accuracy'] = True
        
        return results
    
    def _validate_sensor_encoding(self) -> Dict[str, Any]:
        """Validate sensor encoding functionality."""
        results = {}
        
        encoder = SensorEncoder(1000)
        
        # Test LIDAR encoding
        lidar_ranges = [1.0, 2.5, 0.8, 3.0, 1.5] * 20
        lidar_hv = encoder.encode_lidar_scan(lidar_ranges)
        assert lidar_hv.dimension == 1000, "LIDAR encoding dimension error"
        
        # Test IMU encoding
        imu_hv = encoder.encode_imu_data(
            linear_accel=(0.1, -0.2, 9.8),
            angular_vel=(0.01, 0.02, -0.005)
        )
        assert imu_hv.dimension == 1000, "IMU encoding dimension error"
        
        # Test joint state encoding
        joint_positions = [0.1, -0.5, 1.2, 0.0, 0.8, -0.3]
        joint_hv = encoder.encode_joint_states(joint_positions)
        assert joint_hv.dimension == 1000, "Joint state encoding dimension error"
        
        # Test multi-modal fusion
        multimodal_data = {
            'lidar': lidar_ranges,
            'imu': ((0.1, -0.2, 9.8), (0.01, 0.02, -0.005)),
            'joints': joint_positions
        }
        fused_hv = encoder.encode_multimodal_state(multimodal_data)
        assert fused_hv.dimension == 1000, "Multi-modal fusion dimension error"
        
        # Test consistency
        fused_hv2 = encoder.encode_multimodal_state(multimodal_data)
        similarity = fused_hv.similarity(fused_hv2)
        assert similarity > 0.99, f"Encoding consistency issue: {similarity}"
        
        results['lidar_encoding'] = True
        results['imu_encoding'] = True
        results['joint_encoding'] = True
        results['multimodal_fusion'] = True
        results['encoding_consistency'] = True
        
        return results
    
    def _validate_behavior_learning(self) -> Dict[str, Any]:
        """Validate behavior learning functionality."""
        results = {}
        
        learner = BehaviorLearner(1000, similarity_threshold=0.7)
        
        # Create sample demonstration
        demo_states = []
        demo_actions = []
        
        for i in range(5):
            state = {
                'lidar': [1.0, 2.0, 1.5] * 10,
                'joints': [0.1 * i, -0.2 * i, 0.3 * i]
            }
            action = [0.1, 0.0, -0.1, 0.05 * i, 0.0, 0.02]
            
            demo_states.append(state)
            demo_actions.append(action)
        
        demonstration = {
            'states': demo_states,
            'actions': demo_actions
        }
        
        # Test one-shot learning
        confidence = learner.learn_from_demonstration(demonstration, 'test_behavior')
        assert confidence > 0.2, f"Learning confidence too low: {confidence}"
        
        # Test behavior query
        query_results = learner.query_behavior(demo_states[0], top_k=1)
        if query_results:  # May be empty if similarity threshold not met
            assert len(query_results) <= 1, "Query returned too many results"
        
        # Test behavior execution
        execution_result = learner.execute_behavior('test_behavior', demo_states[0])
        assert 'action' in execution_result, "Execution result missing action"
        assert 'confidence' in execution_result, "Execution result missing confidence"
        
        # Test analytics
        analytics = learner.analyze_learning_patterns()
        assert analytics['total_behaviors_learned'] == 1, "Analytics behavior count incorrect"
        
        results['one_shot_learning'] = True
        results['behavior_query'] = True
        results['behavior_execution'] = True
        results['learning_analytics'] = True
        
        return results
    
    def _validate_error_recovery(self) -> Dict[str, Any]:
        """Validate error recovery and fault tolerance."""
        results = {}
        
        recovery = AdvancedErrorRecovery(1000, enable_learning=False)
        
        # Test fault reporting
        fault_id = recovery.report_fault(
            'test_component',
            FaultSeverity.MEDIUM,
            'Test fault for validation',
            auto_recover=False
        )
        
        assert fault_id is not None, "Fault reporting failed"
        
        # Test health reporting
        health_report = recovery.get_system_health_report()
        assert 'system_state' in health_report, "Health report missing system state"
        assert health_report['active_faults_count'] == 1, "Active fault count incorrect"
        
        # Test fault resolution
        recovery.resolve_fault(fault_id, "Validation test resolved")
        
        updated_health = recovery.get_system_health_report()
        assert updated_health['active_faults_count'] == 0, "Fault resolution failed"
        
        results['fault_reporting'] = True
        results['health_monitoring'] = True
        results['fault_resolution'] = True
        
        return results
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance benchmarks."""
        results = {}
        
        # Vector operation benchmarks
        vectors = [HyperVector.random(1000, seed=i) for i in range(100)]
        
        # Bundle performance
        start_time = time.time()
        bundled = HyperVector.bundle_vectors(vectors)
        bundle_time = time.time() - start_time
        
        # Similarity performance
        query_vector = HyperVector.random(1000, seed=999)
        start_time = time.time()
        similarities = [query_vector.similarity(v) for v in vectors[:50]]
        similarity_time = time.time() - start_time
        
        # Memory performance
        memory = HierarchicalMemory(1000)
        start_time = time.time()
        for i, vector in enumerate(vectors[:20]):
            memory.store_experience(vector, vector, f'pattern_{i}', 0.8)
        memory_time = time.time() - start_time
        
        # Performance thresholds (adjust based on requirements)
        assert bundle_time < 0.1, f"Bundle operation too slow: {bundle_time:.4f}s"
        assert similarity_time < 0.1, f"Similarity batch too slow: {similarity_time:.4f}s"
        assert memory_time < 0.5, f"Memory operations too slow: {memory_time:.4f}s"
        
        self.performance_metrics.update({
            'bundle_100_vectors_ms': bundle_time * 1000,
            'similarity_50_vectors_ms': similarity_time * 1000,
            'memory_20_operations_ms': memory_time * 1000
        })
        
        results['vector_operations_performance'] = True
        results['memory_performance'] = True
        results['performance_thresholds_met'] = True
        
        return results
    
    def _validate_scalability(self) -> Dict[str, Any]:
        """Validate scalability features."""
        results = {}
        
        # Test GPU accelerator (falls back to CPU if no GPU)
        gpu_accel = get_gpu_accelerator()
        
        vectors = [HyperVector.random(1000, seed=i) for i in range(50)]
        
        # Test accelerated operations
        bundled = gpu_accel.accelerated_bundle(vectors)
        assert bundled.dimension == 1000, "GPU bundle dimension error"
        
        query_vector = HyperVector.random(1000, seed=999)
        similarities = gpu_accel.accelerated_similarity_batch(query_vector, vectors[:20])
        assert len(similarities) == 20, "GPU similarity batch size error"
        
        # Test distributed coordinator (basic initialization)
        coordinator = DistributedCoordinator(coordinator_port=8890, max_workers=2)
        cluster_status = coordinator.get_cluster_status()
        assert 'total_nodes' in cluster_status, "Cluster status missing total_nodes"
        
        results['gpu_acceleration'] = True
        results['distributed_coordination'] = True
        results['scalability_features'] = True
        
        return results
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security features."""
        results = {}
        
        # Test input validation
        try:
            invalid_hv = HyperVector(-100)  # Should raise ValueError
            assert False, "Invalid dimension not caught"
        except ValueError:
            pass  # Expected
        
        try:
            invalid_data = [2, 3, 4]  # Invalid bipolar data
            HyperVector(3, invalid_data)
            assert False, "Invalid data not caught"
        except ValueError:
            pass  # Expected
        
        # Test memory bounds
        memory = HierarchicalMemory(1000)
        assoc_mem = memory.get_associative_memory()
        
        try:
            wrong_dim_vector = HyperVector.random(500)
            assoc_mem.store("test", wrong_dim_vector)
            assert False, "Dimension mismatch not caught"
        except ValueError:
            pass  # Expected
        
        results['input_validation'] = True
        results['dimension_checking'] = True
        results['memory_bounds_checking'] = True
        
        return results
    
    def _validate_integration(self) -> Dict[str, Any]:
        """Validate end-to-end integration."""
        results = {}
        
        # Create integrated system
        memory = HierarchicalMemory(1000)
        encoder = SensorEncoder(1000)
        learner = BehaviorLearner(1000)
        
        # Simulate sensor data
        sensor_data = {
            'lidar': [1.0, 2.0, 1.5, 0.8, 2.5] * 20,
            'imu': ((0.1, -0.2, 9.8), (0.01, 0.02, -0.005)),
            'joints': [0.1, -0.5, 1.2, 0.0, 0.8, -0.3]
        }
        
        # Encode sensor data
        encoded_state = encoder.encode_multimodal_state(sensor_data)
        
        # Store in memory
        memory.store_experience(encoded_state, encoded_state, 'integrated_test', 0.9)
        
        # Learn behavior
        demo = {
            'states': [sensor_data] * 3,
            'actions': [[0.1, 0.0, -0.1, 0.05, 0.0, 0.02]] * 3
        }
        confidence = learner.learn_from_demonstration(demo, 'integrated_behavior')
        
        # Query and execute
        query_results = memory.query_experience(encoded_state, 1)
        assert len(query_results) > 0, "Integration query failed"
        
        execution = learner.execute_behavior('integrated_behavior', sensor_data)
        assert 'action' in execution, "Integration execution failed"
        
        results['sensor_to_memory_integration'] = True
        results['memory_to_learning_integration'] = True
        results['end_to_end_pipeline'] = True
        
        return results
    
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final validation report."""
        total_time = time.time() - self.start_time
        
        passed_validations = sum(1 for r in self.validation_results.values() if r['status'] == 'PASS')
        total_validations = len(self.validation_results)
        
        failed_validations = [name for name, result in self.validation_results.items() 
                            if result['status'] != 'PASS']
        
        overall_status = "PASS" if passed_validations == total_validations else "FAIL"
        
        report = {
            'validation_timestamp': time.time(),
            'total_validation_time': total_time,
            'overall_status': overall_status,
            'passed_validations': passed_validations,
            'total_validations': total_validations,
            'success_rate': passed_validations / total_validations,
            'failed_validations': failed_validations,
            'detailed_results': self.validation_results,
            'performance_metrics': self.performance_metrics,
            'system_info': {
                'python_version': sys.version,
                'platform': sys.platform,
            }
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"🎯 VALIDATION SUMMARY")
        print(f"{'='*60}")
        print(f"Overall Status: {'✅ PASS' if overall_status == 'PASS' else '❌ FAIL'}")
        print(f"Success Rate: {passed_validations}/{total_validations} ({report['success_rate']*100:.1f}%)")
        print(f"Total Time: {total_time:.2f}s")
        
        if failed_validations:
            print(f"\nFailed Validations: {', '.join(failed_validations)}")
        
        if self.performance_metrics:
            print(f"\nPerformance Metrics:")
            for metric, value in self.performance_metrics.items():
                print(f"  {metric}: {value:.2f}")
        
        return report


def main():
    """Main validation entry point."""
    try:
        validator = ProductionValidator()
        report = validator.run_all_validations()
        
        # Save report
        report_file = Path("/tmp/hdc_validation_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📋 Detailed report saved to: {report_file}")
        
        # Exit with appropriate code
        sys.exit(0 if report['overall_status'] == 'PASS' else 1)
        
    except Exception as e:
        print(f"\n❌ Validation system error: {str(e)}")
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()