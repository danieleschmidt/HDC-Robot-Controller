"""
Comprehensive Quality Gates Testing Suite
Enterprise-grade testing for HDC Robot Controller.
"""

import pytest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import logging
import json

# Import all core modules for testing
from hdc_robot_controller.core.hypervector import HyperVector, weighted_bundle
from hdc_robot_controller.core.adaptive_learning import (
    AdaptiveLearningEngine, LearningContext, LearningExample
)
from hdc_robot_controller.core.sensor_fusion import (
    MultiModalSensorFusion, SensorReading, SensorModality
)
from hdc_robot_controller.robustness.advanced_fault_tolerance import (
    AdvancedFaultTolerantSystem, FailureType, SeverityLevel
)
from hdc_robot_controller.security.enhanced_security import (
    EnhancedSecurityFramework, SecurityLevel
)

logger = logging.getLogger(__name__)


class QualityGateValidator:
    """Comprehensive quality gate validation system."""
    
    def __init__(self):
        self.test_results = {}
        self.performance_benchmarks = {}
        self.security_audit_results = {}
        self.quality_score = 0.0
        
        # Quality thresholds
        self.quality_thresholds = {
            'test_coverage': 95.0,
            'performance_threshold': 0.2,  # 200ms max response time
            'security_score': 90.0,
            'fault_tolerance_rate': 85.0,
            'memory_efficiency': 80.0,
            'overall_quality': 95.0
        }
        
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all quality gate tests."""
        logger.info("Starting comprehensive quality gate validation")
        
        results = {
            'timestamp': time.time(),
            'test_results': {},
            'performance_results': {},
            'security_results': {},
            'fault_tolerance_results': {},
            'quality_score': 0.0,
            'passed_gates': [],
            'failed_gates': [],
            'recommendations': []
        }
        
        try:
            # Core functionality tests
            results['test_results']['core_tests'] = self._test_core_functionality()
            
            # Performance benchmarks
            results['performance_results'] = self._run_performance_benchmarks()
            
            # Security validation
            results['security_results'] = self._run_security_validation()
            
            # Fault tolerance testing
            results['fault_tolerance_results'] = self._test_fault_tolerance()
            
            # Memory and resource tests
            results['resource_tests'] = self._test_resource_efficiency()
            
            # Calculate overall quality score
            results['quality_score'] = self._calculate_quality_score(results)
            
            # Determine passed/failed gates
            self._evaluate_quality_gates(results)
            
            # Generate recommendations
            results['recommendations'] = self._generate_recommendations(results)
            
            logger.info(f"Quality gate validation completed. Score: {results['quality_score']:.1f}/100")
            
        except Exception as e:
            logger.error(f"Quality gate validation failed: {e}")
            results['error'] = str(e)
            
        return results
        
    def _test_core_functionality(self) -> Dict[str, Any]:
        """Test core HDC functionality."""
        core_results = {
            'hypervector_operations': self._test_hypervector_operations(),
            'adaptive_learning': self._test_adaptive_learning(),
            'sensor_fusion': self._test_sensor_fusion(),
            'encoding_decoding': self._test_encoding_decoding(),
            'memory_operations': self._test_memory_operations()
        }
        
        # Calculate success rate
        total_tests = sum(len(test_group) for test_group in core_results.values())
        passed_tests = sum(
            sum(1 for result in test_group.values() if result.get('passed', False))
            for test_group in core_results.values()
        )
        
        core_results['success_rate'] = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        
        return core_results
        
    def _test_hypervector_operations(self) -> Dict[str, Dict[str, Any]]:
        """Test hypervector operations."""
        tests = {}
        
        # Test vector creation
        tests['vector_creation'] = self._run_test(
            "Vector Creation",
            lambda: self._test_vector_creation()
        )
        
        # Test bundling operations
        tests['bundling'] = self._run_test(
            "Bundling Operations",
            lambda: self._test_bundling_operations()
        )
        
        # Test binding operations
        tests['binding'] = self._run_test(
            "Binding Operations", 
            lambda: self._test_binding_operations()
        )
        
        # Test similarity computation
        tests['similarity'] = self._run_test(
            "Similarity Computation",
            lambda: self._test_similarity_computation()
        )
        
        # Test vector properties
        tests['vector_properties'] = self._run_test(
            "Vector Properties",
            lambda: self._test_vector_properties()
        )
        
        return tests
        
    def _test_vector_creation(self):
        """Test hypervector creation."""
        # Test random vector creation
        hv_random = HyperVector.random(10000, seed=42)
        assert hv_random.dimension == 10000
        assert len(hv_random.data) == 10000
        assert np.all(np.isin(hv_random.data, [-1, 1]))
        
        # Test zero vector creation
        hv_zero = HyperVector.zero(5000)
        assert hv_zero.dimension == 5000
        assert np.all(hv_zero.data == 0)
        
        # Test vector from data
        data = np.random.choice([-1, 1], size=1000)
        hv_data = HyperVector(1000, data)
        assert np.array_equal(hv_data.data, data)
        
    def _test_bundling_operations(self):
        """Test vector bundling."""
        vectors = [HyperVector.random(1000, seed=i) for i in range(5)]
        
        # Test basic bundling
        bundled = HyperVector.bundle_vectors(vectors)
        assert bundled.dimension == 1000
        assert np.all(np.isin(bundled.data, [-1, 1]))
        
        # Test weighted bundling
        weights = [0.1, 0.2, 0.3, 0.2, 0.2]
        weighted_bundled = weighted_bundle(list(zip(vectors, weights)))
        assert weighted_bundled.dimension == 1000
        
        # Test bundle operator
        bundled_op = vectors[0] + vectors[1]
        assert bundled_op.dimension == 1000
        
    def _test_binding_operations(self):
        """Test vector binding."""
        hv1 = HyperVector.random(1000, seed=1)
        hv2 = HyperVector.random(1000, seed=2)
        
        # Test basic binding
        bound = hv1.bind(hv2)
        assert bound.dimension == 1000
        assert np.all(np.isin(bound.data, [-1, 1]))
        
        # Test bind operator
        bound_op = hv1 * hv2
        assert np.array_equal(bound.data, bound_op.data)
        
        # Test binding properties (involution)
        unbound = bound.bind(hv2)
        similarity = hv1.similarity(unbound)
        assert similarity > 0.8  # Should be highly similar to original
        
    def _test_similarity_computation(self):
        """Test similarity computation."""
        hv1 = HyperVector.random(1000, seed=1)
        hv2 = HyperVector.random(1000, seed=2)
        
        # Test self-similarity
        self_sim = hv1.similarity(hv1)
        assert abs(self_sim - 1.0) < 1e-10
        
        # Test different vector similarity
        diff_sim = hv1.similarity(hv2)
        assert -1.0 <= diff_sim <= 1.0
        assert abs(diff_sim) < 0.2  # Should be close to 0 for random vectors
        
        # Test inverted vector similarity
        hv1_inv = hv1.invert()
        inv_sim = hv1.similarity(hv1_inv)
        assert abs(inv_sim + 1.0) < 1e-10  # Should be -1
        
    def _test_vector_properties(self):
        """Test vector properties and methods."""
        hv = HyperVector.random(1000, seed=42)
        
        # Test permutation
        permuted = hv.permute(100)
        assert permuted.dimension == hv.dimension
        assert not np.array_equal(hv.data, permuted.data)
        
        # Test noise addition
        noisy = hv.add_noise(0.1)
        assert noisy.dimension == hv.dimension
        similarity = hv.similarity(noisy)
        assert 0.6 < similarity < 1.0  # Should be similar but not identical
        
        # Test serialization
        hv_bytes = hv.to_bytes()
        assert isinstance(hv_bytes, bytes)
        
        # Test sparsity
        sparsity = hv.sparsity()
        assert 0.0 <= sparsity <= 1.0
        
    def _test_adaptive_learning(self) -> Dict[str, Dict[str, Any]]:
        """Test adaptive learning functionality."""
        tests = {}
        
        tests['learning_engine_creation'] = self._run_test(
            "Learning Engine Creation",
            lambda: self._test_learning_engine_creation()
        )
        
        tests['one_shot_learning'] = self._run_test(
            "One-Shot Learning",
            lambda: self._test_one_shot_learning()
        )
        
        tests['behavior_adaptation'] = self._run_test(
            "Behavior Adaptation",
            lambda: self._test_behavior_adaptation()
        )
        
        tests['behavior_execution'] = self._run_test(
            "Behavior Execution",
            lambda: self._test_behavior_execution()
        )
        
        return tests
        
    def _test_learning_engine_creation(self):
        """Test adaptive learning engine creation."""
        engine = AdaptiveLearningEngine(
            dimension=1000,
            memory_capacity=100,
            adaptation_rate=0.1
        )
        
        assert engine.dimension == 1000
        assert engine.memory_capacity == 100
        assert engine.adaptation_rate == 0.1
        assert hasattr(engine, 'behavior_library')
        assert hasattr(engine, 'episodic_memory')
        
    def _test_one_shot_learning(self):
        """Test one-shot learning capability."""
        engine = AdaptiveLearningEngine(dimension=1000)
        
        # Create demonstration data
        state_action_pairs = [
            (HyperVector.random(1000, seed=i), HyperVector.random(1000, seed=i+100))
            for i in range(5)
        ]
        
        context = LearningContext(
            environment_id="test_env",
            task_type="manipulation",
            sensor_modalities=["camera", "lidar"]
        )
        
        # Learn from demonstration
        success = engine.learn_from_demonstration(
            state_action_pairs=state_action_pairs,
            task_name="pick_object",
            context=context,
            reward_signal=1.0
        )
        
        assert success is True
        assert "pick_object" in engine.behavior_library
        
        # Check learned behavior properties
        behavior_data = engine.behavior_library["pick_object"]
        assert behavior_data['reward'] == 1.0
        assert behavior_data['context'] == context
        assert len(behavior_data['examples']) == 5
        
    def _test_behavior_adaptation(self):
        """Test behavior adaptation capability."""
        engine = AdaptiveLearningEngine(dimension=1000)
        
        # First learn a base behavior
        base_pairs = [
            (HyperVector.random(1000, seed=i), HyperVector.random(1000, seed=i+50))
            for i in range(3)
        ]
        
        context = LearningContext(
            environment_id="test_env",
            task_type="manipulation",
            sensor_modalities=["camera"]
        )
        
        engine.learn_from_demonstration(base_pairs, "base_behavior", context)
        
        # Create adaptation examples
        adaptation_examples = [
            LearningExample(
                state_vector=HyperVector.random(1000, seed=i+200),
                action_vector=HyperVector.random(1000, seed=i+250),
                reward=0.8,
                context=context
            ) for i in range(2)
        ]
        
        # Adapt behavior
        success = engine.adapt_behavior(
            base_behavior="base_behavior",
            adaptation_examples=adaptation_examples,
            new_behavior_name="adapted_behavior",
            adaptation_strength=0.3
        )
        
        assert success is True
        assert "adapted_behavior" in engine.behavior_library
        
        # Check adaptation properties
        adapted_data = engine.behavior_library["adapted_behavior"]
        assert adapted_data['base_behavior'] == "base_behavior"
        assert adapted_data['adaptation_strength'] == 0.3
        
    def _test_behavior_execution(self):
        """Test behavior execution."""
        engine = AdaptiveLearningEngine(dimension=1000)
        
        # Learn a behavior first
        state_action_pairs = [
            (HyperVector.random(1000, seed=i), HyperVector.random(1000, seed=i+100))
            for i in range(3)
        ]
        
        context = LearningContext(
            environment_id="test_env",
            task_type="test",
            sensor_modalities=["test"]
        )
        
        engine.learn_from_demonstration(state_action_pairs, "test_behavior", context)
        
        # Execute behavior
        current_state = HyperVector.random(1000, seed=999)
        action = engine.execute_behavior("test_behavior", current_state, context)
        
        assert action is not None
        assert isinstance(action, HyperVector)
        assert action.dimension == 1000
        
    def _test_sensor_fusion(self) -> Dict[str, Dict[str, Any]]:
        """Test sensor fusion functionality."""
        tests = {}
        
        tests['fusion_engine_creation'] = self._run_test(
            "Fusion Engine Creation",
            lambda: self._test_fusion_engine_creation()
        )
        
        tests['sensor_encoding'] = self._run_test(
            "Sensor Encoding",
            lambda: self._test_sensor_encoding()
        )
        
        tests['multimodal_fusion'] = self._run_test(
            "Multimodal Fusion",
            lambda: self._test_multimodal_fusion()
        )
        
        tests['temporal_fusion'] = self._run_test(
            "Temporal Fusion",
            lambda: self._test_temporal_fusion()
        )
        
        return tests
        
    def _test_fusion_engine_creation(self):
        """Test sensor fusion engine creation."""
        fusion_engine = MultiModalSensorFusion(
            dimension=1000,
            temporal_window=5,
            confidence_threshold=0.3
        )
        
        assert fusion_engine.dimension == 1000
        assert fusion_engine.temporal_window == 5
        assert fusion_engine.confidence_threshold == 0.3
        assert hasattr(fusion_engine, 'encoders')
        assert hasattr(fusion_engine, 'sensor_history')
        
    def _test_sensor_encoding(self):
        """Test individual sensor encoding."""
        fusion_engine = MultiModalSensorFusion(dimension=1000)
        
        # Test LIDAR encoding
        lidar_data = np.random.uniform(0.1, 10.0, size=360)  # 360-degree scan
        lidar_reading = SensorReading(
            modality=SensorModality.LIDAR,
            data=lidar_data,
            timestamp=time.time(),
            confidence=0.9
        )
        
        fusion_engine.add_sensor_reading(lidar_reading)
        assert len(fusion_engine.sensor_history[SensorModality.LIDAR]) == 1
        
        # Test camera encoding
        camera_data = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        camera_reading = SensorReading(
            modality=SensorModality.CAMERA,
            data=camera_data,
            timestamp=time.time(),
            confidence=0.8
        )
        
        fusion_engine.add_sensor_reading(camera_reading)
        assert len(fusion_engine.sensor_history[SensorModality.CAMERA]) == 1
        
    def _test_multimodal_fusion(self):
        """Test multimodal sensor fusion."""
        fusion_engine = MultiModalSensorFusion(dimension=1000)
        
        # Add multiple sensor readings
        sensors_data = {
            SensorModality.LIDAR: np.random.uniform(0.1, 10.0, size=360),
            SensorModality.CAMERA: np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
            SensorModality.IMU: {
                'accel': np.random.normal(0, 1, size=3),
                'gyro': np.random.normal(0, 0.1, size=3)
            }
        }
        
        for modality, data in sensors_data.items():
            reading = SensorReading(
                modality=modality,
                data=data,
                timestamp=time.time(),
                confidence=0.9
            )
            fusion_engine.add_sensor_reading(reading)
            
        # Perform fusion
        fused_percept = fusion_engine.fuse_sensors()
        
        assert fused_percept is not None
        assert isinstance(fused_percept.perception_vector, HyperVector)
        assert fused_percept.perception_vector.dimension == 1000
        assert len(fused_percept.contributing_modalities) > 0
        assert 0.0 <= fused_percept.fusion_confidence <= 1.0
        
    def _test_temporal_fusion(self):
        """Test temporal fusion capability."""
        fusion_engine = MultiModalSensorFusion(dimension=1000, temporal_window=3)
        
        # Add multiple timesteps of sensor data
        for t in range(5):
            lidar_reading = SensorReading(
                modality=SensorModality.LIDAR,
                data=np.random.uniform(0.1, 10.0, size=360),
                timestamp=time.time() + t * 0.1,
                confidence=0.9
            )
            fusion_engine.add_sensor_reading(lidar_reading)
            
        # Test temporal fusion
        fused_percept = fusion_engine.fuse_sensors(temporal_fusion=True)
        
        assert fused_percept.temporal_context is not None
        assert len(fusion_engine.sensor_history[SensorModality.LIDAR]) <= 3  # Temporal window
        
    def _test_encoding_decoding(self) -> Dict[str, Dict[str, Any]]:
        """Test encoding and decoding operations."""
        tests = {}
        
        tests['data_serialization'] = self._run_test(
            "Data Serialization",
            lambda: self._test_data_serialization()
        )
        
        tests['hypervector_conversion'] = self._run_test(
            "Hypervector Conversion", 
            lambda: self._test_hypervector_conversion()
        )
        
        return tests
        
    def _test_data_serialization(self):
        """Test data serialization and deserialization."""
        hv = HyperVector.random(1000, seed=42)
        
        # Test byte conversion
        hv_bytes = hv.to_bytes()
        assert isinstance(hv_bytes, bytes)
        
        # Test restoration from bytes
        hv_restored = HyperVector.zero(1000)
        hv_restored.from_bytes(hv_bytes)
        
        # Should be very similar (not exact due to bit packing)
        similarity = hv.similarity(hv_restored)
        assert similarity > 0.9
        
    def _test_hypervector_conversion(self):
        """Test hypervector conversion methods."""
        data = np.random.choice([-1, 1], size=1000)
        
        # Test numpy conversion
        hv = HyperVector.from_numpy(data)
        assert hv.dimension == 1000
        assert np.array_equal(hv.data, data)
        
        # Test to numpy
        numpy_data = hv.to_numpy()
        assert np.array_equal(numpy_data, data)
        
        # Test copy
        hv_copy = hv.copy()
        assert hv == hv_copy
        assert hv is not hv_copy  # Different objects
        
    def _test_memory_operations(self) -> Dict[str, Dict[str, Any]]:
        """Test memory operations."""
        tests = {}
        
        tests['memory_efficiency'] = self._run_test(
            "Memory Efficiency",
            lambda: self._test_memory_efficiency()
        )
        
        tests['large_vector_operations'] = self._run_test(
            "Large Vector Operations",
            lambda: self._test_large_vector_operations()
        )
        
        return tests
        
    def _test_memory_efficiency(self):
        """Test memory efficiency of operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create many vectors
        vectors = []
        for i in range(100):
            hv = HyperVector.random(10000, seed=i)
            vectors.append(hv)
            
        # Perform operations
        bundled = HyperVector.bundle_vectors(vectors[:50])
        
        # Check memory usage
        final_memory = process.memory_info().rss
        memory_increase = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        # Memory increase should be reasonable (< 100MB for this test)
        assert memory_increase < 100
        
    def _test_large_vector_operations(self):
        """Test operations with large vectors."""
        # Test with large dimension
        large_hv1 = HyperVector.random(50000, seed=1)
        large_hv2 = HyperVector.random(50000, seed=2)
        
        # Test operations complete without error
        bundled = large_hv1.bundle(large_hv2)
        assert bundled.dimension == 50000
        
        bound = large_hv1.bind(large_hv2)
        assert bound.dimension == 50000
        
        similarity = large_hv1.similarity(large_hv2)
        assert -1.0 <= similarity <= 1.0
        
    def _run_performance_benchmarks(self) -> Dict[str, Any]:
        """Run performance benchmarks."""
        benchmarks = {}
        
        # Hypervector operations benchmark
        benchmarks['hypervector_ops'] = self._benchmark_hypervector_operations()
        
        # Learning operations benchmark
        benchmarks['learning_ops'] = self._benchmark_learning_operations()
        
        # Sensor fusion benchmark
        benchmarks['sensor_fusion'] = self._benchmark_sensor_fusion()
        
        # Memory operations benchmark
        benchmarks['memory_ops'] = self._benchmark_memory_operations()
        
        return benchmarks
        
    def _benchmark_hypervector_operations(self) -> Dict[str, float]:
        """Benchmark core hypervector operations."""
        dimension = 10000
        num_operations = 1000
        
        # Bundle benchmark
        vectors = [HyperVector.random(dimension) for _ in range(100)]
        start_time = time.time()
        for _ in range(num_operations // 100):
            HyperVector.bundle_vectors(vectors)
        bundle_time = (time.time() - start_time) / (num_operations // 100)
        
        # Bind benchmark
        hv1 = HyperVector.random(dimension)
        hv2 = HyperVector.random(dimension)
        start_time = time.time()
        for _ in range(num_operations):
            hv1.bind(hv2)
        bind_time = (time.time() - start_time) / num_operations
        
        # Similarity benchmark
        start_time = time.time()
        for _ in range(num_operations):
            hv1.similarity(hv2)
        similarity_time = (time.time() - start_time) / num_operations
        
        return {
            'bundle_time_ms': bundle_time * 1000,
            'bind_time_ms': bind_time * 1000,
            'similarity_time_ms': similarity_time * 1000,
            'bundle_throughput': 1.0 / bundle_time,
            'bind_throughput': 1.0 / bind_time,
            'similarity_throughput': 1.0 / similarity_time
        }
        
    def _benchmark_learning_operations(self) -> Dict[str, float]:
        """Benchmark learning operations."""
        engine = AdaptiveLearningEngine(dimension=1000)
        
        # One-shot learning benchmark
        state_action_pairs = [
            (HyperVector.random(1000, seed=i), HyperVector.random(1000, seed=i+100))
            for i in range(10)
        ]
        
        context = LearningContext(
            environment_id="benchmark",
            task_type="test",
            sensor_modalities=["test"]
        )
        
        start_time = time.time()
        engine.learn_from_demonstration(state_action_pairs, "benchmark_task", context)
        learning_time = time.time() - start_time
        
        # Behavior execution benchmark
        current_state = HyperVector.random(1000)
        start_time = time.time()
        for _ in range(100):
            engine.execute_behavior("benchmark_task", current_state)
        execution_time = (time.time() - start_time) / 100
        
        return {
            'learning_time_ms': learning_time * 1000,
            'execution_time_ms': execution_time * 1000,
            'learning_throughput': 1.0 / learning_time,
            'execution_throughput': 1.0 / execution_time
        }
        
    def _benchmark_sensor_fusion(self) -> Dict[str, float]:
        """Benchmark sensor fusion operations."""
        fusion_engine = MultiModalSensorFusion(dimension=1000)
        
        # Add sensor data
        lidar_reading = SensorReading(
            modality=SensorModality.LIDAR,
            data=np.random.uniform(0.1, 10.0, size=360),
            timestamp=time.time(),
            confidence=0.9
        )
        
        camera_reading = SensorReading(
            modality=SensorModality.CAMERA,
            data=np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8),
            timestamp=time.time(),
            confidence=0.8
        )
        
        imu_reading = SensorReading(
            modality=SensorModality.IMU,
            data={'accel': np.random.normal(0, 1, size=3), 'gyro': np.random.normal(0, 0.1, size=3)},
            timestamp=time.time(),
            confidence=0.95
        )
        
        fusion_engine.add_sensor_reading(lidar_reading)
        fusion_engine.add_sensor_reading(camera_reading)
        fusion_engine.add_sensor_reading(imu_reading)
        
        # Benchmark fusion
        start_time = time.time()
        for _ in range(100):
            fusion_engine.fuse_sensors()
        fusion_time = (time.time() - start_time) / 100
        
        return {
            'fusion_time_ms': fusion_time * 1000,
            'fusion_throughput': 1.0 / fusion_time
        }
        
    def _benchmark_memory_operations(self) -> Dict[str, float]:
        """Benchmark memory operations."""
        # Memory allocation benchmark
        start_time = time.time()
        vectors = [HyperVector.random(10000) for _ in range(1000)]
        allocation_time = time.time() - start_time
        
        # Memory access benchmark
        start_time = time.time()
        for _ in range(1000):
            _ = vectors[np.random.randint(0, len(vectors))].data[np.random.randint(0, 10000)]
        access_time = (time.time() - start_time) / 1000
        
        return {
            'allocation_time_ms': allocation_time * 1000,
            'access_time_us': access_time * 1000000,
            'allocation_throughput': 1000 / allocation_time,
            'access_throughput': 1.0 / access_time
        }
        
    def _run_security_validation(self) -> Dict[str, Any]:
        """Run security validation tests."""
        security_results = {}
        
        try:
            security_framework = EnhancedSecurityFramework(dimension=1000)
            
            # Test authentication
            auth_token = security_framework.authenticate("admin", "admin")  # Use default admin
            security_results['authentication_test'] = {
                'passed': auth_token is not None,
                'has_token': auth_token is not None
            }
            
            # Test encryption
            test_hv = HyperVector.random(1000)
            encrypted_data, key_id = security_framework.encrypt_hypervector(test_hv)
            security_results['encryption_test'] = {
                'passed': encrypted_data is not None and key_id is not None,
                'encrypted_size': len(encrypted_data) if encrypted_data else 0
            }
            
            # Test security audit
            audit_results = security_framework.run_security_audit()
            security_results['security_audit'] = {
                'passed': 'error' not in audit_results,
                'recommendations_count': len(audit_results.get('recommendations', []))
            }
            
            # Calculate security score
            passed_tests = sum(1 for test in security_results.values() if test.get('passed', False))
            total_tests = len(security_results)
            security_results['security_score'] = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
        except Exception as e:
            security_results['error'] = str(e)
            security_results['security_score'] = 0
            
        return security_results
        
    def _test_fault_tolerance(self) -> Dict[str, Any]:
        """Test fault tolerance capabilities."""
        fault_tolerance_results = {}
        
        try:
            ft_system = AdvancedFaultTolerantSystem(
                dimension=1000,
                monitoring_interval=0.1,
                enable_predictive_detection=True,
                enable_self_healing=True
            )
            
            # Test failure simulation
            simulated_failure = ft_system.simulate_failure(FailureType.SENSOR_DROPOUT, SeverityLevel.MEDIUM)
            fault_tolerance_results['failure_simulation'] = {
                'passed': simulated_failure is not None,
                'failure_type': simulated_failure.failure_type.value if simulated_failure else None
            }
            
            # Test diagnostic
            diagnostic_results = ft_system.run_diagnostic()
            fault_tolerance_results['diagnostic_test'] = {
                'passed': 'error' not in diagnostic_results,
                'monitoring_active': diagnostic_results.get('monitoring_active', False)
            }
            
            # Test statistics
            stats = ft_system.get_fault_tolerance_statistics()
            fault_tolerance_results['statistics_test'] = {
                'passed': isinstance(stats, dict) and len(stats) > 0,
                'stats_count': len(stats)
            }
            
            # Calculate fault tolerance score
            passed_tests = sum(1 for test in fault_tolerance_results.values() 
                             if isinstance(test, dict) and test.get('passed', False))
            total_tests = sum(1 for test in fault_tolerance_results.values() if isinstance(test, dict))
            fault_tolerance_results['fault_tolerance_score'] = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
        except Exception as e:
            fault_tolerance_results['error'] = str(e)
            fault_tolerance_results['fault_tolerance_score'] = 0
            
        return fault_tolerance_results
        
    def _test_resource_efficiency(self) -> Dict[str, Any]:
        """Test resource efficiency and optimization."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        initial_cpu = process.cpu_percent()
        
        # Perform resource-intensive operations
        start_time = time.time()
        
        # Create and manipulate many vectors
        vectors = []
        for i in range(500):
            hv = HyperVector.random(5000, seed=i)
            vectors.append(hv)
            
        # Bundle operations
        for i in range(0, len(vectors), 50):
            HyperVector.bundle_vectors(vectors[i:i+50])
            
        execution_time = time.time() - start_time
        
        # Check final resource usage
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        final_cpu = process.cpu_percent()
        
        memory_increase = final_memory - initial_memory
        memory_efficiency = 100 - min(100, (memory_increase / 500) * 100)  # Efficiency percentage
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': memory_increase,
            'memory_efficiency_score': max(0, memory_efficiency),
            'execution_time_seconds': execution_time,
            'cpu_usage': final_cpu,
            'operations_per_second': 500 / execution_time if execution_time > 0 else 0
        }
        
    def _calculate_quality_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        scores = []
        weights = []
        
        # Core functionality score (40% weight)
        if 'test_results' in results:
            core_score = results['test_results'].get('success_rate', 0)
            scores.append(core_score)
            weights.append(0.4)
            
        # Performance score (25% weight)
        if 'performance_results' in results:
            perf_results = results['performance_results']
            
            # Check if key operations meet performance thresholds
            bundle_time = perf_results.get('hypervector_ops', {}).get('bundle_time_ms', 1000)
            fusion_time = perf_results.get('sensor_fusion', {}).get('fusion_time_ms', 1000)
            
            # Performance score based on meeting thresholds
            perf_score = 0
            if bundle_time < 50:  # 50ms threshold
                perf_score += 50
            if fusion_time < 100:  # 100ms threshold
                perf_score += 50
                
            scores.append(perf_score)
            weights.append(0.25)
            
        # Security score (20% weight)
        if 'security_results' in results:
            security_score = results['security_results'].get('security_score', 0)
            scores.append(security_score)
            weights.append(0.2)
            
        # Fault tolerance score (10% weight)
        if 'fault_tolerance_results' in results:
            ft_score = results['fault_tolerance_results'].get('fault_tolerance_score', 0)
            scores.append(ft_score)
            weights.append(0.1)
            
        # Resource efficiency score (5% weight)
        if 'resource_tests' in results:
            resource_score = results['resource_tests'].get('memory_efficiency_score', 0)
            scores.append(resource_score)
            weights.append(0.05)
            
        # Calculate weighted average
        if scores and weights:
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            total_weight = sum(weights)
            return weighted_sum / total_weight if total_weight > 0 else 0
        else:
            return 0
            
    def _evaluate_quality_gates(self, results: Dict[str, Any]):
        """Evaluate which quality gates passed or failed."""
        quality_score = results['quality_score']
        
        # Core functionality gate
        core_success_rate = results.get('test_results', {}).get('success_rate', 0)
        if core_success_rate >= self.quality_thresholds['test_coverage']:
            results['passed_gates'].append('Core Functionality')
        else:
            results['failed_gates'].append('Core Functionality')
            
        # Performance gate
        perf_results = results.get('performance_results', {})
        bundle_time = perf_results.get('hypervector_ops', {}).get('bundle_time_ms', 1000)
        if bundle_time <= self.quality_thresholds['performance_threshold'] * 1000:
            results['passed_gates'].append('Performance')
        else:
            results['failed_gates'].append('Performance')
            
        # Security gate
        security_score = results.get('security_results', {}).get('security_score', 0)
        if security_score >= self.quality_thresholds['security_score']:
            results['passed_gates'].append('Security')
        else:
            results['failed_gates'].append('Security')
            
        # Overall quality gate
        if quality_score >= self.quality_thresholds['overall_quality']:
            results['passed_gates'].append('Overall Quality')
        else:
            results['failed_gates'].append('Overall Quality')
            
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations based on test results."""
        recommendations = []
        
        # Core functionality recommendations
        core_success_rate = results.get('test_results', {}).get('success_rate', 0)
        if core_success_rate < 95:
            recommendations.append("Improve core functionality test coverage - some tests are failing")
            
        # Performance recommendations
        perf_results = results.get('performance_results', {})
        bundle_time = perf_results.get('hypervector_ops', {}).get('bundle_time_ms', 0)
        if bundle_time > 200:
            recommendations.append("Optimize hypervector bundling operations for better performance")
            
        fusion_time = perf_results.get('sensor_fusion', {}).get('fusion_time_ms', 0)
        if fusion_time > 100:
            recommendations.append("Optimize sensor fusion pipeline for real-time performance")
            
        # Security recommendations
        security_score = results.get('security_results', {}).get('security_score', 0)
        if security_score < 90:
            recommendations.append("Enhance security measures - some security tests failed")
            
        # Resource efficiency recommendations
        memory_efficiency = results.get('resource_tests', {}).get('memory_efficiency_score', 100)
        if memory_efficiency < 80:
            recommendations.append("Optimize memory usage - high memory consumption detected")
            
        # Overall recommendations
        if results['quality_score'] < 95:
            recommendations.append("Overall system quality below enterprise standards - review failed components")
            
        return recommendations
        
    def _run_test(self, test_name: str, test_func: callable) -> Dict[str, Any]:
        """Run individual test with error handling."""
        try:
            start_time = time.time()
            test_func()
            execution_time = time.time() - start_time
            
            return {
                'test_name': test_name,
                'passed': True,
                'execution_time': execution_time,
                'error': None
            }
        except Exception as e:
            return {
                'test_name': test_name,
                'passed': False,
                'execution_time': 0,
                'error': str(e)
            }


# Main test functions for pytest
def test_comprehensive_quality_gates():
    """Main comprehensive quality gates test."""
    validator = QualityGateValidator()
    results = validator.run_comprehensive_tests()
    
    # Assert that overall quality meets minimum standards
    assert results['quality_score'] >= 85.0, f"Quality score {results['quality_score']:.1f} below minimum threshold of 85.0"
    
    # Assert that critical gates pass
    assert 'Core Functionality' in results['passed_gates'], "Core functionality quality gate failed"
    
    # Print results for debugging
    print(f"\nQuality Gate Results:")
    print(f"Overall Quality Score: {results['quality_score']:.1f}/100")
    print(f"Passed Gates: {', '.join(results['passed_gates'])}")
    print(f"Failed Gates: {', '.join(results['failed_gates'])}")
    
    if results['recommendations']:
        print(f"Recommendations:")
        for rec in results['recommendations']:
            print(f"  - {rec}")


def test_hypervector_operations():
    """Test core hypervector operations."""
    validator = QualityGateValidator()
    results = validator._test_hypervector_operations()
    
    # Ensure all hypervector tests pass
    for test_name, test_result in results.items():
        assert test_result['passed'], f"Hypervector test '{test_name}' failed: {test_result.get('error')}"


def test_adaptive_learning():
    """Test adaptive learning functionality."""
    validator = QualityGateValidator()
    results = validator._test_adaptive_learning()
    
    # Ensure all learning tests pass
    for test_name, test_result in results.items():
        assert test_result['passed'], f"Learning test '{test_name}' failed: {test_result.get('error')}"


def test_sensor_fusion():
    """Test sensor fusion functionality."""
    validator = QualityGateValidator()
    results = validator._test_sensor_fusion()
    
    # Ensure all sensor fusion tests pass
    for test_name, test_result in results.items():
        assert test_result['passed'], f"Sensor fusion test '{test_name}' failed: {test_result.get('error')}"


def test_performance_benchmarks():
    """Test performance meets requirements."""
    validator = QualityGateValidator()
    results = validator._run_performance_benchmarks()
    
    # Check performance thresholds
    bundle_time = results.get('hypervector_ops', {}).get('bundle_time_ms', 1000)
    assert bundle_time < 200, f"Bundle operation too slow: {bundle_time:.2f}ms"
    
    fusion_time = results.get('sensor_fusion', {}).get('fusion_time_ms', 1000)
    assert fusion_time < 100, f"Sensor fusion too slow: {fusion_time:.2f}ms"


if __name__ == "__main__":
    # Run comprehensive quality gates validation
    validator = QualityGateValidator()
    results = validator.run_comprehensive_tests()
    
    # Save results to file
    output_file = Path(__file__).parent / "quality_gate_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Quality gate results saved to: {output_file}")
    print(f"Overall Quality Score: {results['quality_score']:.1f}/100")
    
    if results['quality_score'] >= 95.0:
        print("✅ All quality gates PASSED - Production ready!")
    else:
        print("❌ Some quality gates FAILED - Review recommendations")
        for rec in results.get('recommendations', []):
            print(f"  - {rec}")