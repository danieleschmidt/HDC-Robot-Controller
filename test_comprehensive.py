#!/usr/bin/env python3
"""
Comprehensive Test Suite for HDC Robot Controller

Validates all major components and research contributions with extensive testing.
This serves as the quality gate for production deployment.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_functionality():
    """Test core HDC functionality."""
    print("🔍 Testing Basic HDC Functionality...")
    
    try:
        from hdc_robot_controller.core.hypervector import HyperVector
        from hdc_robot_controller.core.operations import HDCOperations
        
        # Test vector creation
        hv1 = HyperVector.random(1000, seed=42)
        hv2 = HyperVector.random(1000, seed=43)
        
        # Test basic operations
        bundled = hv1.bundle(hv2)
        bound = hv1.bind(hv2)
        similarity = hv1.similarity(hv2)
        
        # Validate results
        assert bundled.dimension == 1000, "Bundle dimension incorrect"
        assert bound.dimension == 1000, "Bind dimension incorrect"
        assert -1.0 <= similarity <= 1.0, "Similarity out of range"
        
        print("  ✅ HyperVector operations working correctly")
        
        # Test bundling multiple vectors
        vectors = [HyperVector.random(1000) for _ in range(10)]
        multi_bundle = HyperVector.bundle_vectors(vectors)
        assert multi_bundle.dimension == 1000, "Multi-bundle dimension incorrect"
        
        print("  ✅ Multi-vector bundling working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Basic functionality test failed: {e}")
        return False

def test_research_modules():
    """Test research module implementations."""
    print("🔬 Testing Research Modules...")
    
    success = True
    
    # Test meta-learning
    try:
        from hdc_robot_controller.research.meta_learning import MetaHDCLearner, MetaTask
        from hdc_robot_controller.core.hypervector import HyperVector
        
        learner = MetaHDCLearner(dimension=1000)
        
        # Create a simple test task
        context_vectors = [HyperVector.random(1000) for _ in range(5)]
        target_vectors = [HyperVector.random(1000) for _ in range(5)]
        
        task = MetaTask(
            task_id="test_task",
            context_vectors=context_vectors,
            target_vectors=target_vectors,
            task_description="Test task",
            domain="test"
        )
        
        # Test fast adaptation
        query_vectors = [HyperVector.random(1000) for _ in range(2)]
        predictions, stats = learner.fast_adapt(task, query_vectors)
        
        assert len(predictions) == 2, "Incorrect number of predictions"
        assert stats.adaptation_time > 0, "Invalid adaptation time"
        
        print("  ✅ Meta-learning module working correctly")
        
    except Exception as e:
        print(f"  ❌ Meta-learning test failed: {e}")
        success = False
    
    # Test quantum HDC
    try:
        from hdc_robot_controller.research.quantum_hdc import QuantumHyperVector, QuantumHDCProcessor
        from hdc_robot_controller.core.hypervector import HyperVector
        
        # Create quantum hypervector
        classical_hv = HyperVector.random(500)
        quantum_hv = QuantumHyperVector.from_classical(classical_hv)
        
        assert quantum_hv.dimension == 500, "Quantum dimension incorrect"
        
        # Test quantum operations
        quantum_hv2 = QuantumHyperVector.from_classical(HyperVector.random(500))
        bundled = quantum_hv.quantum_bundle(quantum_hv2)
        
        assert bundled.dimension == 500, "Quantum bundle dimension incorrect"
        
        # Test classical conversion
        converted_back = bundled.to_classical()
        assert converted_back.dimension == 500, "Classical conversion failed"
        
        print("  ✅ Quantum HDC module working correctly")
        
    except Exception as e:
        print(f"  ❌ Quantum HDC test failed: {e}")
        success = False
    
    # Test neuromorphic HDC
    try:
        from hdc_robot_controller.research.neuromorphic_hdc import SpikingHyperVector, NeuromorphicHDCProcessor
        from hdc_robot_controller.core.hypervector import HyperVector
        
        # Create spiking hypervector
        classical_hv = HyperVector.random(500)
        spiking_hv = SpikingHyperVector.from_classical(classical_hv, encoding='rate')
        
        assert spiking_hv.dimension == 500, "Spiking dimension incorrect"
        
        # Test neuromorphic processor
        processor = NeuromorphicHDCProcessor(dimension=500)
        
        # Test bundling
        spiking_hv2 = SpikingHyperVector.from_classical(HyperVector.random(500), encoding='rate')
        bundled = processor.neuromorphic_bundle([spiking_hv, spiking_hv2])
        
        assert bundled.dimension == 500, "Neuromorphic bundle dimension incorrect"
        
        print("  ✅ Neuromorphic HDC module working correctly")
        
    except Exception as e:
        print(f"  ❌ Neuromorphic HDC test failed: {e}")
        success = False
    
    return success

def test_fault_tolerance():
    """Test fault tolerance mechanisms."""
    print("🛡️ Testing Fault Tolerance...")
    
    try:
        from hdc_robot_controller.robustness.fault_tolerance import (
            FaultToleranceOrchestrator, 
            SensorDropoutCompensator,
            CircuitBreaker
        )
        from hdc_robot_controller.core.hypervector import HyperVector
        
        # Test sensor dropout compensation
        compensator = SensorDropoutCompensator(dimension=500)
        
        # Simulate sensor data
        sensor_data = {
            'lidar': HyperVector.random(500),
            'camera': HyperVector.random(500),
            'imu': HyperVector.random(500)
        }
        
        # Learn correlations
        compensator.learn_sensor_correlations(sensor_data)
        
        # Test compensation with missing sensor
        available_sensors = {'lidar': sensor_data['lidar'], 'camera': sensor_data['camera']}
        missing_sensors = ['imu']
        
        compensated = compensator.compensate_sensor_dropout(available_sensors, missing_sensors)
        
        assert 'imu' in compensated, "Sensor compensation failed"
        assert compensated['imu'].dimension == 500, "Compensated sensor dimension incorrect"
        
        print("  ✅ Sensor dropout compensation working correctly")
        
        # Test circuit breaker
        def failing_function():
            raise Exception("Simulated failure")
        
        def working_function():
            return "success"
        
        circuit_breaker = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Test failures
        for _ in range(3):
            result, success = circuit_breaker.call(failing_function)
            assert not success, "Circuit breaker should detect failures"
        
        # Circuit should now be open
        result, success = circuit_breaker.call(working_function)
        assert not success, "Circuit breaker should be open"
        
        print("  ✅ Circuit breaker working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Fault tolerance test failed: {e}")
        return False

def test_security_framework():
    """Test security framework."""
    print("🔐 Testing Security Framework...")
    
    try:
        from hdc_robot_controller.security.security_framework import (
            SecurityFramework,
            AccessControlManager,
            SecurityLevel,
            Permission
        )
        from hdc_robot_controller.core.hypervector import HyperVector
        
        # Test access control
        access_control = AccessControlManager()
        
        # Create test user with secure test credentials
        import os
        test_password = os.getenv('TEST_PASSWORD', 'SecureTestPassword123!')
        user_id = access_control.create_user(
            username="testuser",
            email="test@example.com", 
            password=test_password,
            security_level=SecurityLevel.RESTRICTED,
            roles=["user"]
        )
        
        assert user_id is not None, "User creation failed"
        
        # Test authentication
        session_token = access_control.authenticate_user("testuser", "SecurePass123!")
        assert session_token is not None, "Authentication failed"
        
        # Test permission checking
        has_read = access_control.check_permission(user_id, Permission.READ)
        has_admin = access_control.check_permission(user_id, Permission.ADMIN)
        
        assert has_read, "User should have read permission"
        assert not has_admin, "User should not have admin permission"
        
        print("  ✅ Access control working correctly")
        
        # Test cryptographic operations
        security_framework = SecurityFramework()
        
        # Test secure HDC vector
        test_vector = HyperVector.random(500)
        encrypted_data = security_framework.create_secure_hdc_vector(
            test_vector, user_id, SecurityLevel.RESTRICTED
        )
        
        assert len(encrypted_data) > 0, "Encryption failed"
        
        # Test decryption
        decrypted_vector = security_framework.access_secure_hdc_vector(encrypted_data, user_id)
        assert decrypted_vector.dimension == 500, "Decryption failed"
        
        print("  ✅ Cryptographic operations working correctly")
        return True
        
    except Exception as e:
        print(f"  ❌ Security framework test failed: {e}")
        return False

def test_performance_optimization():
    """Test performance optimization."""
    print("⚡ Testing Performance Optimization...")
    
    try:
        from hdc_robot_controller.scaling.performance_optimizer import (
            PerformanceOptimizer,
            IntelligentCache,
            HardwareAccelerator
        )
        from hdc_robot_controller.core.hypervector import HyperVector
        
        # Test intelligent cache
        cache = IntelligentCache(max_size_mb=10)
        
        # Test cache operations
        test_data = HyperVector.random(500)
        cache.put("test_key", test_data)
        
        retrieved = cache.get("test_key")
        assert retrieved is not None, "Cache retrieval failed"
        assert retrieved.dimension == 500, "Cached data corrupted"
        
        # Test cache miss
        missing = cache.get("nonexistent_key")
        assert missing is None, "Cache should return None for missing keys"
        
        print("  ✅ Intelligent cache working correctly")
        
        # Test hardware accelerator
        accelerator = HardwareAccelerator()
        
        # Test vector bundling
        test_vectors = [HyperVector.random(500) for _ in range(10)]
        result = accelerator.accelerated_bundle(test_vectors, hardware='cpu')
        
        assert result.dimension == 500, "Hardware acceleration failed"
        
        print("  ✅ Hardware accelerator working correctly")
        
        # Test performance optimizer
        optimizer = PerformanceOptimizer(cache_size_mb=10)
        
        def test_operation(x, y):
            return x.bundle(y)
        
        hv1 = HyperVector.random(500)
        hv2 = HyperVector.random(500)
        
        result, metrics = optimizer.optimize_operation(
            "test_bundle", test_operation, {}, hv1, hv2
        )
        
        assert result.dimension == 500, "Optimized operation failed"
        assert metrics.execution_time > 0, "Metrics collection failed"
        
        print("  ✅ Performance optimizer working correctly")
        
        optimizer.shutdown()
        return True
        
    except Exception as e:
        print(f"  ❌ Performance optimization test failed: {e}")
        return False

def performance_benchmark():
    """Run performance benchmarks."""
    print("📊 Running Performance Benchmarks...")
    
    try:
        from hdc_robot_controller.core.hypervector import HyperVector
        import time
        
        # Benchmark parameters
        dimension = 10000
        num_vectors = 100
        
        print(f"  📈 Benchmarking {num_vectors} vectors of dimension {dimension}")
        
        # Generate test data
        vectors = [HyperVector.random(dimension) for _ in range(num_vectors)]
        
        # Benchmark bundling
        start_time = time.time()
        bundled = HyperVector.bundle_vectors(vectors)
        bundle_time = time.time() - start_time
        
        vectors_per_second = num_vectors / bundle_time
        print(f"  ⚡ Bundling: {bundle_time:.3f}s ({vectors_per_second:.1f} vectors/sec)")
        
        # Benchmark similarity computation
        start_time = time.time()
        similarities = []
        for i in range(min(50, len(vectors))):  # Limit to 50 for speed
            sim = vectors[0].similarity(vectors[i])
            similarities.append(sim)
        similarity_time = time.time() - start_time
        
        comparisons_per_second = len(similarities) / similarity_time
        print(f"  🔍 Similarity: {similarity_time:.3f}s ({comparisons_per_second:.1f} comparisons/sec)")
        
        # Memory efficiency test
        import sys
        vector_size = sys.getsizeof(vectors[0])
        total_memory = vector_size * num_vectors / (1024 * 1024)  # MB
        print(f"  💾 Memory usage: {total_memory:.2f} MB for {num_vectors} vectors")
        
        # Success criteria
        assert bundle_time < 10.0, "Bundling too slow"
        assert similarity_time < 5.0, "Similarity computation too slow"
        assert total_memory < 100.0, "Memory usage too high"
        
        print("  ✅ Performance benchmarks passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Performance benchmark failed: {e}")
        return False

def test_integration():
    """Test integration between modules."""
    print("🔗 Testing Module Integration...")
    
    try:
        from hdc_robot_controller.core.hypervector import HyperVector
        from hdc_robot_controller.research.meta_learning import MetaHDCLearner, MetaTask
        from hdc_robot_controller.robustness.fault_tolerance import SensorDropoutCompensator
        from hdc_robot_controller.scaling.performance_optimizer import PerformanceOptimizer
        
        # Create integrated test scenario
        dimension = 1000
        
        # Initialize components
        meta_learner = MetaHDCLearner(dimension=dimension)
        sensor_compensator = SensorDropoutCompensator(dimension=dimension)
        optimizer = PerformanceOptimizer(cache_size_mb=50)
        
        # Simulate sensor data with dropout
        sensor_data = {
            'lidar': HyperVector.random(dimension),
            'camera': None,  # Simulated dropout
            'imu': HyperVector.random(dimension)
        }
        
        # Compensate for sensor dropout
        available_sensors = {k: v for k, v in sensor_data.items() if v is not None}
        missing_sensors = [k for k, v in sensor_data.items() if v is None]
        
        def compensation_operation():
            return sensor_compensator.compensate_sensor_dropout(available_sensors, missing_sensors)
        
        # Use optimizer for sensor compensation
        compensated_data, metrics = optimizer.optimize_operation(
            "sensor_compensation", compensation_operation, {}
        )
        
        assert 'camera' in compensated_data, "Integration failed - sensor not compensated"
        assert metrics.execution_time > 0, "Integration failed - no metrics"
        
        print("  ✅ Sensor compensation integration working")
        
        # Test meta-learning with compensated data
        context_vectors = list(compensated_data.values())[:3]
        target_vectors = [HyperVector.random(dimension) for _ in range(3)]
        
        task = MetaTask(
            task_id="integration_task",
            context_vectors=context_vectors,
            target_vectors=target_vectors,
            task_description="Integration test task",
            domain="integration"
        )
        
        def meta_learning_operation():
            return meta_learner.fast_adapt(task, context_vectors[:2])
        
        predictions, ml_metrics = optimizer.optimize_operation(
            "meta_learning", meta_learning_operation, {}
        )
        
        assert len(predictions) == 2, "Integration failed - wrong prediction count"
        assert ml_metrics.adaptation_time > 0, "Integration failed - no adaptation metrics"
        
        print("  ✅ Meta-learning integration working")
        
        optimizer.shutdown()
        print("  ✅ Full integration test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Integration test failed: {e}")
        return False

def main():
    """Run comprehensive test suite."""
    print("🚀 HDC Robot Controller - Comprehensive Test Suite")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Research Modules", test_research_modules),
        ("Fault Tolerance", test_fault_tolerance),
        ("Security Framework", test_security_framework),
        ("Performance Optimization", test_performance_optimization),
        ("Performance Benchmarks", performance_benchmark),
        ("Integration Testing", test_integration)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        result = test_func()
        test_results.append((test_name, result))
        
        if result:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED - REVIEW REQUIRED")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)