"""
Simplified Quality Gates - Core HDC Validation
==============================================

Validates core HDC functionality that is actually implemented.
"""

import numpy as np
import time
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print("🛡️ HDC ROBOT CONTROLLER - CORE QUALITY VALIDATION")
print("=" * 60)

def test_basic_imports():
    """Test basic package imports."""
    print("\n🔍 Testing basic imports...")
    
    try:
        from hdc_robot_controller.core.hypervector import HyperVector
        print("   ✅ HyperVector import successful")
        
        from hdc_robot_controller.core.memory import AssociativeMemory
        print("   ✅ AssociativeMemory import successful")
        
        from hdc_robot_controller.core.operations import HDCOperations
        print("   ✅ HDCOperations import successful")
        
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_hypervector_operations():
    """Test basic hypervector operations."""
    print("\n🔍 Testing HyperVector operations...")
    
    try:
        from hdc_robot_controller.core.hypervector import HyperVector
        
        # Test creation
        dimension = 10000
        hv1 = HyperVector.random(dimension, seed=42)
        hv2 = HyperVector.random(dimension, seed=43)
        print(f"   ✅ Created hypervectors (dimension: {dimension})")
        
        # Test bundling
        bundled = hv1.bundle(hv2)
        assert bundled.get_dimension() == dimension
        print("   ✅ Bundle operation successful")
        
        # Test binding
        bound = hv1.bind(hv2)
        assert bound.get_dimension() == dimension
        print("   ✅ Bind operation successful")
        
        # Test similarity
        similarity = hv1.similarity(hv2)
        assert -1.0 <= similarity <= 1.0
        print(f"   ✅ Similarity computation successful ({similarity:.3f})")
        
        # Test self-similarity
        self_sim = hv1.similarity(hv1)
        assert abs(self_sim - 1.0) < 0.1
        print(f"   ✅ Self-similarity correct ({self_sim:.3f})")
        
        return True
    except Exception as e:
        print(f"   ❌ HyperVector test failed: {e}")
        return False

def test_memory_operations():
    """Test associative memory operations."""
    print("\n🔍 Testing Memory operations...")
    
    try:
        from hdc_robot_controller.core.hypervector import HyperVector
        from hdc_robot_controller.core.memory import AssociativeMemory
        
        dimension = 5000  # Smaller for testing
        memory = AssociativeMemory(dimension)
        print(f"   ✅ Created associative memory (dimension: {dimension})")
        
        # Test storage and retrieval
        test_vectors = []
        for i in range(10):
            hv = HyperVector.random(dimension, seed=i)
            key = f"test_vector_{i}"
            memory.store(key, hv)
            test_vectors.append((key, hv))
        
        print(f"   ✅ Stored {len(test_vectors)} hypervectors")
        
        # Test retrieval
        success_count = 0
        for key, original_hv in test_vectors:
            if memory.contains(key):
                retrieved = memory.recall(key)
                if retrieved and retrieved.similarity(original_hv) > 0.9:
                    success_count += 1
        
        success_rate = success_count / len(test_vectors)
        print(f"   ✅ Retrieved {success_count}/{len(test_vectors)} vectors (rate: {success_rate:.1%})")
        
        if success_rate >= 0.8:
            print("   ✅ Memory operations successful")
            return True
        else:
            print(f"   ⚠️  Low success rate: {success_rate:.1%}")
            return False
            
    except Exception as e:
        print(f"   ❌ Memory test failed: {e}")
        return False

def test_performance_benchmarks():
    """Test performance benchmarks."""
    print("\n🔍 Testing Performance benchmarks...")
    
    try:
        from hdc_robot_controller.core.hypervector import HyperVector
        from hdc_robot_controller.core.memory import AssociativeMemory
        
        dimension = 10000
        iterations = 100
        
        # Benchmark hypervector creation
        start_time = time.perf_counter()
        for i in range(iterations):
            hv = HyperVector.random(dimension)
        creation_time = (time.perf_counter() - start_time) / iterations * 1000  # ms
        
        print(f"   ✅ HyperVector creation: {creation_time:.2f} ms/vector")
        
        # Benchmark similarity computation
        hv1 = HyperVector.random(dimension)
        hv2 = HyperVector.random(dimension)
        
        start_time = time.perf_counter()
        for i in range(iterations):
            similarity = hv1.similarity(hv2)
        similarity_time = (time.perf_counter() - start_time) / iterations * 1000  # ms
        
        print(f"   ✅ Similarity computation: {similarity_time:.2f} ms/operation")
        
        # Memory throughput benchmark
        memory = AssociativeMemory(dimension)
        test_data = [(f"key_{i}", HyperVector.random(dimension)) for i in range(50)]
        
        start_time = time.perf_counter()
        for key, hv in test_data:
            memory.store(key, hv)
        storage_time = time.perf_counter() - start_time
        
        storage_throughput = len(test_data) / storage_time  # items/second
        print(f"   ✅ Memory storage: {storage_throughput:.1f} items/second")
        
        # Performance thresholds
        performance_ok = True
        
        if creation_time > 20.0:  # 20ms critical threshold
            print(f"   ⚠️  HyperVector creation slow: {creation_time:.2f}ms")
            performance_ok = False
        
        if similarity_time > 5.0:  # 5ms critical threshold
            print(f"   ⚠️  Similarity computation slow: {similarity_time:.2f}ms")
            performance_ok = False
            
        if storage_throughput < 10.0:  # 10 items/sec critical threshold
            print(f"   ⚠️  Memory storage slow: {storage_throughput:.1f} items/sec")
            performance_ok = False
        
        if performance_ok:
            print("   ✅ All performance benchmarks within acceptable thresholds")
        
        return performance_ok
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def test_advanced_features():
    """Test advanced features that are implemented."""
    print("\n🔍 Testing Advanced features...")
    
    try:
        # Test evolutionary components
        try:
            from hdc_robot_controller.evolution.genetic_optimizer import GeneticOptimizer
            print("   ✅ Genetic optimizer available")
            genetic_available = True
        except ImportError:
            print("   ℹ️  Genetic optimizer not available")
            genetic_available = False
        
        # Test self-improving components
        try:
            from hdc_robot_controller.evolution.self_improving_algorithms import SelfImprovingHDC
            print("   ✅ Self-improving algorithms available")
            self_improving_available = True
        except ImportError:
            print("   ℹ️  Self-improving algorithms not available")
            self_improving_available = False
        
        # Test quantum transcendence components
        try:
            from hdc_robot_controller.quantum_transcendence.quantum_consciousness_engine import QuantumConsciousnessEngine
            print("   ✅ Quantum consciousness available")
            quantum_available = True
        except ImportError:
            print("   ℹ️  Quantum consciousness not available")
            quantum_available = False
        
        # Test cosmic intelligence
        try:
            from hdc_robot_controller.cosmic_intelligence.universal_knowledge_integrator import UniversalKnowledgeIntegrator
            print("   ✅ Cosmic intelligence available")
            cosmic_available = True
        except ImportError:
            print("   ℹ️  Cosmic intelligence not available") 
            cosmic_available = False
        
        advanced_count = sum([genetic_available, self_improving_available, quantum_available, cosmic_available])
        print(f"   📊 Advanced modules available: {advanced_count}/4")
        
        return advanced_count >= 2  # At least 2 advanced modules working
        
    except Exception as e:
        print(f"   ❌ Advanced features test failed: {e}")
        return False

def run_comprehensive_validation():
    """Run comprehensive validation suite."""
    
    print("🚀 Starting comprehensive HDC validation...")
    
    # Test results
    test_results = []
    
    # Run tests
    test_results.append(("Basic Imports", test_basic_imports()))
    test_results.append(("HyperVector Operations", test_hypervector_operations()))
    test_results.append(("Memory Operations", test_memory_operations()))
    test_results.append(("Performance Benchmarks", test_performance_benchmarks()))
    test_results.append(("Advanced Features", test_advanced_features()))
    
    # Calculate results
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    success_rate = passed_tests / total_tests
    
    print(f"\n🏁 VALIDATION COMPLETE")
    print("=" * 60)
    print(f"📊 RESULTS SUMMARY:")
    print(f"   Total Tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success Rate: {success_rate:.1%}")
    
    # Detailed results
    print(f"\n📋 DETAILED RESULTS:")
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    # Quality assessment
    if success_rate >= 0.9:
        overall_status = "✅ EXCELLENT"
        quality_score = 100
        exit_code = 0
    elif success_rate >= 0.8:
        overall_status = "✅ GOOD"  
        quality_score = 85
        exit_code = 0
    elif success_rate >= 0.6:
        overall_status = "⚠️  ACCEPTABLE"
        quality_score = 70
        exit_code = 1
    else:
        overall_status = "❌ NEEDS IMPROVEMENT"
        quality_score = 50
        exit_code = 2
    
    print(f"\n🎯 OVERALL ASSESSMENT:")
    print(f"   Status: {overall_status}")
    print(f"   Quality Score: {quality_score}/100")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    
    failed_tests = [name for name, result in test_results if not result]
    if failed_tests:
        print(f"   - Fix failing tests: {', '.join(failed_tests)}")
    
    if success_rate < 1.0:
        print(f"   - Investigate and resolve failed components")
        print(f"   - Run additional diagnostic tests")
    
    if success_rate >= 0.8:
        print(f"   - System ready for advanced functionality")
        print(f"   - Consider deploying to production")
    
    print("=" * 60)
    
    return exit_code

if __name__ == "__main__":
    exit_code = run_comprehensive_validation()
    exit(exit_code)