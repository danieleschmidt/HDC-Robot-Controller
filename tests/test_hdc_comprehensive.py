#!/usr/bin/env python3
"""
Comprehensive test suite for HDC Robot Controller.
Tests all major components including fault tolerance, security, and performance.
"""

import unittest
import numpy as np
import time
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import threading
import json

# Import HDC components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from hdc_robot_controller.core.hypervector import HyperVector
from hdc_robot_controller.core.operations import HDCOperations, BasisVectors
from hdc_robot_controller.core.memory import AssociativeMemory
from hdc_robot_controller.core.error_handling import (
    ErrorRecoveryManager, HDCException, DimensionMismatchError,
    validate_hypervector, validate_dimension, robust_hdc_operation
)
from hdc_robot_controller.core.security import (
    SecurityManager, InputSanitizer, AccessController, 
    Permission, SecurityLevel, SecurityError
)


class TestHyperVector(unittest.TestCase):
    """Test HyperVector core functionality."""
    
    def setUp(self):
        self.dimension = 1000
        self.hv1 = HyperVector.random(self.dimension, seed=42)
        self.hv2 = HyperVector.random(self.dimension, seed=43)
    
    def test_creation(self):
        """Test HyperVector creation."""
        # Test default creation
        hv = HyperVector()
        self.assertEqual(hv.dimension, HyperVector.DEFAULT_DIMENSION)
        self.assertTrue(np.all(hv.data == 0))
        
        # Test with specific dimension
        hv = HyperVector(self.dimension)
        self.assertEqual(hv.dimension, self.dimension)
        
        # Test with data
        data = np.random.choice([-1, 1], size=self.dimension).astype(np.int8)
        hv = HyperVector(self.dimension, data)
        self.assertEqual(hv.dimension, self.dimension)
        np.testing.assert_array_equal(hv.data, data)
    
    def test_random_creation(self):
        """Test random HyperVector creation."""
        hv1 = HyperVector.random(self.dimension, seed=42)
        hv2 = HyperVector.random(self.dimension, seed=42)
        hv3 = HyperVector.random(self.dimension, seed=43)
        
        # Same seed should produce identical vectors
        np.testing.assert_array_equal(hv1.data, hv2.data)
        
        # Different seeds should produce different vectors
        self.assertFalse(np.array_equal(hv1.data, hv3.data))
        
        # Should be bipolar
        self.assertTrue(np.all(np.isin(hv1.data, [-1, 1])))
    
    def test_bundling(self):
        """Test vector bundling operation."""
        result = self.hv1.bundle(self.hv2)
        self.assertEqual(result.dimension, self.dimension)
        self.assertTrue(np.all(np.isin(result.data, [-1, 1])))
        
        # Test multiple vector bundling
        hv3 = HyperVector.random(self.dimension, seed=44)
        bundle_result = HyperVector.bundle_vectors([self.hv1, self.hv2, hv3])
        self.assertEqual(bundle_result.dimension, self.dimension)
        self.assertTrue(np.all(np.isin(bundle_result.data, [-1, 1])))
    
    def test_binding(self):
        """Test vector binding operation."""
        result = self.hv1.bind(self.hv2)
        self.assertEqual(result.dimension, self.dimension)
        self.assertTrue(np.all(np.isin(result.data, [-1, 1])))
        
        # Test binding properties
        # Should be commutative: A * B = B * A
        result1 = self.hv1.bind(self.hv2)
        result2 = self.hv2.bind(self.hv1)
        np.testing.assert_array_equal(result1.data, result2.data)
        
        # Inverse property: A * A = identity-like
        self_bind = self.hv1.bind(self.hv1)
        # Self-binding should produce a vector with high positive similarity to identity
        self.assertGreater(np.sum(self_bind.data), 0)
    
    def test_similarity(self):
        """Test similarity calculation."""
        # Self similarity should be 1.0
        self.assertAlmostEqual(self.hv1.similarity(self.hv1), 1.0, places=5)
        
        # Inverse similarity should be -1.0
        inv_hv1 = self.hv1.invert()
        self.assertAlmostEqual(self.hv1.similarity(inv_hv1), -1.0, places=5)
        
        # Random vectors should have low similarity
        similarity = self.hv1.similarity(self.hv2)
        self.assertLess(abs(similarity), 0.2)  # Should be close to 0
    
    def test_permutation(self):
        """Test vector permutation."""
        shifted = self.hv1.permute(1)
        self.assertEqual(shifted.dimension, self.dimension)
        
        # Permuting back should restore original
        restored = shifted.permute(-1)
        np.testing.assert_array_equal(self.hv1.data, restored.data)
        
        # Multiple shifts
        double_shifted = self.hv1.permute(2)
        single_twice = self.hv1.permute(1).permute(1)
        np.testing.assert_array_equal(double_shifted.data, single_twice.data)
    
    def test_serialization(self):
        """Test vector serialization."""
        # Test to_bytes and from_bytes
        byte_data = self.hv1.to_bytes()
        self.assertIsInstance(byte_data, bytes)
        
        hv_restored = HyperVector(self.dimension)
        hv_restored.from_bytes(byte_data)
        np.testing.assert_array_equal(self.hv1.data, hv_restored.data)
    
    def test_error_conditions(self):
        """Test error handling."""
        with self.assertRaises(ValueError):
            HyperVector(-1)  # Invalid dimension
        
        with self.assertRaises(ValueError):
            HyperVector(100, np.array([1, 2, 3]))  # Mismatched dimension
        
        # Dimension mismatch in operations
        hv_small = HyperVector.random(100)
        with self.assertRaises(ValueError):
            self.hv1.bundle(hv_small)
        
        with self.assertRaises(ValueError):
            self.hv1.bind(hv_small)


class TestHDCOperations(unittest.TestCase):
    """Test HDC operations and utilities."""
    
    def setUp(self):
        self.dimension = 1000
        self.hv1 = HyperVector.random(self.dimension, seed=42)
        self.hv2 = HyperVector.random(self.dimension, seed=43)
        self.hv3 = HyperVector.random(self.dimension, seed=44)
    
    def test_majority_bundle(self):
        """Test majority bundling."""
        result = HDCOperations.majority_bundle([self.hv1, self.hv2, self.hv3])
        self.assertEqual(result.dimension, self.dimension)
        self.assertTrue(np.all(np.isin(result.data, [-1, 1])))
    
    def test_weighted_bundle(self):
        """Test weighted bundling."""
        weights = [0.5, 0.3, 0.2]
        result = HDCOperations.weighted_bundle([self.hv1, self.hv2, self.hv3], weights)
        self.assertEqual(result.dimension, self.dimension)
    
    def test_similarity_measures(self):
        """Test various similarity measures."""
        # Cosine similarity
        cos_sim = HDCOperations.cosine_similarity(self.hv1, self.hv2)
        self.assertIsInstance(cos_sim, float)
        self.assertLessEqual(abs(cos_sim), 1.0)
        
        # Hamming similarity
        ham_sim = HDCOperations.hamming_similarity(self.hv1, self.hv2)
        self.assertIsInstance(ham_sim, float)
        self.assertGreaterEqual(ham_sim, 0.0)
        self.assertLessEqual(ham_sim, 1.0)
        
        # Jaccard similarity
        jac_sim = HDCOperations.jaccard_similarity(self.hv1, self.hv2)
        self.assertIsInstance(jac_sim, float)
        self.assertGreaterEqual(jac_sim, 0.0)
        self.assertLessEqual(jac_sim, 1.0)
    
    def test_basis_vectors(self):
        """Test basis vector encoding."""
        basis = BasisVectors(self.dimension)
        
        # Test integer encoding
        int_hv = basis.encode_integer(42)
        self.assertEqual(int_hv.dimension, self.dimension)
        
        # Same integer should produce same vector
        int_hv2 = basis.encode_integer(42)
        np.testing.assert_array_equal(int_hv.data, int_hv2.data)
        
        # Different integers should produce different vectors
        int_hv3 = basis.encode_integer(43)
        self.assertFalse(np.array_equal(int_hv.data, int_hv3.data))
        
        # Test float encoding
        float_hv = basis.encode_float(3.14)
        self.assertEqual(float_hv.dimension, self.dimension)
        
        # Test category encoding
        cat_hv = basis.encode_category("test_category")
        self.assertEqual(cat_hv.dimension, self.dimension)
        
        # Test position encoding
        pos_hv = basis.encode_2d_position(1.0, 2.0)
        self.assertEqual(pos_hv.dimension, self.dimension)


class TestAssociativeMemory(unittest.TestCase):
    """Test associative memory functionality."""
    
    def setUp(self):
        self.dimension = 1000
        self.memory = AssociativeMemory(self.dimension)
        self.test_vectors = {
            "vector1": HyperVector.random(self.dimension, seed=42),
            "vector2": HyperVector.random(self.dimension, seed=43),
            "vector3": HyperVector.random(self.dimension, seed=44)
        }
    
    def test_store_and_retrieve(self):
        """Test basic store and retrieve operations."""
        key = "test_key"
        vector = self.test_vectors["vector1"]
        
        # Store vector
        self.memory.store(key, vector)
        self.assertTrue(self.memory.contains(key))
        
        # Retrieve vector
        retrieved = self.memory.retrieve(key)
        np.testing.assert_array_equal(vector.data, retrieved.data)
    
    def test_query_by_similarity(self):
        """Test similarity-based querying."""
        # Store test vectors
        for key, vector in self.test_vectors.items():
            self.memory.store(key, vector)
        
        # Query with exact match
        query_vector = self.test_vectors["vector1"]
        results = self.memory.query(query_vector, max_results=3)
        
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].key, "vector1")
        self.assertAlmostEqual(results[0].similarity, 1.0, places=5)
        
        # Query with noisy version
        noisy_vector = query_vector.add_noise(0.1, seed=100)
        results = self.memory.query(noisy_vector, max_results=1, min_similarity=0.5)
        
        if results:  # Should find the original despite noise
            self.assertEqual(results[0].key, "vector1")
            self.assertGreater(results[0].similarity, 0.5)
    
    def test_best_match(self):
        """Test best match retrieval."""
        for key, vector in self.test_vectors.items():
            self.memory.store(key, vector)
        
        query_vector = self.test_vectors["vector2"]
        best = self.memory.best_match(query_vector)
        
        # Should return the exact match
        np.testing.assert_array_equal(query_vector.data, best.data)
    
    def test_memory_management(self):
        """Test memory size and cleanup."""
        initial_size = self.memory.size()
        
        # Add vectors
        for i in range(10):
            key = f"vector_{i}"
            vector = HyperVector.random(self.dimension, seed=i)
            self.memory.store(key, vector)
        
        self.assertEqual(self.memory.size(), initial_size + 10)
        
        # Clear memory
        self.memory.clear()
        self.assertEqual(self.memory.size(), 0)
    
    def test_batch_operations(self):
        """Test batch store and retrieve."""
        # Batch store
        self.memory.batch_store(self.test_vectors)
        self.assertEqual(self.memory.size(), len(self.test_vectors))
        
        # Batch retrieve
        keys = list(self.test_vectors.keys())
        retrieved = self.memory.batch_retrieve(keys)
        
        self.assertEqual(len(retrieved), len(self.test_vectors))
        for key in keys:
            self.assertIn(key, retrieved)
    
    def test_persistence(self):
        """Test save and load functionality."""
        # Store test data
        for key, vector in self.test_vectors.items():
            self.memory.store(key, vector)
        
        # Save to file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.hdc') as f:
            temp_filename = f.name
        
        try:
            self.memory.save_to_file(temp_filename)
            
            # Create new memory and load
            new_memory = AssociativeMemory(self.dimension)
            new_memory.load_from_file(temp_filename)
            
            # Verify data
            self.assertEqual(new_memory.size(), len(self.test_vectors))
            for key, original_vector in self.test_vectors.items():
                retrieved = new_memory.retrieve(key)
                np.testing.assert_array_equal(original_vector.data, retrieved.data)
        
        finally:
            if os.path.exists(temp_filename):
                os.unlink(temp_filename)


class TestErrorHandling(unittest.TestCase):
    """Test error handling and recovery."""
    
    def setUp(self):
        self.error_manager = ErrorRecoveryManager()
    
    def test_dimension_validation(self):
        """Test dimension validation."""
        # Valid dimensions
        self.assertEqual(validate_dimension(1000), 1000)
        
        # Invalid dimensions
        with self.assertRaises(Exception):
            validate_dimension(-1)
        
        with self.assertRaises(Exception):
            validate_dimension(0)
        
        with self.assertRaises(Exception):
            validate_dimension(200000)  # Too large
    
    def test_hypervector_validation(self):
        """Test hypervector validation."""
        # Valid hypervector
        hv = HyperVector.random(1000)
        validated = validate_hypervector(hv)
        self.assertEqual(validated.dimension, 1000)
        
        # Valid numpy array
        data = np.random.choice([-1, 1], size=1000).astype(np.int8)
        validated = validate_hypervector(data)
        self.assertEqual(validated.dimension, 1000)
        
        # Invalid input
        with self.assertRaises(Exception):
            validate_hypervector(None)
        
        with self.assertRaises(Exception):
            validate_hypervector("invalid")
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Test handling of HDC exception
        try:
            raise DimensionMismatchError(1000, 500, "test_operation")
        except Exception as e:
            success = self.error_manager.handle_error(e)
            self.assertIsInstance(success, bool)
        
        # Test handling of general exception
        try:
            raise ValueError("Test error")
        except Exception as e:
            success = self.error_manager.handle_error(e)
            self.assertIsInstance(success, bool)
    
    def test_robust_operation_decorator(self):
        """Test robust operation decorator."""
        
        @robust_hdc_operation(max_retries=3)
        def failing_operation(fail_count=2):
            if hasattr(failing_operation, 'attempt_count'):
                failing_operation.attempt_count += 1
            else:
                failing_operation.attempt_count = 1
            
            if failing_operation.attempt_count <= fail_count:
                raise ValueError("Simulated failure")
            
            return "success"
        
        # Should succeed after retries
        result = failing_operation(fail_count=2)
        self.assertEqual(result, "success")
        
        # Reset for next test
        if hasattr(failing_operation, 'attempt_count'):
            delattr(failing_operation, 'attempt_count')


class TestSecurity(unittest.TestCase):
    """Test security features."""
    
    def setUp(self):
        self.security_manager = SecurityManager()
        self.sanitizer = InputSanitizer()
        self.access_controller = AccessController()
    
    def test_input_sanitization(self):
        """Test input sanitization."""
        # String sanitization
        clean_string = self.sanitizer.sanitize_string("Hello World!")
        self.assertEqual(clean_string, "Hello World!")
        
        # Dangerous string
        with self.assertRaises(SecurityError):
            self.sanitizer.sanitize_string("<script>alert('xss')</script>")
        
        # Numeric sanitization
        clean_num = self.sanitizer.sanitize_numeric("42.5")
        self.assertEqual(clean_num, 42.5)
        
        # Invalid numeric
        with self.assertRaises(SecurityError):
            self.sanitizer.sanitize_numeric("not_a_number")
        
        # Array sanitization
        test_array = [1, 2, 3, 4, 5]
        clean_array = self.sanitizer.sanitize_array(test_array)
        np.testing.assert_array_equal(clean_array, np.array(test_array))
    
    def test_access_control(self):
        """Test access control system."""
        user_id = "test_user"
        permissions = {Permission.READ_PERCEPTION, Permission.WRITE_CONTROL}
        
        # Create session
        token = self.access_controller.create_session(user_id, permissions)
        self.assertIsNotNone(token)
        
        # Validate session
        context = self.access_controller.validate_session(token)
        self.assertIsNotNone(context)
        self.assertEqual(context.user_id, user_id)
        
        # Check permissions
        self.assertTrue(self.access_controller.check_permission(token, Permission.READ_PERCEPTION))
        self.assertFalse(self.access_controller.check_permission(token, Permission.SYSTEM_CONFIG))
        
        # Revoke session
        success = self.access_controller.revoke_session(token)
        self.assertTrue(success)
        
        # Should no longer be valid
        context = self.access_controller.validate_session(token)
        self.assertIsNone(context)
    
    def test_security_manager_integration(self):
        """Test security manager integration."""
        # Create user session
        permissions = ["read_perception", "write_control"]
        token = self.security_manager.create_user_session("test_user", permissions)
        self.assertIsNotNone(token)
        
        # Check access
        self.assertTrue(self.security_manager.check_access(token, "read_perception"))
        self.assertFalse(self.security_manager.check_access(token, "system_config"))
        
        # Validate and sanitize input
        clean_data = self.security_manager.validate_and_sanitize_input("Hello World!")
        self.assertEqual(clean_data, "Hello World!")
    
    def test_rate_limiting(self):
        """Test rate limiting."""
        rate_limiter = self.security_manager.rate_limiter
        identifier = "test_user"
        
        # Should allow initial requests
        for _ in range(10):
            self.assertTrue(rate_limiter.is_allowed(identifier))
        
        # Check remaining requests
        remaining = rate_limiter.get_remaining_requests(identifier)
        self.assertLess(remaining, rate_limiter.max_requests)


class TestPerformance(unittest.TestCase):
    """Test performance characteristics."""
    
    def test_vector_operations_performance(self):
        """Test performance of vector operations."""
        dimension = 10000
        num_operations = 1000
        
        # Create test vectors
        vectors = [HyperVector.random(dimension, seed=i) for i in range(10)]
        
        # Time bundling operations
        start_time = time.time()
        for i in range(num_operations):
            v1, v2 = vectors[i % len(vectors)], vectors[(i + 1) % len(vectors)]
            result = v1.bundle(v2)
        bundling_time = time.time() - start_time
        
        # Time binding operations
        start_time = time.time()
        for i in range(num_operations):
            v1, v2 = vectors[i % len(vectors)], vectors[(i + 1) % len(vectors)]
            result = v1.bind(v2)
        binding_time = time.time() - start_time
        
        # Time similarity operations
        start_time = time.time()
        for i in range(num_operations):
            v1, v2 = vectors[i % len(vectors)], vectors[(i + 1) % len(vectors)]
            sim = v1.similarity(v2)
        similarity_time = time.time() - start_time
        
        # Print performance metrics
        print(f"\nPerformance Results (dimension={dimension}, operations={num_operations}):")
        print(f"Bundling: {bundling_time:.3f}s ({num_operations/bundling_time:.1f} ops/sec)")
        print(f"Binding: {binding_time:.3f}s ({num_operations/binding_time:.1f} ops/sec)")
        print(f"Similarity: {similarity_time:.3f}s ({num_operations/similarity_time:.1f} ops/sec)")
        
        # Performance assertions (adjust based on hardware)
        self.assertLess(bundling_time, 10.0)  # Should complete within 10 seconds
        self.assertLess(binding_time, 10.0)
        self.assertLess(similarity_time, 10.0)
    
    def test_memory_performance(self):
        """Test memory system performance."""
        dimension = 10000
        num_vectors = 1000
        
        memory = AssociativeMemory(dimension)
        
        # Create test vectors
        test_vectors = {}
        for i in range(num_vectors):
            key = f"vector_{i}"
            vector = HyperVector.random(dimension, seed=i)
            test_vectors[key] = vector
        
        # Time storage operations
        start_time = time.time()
        for key, vector in test_vectors.items():
            memory.store(key, vector)
        storage_time = time.time() - start_time
        
        # Time retrieval operations
        keys = list(test_vectors.keys())
        start_time = time.time()
        for key in keys[:100]:  # Test first 100
            retrieved = memory.retrieve(key)
        retrieval_time = time.time() - start_time
        
        # Time query operations
        query_vectors = [test_vectors[f"vector_{i}"] for i in range(0, 50, 5)]
        start_time = time.time()
        for query_vector in query_vectors:
            results = memory.query(query_vector, max_results=5)
        query_time = time.time() - start_time
        
        print(f"\nMemory Performance (dimension={dimension}, vectors={num_vectors}):")
        print(f"Storage: {storage_time:.3f}s ({num_vectors/storage_time:.1f} vectors/sec)")
        print(f"Retrieval: {retrieval_time:.3f}s ({100/retrieval_time:.1f} retrievals/sec)")
        print(f"Query: {query_time:.3f}s ({len(query_vectors)/query_time:.1f} queries/sec)")
        
        # Performance assertions
        self.assertLess(storage_time, 30.0)
        self.assertLess(retrieval_time, 5.0)
        self.assertLess(query_time, 10.0)


class TestFaultTolerance(unittest.TestCase):
    """Test fault tolerance capabilities."""
    
    def test_noise_resilience(self):
        """Test resilience to noise."""
        dimension = 10000
        original_vector = HyperVector.random(dimension, seed=42)
        
        # Test various noise levels
        noise_levels = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
        
        for noise_level in noise_levels:
            noisy_vector = original_vector.add_noise(noise_level, seed=100)
            similarity = original_vector.similarity(noisy_vector)
            
            # Similarity should degrade gracefully
            expected_similarity = 1.0 - (2 * noise_level)
            self.assertGreater(similarity, expected_similarity - 0.1)
            
            print(f"Noise {noise_level*100}%: similarity = {similarity:.3f}")
    
    def test_dimension_robustness(self):
        """Test robustness across different dimensions."""
        dimensions = [1000, 5000, 10000, 20000]
        
        for dim in dimensions:
            hv1 = HyperVector.random(dim, seed=42)
            hv2 = HyperVector.random(dim, seed=43)
            
            # Test basic operations
            bundled = hv1.bundle(hv2)
            bound = hv1.bind(hv2)
            similarity = hv1.similarity(hv2)
            
            # Verify properties hold across dimensions
            self.assertEqual(bundled.dimension, dim)
            self.assertEqual(bound.dimension, dim)
            self.assertLess(abs(similarity), 0.2)  # Should be low for random vectors
    
    def test_memory_corruption_recovery(self):
        """Test recovery from memory corruption."""
        dimension = 1000
        memory = AssociativeMemory(dimension)
        
        # Store test vectors
        test_vectors = {}
        for i in range(10):
            key = f"vector_{i}"
            vector = HyperVector.random(dimension, seed=i)
            memory.store(key, vector)
            test_vectors[key] = vector
        
        # Simulate corruption by modifying internal data
        # In a real system, this might be detected by checksums
        for key in list(memory.memory_.keys())[:3]:  # Corrupt first 3 vectors
            corrupted_data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
            memory.memory_[key] = HyperVector(dimension, corrupted_data)
        
        # Test that uncorrupted vectors are still retrievable
        uncorrupted_keys = list(test_vectors.keys())[3:]
        for key in uncorrupted_keys:
            retrieved = memory.retrieve(key)
            expected = test_vectors[key]
            # Should have high similarity (exact match)
            similarity = retrieved.similarity(expected)
            self.assertGreater(similarity, 0.99)


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of components."""
    
    def test_concurrent_memory_access(self):
        """Test concurrent access to associative memory."""
        dimension = 1000
        memory = AssociativeMemory(dimension)
        num_threads = 5
        operations_per_thread = 100
        
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(operations_per_thread):
                    key = f"thread_{thread_id}_vector_{i}"
                    vector = HyperVector.random(dimension, seed=thread_id * 1000 + i)
                    
                    # Store vector
                    memory.store(key, vector)
                    
                    # Retrieve vector
                    retrieved = memory.retrieve(key)
                    
                    # Verify integrity
                    if not np.array_equal(vector.data, retrieved.data):
                        errors.append(f"Data mismatch in thread {thread_id}, operation {i}")
                    else:
                        results.append(f"Success: thread {thread_id}, operation {i}")
                        
            except Exception as e:
                errors.append(f"Exception in thread {thread_id}: {e}")
        
        # Start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        expected_operations = num_threads * operations_per_thread
        print(f"\nThread Safety Test: {len(results)} successes, {len(errors)} errors")
        
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), expected_operations)
    
    def test_concurrent_vector_operations(self):
        """Test concurrent vector operations."""
        dimension = 1000
        num_threads = 10
        operations_per_thread = 50
        
        # Shared vectors
        shared_vectors = [HyperVector.random(dimension, seed=i) for i in range(5)]
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(operations_per_thread):
                    v1 = shared_vectors[i % len(shared_vectors)]
                    v2 = shared_vectors[(i + 1) % len(shared_vectors)]
                    
                    # Perform operations
                    bundled = v1.bundle(v2)
                    bound = v1.bind(v2)
                    similarity = v1.similarity(v2)
                    
                    # Verify results
                    if bundled.dimension != dimension or bound.dimension != dimension:
                        errors.append(f"Dimension error in thread {thread_id}")
                    elif not (-1 <= similarity <= 1):
                        errors.append(f"Similarity out of range in thread {thread_id}")
                    else:
                        results.append(f"Success: thread {thread_id}, operation {i}")
                        
            except Exception as e:
                errors.append(f"Exception in thread {thread_id}: {e}")
        
        # Start threads
        threads = []
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        expected_operations = num_threads * operations_per_thread
        print(f"Concurrent Operations Test: {len(results)} successes, {len(errors)} errors")
        
        self.assertEqual(len(errors), 0, f"Concurrent operation errors: {errors}")
        self.assertEqual(len(results), expected_operations)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestHyperVector,
        TestHDCOperations,
        TestAssociativeMemory,
        TestErrorHandling,
        TestSecurity,
        TestPerformance,
        TestFaultTolerance,
        TestThreadSafety
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    # Exit with appropriate code
    exit_code = 0 if (len(result.failures) == 0 and len(result.errors) == 0) else 1
    exit(exit_code)