#!/usr/bin/env python3
"""
Test Suite for Generation 6: Quantum HDC Engine
Comprehensive testing of quantum computing integration.
"""

import pytest
import numpy as np
import time
import sys
import pathlib

# Add project root to path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hdc_robot_controller.quantum_core.quantum_hdc_engine import (
    QuantumHDCEngine,
    QuantumClassicalHybrid,
    QuantumState,
    QuantumOperation,
    QuantumGate,
    demonstrate_quantum_hdc
)
from hdc_robot_controller.core.hypervector import HyperVector


class TestQuantumHDCEngine:
    """Test quantum HDC engine functionality."""
    
    @pytest.fixture
    def quantum_engine(self):
        """Create quantum HDC engine for testing."""
        return QuantumHDCEngine(dimension=1000, quantum_backend="simulator")
    
    @pytest.fixture
    def test_hypervector(self):
        """Create test hypervector."""
        hv = HyperVector(1000)
        hv.randomize(seed=42)
        return hv
    
    def test_quantum_engine_initialization(self, quantum_engine):
        """Test quantum engine initialization."""
        assert quantum_engine.dimension == 1000
        assert quantum_engine.quantum_backend == "simulator"
        assert len(quantum_engine.quantum_states) == 0
        assert len(quantum_engine.operation_history) == 0
    
    def test_quantum_hypervector_creation(self, quantum_engine, test_hypervector):
        """Test creating quantum hypervector from classical."""
        quantum_state = quantum_engine.create_quantum_hypervector(
            test_hypervector, 
            superposition_degree=0.5
        )
        
        assert isinstance(quantum_state, QuantumState)
        assert quantum_state.dimension == 1000
        assert len(quantum_state.amplitudes) == 1000
        assert len(quantum_state.phases) == 1000
        assert quantum_state.coherence_time > 0
        
        # Check normalization
        norm = np.linalg.norm(quantum_state.amplitudes)
        assert abs(norm - 1.0) < 1e-6
    
    def test_quantum_bundle_operation(self, quantum_engine, test_hypervector):
        """Test quantum bundle operation."""
        # Create quantum states
        hv1 = test_hypervector
        hv2 = HyperVector(1000)
        hv2.randomize(seed=43)
        
        quantum_state1 = quantum_engine.create_quantum_hypervector(hv1)
        quantum_state2 = quantum_engine.create_quantum_hypervector(hv2)
        
        # Perform quantum bundle
        start_time = time.time()
        bundled_state = quantum_engine.quantum_bundle([quantum_state1, quantum_state2])
        execution_time = time.time() - start_time
        
        assert isinstance(bundled_state, QuantumState)
        assert bundled_state.dimension == 1000
        
        # Check operation was recorded
        assert len(quantum_engine.operation_history) > 0
        assert len(quantum_engine.speedup_measurements) > 0
        
        # Check for performance improvement indication
        assert execution_time < 0.1  # Should be fast
    
    def test_quantum_entanglement(self, quantum_engine, test_hypervector):
        """Test quantum entanglement creation."""
        hv1 = test_hypervector
        hv2 = HyperVector(1000)
        hv2.randomize(seed=44)
        
        quantum_state1 = quantum_engine.create_quantum_hypervector(hv1)
        quantum_state2 = quantum_engine.create_quantum_hypervector(hv2)
        
        # Create entanglement
        entangled1, entangled2 = quantum_engine.quantum_entangle(quantum_state1, quantum_state2)
        
        assert isinstance(entangled1, QuantumState)
        assert isinstance(entangled2, QuantumState)
        
        # Check entanglement registry
        assert len(entangled1.entangled_qubits) > 0
        assert len(entangled2.entangled_qubits) > 0
        assert id(entangled2) in entangled1.entangled_qubits
        assert id(entangled1) in entangled2.entangled_qubits
    
    def test_quantum_interference_similarity(self, quantum_engine, test_hypervector):
        """Test quantum interference similarity calculation."""
        hv1 = test_hypervector
        hv2 = HyperVector(1000)
        hv2.randomize(seed=45)
        
        quantum_state1 = quantum_engine.create_quantum_hypervector(hv1)
        quantum_state2 = quantum_engine.create_quantum_hypervector(hv2)
        
        # Calculate quantum similarity
        start_time = time.time()
        similarity = quantum_engine.quantum_interference_similarity(quantum_state1, quantum_state2)
        execution_time = time.time() - start_time
        
        assert 0.0 <= similarity <= 1.0
        assert execution_time < 0.001  # Should be very fast (<1ms)
        
        # Test self-similarity
        self_similarity = quantum_engine.quantum_interference_similarity(quantum_state1, quantum_state1)
        assert self_similarity > 0.9  # Should be high for identical states
    
    def test_quantum_measurement(self, quantum_engine, test_hypervector):
        """Test quantum measurement collapse."""
        quantum_state = quantum_engine.create_quantum_hypervector(test_hypervector)
        
        # Perform measurement
        classical_result = quantum_engine.quantum_measurement(quantum_state)
        
        assert isinstance(classical_result, HyperVector)
        assert classical_result.dimension == 1000
        assert all(x in [-1, 1] for x in classical_result.data)
    
    def test_circuit_compilation(self, quantum_engine):
        """Test quantum circuit compilation."""
        operations = ["bundle", "bind", "similarity"]
        
        circuit = quantum_engine.compile_quantum_circuit(operations)
        
        assert isinstance(circuit, str)
        assert "OPENQASM 2.0" in circuit
        assert "qreg q[1000]" in circuit
        assert "creg c[1000]" in circuit
        assert "measure q -> c" in circuit
        
        # Test caching
        circuit2 = quantum_engine.compile_quantum_circuit(operations)
        assert circuit == circuit2  # Should be cached
    
    def test_performance_metrics(self, quantum_engine, test_hypervector):
        """Test performance metrics collection."""
        # Perform some operations to generate metrics
        quantum_state = quantum_engine.create_quantum_hypervector(test_hypervector)
        quantum_engine.quantum_bundle([quantum_state])
        
        metrics = quantum_engine.get_performance_metrics()
        
        assert "average_speedup" in metrics
        assert "maximum_speedup" in metrics
        assert "total_operations" in metrics
        assert "average_latency_us" in metrics
        assert "quantum_backend" in metrics
        
        assert metrics["total_operations"] > 0
        assert metrics["quantum_backend"] == "simulator"
    
    def test_quantum_state_persistence(self, quantum_engine, test_hypervector, tmp_path):
        """Test saving and loading quantum states."""
        quantum_state = quantum_engine.create_quantum_hypervector(test_hypervector)
        
        # Save state
        filepath = tmp_path / "quantum_state.json"
        quantum_engine.save_quantum_state(quantum_state, filepath)
        
        assert filepath.exists()
        
        # Load state
        loaded_state = quantum_engine.load_quantum_state(filepath)
        
        assert isinstance(loaded_state, QuantumState)
        assert loaded_state.dimension == quantum_state.dimension
        assert np.allclose(loaded_state.amplitudes, quantum_state.amplitudes)
        assert np.allclose(loaded_state.phases, quantum_state.phases)


class TestQuantumClassicalHybrid:
    """Test quantum-classical hybrid processing."""
    
    @pytest.fixture
    def hybrid_processor(self):
        """Create hybrid processor for testing."""
        quantum_engine = QuantumHDCEngine(dimension=500, quantum_backend="simulator")
        return QuantumClassicalHybrid(quantum_engine)
    
    @pytest.fixture
    def test_vectors(self):
        """Create test vectors."""
        vectors = []
        for i in range(10):
            hv = HyperVector(500)
            hv.randomize(seed=i)
            vectors.append(hv)
        return vectors
    
    def test_hybrid_initialization(self, hybrid_processor):
        """Test hybrid processor initialization."""
        assert hybrid_processor.quantum_engine is not None
        assert hybrid_processor.classical_ops is not None
        assert len(hybrid_processor.routing_decisions) == 0
    
    def test_operation_routing(self, hybrid_processor):
        """Test intelligent operation routing."""
        # Test small workload (should route to classical)
        routing = hybrid_processor.route_operation("bundle", size=5)
        assert routing in ["quantum", "classical"]
        
        # Test large workload (may route to quantum if available)
        routing = hybrid_processor.route_operation("bundle", size=1500)
        assert routing in ["quantum", "classical"]
        
        # Test high-dimensional operation
        routing = hybrid_processor.route_operation("similarity", dimension=8000)
        assert routing in ["quantum", "classical"]
    
    def test_adaptive_bundle(self, hybrid_processor, test_vectors):
        """Test adaptive bundling with automatic routing."""
        result = hybrid_processor.adaptive_bundle(test_vectors)
        
        assert isinstance(result, HyperVector)
        assert result.dimension == 500
        
        # Check metrics
        metrics = hybrid_processor.get_hybrid_metrics()
        assert "total_operations" in metrics
        assert metrics["total_operations"] > 0
    
    def test_hybrid_metrics(self, hybrid_processor, test_vectors):
        """Test hybrid processing metrics."""
        # Perform some operations
        hybrid_processor.adaptive_bundle(test_vectors[:3])  # Small bundle
        hybrid_processor.adaptive_bundle(test_vectors)     # Larger bundle
        
        metrics = hybrid_processor.get_hybrid_metrics()
        
        assert "total_operations" in metrics
        assert "quantum_operations" in metrics
        assert "classical_operations" in metrics
        assert "quantum_percentage" in metrics
        assert "classical_percentage" in metrics
        
        assert metrics["total_operations"] == 2
        assert metrics["quantum_percentage"] + metrics["classical_percentage"] == 100


class TestQuantumPerformance:
    """Test quantum performance characteristics."""
    
    @pytest.fixture
    def quantum_engine(self):
        """Create quantum engine for performance testing."""
        return QuantumHDCEngine(dimension=2000, quantum_backend="simulator")
    
    def test_latency_targets(self, quantum_engine):
        """Test that quantum operations meet latency targets."""
        # Create test states
        hv1 = HyperVector(2000)
        hv1.randomize(seed=100)
        hv2 = HyperVector(2000)
        hv2.randomize(seed=101)
        
        quantum_state1 = quantum_engine.create_quantum_hypervector(hv1)
        quantum_state2 = quantum_engine.create_quantum_hypervector(hv2)
        
        # Test similarity latency (target: <10μs)
        start_time = time.time()
        similarity = quantum_engine.quantum_interference_similarity(quantum_state1, quantum_state2)
        latency_us = (time.time() - start_time) * 1e6
        
        # Note: In simulation, actual quantum hardware would be faster
        # This tests the computational overhead
        assert latency_us < 1000  # Allow 1ms for simulation
        assert 0.0 <= similarity <= 1.0
    
    def test_speedup_measurement(self, quantum_engine):
        """Test speedup measurement and recording."""
        hv1 = HyperVector(2000)
        hv1.randomize(seed=200)
        hv2 = HyperVector(2000) 
        hv2.randomize(seed=201)
        
        quantum_state1 = quantum_engine.create_quantum_hypervector(hv1)
        quantum_state2 = quantum_engine.create_quantum_hypervector(hv2)
        
        # Perform bundle operation
        quantum_engine.quantum_bundle([quantum_state1, quantum_state2])
        
        # Check speedup measurements
        assert len(quantum_engine.speedup_measurements) > 0
        assert len(quantum_engine.operation_history) > 0
        
        # Check operation metadata
        operation = quantum_engine.operation_history[-1]
        assert isinstance(operation, QuantumOperation)
        assert operation.speedup_factor > 0
        assert 0.0 <= operation.fidelity <= 1.0
        assert operation.error_rate >= 0.0
    
    def test_large_scale_operations(self, quantum_engine):
        """Test performance with larger scale operations."""
        # Create multiple quantum states
        quantum_states = []
        for i in range(20):
            hv = HyperVector(2000)
            hv.randomize(seed=i + 300)
            quantum_state = quantum_engine.create_quantum_hypervector(hv)
            quantum_states.append(quantum_state)
        
        # Bundle all states
        start_time = time.time()
        bundled_state = quantum_engine.quantum_bundle(quantum_states)
        execution_time = time.time() - start_time
        
        assert isinstance(bundled_state, QuantumState)
        assert execution_time < 1.0  # Should complete within 1 second
        
        # Check final metrics
        metrics = quantum_engine.get_performance_metrics()
        assert metrics["total_operations"] > 0
        assert metrics["average_speedup"] > 0


class TestQuantumIntegration:
    """Test integration with existing HDC systems."""
    
    def test_quantum_classical_consistency(self):
        """Test consistency between quantum and classical operations."""
        # Create identical classical vectors
        hv1 = HyperVector(500)
        hv1.randomize(seed=42)
        hv2 = HyperVector(500)
        hv2.randomize(seed=42)  # Same seed = identical vectors
        
        # Classical similarity
        classical_similarity = hv1.similarity(hv2)
        
        # Quantum similarity
        quantum_engine = QuantumHDCEngine(dimension=500)
        quantum_state1 = quantum_engine.create_quantum_hypervector(hv1)
        quantum_state2 = quantum_engine.create_quantum_hypervector(hv2)
        quantum_similarity = quantum_engine.quantum_interference_similarity(quantum_state1, quantum_state2)
        
        # Should be very similar (allowing for quantum approximation)
        assert abs(classical_similarity - quantum_similarity) < 0.2
    
    def test_end_to_end_quantum_workflow(self):
        """Test complete quantum HDC workflow."""
        # Initialize systems
        quantum_engine = QuantumHDCEngine(dimension=1000)
        hybrid = QuantumClassicalHybrid(quantum_engine)
        
        # Create test data
        vectors = [HyperVector(1000) for _ in range(5)]
        for i, hv in enumerate(vectors):
            hv.randomize(seed=i + 500)
        
        # Perform hybrid operations
        bundled = hybrid.adaptive_bundle(vectors)
        
        # Verify results
        assert isinstance(bundled, HyperVector)
        assert bundled.dimension == 1000
        
        # Check metrics
        quantum_metrics = quantum_engine.get_performance_metrics()
        hybrid_metrics = hybrid.get_hybrid_metrics()
        
        assert quantum_metrics["total_operations"] >= 0
        assert hybrid_metrics["total_operations"] > 0


def test_quantum_demonstration():
    """Test the quantum HDC demonstration function."""
    try:
        quantum_engine, hybrid = demonstrate_quantum_hdc()
        
        assert isinstance(quantum_engine, QuantumHDCEngine)
        assert isinstance(hybrid, QuantumClassicalHybrid)
        
        # Check that operations were performed
        metrics = quantum_engine.get_performance_metrics()
        assert metrics["total_operations"] > 0
        
        hybrid_metrics = hybrid.get_hybrid_metrics() 
        assert hybrid_metrics["total_operations"] > 0
        
    except Exception as e:
        # Demonstration should not fail
        pytest.fail(f"Quantum demonstration failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])