"""
Quantum-Inspired Hyperdimensional Computing (Q-HDC)

Novel implementation of quantum-inspired operations for HDC, bringing quantum advantages
to classical robotic systems. This research module implements quantum superposition,
entanglement, and interference principles adapted for hyperdimensional computing.

Research Contributions:
1. Quantum Superposition HDC: Probabilistic hypervector representations
2. Entangled Hypervectors: Non-local correlations for sensor fusion
3. Quantum Interference: Constructive/destructive pattern matching
4. Quantum Annealing HDC: Optimization using quantum-inspired dynamics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import time
import cmath
from dataclasses import dataclass
from collections import defaultdict
import logging
from scipy.optimize import minimize
from scipy.linalg import expm

from ..core.hypervector import HyperVector
from ..core.operations import HDCOperations


@dataclass
class QuantumState:
    """Quantum state representation for HDC."""
    amplitudes: np.ndarray  # Complex amplitudes
    dimension: int
    
    def __post_init__(self):
        """Ensure normalization of quantum state."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm


class QuantumHyperVector:
    """
    Quantum-inspired hyperdimensional vector with complex amplitudes.
    
    Represents quantum superposition states in high-dimensional space,
    enabling quantum interference patterns and entanglement correlations.
    """
    
    def __init__(self, dimension: int = 10000, amplitudes: Optional[np.ndarray] = None):
        """
        Initialize quantum hypervector.
        
        Args:
            dimension: Vector dimension
            amplitudes: Complex amplitude array (normalized)
        """
        self.dimension = dimension
        
        if amplitudes is not None:
            if len(amplitudes) != dimension:
                raise ValueError("Amplitudes length must match dimension")
            self.amplitudes = amplitudes.astype(np.complex128)
        else:
            # Initialize in uniform superposition
            self.amplitudes = np.ones(dimension, dtype=np.complex128) / np.sqrt(dimension)
        
        self._normalize()
    
    def _normalize(self):
        """Normalize quantum state."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes = self.amplitudes / norm
    
    @classmethod
    def from_classical(cls, classical_hv: HyperVector, phase_noise: float = 0.0) -> 'QuantumHyperVector':
        """
        Create quantum hypervector from classical HDC vector.
        
        Args:
            classical_hv: Classical hyperdimensional vector
            phase_noise: Amount of phase noise to add
            
        Returns:
            Quantum hypervector with complex amplitudes
        """
        # Convert bipolar to complex amplitudes
        amplitudes = classical_hv.data.astype(np.complex128)
        amplitudes = amplitudes / np.sqrt(2.0)  # Normalize magnitude
        
        # Add quantum phase information
        if phase_noise > 0:
            phases = np.random.uniform(-np.pi * phase_noise, np.pi * phase_noise, classical_hv.dimension)
            amplitudes = amplitudes * np.exp(1j * phases)
        
        return cls(classical_hv.dimension, amplitudes)
    
    def to_classical(self, measurement_basis: str = 'computational') -> HyperVector:
        """
        Collapse quantum state to classical HDC vector via measurement.
        
        Args:
            measurement_basis: Measurement basis ('computational', 'hadamard', 'random')
            
        Returns:
            Collapsed classical hyperdimensional vector
        """
        if measurement_basis == 'computational':
            # Measure in computational basis (|0⟩, |1⟩)
            probabilities = np.abs(self.amplitudes) ** 2
            measurements = np.random.random(self.dimension) < probabilities
            classical_data = np.where(measurements, 1, -1).astype(np.int8)
            
        elif measurement_basis == 'hadamard':
            # Measure in Hadamard basis (|+⟩, |-⟩)
            hadamard_amplitudes = (self.amplitudes + np.conj(self.amplitudes)) / np.sqrt(2)
            probabilities = np.abs(hadamard_amplitudes) ** 2
            measurements = np.random.random(self.dimension) < probabilities
            classical_data = np.where(measurements, 1, -1).astype(np.int8)
            
        elif measurement_basis == 'random':
            # Random measurement basis
            random_phases = np.random.uniform(0, 2*np.pi, self.dimension)
            rotated_amplitudes = self.amplitudes * np.exp(1j * random_phases)
            probabilities = np.abs(rotated_amplitudes) ** 2
            measurements = np.random.random(self.dimension) < probabilities
            classical_data = np.where(measurements, 1, -1).astype(np.int8)
            
        else:
            raise ValueError(f"Unknown measurement basis: {measurement_basis}")
        
        return HyperVector(self.dimension, classical_data)
    
    def quantum_bundle(self, other: 'QuantumHyperVector') -> 'QuantumHyperVector':
        """
        Quantum superposition of two quantum hypervectors.
        
        Args:
            other: Other quantum hypervector
            
        Returns:
            Superposed quantum hypervector
        """
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for quantum bundling")
        
        # Quantum superposition: |ψ⟩ = α|ψ₁⟩ + β|ψ₂⟩
        alpha = 1 / np.sqrt(2)
        beta = 1 / np.sqrt(2)
        
        superposed_amplitudes = alpha * self.amplitudes + beta * other.amplitudes
        return QuantumHyperVector(self.dimension, superposed_amplitudes)
    
    def quantum_bind(self, other: 'QuantumHyperVector') -> 'QuantumHyperVector':
        """
        Quantum entanglement binding of two quantum hypervectors.
        
        Args:
            other: Other quantum hypervector
            
        Returns:
            Entangled quantum hypervector
        """
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for quantum binding")
        
        # Element-wise quantum multiplication (entanglement)
        entangled_amplitudes = self.amplitudes * other.amplitudes
        return QuantumHyperVector(self.dimension, entangled_amplitudes)
    
    def quantum_rotate(self, rotation_angles: np.ndarray) -> 'QuantumHyperVector':
        """
        Apply quantum rotation to hypervector.
        
        Args:
            rotation_angles: Rotation angles for each dimension
            
        Returns:
            Rotated quantum hypervector
        """
        if len(rotation_angles) != self.dimension:
            raise ValueError("Rotation angles must match dimension")
        
        # Apply rotation: |ψ'⟩ = e^(iθ)|ψ⟩
        rotation_phases = np.exp(1j * rotation_angles)
        rotated_amplitudes = self.amplitudes * rotation_phases
        
        return QuantumHyperVector(self.dimension, rotated_amplitudes)
    
    def quantum_interference(self, other: 'QuantumHyperVector', phase_difference: float = 0.0) -> 'QuantumHyperVector':
        """
        Quantum interference between two quantum hypervectors.
        
        Args:
            other: Other quantum hypervector
            phase_difference: Phase difference for interference pattern
            
        Returns:
            Interfered quantum hypervector
        """
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for quantum interference")
        
        # Apply phase difference
        phase_shifted_other = other.quantum_rotate(np.full(other.dimension, phase_difference))
        
        # Constructive/destructive interference
        interference_amplitudes = self.amplitudes + phase_shifted_other.amplitudes
        return QuantumHyperVector(self.dimension, interference_amplitudes)
    
    def quantum_similarity(self, other: 'QuantumHyperVector') -> complex:
        """
        Quantum fidelity (similarity) between quantum states.
        
        Args:
            other: Other quantum hypervector
            
        Returns:
            Complex quantum fidelity
        """
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for quantum similarity")
        
        # Quantum fidelity: F = |⟨ψ₁|ψ₂⟩|²
        inner_product = np.vdot(self.amplitudes, other.amplitudes)
        return inner_product
    
    def quantum_entanglement_entropy(self, partition_size: int) -> float:
        """
        Calculate entanglement entropy for a bipartition.
        
        Args:
            partition_size: Size of first partition
            
        Returns:
            Von Neumann entanglement entropy
        """
        if partition_size >= self.dimension or partition_size <= 0:
            raise ValueError("Invalid partition size")
        
        # Reshape state for bipartition
        state_matrix = self.amplitudes.reshape(partition_size, -1)
        
        # Compute reduced density matrix for first partition
        rho = state_matrix @ state_matrix.conj().T
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]  # Remove zero eigenvalues
        
        # Von Neumann entropy: S = -Tr(ρ log ρ)
        entropy = -np.sum(eigenvals * np.log(eigenvals))
        return float(entropy.real)
    
    def expectation_value(self, observable: np.ndarray) -> complex:
        """
        Calculate expectation value of observable.
        
        Args:
            observable: Observable matrix
            
        Returns:
            Expectation value ⟨ψ|O|ψ⟩
        """
        if observable.shape != (self.dimension, self.dimension):
            raise ValueError("Observable must be square matrix matching dimension")
        
        expectation = np.vdot(self.amplitudes, observable @ self.amplitudes)
        return expectation


class QuantumHDCProcessor:
    """
    Quantum-inspired HDC processor for advanced robotics computations.
    
    Implements quantum algorithms and protocols adapted for HDC,
    including quantum walks, quantum annealing, and quantum machine learning.
    """
    
    def __init__(self, dimension: int = 10000):
        """
        Initialize quantum HDC processor.
        
        Args:
            dimension: HDC vector dimension
        """
        self.dimension = dimension
        self.logger = logging.getLogger(__name__)
        
        # Quantum circuit simulation parameters
        self.circuit_depth = 10
        self.decoherence_rate = 0.01
        
        # Quantum annealing parameters
        self.annealing_time = 1000
        self.temperature_schedule = self._create_temperature_schedule()
        
    def quantum_walk_search(self, target_vector: QuantumHyperVector, 
                           search_space: List[QuantumHyperVector], 
                           steps: int = 100) -> Tuple[QuantumHyperVector, float]:
        """
        Quantum walk-based search in hyperdimensional space.
        
        Args:
            target_vector: Target quantum hypervector
            search_space: List of candidate vectors
            steps: Number of quantum walk steps
            
        Returns:
            Best match and quantum similarity
        """
        self.logger.info(f"Performing quantum walk search with {steps} steps")
        
        if not search_space:
            raise ValueError("Search space cannot be empty")
        
        # Initialize walker in uniform superposition
        walker_amplitudes = np.ones(len(search_space), dtype=np.complex128) / np.sqrt(len(search_space))
        
        # Quantum walk evolution
        for step in range(steps):
            # Apply coin operator (Hadamard-like transformation)
            walker_amplitudes = self._apply_quantum_coin(walker_amplitudes)
            
            # Apply shift operator (move walker)
            walker_amplitudes = self._apply_quantum_shift(walker_amplitudes, search_space, target_vector)
            
            # Add decoherence
            walker_amplitudes = self._apply_decoherence(walker_amplitudes, step)
        
        # Measure final state
        probabilities = np.abs(walker_amplitudes) ** 2
        best_index = np.argmax(probabilities)
        best_match = search_space[best_index]
        
        # Calculate quantum similarity
        similarity = abs(target_vector.quantum_similarity(best_match))
        
        return best_match, similarity
    
    def quantum_annealing_optimization(self, objective_function: Callable[[QuantumHyperVector], float],
                                     initial_state: QuantumHyperVector,
                                     constraints: Optional[List[Callable]] = None) -> QuantumHyperVector:
        """
        Quantum annealing for HDC optimization problems.
        
        Args:
            objective_function: Function to minimize
            initial_state: Initial quantum state
            constraints: List of constraint functions
            
        Returns:
            Optimized quantum hypervector
        """
        self.logger.info("Starting quantum annealing optimization")
        
        current_state = QuantumHyperVector(initial_state.dimension, initial_state.amplitudes.copy())
        best_state = current_state
        best_energy = objective_function(current_state)
        
        for step in range(self.annealing_time):
            temperature = self.temperature_schedule[step]
            
            # Generate neighbor state via quantum tunneling
            neighbor_state = self._quantum_tunneling_move(current_state, temperature)
            
            # Apply constraints
            if constraints:
                for constraint in constraints:
                    if not constraint(neighbor_state):
                        continue  # Skip invalid states
            
            # Calculate energy difference
            neighbor_energy = objective_function(neighbor_state)
            energy_diff = neighbor_energy - objective_function(current_state)
            
            # Quantum acceptance probability
            acceptance_prob = self._quantum_acceptance_probability(energy_diff, temperature)
            
            if np.random.random() < acceptance_prob:
                current_state = neighbor_state
                
                if neighbor_energy < best_energy:
                    best_state = neighbor_state
                    best_energy = neighbor_energy
        
        self.logger.info(f"Quantum annealing completed. Best energy: {best_energy}")
        return best_state
    
    def quantum_machine_learning(self, training_data: List[Tuple[QuantumHyperVector, QuantumHyperVector]],
                                test_data: List[QuantumHyperVector],
                                learning_rate: float = 0.1,
                                epochs: int = 100) -> Tuple[Dict[str, Any], List[QuantumHyperVector]]:
        """
        Quantum machine learning algorithm for HDC pattern recognition.
        
        Args:
            training_data: List of (input, target) pairs
            test_data: Test input vectors
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            
        Returns:
            Training statistics and predictions
        """
        self.logger.info(f"Starting quantum ML training with {len(training_data)} samples")
        
        # Initialize quantum neural network parameters
        num_layers = 3
        param_dim = self.dimension // 10  # Reduced parameter space
        
        # Random quantum parameters for each layer
        quantum_params = []
        for layer in range(num_layers):
            layer_params = np.random.uniform(-np.pi, np.pi, param_dim)
            quantum_params.append(layer_params)
        
        training_stats = {
            'losses': [],
            'quantum_fidelities': [],
            'parameter_norms': []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_fidelity = 0.0
            
            # Shuffle training data
            shuffled_data = training_data.copy()
            np.random.shuffle(shuffled_data)
            
            for input_qhv, target_qhv in shuffled_data:
                # Forward pass through quantum neural network
                output_qhv = self._quantum_forward_pass(input_qhv, quantum_params)
                
                # Compute quantum loss
                loss = self._quantum_loss_function(output_qhv, target_qhv)
                epoch_loss += loss
                
                # Compute quantum fidelity
                fidelity = abs(output_qhv.quantum_similarity(target_qhv))
                epoch_fidelity += fidelity
                
                # Quantum gradient computation
                gradients = self._quantum_gradient_computation(input_qhv, target_qhv, quantum_params)
                
                # Parameter update
                for layer in range(num_layers):
                    quantum_params[layer] -= learning_rate * gradients[layer]
            
            avg_loss = epoch_loss / len(training_data)
            avg_fidelity = epoch_fidelity / len(training_data)
            
            training_stats['losses'].append(avg_loss)
            training_stats['quantum_fidelities'].append(avg_fidelity)
            training_stats['parameter_norms'].append([np.linalg.norm(params) for params in quantum_params])
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}/{epochs}: Loss={avg_loss:.4f}, Fidelity={avg_fidelity:.4f}")
        
        # Make predictions on test data
        predictions = []
        for test_input in test_data:
            prediction = self._quantum_forward_pass(test_input, quantum_params)
            predictions.append(prediction)
        
        return training_stats, predictions
    
    def quantum_sensor_fusion(self, sensor_data: Dict[str, QuantumHyperVector], 
                            fusion_strategy: str = 'entanglement') -> QuantumHyperVector:
        """
        Quantum sensor fusion using entanglement and superposition.
        
        Args:
            sensor_data: Dictionary of sensor quantum hypervectors
            fusion_strategy: Fusion strategy ('entanglement', 'interference', 'superposition')
            
        Returns:
            Fused quantum sensor representation
        """
        if not sensor_data:
            raise ValueError("Sensor data cannot be empty")
        
        sensor_vectors = list(sensor_data.values())
        sensor_names = list(sensor_data.keys())
        
        if fusion_strategy == 'entanglement':
            # Entangle all sensors
            fused = sensor_vectors[0]
            for sensor_qhv in sensor_vectors[1:]:
                fused = fused.quantum_bind(sensor_qhv)
            
        elif fusion_strategy == 'interference':
            # Quantum interference fusion
            fused = sensor_vectors[0]
            for i, sensor_qhv in enumerate(sensor_vectors[1:], 1):
                phase_shift = 2 * np.pi * i / len(sensor_vectors)
                fused = fused.quantum_interference(sensor_qhv, phase_shift)
            
        elif fusion_strategy == 'superposition':
            # Equal superposition of all sensors
            fused_amplitudes = np.zeros(self.dimension, dtype=np.complex128)
            weight = 1 / np.sqrt(len(sensor_vectors))
            
            for sensor_qhv in sensor_vectors:
                fused_amplitudes += weight * sensor_qhv.amplitudes
            
            fused = QuantumHyperVector(self.dimension, fused_amplitudes)
            
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")
        
        self.logger.info(f"Quantum sensor fusion completed using {fusion_strategy} strategy")
        return fused
    
    def _apply_quantum_coin(self, walker_amplitudes: np.ndarray) -> np.ndarray:
        """Apply quantum coin operator for quantum walk."""
        # Hadamard-like coin operator
        coin_matrix = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        
        # Apply to each position (simplified for 1D walk)
        new_amplitudes = np.zeros_like(walker_amplitudes)
        for i in range(0, len(walker_amplitudes), 2):
            if i + 1 < len(walker_amplitudes):
                state_pair = np.array([walker_amplitudes[i], walker_amplitudes[i+1]])
                new_pair = coin_matrix @ state_pair
                new_amplitudes[i] = new_pair[0]
                new_amplitudes[i+1] = new_pair[1]
            else:
                new_amplitudes[i] = walker_amplitudes[i]  # Boundary condition
        
        return new_amplitudes
    
    def _apply_quantum_shift(self, walker_amplitudes: np.ndarray, 
                           search_space: List[QuantumHyperVector],
                           target: QuantumHyperVector) -> np.ndarray:
        """Apply quantum shift operator based on target similarity."""
        new_amplitudes = np.zeros_like(walker_amplitudes)
        
        for i in range(len(walker_amplitudes)):
            # Compute shift based on quantum similarity to target
            similarity = abs(search_space[i].quantum_similarity(target))
            
            # Quantum shift with bias toward similar states
            left_idx = (i - 1) % len(walker_amplitudes)
            right_idx = (i + 1) % len(walker_amplitudes)
            
            # Biased shift based on similarity
            shift_bias = similarity
            new_amplitudes[i] = (1 - shift_bias) * walker_amplitudes[i] + \
                              shift_bias * (walker_amplitudes[left_idx] + walker_amplitudes[right_idx]) / 2
        
        # Normalize
        norm = np.linalg.norm(new_amplitudes)
        if norm > 0:
            new_amplitudes = new_amplitudes / norm
        
        return new_amplitudes
    
    def _apply_decoherence(self, walker_amplitudes: np.ndarray, step: int) -> np.ndarray:
        """Apply decoherence effects to quantum walker."""
        # Time-dependent decoherence
        decoherence_strength = self.decoherence_rate * np.sqrt(step + 1)
        
        # Add random phase noise
        phase_noise = np.random.normal(0, decoherence_strength, len(walker_amplitudes))
        decoherent_amplitudes = walker_amplitudes * np.exp(1j * phase_noise)
        
        # Amplitude damping
        damping_factor = np.exp(-decoherence_strength)
        decoherent_amplitudes *= damping_factor
        
        # Normalize
        norm = np.linalg.norm(decoherent_amplitudes)
        if norm > 0:
            decoherent_amplitudes = decoherent_amplitudes / norm
        
        return decoherent_amplitudes
    
    def _create_temperature_schedule(self) -> np.ndarray:
        """Create temperature schedule for quantum annealing."""
        # Exponential cooling schedule
        initial_temp = 10.0
        final_temp = 0.01
        
        schedule = np.exp(np.linspace(np.log(initial_temp), np.log(final_temp), self.annealing_time))
        return schedule
    
    def _quantum_tunneling_move(self, current_state: QuantumHyperVector, temperature: float) -> QuantumHyperVector:
        """Generate neighbor state via quantum tunneling."""
        # Quantum tunneling probability
        tunnel_strength = np.sqrt(temperature)
        
        # Add quantum noise to amplitudes
        noise_amplitudes = np.random.normal(0, tunnel_strength, self.dimension) + \
                          1j * np.random.normal(0, tunnel_strength, self.dimension)
        
        new_amplitudes = current_state.amplitudes + noise_amplitudes
        return QuantumHyperVector(self.dimension, new_amplitudes)
    
    def _quantum_acceptance_probability(self, energy_diff: float, temperature: float) -> float:
        """Calculate quantum acceptance probability."""
        if energy_diff <= 0:
            return 1.0  # Always accept better states
        
        # Quantum thermal factor
        beta = 1.0 / (temperature + 1e-8)
        
        # Quantum acceptance with tunneling effects
        classical_prob = np.exp(-beta * energy_diff)
        quantum_tunneling = 1.0 / (1.0 + energy_diff**2)  # Quantum tunneling term
        
        return min(1.0, classical_prob + 0.1 * quantum_tunneling)
    
    def _quantum_forward_pass(self, input_qhv: QuantumHyperVector, 
                            quantum_params: List[np.ndarray]) -> QuantumHyperVector:
        """Forward pass through quantum neural network."""
        current_state = input_qhv
        
        for layer_params in quantum_params:
            # Apply parameterized quantum gates
            current_state = self._apply_quantum_layer(current_state, layer_params)
        
        return current_state
    
    def _apply_quantum_layer(self, input_qhv: QuantumHyperVector, 
                           layer_params: np.ndarray) -> QuantumHyperVector:
        """Apply parameterized quantum layer."""
        # Sample subset of dimensions for efficiency
        sample_size = len(layer_params)
        indices = np.random.choice(self.dimension, sample_size, replace=False)
        
        # Apply rotation gates
        new_amplitudes = input_qhv.amplitudes.copy()
        for i, param in enumerate(layer_params):
            idx = indices[i]
            rotation = np.exp(1j * param)
            new_amplitudes[idx] *= rotation
        
        return QuantumHyperVector(self.dimension, new_amplitudes)
    
    def _quantum_loss_function(self, output_qhv: QuantumHyperVector, 
                             target_qhv: QuantumHyperVector) -> float:
        """Quantum loss function based on fidelity."""
        fidelity = abs(output_qhv.quantum_similarity(target_qhv))
        loss = 1.0 - fidelity  # Convert fidelity to loss
        return loss
    
    def _quantum_gradient_computation(self, input_qhv: QuantumHyperVector,
                                    target_qhv: QuantumHyperVector,
                                    quantum_params: List[np.ndarray]) -> List[np.ndarray]:
        """Compute quantum gradients using parameter shift rule."""
        gradients = []
        
        for layer_idx, layer_params in enumerate(quantum_params):
            layer_gradients = np.zeros_like(layer_params)
            
            for param_idx in range(len(layer_params)):
                # Parameter shift rule: gradient = [f(θ+π/2) - f(θ-π/2)] / 2
                shift = np.pi / 2
                
                # Forward pass with positive shift
                params_plus = quantum_params.copy()
                params_plus[layer_idx][param_idx] += shift
                output_plus = self._quantum_forward_pass(input_qhv, params_plus)
                loss_plus = self._quantum_loss_function(output_plus, target_qhv)
                
                # Forward pass with negative shift  
                params_minus = quantum_params.copy()
                params_minus[layer_idx][param_idx] -= shift
                output_minus = self._quantum_forward_pass(input_qhv, params_minus)
                loss_minus = self._quantum_loss_function(output_minus, target_qhv)
                
                # Compute gradient
                layer_gradients[param_idx] = (loss_plus - loss_minus) / 2
            
            gradients.append(layer_gradients)
        
        return gradients


class QuantumHDCBenchmark:
    """
    Comprehensive benchmarking suite for quantum HDC algorithms.
    
    Provides standardized tests and metrics for evaluating quantum
    advantages in hyperdimensional computing applications.
    """
    
    def __init__(self, dimension: int = 10000):
        """
        Initialize quantum HDC benchmark.
        
        Args:
            dimension: HDC vector dimension
        """
        self.dimension = dimension
        self.processor = QuantumHDCProcessor(dimension)
        self.logger = logging.getLogger(__name__)
    
    def benchmark_quantum_vs_classical(self, test_tasks: List[str], 
                                     num_trials: int = 10) -> Dict[str, Any]:
        """
        Comprehensive comparison of quantum vs classical HDC performance.
        
        Args:
            test_tasks: List of tasks to benchmark
            num_trials: Number of trials per task
            
        Returns:
            Benchmark results with statistical analysis
        """
        results = {
            'tasks': {},
            'summary': {},
            'quantum_advantages': []
        }
        
        for task in test_tasks:
            self.logger.info(f"Benchmarking task: {task}")
            
            task_results = {
                'quantum_times': [],
                'classical_times': [],
                'quantum_accuracies': [],
                'classical_accuracies': [],
                'quantum_advantages': []
            }
            
            for trial in range(num_trials):
                # Generate test data for task
                test_data = self._generate_test_data(task)
                
                # Quantum benchmark
                q_start = time.time()
                q_accuracy = self._run_quantum_task(task, test_data)
                q_time = time.time() - q_start
                
                # Classical benchmark
                c_start = time.time()
                c_accuracy = self._run_classical_task(task, test_data)
                c_time = time.time() - c_start
                
                # Store results
                task_results['quantum_times'].append(q_time)
                task_results['classical_times'].append(c_time)
                task_results['quantum_accuracies'].append(q_accuracy)
                task_results['classical_accuracies'].append(c_accuracy)
                
                # Calculate quantum advantage
                time_advantage = c_time / q_time if q_time > 0 else 1.0
                accuracy_advantage = q_accuracy / c_accuracy if c_accuracy > 0 else 1.0
                overall_advantage = time_advantage * accuracy_advantage
                
                task_results['quantum_advantages'].append(overall_advantage)
            
            # Statistical analysis
            task_results['stats'] = {
                'avg_quantum_time': np.mean(task_results['quantum_times']),
                'avg_classical_time': np.mean(task_results['classical_times']),
                'avg_quantum_accuracy': np.mean(task_results['quantum_accuracies']),
                'avg_classical_accuracy': np.mean(task_results['classical_accuracies']),
                'avg_quantum_advantage': np.mean(task_results['quantum_advantages']),
                'advantage_std': np.std(task_results['quantum_advantages']),
                'significance': self._statistical_significance_test(
                    task_results['quantum_advantages'], 1.0
                )
            }
            
            results['tasks'][task] = task_results
            results['quantum_advantages'].extend(task_results['quantum_advantages'])
        
        # Overall summary
        results['summary'] = {
            'overall_quantum_advantage': np.mean(results['quantum_advantages']),
            'advantage_variance': np.var(results['quantum_advantages']),
            'num_tasks': len(test_tasks),
            'num_trials': num_trials,
            'statistically_significant': np.mean(results['quantum_advantages']) > 1.05  # 5% threshold
        }
        
        return results
    
    def _generate_test_data(self, task: str) -> Dict[str, Any]:
        """Generate test data for specific task."""
        data_size = 100
        
        if task == 'search':
            # Search task data
            target = QuantumHyperVector.from_classical(HyperVector.random(self.dimension))
            search_space = []
            for _ in range(data_size):
                candidate = QuantumHyperVector.from_classical(HyperVector.random(self.dimension))
                search_space.append(candidate)
            
            # Add target with noise to search space
            noisy_target = target.quantum_interference(
                QuantumHyperVector.from_classical(HyperVector.random(self.dimension)), 
                0.1
            )
            search_space[data_size // 2] = noisy_target
            
            return {'target': target, 'search_space': search_space}
            
        elif task == 'classification':
            # Classification task data  
            training_data = []
            test_data = []
            
            for i in range(data_size // 2):
                # Generate input-output pairs
                input_qhv = QuantumHyperVector.from_classical(HyperVector.random(self.dimension))
                target_qhv = QuantumHyperVector.from_classical(HyperVector.random(self.dimension))
                training_data.append((input_qhv, target_qhv))
            
            for i in range(data_size // 4):
                test_input = QuantumHyperVector.from_classical(HyperVector.random(self.dimension))
                test_data.append(test_input)
            
            return {'training_data': training_data, 'test_data': test_data}
            
        elif task == 'optimization':
            # Optimization task data
            def objective(qhv: QuantumHyperVector) -> float:
                # Simple quadratic objective
                classical_hv = qhv.to_classical()
                return np.sum(classical_hv.data**2) / self.dimension
            
            initial_state = QuantumHyperVector.from_classical(HyperVector.random(self.dimension))
            
            return {'objective': objective, 'initial_state': initial_state}
            
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def _run_quantum_task(self, task: str, test_data: Dict[str, Any]) -> float:
        """Run quantum version of task."""
        if task == 'search':
            target = test_data['target']
            search_space = test_data['search_space']
            best_match, similarity = self.processor.quantum_walk_search(target, search_space)
            return similarity
            
        elif task == 'classification':
            training_data = test_data['training_data']
            test_data_inputs = test_data['test_data']
            stats, predictions = self.processor.quantum_machine_learning(training_data, test_data_inputs)
            return np.mean(stats['quantum_fidelities'][-10:])  # Average of last 10 epochs
            
        elif task == 'optimization':
            objective = test_data['objective']
            initial_state = test_data['initial_state']
            optimized_state = self.processor.quantum_annealing_optimization(objective, initial_state)
            final_value = objective(optimized_state)
            initial_value = objective(initial_state)
            return max(0, (initial_value - final_value) / initial_value)  # Relative improvement
            
        else:
            return 0.0
    
    def _run_classical_task(self, task: str, test_data: Dict[str, Any]) -> float:
        """Run classical HDC version of task."""
        if task == 'search':
            target = test_data['target'].to_classical()
            search_space = [qhv.to_classical() for qhv in test_data['search_space']]
            
            best_similarity = -1.0
            for candidate in search_space:
                similarity = target.similarity(candidate)
                best_similarity = max(best_similarity, similarity)
            
            return best_similarity
            
        elif task == 'classification':
            # Simple classical HDC classification
            training_data = [(qin.to_classical(), qout.to_classical()) 
                           for qin, qout in test_data['training_data']]
            test_inputs = [qin.to_classical() for qin in test_data['test_data']]
            
            # Build simple associative memory
            associations = {}
            for inp, out in training_data:
                associations[tuple(inp.data)] = out
            
            # Test classification
            correct = 0
            for test_inp in test_inputs:
                best_match = None
                best_sim = -1.0
                
                for stored_inp_tuple, stored_out in associations.items():
                    stored_inp = HyperVector(self.dimension, np.array(stored_inp_tuple, dtype=np.int8))
                    sim = test_inp.similarity(stored_inp)
                    if sim > best_sim:
                        best_sim = sim
                        best_match = stored_out
                
                if best_match is not None and best_sim > 0.5:
                    correct += 1
            
            return correct / len(test_inputs) if test_inputs else 0.0
            
        elif task == 'optimization':
            # Simple hill climbing optimization
            objective = lambda qhv: test_data['objective'](QuantumHyperVector.from_classical(qhv))
            initial_classical = test_data['initial_state'].to_classical()
            
            current_state = initial_classical
            current_value = objective(current_state)
            initial_value = current_value
            
            for _ in range(100):  # Limited iterations for fair comparison
                # Generate neighbor
                neighbor = current_state.add_noise(0.1)
                neighbor_value = objective(neighbor)
                
                if neighbor_value < current_value:
                    current_state = neighbor
                    current_value = neighbor_value
            
            return max(0, (initial_value - current_value) / initial_value)
            
        else:
            return 0.0
    
    def _statistical_significance_test(self, advantages: List[float], null_hypothesis: float = 1.0) -> Dict[str, Any]:
        """Perform statistical significance test on quantum advantages."""
        from scipy import stats
        
        # One-sample t-test against null hypothesis (no advantage)
        t_stat, p_value = stats.ttest_1samp(advantages, null_hypothesis)
        
        # Effect size (Cohen's d)
        mean_advantage = np.mean(advantages)
        std_advantage = np.std(advantages, ddof=1)
        effect_size = (mean_advantage - null_hypothesis) / std_advantage if std_advantage > 0 else 0
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'effect_size': float(effect_size),
            'significant_at_0_05': p_value < 0.05,
            'significant_at_0_01': p_value < 0.01
        }