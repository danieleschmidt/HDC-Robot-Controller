"""
Core HDC operations for robotic control.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import numba
from numba import jit, prange
from .hypervector import HyperVector, HyperVectorSpace
import structlog

logger = structlog.get_logger()


@jit(nopython=True, parallel=True)
def _parallel_similarity_search(query: np.ndarray, 
                               memory: np.ndarray, 
                               top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fast parallel similarity search in hypervector memory."""
    n_vectors, dim = memory.shape
    similarities = np.zeros(n_vectors, dtype=np.float32)
    
    # Compute similarities in parallel
    for i in prange(n_vectors):
        matches = 0
        for j in range(dim):
            if query[j] == memory[i, j]:
                matches += 1
        similarities[i] = matches / dim
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_similarities = similarities[top_indices]
    
    return top_indices, top_similarities


class HDCOperations:
    """
    High-performance HDC operations for robotic control.
    
    Provides optimized implementations of core HDC operations:
    - Bundling and binding
    - Similarity search and retrieval
    - Temporal sequence processing
    - Noise-robust operations
    """
    
    def __init__(self, vector_space: HyperVectorSpace):
        """Initialize with a vector space."""
        self.space = vector_space
        self.dimension = vector_space.dimension
        
        # Performance metrics
        self.operation_counts = {
            'bind': 0,
            'bundle': 0,
            'query': 0,
            'encode': 0
        }
    
    def bind_multimodal(self, 
                       modalities: List[Tuple[HyperVector, str]]) -> HyperVector:
        """
        Bind multiple sensor modalities with role identifiers.
        
        Args:
            modalities: List of (hypervector, role) pairs
            
        Returns:
            Fused multimodal hypervector
        """
        if not modalities:
            return self.space.random_vector("EMPTY_MULTIMODAL")
        
        bound_modalities = []
        
        for hv, role in modalities:
            role_vector = self.space.get_basis(role.upper())
            bound_hv = hv.bind(role_vector)
            bound_modalities.append(bound_hv)
            
        self.operation_counts['bind'] += len(modalities)
        
        # Bundle all bound modalities
        if len(bound_modalities) == 1:
            return bound_modalities[0]
        
        result = bound_modalities[0].bundle(bound_modalities[1:])
        self.operation_counts['bundle'] += 1
        
        return result
    
    def encode_state_action(self, 
                           state_hv: HyperVector, 
                           action_hv: HyperVector,
                           reward: Optional[float] = None) -> HyperVector:
        """
        Encode state-action pair for learning.
        
        Args:
            state_hv: State hypervector
            action_hv: Action hypervector  
            reward: Optional reward signal
            
        Returns:
            Encoded state-action hypervector
        """
        # Bind state with STATE role
        state_bound = state_hv.bind(self.space.get_basis('STATE'))
        
        # Bind action with ACTION role
        action_bound = action_hv.bind(self.space.get_basis('ACTION'))
        
        # Bundle state and action
        components = [state_bound, action_bound]
        
        # Add reward if provided
        if reward is not None:
            reward_hv = self.space.encode_scalar(reward, -1.0, 1.0)
            reward_bound = reward_hv.bind(self.space.get_basis('REWARD'))
            components.append(reward_bound)
        
        self.operation_counts['bind'] += len(components)
        self.operation_counts['encode'] += 1
        
        return components[0].bundle(components[1:]) if len(components) > 1 else components[0]
    
    def decode_action(self, 
                     query_hv: HyperVector,
                     action_memory: Dict[str, HyperVector],
                     threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """
        Decode action from query hypervector.
        
        Args:
            query_hv: Query hypervector
            action_memory: Dictionary of action name -> hypervector
            threshold: Minimum similarity threshold
            
        Returns:
            (action_name, confidence) or (None, 0.0)
        """
        if not action_memory:
            return None, 0.0
        
        best_action = None
        best_similarity = 0.0
        
        for action_name, action_hv in action_memory.items():
            similarity = query_hv.similarity(action_hv)
            if similarity > best_similarity:
                best_similarity = similarity
                best_action = action_name
        
        self.operation_counts['query'] += len(action_memory)
        
        if best_similarity >= threshold:
            return best_action, best_similarity
        else:
            return None, best_similarity
    
    def temporal_binding(self, 
                        sequence: List[HyperVector],
                        temporal_weight: float = 0.8) -> HyperVector:
        """
        Bind sequence with temporal decay.
        
        Args:
            sequence: List of hypervectors in temporal order
            temporal_weight: Decay factor for older elements
            
        Returns:
            Temporally weighted sequence hypervector
        """
        if not sequence:
            return self.space.random_vector("EMPTY_TEMPORAL")
        
        if len(sequence) == 1:
            return sequence[0]
        
        # Apply temporal weighting
        weighted_sequence = []
        for i, hv in enumerate(sequence):
            # More recent elements get higher weight
            weight = temporal_weight ** (len(sequence) - 1 - i)
            
            # Simulate weighting by noise addition
            if weight < 1.0:
                noise_rate = (1.0 - weight) * 0.1  # Max 10% noise
                weighted_hv = hv.flip_noise(noise_rate)
            else:
                weighted_hv = hv
            
            # Bind with temporal position
            pos_hv = self.space.random_vector(f"TEMPORAL_POS_{i}")
            temporal_hv = weighted_hv.bind(pos_hv)
            weighted_sequence.append(temporal_hv)
        
        self.operation_counts['bind'] += len(sequence)
        
        # Bundle weighted sequence
        return weighted_sequence[0].bundle(weighted_sequence[1:])
    
    def similarity_search(self, 
                         query_hv: HyperVector,
                         memory_vectors: List[HyperVector],
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Fast similarity search in vector memory.
        
        Args:
            query_hv: Query hypervector
            memory_vectors: List of stored hypervectors
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity) pairs sorted by similarity
        """
        if not memory_vectors:
            return []
        
        # Convert to numpy arrays for fast computation
        memory_array = np.vstack([hv.data for hv in memory_vectors])
        
        # Fast parallel search
        indices, similarities = _parallel_similarity_search(
            query_hv.data, memory_array, min(top_k, len(memory_vectors))
        )
        
        self.operation_counts['query'] += len(memory_vectors)
        
        return [(int(idx), float(sim)) for idx, sim in zip(indices, similarities)]
    
    def consensus_binding(self, 
                         vectors: List[HyperVector],
                         weights: Optional[List[float]] = None,
                         threshold: float = 0.6) -> HyperVector:
        """
        Weighted consensus binding with confidence thresholding.
        
        Args:
            vectors: List of hypervectors to combine
            weights: Optional weights for each vector
            threshold: Consensus threshold
            
        Returns:
            Consensus hypervector
        """
        if not vectors:
            return self.space.random_vector("EMPTY_CONSENSUS")
        
        if len(vectors) == 1:
            return vectors[0]
        
        if weights is None:
            weights = [1.0] * len(vectors)
        
        if len(weights) != len(vectors):
            raise ValueError("Weights length must match vectors length")
        
        # Weighted voting for each dimension
        n_vectors = len(vectors)
        consensus_data = np.zeros(self.dimension, dtype=np.float32)
        
        for i, (hv, weight) in enumerate(zip(vectors, weights)):
            consensus_data += hv.data.astype(np.float32) * weight
        
        # Apply threshold
        total_weight = sum(weights)
        threshold_value = total_weight * (2 * threshold - 1)  # Convert to bipolar threshold
        
        result_data = np.where(consensus_data > threshold_value, 1, -1).astype(np.int8)
        
        self.operation_counts['bundle'] += 1
        
        return HyperVector(data=result_data, dimension=self.dimension)
    
    def fault_tolerant_query(self,
                           query_hv: HyperVector,
                           memory: Dict[str, HyperVector],
                           missing_modalities: Optional[List[str]] = None,
                           noise_tolerance: float = 0.1) -> Tuple[Optional[str], float]:
        """
        Query memory with fault tolerance for missing/noisy modalities.
        
        Args:
            query_hv: Query hypervector (possibly degraded)
            memory: Memory dictionary
            missing_modalities: List of missing sensor modalities
            noise_tolerance: Tolerance for noisy measurements
            
        Returns:
            (best_match, confidence)
        """
        if not memory:
            return None, 0.0
        
        # Add noise to simulate sensor degradation
        if missing_modalities or noise_tolerance > 0:
            degraded_query = query_hv.flip_noise(noise_tolerance)
            logger.info("Using degraded query for fault tolerance",
                       missing_modalities=missing_modalities,
                       noise_tolerance=noise_tolerance)
        else:
            degraded_query = query_hv
        
        # Find best match with degraded query
        best_key = None
        best_similarity = 0.0
        
        for key, stored_hv in memory.items():
            similarity = degraded_query.similarity(stored_hv)
            if similarity > best_similarity:
                best_similarity = similarity
                best_key = key
        
        self.operation_counts['query'] += len(memory)
        
        # Adjust confidence based on degradation
        confidence_penalty = len(missing_modalities or []) * 0.05 + noise_tolerance * 0.2
        adjusted_confidence = max(0.0, best_similarity - confidence_penalty)
        
        return best_key, adjusted_confidence
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        total_ops = sum(self.operation_counts.values())
        
        stats = {
            'total_operations': total_ops,
            'operation_breakdown': self.operation_counts.copy(),
            'dimension': self.dimension,
            'memory_usage_mb': self.dimension * total_ops * 1e-6  # Rough estimate
        }
        
        if total_ops > 0:
            stats['operation_percentages'] = {
                op: count / total_ops * 100 
                for op, count in self.operation_counts.items()
            }
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance counters."""
        for key in self.operation_counts:
            self.operation_counts[key] = 0