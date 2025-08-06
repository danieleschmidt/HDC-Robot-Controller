"""
HDC Operations Module

Provides high-level operations for hyperdimensional computing including
bundling, binding, similarity, and various utility functions.
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from .hypervector import HyperVector


class HDCOperations:
    """High-level HDC operations and utilities."""
    
    @staticmethod
    def majority_bundle(vectors: List[HyperVector]) -> HyperVector:
        """Bundle multiple vectors using majority rule."""
        return HyperVector.bundle_vectors(vectors)
    
    @staticmethod
    def weighted_bundle(vectors: List[HyperVector], weights: List[float]) -> HyperVector:
        """Bundle vectors with weights."""
        if len(vectors) != len(weights):
            raise ValueError("Vector and weight counts must match")
        
        weighted_pairs = [(v, w) for v, w in zip(vectors, weights)]
        return weighted_bundle(weighted_pairs)
    
    @staticmethod
    def elementwise_bind(a: HyperVector, b: HyperVector) -> HyperVector:
        """Element-wise binding (multiplication)."""
        return a.bind(b)
    
    @staticmethod
    def circular_bind(a: HyperVector, b: HyperVector) -> HyperVector:
        """Circular convolution binding."""
        if a.dimension != b.dimension:
            raise ValueError("Dimensions must match")
        
        dimension = a.dimension
        result_data = np.zeros(dimension, dtype=np.int8)
        
        for i in range(dimension):
            sum_val = 0
            for j in range(dimension):
                sum_val += a.data[j] * b.data[(i - j) % dimension]
            result_data[i] = 1 if sum_val > 0 else -1
        
        return HyperVector(dimension, result_data)
    
    @staticmethod
    def left_rotate(v: HyperVector, positions: int) -> HyperVector:
        """Rotate vector left by positions."""
        return v.permute(positions)
    
    @staticmethod
    def right_rotate(v: HyperVector, positions: int) -> HyperVector:
        """Rotate vector right by positions."""
        return v.permute(-positions)
    
    @staticmethod
    def cosine_similarity(a: HyperVector, b: HyperVector) -> float:
        """Calculate cosine similarity."""
        return a.similarity(b)
    
    @staticmethod
    def hamming_similarity(a: HyperVector, b: HyperVector) -> float:
        """Calculate Hamming similarity."""
        return 1.0 - a.hamming_distance(b)
    
    @staticmethod
    def jaccard_similarity(a: HyperVector, b: HyperVector) -> float:
        """Calculate Jaccard similarity."""
        if a.dimension != b.dimension:
            raise ValueError("Dimensions must match")
        
        intersection = np.sum((a.data > 0) & (b.data > 0))
        union = np.sum((a.data > 0) | (b.data > 0))
        
        return float(intersection) / union if union > 0 else 0.0
    
    @staticmethod
    def cleanup(v: HyperVector, threshold: float = 0.0) -> HyperVector:
        """Clean up vector by applying threshold."""
        result = v.copy()
        result.threshold()
        return result
    
    @staticmethod
    def add_noise(v: HyperVector, noise_ratio: float = 0.1, seed: Optional[int] = None) -> HyperVector:
        """Add noise to vector."""
        return v.add_noise(noise_ratio, seed)
    
    @staticmethod
    def flip_bits(v: HyperVector, num_flips: int, seed: Optional[int] = None) -> HyperVector:
        """Flip specified number of random bits."""
        if seed is not None:
            np.random.seed(seed)
        
        result = v.copy()
        flip_indices = np.random.choice(v.dimension, num_flips, replace=False)
        for idx in flip_indices:
            result.data[idx] *= -1
        
        return result
    
    @staticmethod
    def encode_sequence(sequence: List[HyperVector], use_positions: bool = True) -> HyperVector:
        """Encode sequence of vectors."""
        from . import create_sequence
        if use_positions:
            return create_sequence(sequence)
        else:
            return HyperVector.bundle_vectors(sequence)
    
    @staticmethod
    def decode_sequence_element(sequence_hv: HyperVector, candidates: List[HyperVector], 
                               position: int) -> HyperVector:
        """Decode element at position from sequence."""
        # Create position vector
        pos_vec = HyperVector.random(sequence_hv.dimension, seed=position + 1000)
        
        # Unbind position
        unbound = sequence_hv.bind(pos_vec)
        
        # Find best match
        best_similarity = -1.0
        best_match = candidates[0]
        
        for candidate in candidates:
            sim = unbound.similarity(candidate)
            if sim > best_similarity:
                best_similarity = sim
                best_match = candidate
        
        return best_match
    
    @staticmethod
    def set_union(vectors: List[HyperVector]) -> HyperVector:
        """Set union operation."""
        return HyperVector.bundle_vectors(vectors)
    
    @staticmethod
    def set_intersection(vectors: List[HyperVector], threshold: float = 0.5) -> HyperVector:
        """Set intersection operation."""
        if not vectors:
            raise ValueError("Cannot intersect empty vector set")
        
        dimension = vectors[0].dimension
        sums = np.zeros(dimension, dtype=np.int32)
        
        for v in vectors:
            sums += (v.data > 0).astype(np.int32)
        
        min_votes = int(threshold * len(vectors))
        result_data = np.where(sums >= min_votes, 1, -1).astype(np.int8)
        
        return HyperVector(dimension, result_data)
    
    @staticmethod
    def entropy(v: HyperVector) -> float:
        """Calculate entropy of vector."""
        return v.entropy()
    
    @staticmethod
    def get_active_dimensions(v: HyperVector) -> List[int]:
        """Get active dimensions."""
        return v.get_active_dimensions()
    
    @staticmethod
    def sparsity(v: HyperVector) -> float:
        """Calculate sparsity."""
        return v.sparsity()
    
    @staticmethod
    def euclidean_distance(a: HyperVector, b: HyperVector) -> float:
        """Calculate Euclidean distance."""
        if a.dimension != b.dimension:
            raise ValueError("Dimensions must match")
        
        diff = a.data.astype(np.float32) - b.data.astype(np.float32)
        return float(np.sqrt(np.sum(diff * diff)))
    
    @staticmethod
    def manhattan_distance(a: HyperVector, b: HyperVector) -> float:
        """Calculate Manhattan distance."""
        if a.dimension != b.dimension:
            raise ValueError("Dimensions must match")
        
        return float(np.sum(np.abs(a.data - b.data)))
    
    @staticmethod
    def chebyshev_distance(a: HyperVector, b: HyperVector) -> float:
        """Calculate Chebyshev distance."""
        if a.dimension != b.dimension:
            raise ValueError("Dimensions must match")
        
        return float(np.max(np.abs(a.data - b.data)))


def weighted_bundle(weighted_vectors: List[Tuple[HyperVector, float]]) -> HyperVector:
    """Bundle vectors with weights."""
    if not weighted_vectors:
        raise ValueError("Cannot bundle empty weighted vector list")
    
    dimension = weighted_vectors[0][0].dimension
    weighted_sum = np.zeros(dimension, dtype=np.float64)
    
    for vector, weight in weighted_vectors:
        if vector.dimension != dimension:
            raise ValueError("All vectors must have same dimension")
        weighted_sum += vector.data * weight
    
    result_data = np.where(weighted_sum > 0, 1, -1).astype(np.int8)
    return HyperVector(dimension, result_data)


# Utility classes for basis vectors and position encoding
class BasisVectors:
    """Basis vectors for encoding different data types."""
    
    def __init__(self, dimension: int = 10000):
        self.dimension = dimension
        self._integer_cache = {}
        self._category_cache = {}
    
    def encode_integer(self, value: int, min_val: int = -1000, max_val: int = 1000) -> HyperVector:
        """Encode integer value."""
        if not min_val <= value <= max_val:
            raise ValueError(f"Value {value} outside range [{min_val}, {max_val}]")
        
        cache_key = (value, min_val, max_val)
        if cache_key not in self._integer_cache:
            seed = hash(cache_key) % (2**32)
            self._integer_cache[cache_key] = HyperVector.random(self.dimension, seed)
        
        return self._integer_cache[cache_key]
    
    def encode_float(self, value: float, min_val: float = -10.0, max_val: float = 10.0, 
                    precision: int = 100) -> HyperVector:
        """Encode float value with discretization."""
        if not min_val <= value <= max_val:
            raise ValueError(f"Value {value} outside range [{min_val}, {max_val}]")
        
        discrete_val = int((value - min_val) / (max_val - min_val) * precision)
        discrete_val = max(0, min(precision - 1, discrete_val))
        
        return self.encode_integer(discrete_val, 0, precision - 1)
    
    def encode_category(self, category: str) -> HyperVector:
        """Encode categorical value."""
        if category not in self._category_cache:
            seed = hash(category) % (2**32)
            self._category_cache[category] = HyperVector.random(self.dimension, seed)
        
        return self._category_cache[category]
    
    def encode_2d_position(self, x: float, y: float, resolution: float = 0.1) -> HyperVector:
        """Encode 2D position."""
        grid_x = int(x / resolution)
        grid_y = int(y / resolution)
        
        x_vec = self.encode_integer(grid_x, -1000, 1000)
        y_vec = self.encode_integer(grid_y, -1000, 1000)
        
        return x_vec.bind(y_vec.permute(1))
    
    def encode_3d_position(self, x: float, y: float, z: float, resolution: float = 0.1) -> HyperVector:
        """Encode 3D position."""
        grid_x = int(x / resolution)
        grid_y = int(y / resolution)
        grid_z = int(z / resolution)
        
        x_vec = self.encode_integer(grid_x, -1000, 1000)
        y_vec = self.encode_integer(grid_y, -1000, 1000)
        z_vec = self.encode_integer(grid_z, -1000, 1000)
        
        return x_vec.bind(y_vec.permute(1)).bind(z_vec.permute(2))
    
    def encode_angle(self, angle_rad: float) -> HyperVector:
        """Encode angle in radians."""
        angle_deg = angle_rad * 180.0 / np.pi
        angle_deg = angle_deg % 360.0  # Normalize to [0, 360)
        
        discrete_angle = int(angle_deg)
        return self.encode_integer(discrete_angle, 0, 359)


class PositionVectors:
    """Position vectors for sequence encoding."""
    
    def __init__(self, dimension: int = 10000, max_positions: int = 1000):
        self.dimension = dimension
        self.max_positions = max_positions
        self._position_cache = {}
    
    def get_position_vector(self, position: int) -> HyperVector:
        """Get position vector for given position."""
        if position < 0:
            raise ValueError("Position must be non-negative")
        
        if position not in self._position_cache:
            seed = position + 1000
            self._position_cache[position] = HyperVector.random(self.dimension, seed)
        
        return self._position_cache[position]