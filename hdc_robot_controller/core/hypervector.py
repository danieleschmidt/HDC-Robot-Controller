"""
Python wrapper for C++ HyperVector implementation.
Provides high-level interface for hyperdimensional computing operations.
"""

import ctypes
import numpy as np
from typing import List, Optional, Union, Tuple
import os
import sys
from pathlib import Path

# Try to load the C++ library
try:
    # Find the compiled library
    lib_path = None
    possible_paths = [
        Path(__file__).parent.parent.parent / "build" / "libhdc_core.so",
        Path(__file__).parent.parent.parent / "install" / "lib" / "libhdc_core.so",
        "/opt/ros/humble/lib/libhdc_core.so"
    ]
    
    for path in possible_paths:
        if os.path.exists(str(path)):
            lib_path = str(path)
            break
    
    if lib_path:
        hdc_lib = ctypes.CDLL(lib_path)
        CPP_AVAILABLE = True
    else:
        CPP_AVAILABLE = False
        hdc_lib = None
        
except (OSError, ImportError):
    CPP_AVAILABLE = False
    hdc_lib = None


class HyperVector:
    """
    Hyperdimensional vector for HDC operations.
    
    This class provides a Python interface to hyperdimensional computing
    operations, with optional C++ acceleration for performance.
    """
    
    DEFAULT_DIMENSION = 10000
    
    def __init__(self, dimension: int = DEFAULT_DIMENSION, data: Optional[np.ndarray] = None):
        """
        Initialize a hypervector.
        
        Args:
            dimension: The dimension of the hypervector
            data: Optional initial data (bipolar: -1, +1)
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")
            
        self.dimension = dimension
        
        if data is not None:
            if len(data) != dimension:
                raise ValueError("Data length must match dimension")
            self.data = np.array(data, dtype=np.int8)
        else:
            self.data = np.zeros(dimension, dtype=np.int8)
            
        # Validate bipolar representation
        if not np.all(np.isin(self.data, [-1, 0, 1])):
            raise ValueError("HyperVector data must be bipolar (-1, +1) or zero")
    
    def get_dimension(self) -> int:
        """Get the dimension of this hypervector."""
        return self.dimension
    
    @classmethod
    def random(cls, dimension: int = DEFAULT_DIMENSION, seed: Optional[int] = None) -> 'HyperVector':
        """Create a random hypervector."""
        if seed is not None:
            np.random.seed(seed)
        
        data = np.random.choice([-1, 1], size=dimension).astype(np.int8)
        return cls(dimension, data)
    
    @classmethod
    def zero(cls, dimension: int = DEFAULT_DIMENSION) -> 'HyperVector':
        """Create a zero hypervector."""
        return cls(dimension)
    
    @classmethod
    def bundle_vectors(cls, vectors: List['HyperVector']) -> 'HyperVector':
        """Bundle multiple vectors using majority rule."""
        if not vectors:
            raise ValueError("Cannot bundle empty vector list")
        
        # Validate dimensions
        dimension = vectors[0].dimension
        for v in vectors:
            if v.dimension != dimension:
                raise ValueError("All vectors must have same dimension")
        
        # Sum all vectors
        total = np.zeros(dimension, dtype=np.int32)
        for v in vectors:
            total += v.data
        
        # Apply majority rule
        result_data = np.where(total > 0, 1, -1).astype(np.int8)
        return cls(dimension, result_data)
    
    def bundle(self, other: 'HyperVector') -> 'HyperVector':
        """Bundle this vector with another."""
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for bundling")
        
        sum_data = self.data.astype(np.int32) + other.data.astype(np.int32)
        result_data = np.where(sum_data > 0, 1, -1).astype(np.int8)
        return HyperVector(self.dimension, result_data)
    
    def bind(self, other: 'HyperVector') -> 'HyperVector':
        """Bind this vector with another (element-wise multiplication)."""
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for binding")
        
        result_data = (self.data * other.data).astype(np.int8)
        return HyperVector(self.dimension, result_data)
    
    def permute(self, shift: int = 1) -> 'HyperVector':
        """Permute (rotate) the vector elements."""
        result_data = np.roll(self.data, shift)
        return HyperVector(self.dimension, result_data)
    
    def invert(self) -> 'HyperVector':
        """Invert the vector (flip all bits)."""
        result_data = (-self.data).astype(np.int8)
        return HyperVector(self.dimension, result_data)
    
    def similarity(self, other: 'HyperVector') -> float:
        """Calculate cosine similarity with another vector."""
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for similarity")
        
        # Use int32 to prevent overflow with large dimensions
        dot_product = np.dot(self.data.astype(np.int32), other.data.astype(np.int32))
        return float(dot_product) / self.dimension
    
    def hamming_distance(self, other: 'HyperVector') -> float:
        """Calculate normalized Hamming distance."""
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for Hamming distance")
        
        differences = np.sum(self.data != other.data)
        return float(differences) / self.dimension
    
    def threshold(self) -> None:
        """Apply threshold to make vector bipolar."""
        self.data = np.where(self.data > 0, 1, -1).astype(np.int8)
    
    def normalize(self) -> None:
        """Normalize vector (same as threshold for bipolar vectors)."""
        self.threshold()
    
    def is_zero_vector(self) -> bool:
        """Check if this is a zero vector."""
        return np.all(self.data == 0)
    
    def to_bytes(self) -> bytes:
        """Convert to byte representation for storage/transmission."""
        # Pack bits efficiently
        byte_data = np.packbits(self.data > 0).tobytes()
        return byte_data
    
    def from_bytes(self, byte_data: bytes) -> None:
        """Load from byte representation."""
        bits = np.unpackbits(np.frombuffer(byte_data, dtype=np.uint8))
        if len(bits) < self.dimension:
            # Pad with zeros if needed
            bits = np.pad(bits, (0, self.dimension - len(bits)))
        else:
            bits = bits[:self.dimension]
        
        self.data = np.where(bits, 1, -1).astype(np.int8)
    
    def get_active_dimensions(self) -> List[int]:
        """Get indices of active (positive) dimensions."""
        return np.where(self.data > 0)[0].tolist()
    
    def sparsity(self) -> float:
        """Calculate sparsity (fraction of positive elements)."""
        return float(np.sum(self.data > 0)) / self.dimension
    
    def entropy(self) -> float:
        """Calculate binary entropy of the vector."""
        p1 = self.sparsity()
        p0 = 1.0 - p1
        
        if p1 == 0.0 or p0 == 0.0:
            return 0.0
        
        return -(p1 * np.log2(p1) + p0 * np.log2(p0))
    
    def add_noise(self, noise_ratio: float = 0.1, seed: Optional[int] = None) -> 'HyperVector':
        """Add noise by flipping random bits."""
        if not 0.0 <= noise_ratio <= 1.0:
            raise ValueError("Noise ratio must be between 0 and 1")
        
        if seed is not None:
            np.random.seed(seed)
        
        result_data = self.data.copy()
        num_flips = int(self.dimension * noise_ratio)
        flip_indices = np.random.choice(self.dimension, num_flips, replace=False)
        result_data[flip_indices] *= -1
        
        return HyperVector(self.dimension, result_data)
    
    def __add__(self, other: 'HyperVector') -> 'HyperVector':
        """Bundle operator (+)."""
        return self.bundle(other)
    
    def __mul__(self, other: 'HyperVector') -> 'HyperVector':
        """Bind operator (*)."""
        return self.bind(other)
    
    def __xor__(self, other: 'HyperVector') -> 'HyperVector':
        """XOR operation (element-wise)."""
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for XOR")
        
        result_data = np.where(self.data == other.data, 1, -1).astype(np.int8)
        return HyperVector(self.dimension, result_data)
    
    def __eq__(self, other) -> bool:
        """Equality comparison."""
        if not isinstance(other, HyperVector):
            return False
        return self.dimension == other.dimension and np.array_equal(self.data, other.data)
    
    def __ne__(self, other) -> bool:
        """Inequality comparison."""
        return not self.__eq__(other)
    
    def __repr__(self) -> str:
        """String representation."""
        active_count = np.sum(self.data > 0)
        return f"HyperVector(dim={self.dimension}, active={active_count}/{self.dimension})"
    
    def __str__(self) -> str:
        """Human-readable string."""
        if self.dimension <= 20:
            data_str = str(self.data.tolist())
        else:
            preview = self.data[:10].tolist()
            data_str = str(preview)[:-1] + f", ... ({self.dimension-10} more)]"
        
        return f"HyperVector(dimension={self.dimension}, data={data_str})"
    
    def __len__(self) -> int:
        """Length (dimension) of the vector."""
        return self.dimension
    
    def __getitem__(self, index: Union[int, slice]) -> Union[int, np.ndarray]:
        """Get element(s) by index."""
        return self.data[index]
    
    def __setitem__(self, index: Union[int, slice], value: Union[int, np.ndarray]) -> None:
        """Set element(s) by index."""
        self.data[index] = value
    
    def copy(self) -> 'HyperVector':
        """Create a copy of this hypervector."""
        return HyperVector(self.dimension, self.data.copy())
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array."""
        return self.data.copy()
    
    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'HyperVector':
        """Create from numpy array."""
        return cls(len(array), array.astype(np.int8))


def weighted_bundle(vectors_and_weights: List[Tuple['HyperVector', float]]) -> 'HyperVector':
    """Bundle vectors with weights."""
    if not vectors_and_weights:
        raise ValueError("Cannot bundle empty weighted vector list")
    
    dimension = vectors_and_weights[0][0].dimension
    weighted_sum = np.zeros(dimension, dtype=np.float64)
    
    for vector, weight in vectors_and_weights:
        if vector.dimension != dimension:
            raise ValueError("All vectors must have same dimension")
        weighted_sum += vector.data * weight
    
    result_data = np.where(weighted_sum > 0, 1, -1).astype(np.int8)
    return HyperVector(dimension, result_data)


def create_sequence(vectors: List['HyperVector']) -> 'HyperVector':
    """Create sequence hypervector by binding vectors with positions."""
    if not vectors:
        raise ValueError("Cannot create sequence from empty vector list")
    
    dimension = vectors[0].dimension
    result = HyperVector.zero(dimension)
    
    for i, vector in enumerate(vectors):
        # Create position vector
        position_vector = HyperVector.random(dimension, seed=i + 1000)
        
        # Bind element with position and bundle into result
        bound = vector.bind(position_vector)
        result = result.bundle(bound)
    
    return result


def create_ngram(sequence: List['HyperVector'], n: int = 3) -> 'HyperVector':
    """Create n-gram representation of a sequence."""
    if len(sequence) < n:
        raise ValueError("Sequence too short for n-gram")
    
    ngrams = []
    for i in range(len(sequence) - n + 1):
        gram = sequence[i:i + n]
        ngrams.append(create_sequence(gram))
    
    return HyperVector.bundle_vectors(ngrams)


# Utility functions for common operations
def bundle(*vectors: 'HyperVector') -> 'HyperVector':
    """Convenience function to bundle multiple vectors."""
    return HyperVector.bundle_vectors(list(vectors))


def bind(a: 'HyperVector', b: 'HyperVector') -> 'HyperVector':
    """Convenience function to bind two vectors."""
    return a.bind(b)


def similarity(a: 'HyperVector', b: 'HyperVector') -> float:
    """Convenience function to calculate similarity."""
    return a.similarity(b)


def random_hypervector(dimension: int = HyperVector.DEFAULT_DIMENSION, 
                      seed: Optional[int] = None) -> 'HyperVector':
    """Create a random hypervector."""
    return HyperVector.random(dimension, seed)


def zero_hypervector(dimension: int = HyperVector.DEFAULT_DIMENSION) -> 'HyperVector':
    """Create a zero hypervector."""
    return HyperVector.zero(dimension)