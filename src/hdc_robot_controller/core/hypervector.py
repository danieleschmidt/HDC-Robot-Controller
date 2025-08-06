"""
Core hypervector implementation with optimized operations.
"""

import numpy as np
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass
import numba
from numba import jit, prange
import hashlib
import pickle


@jit(nopython=True, parallel=True)
def _bundle_vectors(vectors: np.ndarray) -> np.ndarray:
    """Fast parallel bundling of multiple hypervectors."""
    result = np.zeros(vectors.shape[1], dtype=np.int8)
    
    for i in prange(vectors.shape[1]):
        sum_val = 0
        for j in range(vectors.shape[0]):
            sum_val += vectors[j, i]
        result[i] = 1 if sum_val > 0 else -1
    
    return result


@jit(nopython=True, parallel=True)
def _bind_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fast element-wise binding (XOR-like operation)."""
    result = np.zeros_like(a, dtype=np.int8)
    
    for i in prange(len(a)):
        result[i] = a[i] * b[i]
    
    return result


@jit(nopython=True)
def _hamming_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute normalized Hamming similarity between two hypervectors."""
    matches = 0
    for i in range(len(a)):
        if a[i] == b[i]:
            matches += 1
    return matches / len(a)


class HyperVector:
    """
    High-dimensional binary vector optimized for robotic control applications.
    
    Features:
    - Fast vectorized operations using NumPy and Numba
    - Memory-efficient binary representation
    - Hardware acceleration ready
    - Robust to noise and partial information
    """
    
    def __init__(self, 
                 data: Optional[Union[np.ndarray, List[int]]] = None,
                 dimension: int = 10000,
                 seed: Optional[int] = None):
        """
        Initialize hypervector.
        
        Args:
            data: Vector data (-1 or 1 values)
            dimension: Vector dimensionality (default 10000)
            seed: Random seed for reproducible vectors
        """
        self.dimension = dimension
        
        if data is not None:
            self.data = np.array(data, dtype=np.int8)
            if len(self.data) != dimension:
                raise ValueError(f"Data length {len(self.data)} != dimension {dimension}")
        else:
            # Generate random bipolar vector
            rng = np.random.RandomState(seed)
            self.data = rng.choice([-1, 1], size=dimension).astype(np.int8)
    
    def bind(self, other: 'HyperVector') -> 'HyperVector':
        """Bind with another hypervector (element-wise multiplication)."""
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        
        result_data = _bind_vectors(self.data, other.data)
        return HyperVector(data=result_data, dimension=self.dimension)
    
    def bundle(self, others: List['HyperVector']) -> 'HyperVector':
        """Bundle with multiple hypervectors (majority rule)."""
        all_vectors = [self] + others
        
        # Check dimensions
        for vec in all_vectors:
            if vec.dimension != self.dimension:
                raise ValueError("All vector dimensions must match")
        
        # Stack vectors for efficient processing
        stacked = np.vstack([vec.data for vec in all_vectors])
        result_data = _bundle_vectors(stacked)
        
        return HyperVector(data=result_data, dimension=self.dimension)
    
    def permute(self, shift: int) -> 'HyperVector':
        """Circular permutation of hypervector elements."""
        result_data = np.roll(self.data, shift)
        return HyperVector(data=result_data, dimension=self.dimension)
    
    def similarity(self, other: 'HyperVector') -> float:
        """Compute similarity with another hypervector (0-1 scale)."""
        if self.dimension != other.dimension:
            raise ValueError("Vector dimensions must match")
        
        return _hamming_similarity(self.data, other.data)
    
    def distance(self, other: 'HyperVector') -> float:
        """Compute Hamming distance (complement of similarity)."""
        return 1.0 - self.similarity(other)
    
    def flip_noise(self, noise_rate: float = 0.1) -> 'HyperVector':
        """Add random bit-flip noise to test robustness."""
        result_data = self.data.copy()
        n_flips = int(self.dimension * noise_rate)
        flip_indices = np.random.choice(self.dimension, n_flips, replace=False)
        result_data[flip_indices] *= -1
        
        return HyperVector(data=result_data, dimension=self.dimension)
    
    def __add__(self, other: 'HyperVector') -> 'HyperVector':
        """Bundle operation via + operator."""
        return self.bundle([other])
    
    def __mul__(self, other: 'HyperVector') -> 'HyperVector':
        """Bind operation via * operator."""
        return self.bind(other)
    
    def __eq__(self, other: 'HyperVector') -> bool:
        """Exact equality check."""
        if not isinstance(other, HyperVector):
            return False
        return np.array_equal(self.data, other.data)
    
    def __hash__(self) -> int:
        """Hash for use in sets and dictionaries."""
        return hash(self.data.tobytes())
    
    def __str__(self) -> str:
        """String representation."""
        return f"HyperVector(dim={self.dimension}, hash={hash(self) % 10000:04d})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for transmission."""
        return self.data.tobytes()
    
    @classmethod
    def from_bytes(cls, data: bytes, dimension: int) -> 'HyperVector':
        """Deserialize from bytes."""
        array_data = np.frombuffer(data, dtype=np.int8)
        return cls(data=array_data, dimension=dimension)
    
    def save(self, filepath: str) -> None:
        """Save hypervector to file."""
        with open(filepath, 'wb') as f:
            pickle.dump({'data': self.data, 'dimension': self.dimension}, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'HyperVector':
        """Load hypervector from file."""
        with open(filepath, 'rb') as f:
            saved = pickle.load(f)
        return cls(data=saved['data'], dimension=saved['dimension'])


class HyperVectorSpace:
    """
    Manages a space of hypervectors with shared dimensionality and operations.
    
    Features:
    - Consistent dimension management
    - Basis vector generation
    - Spatial encoding for continuous values
    - Symbol mapping for discrete concepts
    """
    
    def __init__(self, dimension: int = 10000, seed: int = 42):
        """Initialize hypervector space."""
        self.dimension = dimension
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Pre-generated basis vectors for common operations
        self.basis_vectors: Dict[str, HyperVector] = {}
        self._generate_basis_vectors()
    
    def _generate_basis_vectors(self) -> None:
        """Generate basis vectors for fundamental concepts."""
        basis_concepts = [
            'POSITION', 'VELOCITY', 'ACCELERATION',
            'OBJECT', 'ACTION', 'GOAL',
            'TIME', 'SPACE', 'FORCE',
            'SUCCESS', 'FAILURE', 'UNKNOWN',
            'LEFT', 'RIGHT', 'FORWARD', 'BACKWARD', 'UP', 'DOWN'
        ]
        
        for concept in basis_concepts:
            seed_val = int(hashlib.md5(concept.encode()).hexdigest()[:8], 16)
            self.basis_vectors[concept] = HyperVector(dimension=self.dimension, seed=seed_val)
    
    def random_vector(self, name: Optional[str] = None) -> HyperVector:
        """Generate a random hypervector."""
        if name:
            seed_val = int(hashlib.md5(name.encode()).hexdigest()[:8], 16)
            return HyperVector(dimension=self.dimension, seed=seed_val)
        else:
            return HyperVector(dimension=self.dimension, seed=self.rng.randint(0, 2**31))
    
    def encode_scalar(self, value: float, min_val: float = -1.0, max_val: float = 1.0) -> HyperVector:
        """Encode scalar value using distributed representation."""
        # Normalize to [0, 1]
        normalized = (value - min_val) / (max_val - min_val)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Create distributed encoding
        n_active = int(self.dimension * 0.5)  # 50% sparsity
        threshold = normalized
        
        # Create vector with probability-based activation
        probs = np.linspace(0, 1, self.dimension)
        active_mask = probs <= threshold
        
        vector_data = np.where(active_mask, 1, -1).astype(np.int8)
        return HyperVector(data=vector_data, dimension=self.dimension)
    
    def encode_vector(self, values: List[float], 
                     min_vals: Optional[List[float]] = None,
                     max_vals: Optional[List[float]] = None) -> HyperVector:
        """Encode multi-dimensional vector by binding scalar encodings."""
        if min_vals is None:
            min_vals = [-1.0] * len(values)
        if max_vals is None:
            max_vals = [1.0] * len(values)
        
        # Encode each dimension
        encoded_dims = []
        for i, (val, min_val, max_val) in enumerate(zip(values, min_vals, max_vals)):
            scalar_hv = self.encode_scalar(val, min_val, max_val)
            # Bind with dimension identifier
            dim_id = self.random_vector(f"DIM_{i}")
            encoded_dims.append(scalar_hv.bind(dim_id))
        
        # Bundle all dimensions
        if len(encoded_dims) == 1:
            return encoded_dims[0]
        return encoded_dims[0].bundle(encoded_dims[1:])
    
    def encode_sequence(self, vectors: List[HyperVector]) -> HyperVector:
        """Encode temporal sequence using permutation-based binding."""
        if not vectors:
            return self.random_vector("EMPTY_SEQUENCE")
        
        if len(vectors) == 1:
            return vectors[0]
        
        # Bind each vector with its temporal position
        sequence_hv = vectors[0]
        for i, vec in enumerate(vectors[1:], 1):
            # Permute vector based on temporal position
            permuted = vec.permute(i * 100)  # Large shift for distinctiveness
            sequence_hv = sequence_hv.bind(permuted)
        
        return sequence_hv
    
    def get_basis(self, concept: str) -> HyperVector:
        """Get basis vector for a concept."""
        if concept not in self.basis_vectors:
            self.basis_vectors[concept] = self.random_vector(concept)
        return self.basis_vectors[concept]