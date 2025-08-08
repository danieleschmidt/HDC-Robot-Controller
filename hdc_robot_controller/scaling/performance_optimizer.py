"""
Advanced Performance Optimization System for HDC Robotics

Implements adaptive performance optimization including algorithm selection,
resource management, caching strategies, and hardware acceleration.

Performance Features:
1. Adaptive Algorithm Selection: Choose optimal algorithms based on conditions
2. Dynamic Resource Management: Optimize CPU, GPU, and memory usage
3. Intelligent Caching: Multi-level caching with eviction policies
4. Hardware Acceleration: CPU vectorization, GPU compute, and specialized hardware
5. Performance Profiling: Real-time performance monitoring and optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
import time
import threading
import multiprocessing as mp
import concurrent.futures
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import psutil
import logging
import pickle
import hashlib
from pathlib import Path
import gc

# Optional GPU support
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# Optional JIT compilation
try:
    import numba
    from numba import jit, cuda
    JIT_AVAILABLE = True
except ImportError:
    JIT_AVAILABLE = False

from ..core.hypervector import HyperVector
from ..core.operations import HDCOperations


class OptimizationLevel(Enum):
    """Performance optimization levels."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class ResourceType(Enum):
    """System resource types."""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    operation_name: str
    execution_time: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float = 0.0
    cache_hit_rate: float = 0.0
    throughput: float = 0.0
    energy_consumption: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationProfile:
    """Optimization profile for specific scenarios."""
    profile_name: str
    cpu_threads: int
    use_gpu: bool
    use_jit: bool
    cache_size: int
    batch_size: int
    optimization_level: OptimizationLevel
    custom_settings: Dict[str, Any] = field(default_factory=dict)


class IntelligentCache:
    """Multi-level intelligent caching system."""
    
    def __init__(self, max_size_mb: int = 1024, eviction_policy: str = 'lru'):
        """
        Initialize intelligent cache.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            eviction_policy: Cache eviction policy ('lru', 'lfu', 'adaptive')
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        
        # Cache storage
        self.l1_cache = {}  # Hot cache - fast access
        self.l2_cache = {}  # Warm cache - medium access
        self.l3_cache = {}  # Cold cache - slow access
        
        # Cache metadata
        self.access_counts = defaultdict(int)
        self.access_times = defaultdict(float)
        self.cache_sizes = defaultdict(int)
        self.hit_counts = 0
        self.miss_counts = 0
        
        # Cache locks for thread safety
        self.cache_lock = threading.RLock()
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache with intelligent promotion."""
        with self.cache_lock:
            current_time = time.time()
            
            # Check L1 cache first
            if key in self.l1_cache:
                self.access_counts[key] += 1
                self.access_times[key] = current_time
                self.hit_counts += 1
                return self.l1_cache[key]
            
            # Check L2 cache
            if key in self.l2_cache:
                value = self.l2_cache[key]
                
                # Promote to L1 if frequently accessed
                self.access_counts[key] += 1
                self.access_times[key] = current_time
                
                if self.access_counts[key] > 5:  # Promotion threshold
                    del self.l2_cache[key]
                    self.l1_cache[key] = value
                    self._ensure_cache_limits()
                
                self.hit_counts += 1
                return value
            
            # Check L3 cache
            if key in self.l3_cache:
                value = self.l3_cache[key]
                
                # Promote to L2
                self.access_counts[key] += 1
                self.access_times[key] = current_time
                
                del self.l3_cache[key]
                self.l2_cache[key] = value
                self._ensure_cache_limits()
                
                self.hit_counts += 1
                return value
            
            # Cache miss
            self.miss_counts += 1
            return None
    
    def put(self, key: str, value: Any, priority: int = 1):
        """Put item in cache with intelligent placement."""
        with self.cache_lock:
            # Calculate size
            value_size = self._estimate_size(value)
            
            # Remove existing entry if present
            self._remove_key(key)
            
            # Determine cache level based on priority and size
            if priority >= 3 and value_size < self.max_size_bytes * 0.1:  # Hot data, small size
                self.l1_cache[key] = value
            elif priority >= 2 and value_size < self.max_size_bytes * 0.3:  # Warm data
                self.l2_cache[key] = value
            else:  # Cold data or large size
                self.l3_cache[key] = value
            
            # Update metadata
            self.cache_sizes[key] = value_size
            self.access_times[key] = time.time()
            
            # Ensure cache size limits
            self._ensure_cache_limits()
    
    def _remove_key(self, key: str):
        """Remove key from all cache levels."""
        if key in self.l1_cache:
            del self.l1_cache[key]
        if key in self.l2_cache:
            del self.l2_cache[key]
        if key in self.l3_cache:
            del self.l3_cache[key]
        
        # Clean up metadata
        if key in self.cache_sizes:
            del self.cache_sizes[key]
        if key in self.access_counts:
            del self.access_counts[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _ensure_cache_limits(self):
        """Ensure cache doesn't exceed size limits."""
        current_size = sum(self.cache_sizes.values())
        
        if current_size <= self.max_size_bytes:
            return
        
        # Apply eviction policy
        if self.eviction_policy == 'lru':
            self._evict_lru()
        elif self.eviction_policy == 'lfu':
            self._evict_lfu()
        elif self.eviction_policy == 'adaptive':
            self._evict_adaptive()
    
    def _evict_lru(self):
        """Evict least recently used items."""
        # Sort by access time (oldest first)
        all_keys = list(self.access_times.keys())
        sorted_keys = sorted(all_keys, key=lambda k: self.access_times[k])
        
        current_size = sum(self.cache_sizes.values())
        target_size = int(self.max_size_bytes * 0.8)  # Reduce to 80% of limit
        
        for key in sorted_keys:
            if current_size <= target_size:
                break
            
            current_size -= self.cache_sizes.get(key, 0)
            self._remove_key(key)
    
    def _evict_lfu(self):
        """Evict least frequently used items."""
        # Sort by access count (lowest first)
        all_keys = list(self.access_counts.keys())
        sorted_keys = sorted(all_keys, key=lambda k: self.access_counts[k])
        
        current_size = sum(self.cache_sizes.values())
        target_size = int(self.max_size_bytes * 0.8)
        
        for key in sorted_keys:
            if current_size <= target_size:
                break
            
            current_size -= self.cache_sizes.get(key, 0)
            self._remove_key(key)
    
    def _evict_adaptive(self):
        """Adaptive eviction based on access patterns."""
        current_time = time.time()
        
        # Score-based eviction (combination of frequency and recency)
        scored_keys = []
        for key in self.cache_sizes.keys():
            recency_score = 1.0 / (current_time - self.access_times.get(key, 0) + 1)
            frequency_score = self.access_counts.get(key, 0)
            combined_score = recency_score * frequency_score
            scored_keys.append((key, combined_score))
        
        # Sort by score (lowest first)
        scored_keys.sort(key=lambda x: x[1])
        
        current_size = sum(self.cache_sizes.values())
        target_size = int(self.max_size_bytes * 0.8)
        
        for key, score in scored_keys:
            if current_size <= target_size:
                break
            
            current_size -= self.cache_sizes.get(key, 0)
            self._remove_key(key)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimation
            if isinstance(obj, np.ndarray):
                return obj.nbytes
            elif isinstance(obj, (list, tuple)):
                return len(obj) * 8  # Rough estimate
            elif isinstance(obj, dict):
                return len(obj) * 16  # Rough estimate
            else:
                return 64  # Default estimate
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.hit_counts + self.miss_counts
        hit_rate = self.hit_counts / total_requests if total_requests > 0 else 0.0
        
        current_size = sum(self.cache_sizes.values())
        
        return {
            'hit_rate': hit_rate,
            'hit_counts': self.hit_counts,
            'miss_counts': self.miss_counts,
            'current_size_mb': current_size / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': current_size / self.max_size_bytes,
            'l1_entries': len(self.l1_cache),
            'l2_entries': len(self.l2_cache),
            'l3_entries': len(self.l3_cache)
        }


class HardwareAccelerator:
    """Hardware acceleration manager for HDC operations."""
    
    def __init__(self):
        """Initialize hardware accelerator."""
        self.cpu_count = mp.cpu_count()
        self.gpu_available = GPU_AVAILABLE
        self.jit_available = JIT_AVAILABLE
        
        # Performance benchmarks for different hardware
        self.performance_benchmarks = {}
        
        # Hardware utilization tracking
        self.cpu_utilization = deque(maxlen=100)
        self.gpu_utilization = deque(maxlen=100)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU context if available
        if self.gpu_available:
            try:
                cp.cuda.Device().use()
                self.logger.info("GPU acceleration initialized")
            except Exception as e:
                self.gpu_available = False
                self.logger.warning(f"GPU initialization failed: {e}")
    
    def benchmark_hardware(self):
        """Benchmark different hardware configurations."""
        self.logger.info("Running hardware benchmarks...")
        
        # Test data
        dimension = 10000
        num_vectors = 100
        test_vectors = [HyperVector.random(dimension) for _ in range(num_vectors)]
        
        # CPU benchmark
        start_time = time.time()
        for i in range(num_vectors - 1):
            _ = test_vectors[i].bundle(test_vectors[i + 1])
        cpu_time = time.time() - start_time
        self.performance_benchmarks['cpu'] = 1.0 / cpu_time
        
        # CPU with multiprocessing benchmark
        start_time = time.time()
        self._cpu_parallel_bundle(test_vectors)
        cpu_parallel_time = time.time() - start_time
        self.performance_benchmarks['cpu_parallel'] = 1.0 / cpu_parallel_time
        
        # JIT compilation benchmark
        if self.jit_available:
            start_time = time.time()
            self._jit_bundle_operation(test_vectors)
            jit_time = time.time() - start_time
            self.performance_benchmarks['jit'] = 1.0 / jit_time
        
        # GPU benchmark
        if self.gpu_available:
            start_time = time.time()
            self._gpu_bundle_operation(test_vectors)
            gpu_time = time.time() - start_time
            self.performance_benchmarks['gpu'] = 1.0 / gpu_time
        
        self.logger.info(f"Benchmark results: {self.performance_benchmarks}")
    
    def choose_optimal_hardware(self, operation_type: str, data_size: int) -> str:
        """Choose optimal hardware configuration for operation."""
        if not self.performance_benchmarks:
            self.benchmark_hardware()
        
        # Consider current system load
        cpu_usage = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # Decision logic based on operation characteristics
        if operation_type in ['bundle', 'bind'] and data_size > 1000:
            # Large operations benefit from parallelization
            if self.gpu_available and cpu_usage > 80:
                return 'gpu'
            elif self.jit_available and memory_usage < 70:
                return 'jit'
            elif cpu_usage < 50:
                return 'cpu_parallel'
            else:
                return 'cpu'
        
        elif operation_type == 'similarity' and data_size > 500:
            # Similarity computations are often CPU-intensive
            if self.jit_available:
                return 'jit'
            else:
                return 'cpu_parallel'
        
        else:
            # Small operations or general purpose
            return 'cpu'
    
    def accelerated_bundle(self, vectors: List[HyperVector], 
                          hardware: Optional[str] = None) -> HyperVector:
        """Perform hardware-accelerated bundling."""
        if not vectors:
            raise ValueError("Empty vector list")
        
        if hardware is None:
            hardware = self.choose_optimal_hardware('bundle', len(vectors))
        
        if hardware == 'gpu' and self.gpu_available:
            return self._gpu_bundle_operation(vectors)
        elif hardware == 'jit' and self.jit_available:
            return self._jit_bundle_operation(vectors)
        elif hardware == 'cpu_parallel':
            return self._cpu_parallel_bundle(vectors)
        else:
            return self._cpu_bundle_operation(vectors)
    
    def _cpu_bundle_operation(self, vectors: List[HyperVector]) -> HyperVector:
        """CPU-based bundling operation."""
        return HyperVector.bundle_vectors(vectors)
    
    def _cpu_parallel_bundle(self, vectors: List[HyperVector]) -> HyperVector:
        """CPU parallel bundling using multiprocessing."""
        if len(vectors) <= 4:  # Not worth parallelizing small lists
            return self._cpu_bundle_operation(vectors)
        
        # Split vectors into chunks
        num_processes = min(self.cpu_count, len(vectors))
        chunk_size = len(vectors) // num_processes
        
        chunks = []
        for i in range(0, len(vectors), chunk_size):
            chunk = vectors[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
        
        # Process chunks in parallel
        with mp.Pool(processes=num_processes) as pool:
            chunk_results = pool.map(self._bundle_chunk, chunks)
        
        # Combine chunk results
        return HyperVector.bundle_vectors(chunk_results)
    
    def _bundle_chunk(self, chunk: List[HyperVector]) -> HyperVector:
        """Bundle a chunk of vectors."""
        return HyperVector.bundle_vectors(chunk)
    
    def _jit_bundle_operation(self, vectors: List[HyperVector]) -> HyperVector:
        """JIT-compiled bundling operation."""
        if not self.jit_available:
            return self._cpu_bundle_operation(vectors)
        
        # Convert to numpy arrays for JIT compilation
        dimension = vectors[0].dimension
        vector_data = np.array([v.data for v in vectors], dtype=np.int8)
        
        # JIT-compiled bundling
        result_data = self._jit_bundle_kernel(vector_data)
        
        return HyperVector(dimension, result_data)
    
    @staticmethod
    @jit(nopython=True)
    def _jit_bundle_kernel(vector_data: np.ndarray) -> np.ndarray:
        """JIT-compiled bundling kernel."""
        num_vectors, dimension = vector_data.shape
        result = np.zeros(dimension, dtype=np.int32)
        
        # Sum all vectors
        for i in range(num_vectors):
            for j in range(dimension):
                result[j] += vector_data[i, j]
        
        # Apply majority rule
        output = np.zeros(dimension, dtype=np.int8)
        for j in range(dimension):
            output[j] = 1 if result[j] > 0 else -1
        
        return output
    
    def _gpu_bundle_operation(self, vectors: List[HyperVector]) -> HyperVector:
        """GPU-accelerated bundling operation."""
        if not self.gpu_available:
            return self._cpu_bundle_operation(vectors)
        
        try:
            # Convert to GPU arrays
            dimension = vectors[0].dimension
            vector_data = cp.array([v.data for v in vectors], dtype=cp.int8)
            
            # GPU bundling
            result_data = self._gpu_bundle_kernel(vector_data)
            
            # Convert back to CPU
            result_cpu = cp.asnumpy(result_data)
            
            return HyperVector(dimension, result_cpu)
            
        except Exception as e:
            self.logger.warning(f"GPU operation failed, falling back to CPU: {e}")
            return self._cpu_bundle_operation(vectors)
    
    def _gpu_bundle_kernel(self, vector_data: cp.ndarray) -> cp.ndarray:
        """GPU bundling kernel."""
        # Sum along the first axis (across vectors)
        summed = cp.sum(vector_data, axis=0, dtype=cp.int32)
        
        # Apply majority rule
        result = cp.where(summed > 0, 1, -1).astype(cp.int8)
        
        return result
    
    def get_hardware_status(self) -> Dict[str, Any]:
        """Get current hardware utilization status."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        
        self.cpu_utilization.append(cpu_percent)
        
        status = {
            'cpu_count': self.cpu_count,
            'cpu_usage_current': cpu_percent,
            'cpu_usage_avg': np.mean(list(self.cpu_utilization)),
            'memory_total_gb': memory_info.total / (1024**3),
            'memory_used_gb': memory_info.used / (1024**3),
            'memory_percent': memory_info.percent,
            'gpu_available': self.gpu_available,
            'jit_available': self.jit_available
        }
        
        if self.gpu_available:
            try:
                gpu_info = cp.cuda.Device().attributes
                status['gpu_info'] = {
                    'name': cp.cuda.Device().name.decode('utf-8'),
                    'memory_total_gb': cp.cuda.Device().mem_info[1] / (1024**3),
                    'memory_used_gb': (cp.cuda.Device().mem_info[1] - cp.cuda.Device().mem_info[0]) / (1024**3)
                }
            except Exception as e:
                status['gpu_error'] = str(e)
        
        return status


class AdaptiveAlgorithmSelector:
    """Selects optimal algorithms based on current conditions."""
    
    def __init__(self):
        """Initialize adaptive algorithm selector."""
        self.algorithm_performance = defaultdict(lambda: defaultdict(list))
        self.algorithm_conditions = {}
        
        # Algorithm registry
        self.algorithms = {
            'bundle': {
                'standard': self._standard_bundle,
                'weighted': self._weighted_bundle,
                'iterative': self._iterative_bundle,
                'hierarchical': self._hierarchical_bundle
            },
            'similarity': {
                'cosine': self._cosine_similarity,
                'hamming': self._hamming_similarity,
                'jaccard': self._jaccard_similarity,
                'approximate': self._approximate_similarity
            },
            'search': {
                'linear': self._linear_search,
                'approximate': self._approximate_search,
                'hierarchical': self._hierarchical_search
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    def select_algorithm(self, operation_type: str, context: Dict[str, Any]) -> str:
        """Select optimal algorithm based on context."""
        if operation_type not in self.algorithms:
            raise ValueError(f"Unknown operation type: {operation_type}")
        
        # Extract context features
        data_size = context.get('data_size', 0)
        accuracy_requirement = context.get('accuracy_requirement', 0.9)
        time_budget = context.get('time_budget', float('inf'))
        memory_budget = context.get('memory_budget', float('inf'))
        
        # Get available algorithms
        available_algorithms = list(self.algorithms[operation_type].keys())
        
        # Score algorithms based on performance history and constraints
        algorithm_scores = {}
        
        for algorithm_name in available_algorithms:
            score = self._score_algorithm(
                operation_type, algorithm_name, data_size, 
                accuracy_requirement, time_budget, memory_budget
            )
            algorithm_scores[algorithm_name] = score
        
        # Select highest scoring algorithm
        best_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])[0]
        
        self.logger.debug(f"Selected {best_algorithm} for {operation_type} "
                         f"(scores: {algorithm_scores})")
        
        return best_algorithm
    
    def _score_algorithm(self, operation_type: str, algorithm_name: str,
                        data_size: int, accuracy_req: float, 
                        time_budget: float, memory_budget: float) -> float:
        """Score algorithm based on historical performance and constraints."""
        
        # Base score
        score = 1.0
        
        # Historical performance
        if operation_type in self.algorithm_performance:
            if algorithm_name in self.algorithm_performance[operation_type]:
                performance_data = self.algorithm_performance[operation_type][algorithm_name]
                if performance_data:
                    avg_time = np.mean([p[0] for p in performance_data])
                    avg_accuracy = np.mean([p[1] for p in performance_data])
                    
                    # Time score (lower is better)
                    time_score = min(time_budget / (avg_time + 1e-6), 2.0)
                    
                    # Accuracy score
                    accuracy_score = avg_accuracy / accuracy_req if accuracy_req > 0 else 1.0
                    
                    score = 0.6 * time_score + 0.4 * accuracy_score
        
        # Algorithm-specific adjustments based on data size
        if algorithm_name == 'approximate' and data_size > 10000:
            score *= 1.2  # Approximate methods scale better
        elif algorithm_name == 'hierarchical' and data_size > 5000:
            score *= 1.1  # Hierarchical methods good for large data
        elif algorithm_name == 'standard' and data_size < 1000:
            score *= 1.1  # Standard methods fine for small data
        
        return score
    
    def record_performance(self, operation_type: str, algorithm_name: str,
                          execution_time: float, accuracy: float):
        """Record algorithm performance for future selection."""
        self.algorithm_performance[operation_type][algorithm_name].append(
            (execution_time, accuracy)
        )
        
        # Keep only recent performance data
        if len(self.algorithm_performance[operation_type][algorithm_name]) > 100:
            self.algorithm_performance[operation_type][algorithm_name] = \
                self.algorithm_performance[operation_type][algorithm_name][-50:]
    
    # Algorithm implementations
    def _standard_bundle(self, vectors: List[HyperVector]) -> HyperVector:
        """Standard bundling algorithm."""
        return HyperVector.bundle_vectors(vectors)
    
    def _weighted_bundle(self, vectors: List[HyperVector], 
                        weights: Optional[List[float]] = None) -> HyperVector:
        """Weighted bundling algorithm."""
        if weights is None:
            weights = [1.0] * len(vectors)
        
        if len(vectors) != len(weights):
            raise ValueError("Vector and weight counts must match")
        
        dimension = vectors[0].dimension
        weighted_sum = np.zeros(dimension, dtype=np.float64)
        
        for vector, weight in zip(vectors, weights):
            weighted_sum += vector.data * weight
        
        result_data = np.where(weighted_sum > 0, 1, -1).astype(np.int8)
        return HyperVector(dimension, result_data)
    
    def _iterative_bundle(self, vectors: List[HyperVector]) -> HyperVector:
        """Iterative bundling for memory efficiency."""
        if not vectors:
            raise ValueError("Empty vector list")
        
        result = vectors[0].copy()
        for vector in vectors[1:]:
            result = result.bundle(vector)
        
        return result
    
    def _hierarchical_bundle(self, vectors: List[HyperVector]) -> HyperVector:
        """Hierarchical bundling for better accuracy."""
        if len(vectors) <= 2:
            return HyperVector.bundle_vectors(vectors)
        
        # Group vectors into pairs and bundle
        level_vectors = vectors.copy()
        
        while len(level_vectors) > 1:
            next_level = []
            
            # Bundle pairs
            for i in range(0, len(level_vectors), 2):
                if i + 1 < len(level_vectors):
                    bundled = level_vectors[i].bundle(level_vectors[i + 1])
                    next_level.append(bundled)
                else:
                    next_level.append(level_vectors[i])
            
            level_vectors = next_level
        
        return level_vectors[0]
    
    def _cosine_similarity(self, a: HyperVector, b: HyperVector) -> float:
        """Cosine similarity."""
        return a.similarity(b)
    
    def _hamming_similarity(self, a: HyperVector, b: HyperVector) -> float:
        """Hamming similarity."""
        return 1.0 - a.hamming_distance(b)
    
    def _jaccard_similarity(self, a: HyperVector, b: HyperVector) -> float:
        """Jaccard similarity."""
        return HDCOperations.jaccard_similarity(a, b)
    
    def _approximate_similarity(self, a: HyperVector, b: HyperVector) -> float:
        """Fast approximate similarity."""
        # Sample subset of dimensions for faster computation
        sample_size = min(1000, a.dimension // 4)
        indices = np.random.choice(a.dimension, sample_size, replace=False)
        
        a_sample = a.data[indices]
        b_sample = b.data[indices]
        
        dot_product = np.dot(a_sample, b_sample)
        return float(dot_product) / sample_size
    
    def _linear_search(self, query: HyperVector, 
                      candidates: List[HyperVector]) -> Tuple[HyperVector, float]:
        """Linear search through all candidates."""
        best_similarity = -1.0
        best_match = None
        
        for candidate in candidates:
            similarity = query.similarity(candidate)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = candidate
        
        return best_match, best_similarity
    
    def _approximate_search(self, query: HyperVector, 
                           candidates: List[HyperVector]) -> Tuple[HyperVector, float]:
        """Approximate search using sampling."""
        if len(candidates) <= 100:
            return self._linear_search(query, candidates)
        
        # Sample subset of candidates
        sample_size = min(100, len(candidates) // 2)
        sampled_candidates = np.random.choice(candidates, sample_size, replace=False)
        
        return self._linear_search(query, list(sampled_candidates))
    
    def _hierarchical_search(self, query: HyperVector, 
                            candidates: List[HyperVector]) -> Tuple[HyperVector, float]:
        """Hierarchical search using clustering."""
        # Simple hierarchical search - could be improved with actual clustering
        if len(candidates) <= 50:
            return self._linear_search(query, candidates)
        
        # First pass: coarse search
        step = max(1, len(candidates) // 20)
        coarse_candidates = candidates[::step]
        coarse_match, coarse_sim = self._linear_search(query, coarse_candidates)
        
        # Find index of coarse match
        coarse_idx = candidates.index(coarse_match) if coarse_match in candidates else 0
        
        # Second pass: fine search around coarse match
        search_radius = min(25, len(candidates) // 4)
        start_idx = max(0, coarse_idx - search_radius)
        end_idx = min(len(candidates), coarse_idx + search_radius)
        
        fine_candidates = candidates[start_idx:end_idx]
        return self._linear_search(query, fine_candidates)


class PerformanceOptimizer:
    """Main performance optimization orchestrator."""
    
    def __init__(self, cache_size_mb: int = 512):
        """
        Initialize performance optimizer.
        
        Args:
            cache_size_mb: Cache size in megabytes
        """
        # Initialize subsystems
        self.cache = IntelligentCache(cache_size_mb)
        self.hardware_accelerator = HardwareAccelerator()
        self.algorithm_selector = AdaptiveAlgorithmSelector()
        
        # Performance monitoring
        self.performance_history = deque(maxlen=10000)
        self.optimization_profiles = {}
        
        # Current optimization settings
        self.current_profile = OptimizationProfile(
            profile_name="default",
            cpu_threads=mp.cpu_count(),
            use_gpu=GPU_AVAILABLE,
            use_jit=JIT_AVAILABLE,
            cache_size=cache_size_mb,
            batch_size=100,
            optimization_level=OptimizationLevel.BASIC
        )
        
        # Resource monitoring
        self.resource_monitor = threading.Thread(
            target=self._monitor_resources, daemon=True
        )
        self.monitoring_active = True
        self.resource_monitor.start()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Performance optimizer initialized")
    
    def optimize_operation(self, operation_name: str, operation_func: Callable,
                          context: Dict[str, Any], *args, **kwargs) -> Tuple[Any, PerformanceMetrics]:
        """Execute operation with comprehensive optimization."""
        start_time = time.time()
        start_cpu = psutil.cpu_percent()
        start_memory = psutil.virtual_memory().used
        
        # Generate cache key
        cache_key = self._generate_cache_key(operation_name, args, kwargs)
        
        # Check cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=time.time() - start_time,
                cpu_usage=0.0,  # Cached operation uses minimal CPU
                memory_usage=0.0,
                cache_hit_rate=1.0
            )
            return cached_result, metrics
        
        # Select optimal algorithm if applicable
        if operation_name in ['bundle', 'similarity', 'search']:
            algorithm_name = self.algorithm_selector.select_algorithm(operation_name, context)
            context['selected_algorithm'] = algorithm_name
        
        # Execute operation
        try:
            result = operation_func(*args, **kwargs)
            
            # Cache result if beneficial
            result_size = self._estimate_result_size(result)
            if result_size < self.cache.max_size_bytes * 0.1:  # Cache if <10% of cache size
                priority = self._calculate_cache_priority(operation_name, context)
                self.cache.put(cache_key, result, priority)
            
        except Exception as e:
            self.logger.error(f"Optimized operation failed: {operation_name}: {e}")
            raise
        
        # Calculate metrics
        end_time = time.time()
        end_cpu = psutil.cpu_percent()
        end_memory = psutil.virtual_memory().used
        
        execution_time = end_time - start_time
        cpu_usage = (start_cpu + end_cpu) / 2
        memory_delta = max(0, end_memory - start_memory)
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            execution_time=execution_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_delta / (1024**3),  # GB
            cache_hit_rate=0.0,  # Cache miss
            throughput=1.0 / execution_time if execution_time > 0 else 0.0
        )
        
        # Record performance
        self.performance_history.append(metrics)
        
        # Record algorithm performance if applicable
        if 'selected_algorithm' in context:
            # Estimate accuracy (simplified)
            accuracy = context.get('accuracy', 1.0)
            self.algorithm_selector.record_performance(
                operation_name, context['selected_algorithm'], execution_time, accuracy
            )
        
        return result, metrics
    
    def create_optimization_profile(self, profile_name: str, **settings) -> OptimizationProfile:
        """Create custom optimization profile."""
        profile = OptimizationProfile(
            profile_name=profile_name,
            cpu_threads=settings.get('cpu_threads', mp.cpu_count()),
            use_gpu=settings.get('use_gpu', GPU_AVAILABLE),
            use_jit=settings.get('use_jit', JIT_AVAILABLE),
            cache_size=settings.get('cache_size', 512),
            batch_size=settings.get('batch_size', 100),
            optimization_level=OptimizationLevel(settings.get('optimization_level', 'basic')),
            custom_settings=settings.get('custom_settings', {})
        )
        
        self.optimization_profiles[profile_name] = profile
        self.logger.info(f"Created optimization profile: {profile_name}")
        return profile
    
    def apply_profile(self, profile_name: str):
        """Apply optimization profile."""
        if profile_name not in self.optimization_profiles:
            raise ValueError(f"Unknown profile: {profile_name}")
        
        self.current_profile = self.optimization_profiles[profile_name]
        
        # Adjust cache size if needed
        if self.current_profile.cache_size != self.cache.max_size_bytes // (1024**2):
            new_cache = IntelligentCache(self.current_profile.cache_size)
            # Could transfer hot entries from old cache
            self.cache = new_cache
        
        self.logger.info(f"Applied optimization profile: {profile_name}")
    
    def _monitor_resources(self):
        """Monitor system resources continuously."""
        while self.monitoring_active:
            try:
                # Check if system is under stress
                cpu_usage = psutil.cpu_percent(interval=1.0)
                memory_usage = psutil.virtual_memory().percent
                
                # Adaptive optimization based on resource usage
                if cpu_usage > 90:
                    # High CPU usage - reduce parallelization
                    self.current_profile.cpu_threads = max(1, self.current_profile.cpu_threads // 2)
                elif cpu_usage < 30 and self.current_profile.cpu_threads < mp.cpu_count():
                    # Low CPU usage - can increase parallelization
                    self.current_profile.cpu_threads = min(mp.cpu_count(), 
                                                          self.current_profile.cpu_threads * 2)
                
                if memory_usage > 85:
                    # High memory usage - reduce cache size
                    self.cache.max_size_bytes = int(self.cache.max_size_bytes * 0.8)
                    self.cache._ensure_cache_limits()
                
                time.sleep(5.0)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                time.sleep(10.0)
    
    def _generate_cache_key(self, operation_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for operation."""
        # Create a hash of the operation and arguments
        key_data = f"{operation_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _estimate_result_size(self, result: Any) -> int:
        """Estimate size of operation result."""
        return self.cache._estimate_size(result)
    
    def _calculate_cache_priority(self, operation_name: str, context: Dict[str, Any]) -> int:
        """Calculate cache priority for operation result."""
        # Base priority
        priority = 1
        
        # Expensive operations get higher priority
        if context.get('execution_time', 0) > 1.0:  # >1 second
            priority += 2
        
        # Frequently used operations get higher priority
        operation_count = sum(1 for m in self.performance_history 
                             if m.operation_name == operation_name)
        if operation_count > 10:
            priority += 1
        
        # Large result sizes get lower priority
        result_size = context.get('result_size', 0)
        if result_size > self.cache.max_size_bytes * 0.05:  # >5% of cache
            priority = max(1, priority - 1)
        
        return min(3, priority)  # Max priority 3
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.performance_history:
            return {"status": "no_data"}
        
        # Overall statistics
        recent_metrics = list(self.performance_history)[-1000:]  # Last 1000 operations
        
        avg_execution_time = np.mean([m.execution_time for m in recent_metrics])
        avg_cpu_usage = np.mean([m.cpu_usage for m in recent_metrics])
        avg_memory_usage = np.mean([m.memory_usage for m in recent_metrics])
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        
        # Per-operation statistics
        operation_stats = defaultdict(list)
        for metric in recent_metrics:
            operation_stats[metric.operation_name].append(metric)
        
        operation_summary = {}
        for op_name, metrics in operation_stats.items():
            operation_summary[op_name] = {
                'count': len(metrics),
                'avg_time': np.mean([m.execution_time for m in metrics]),
                'avg_cpu': np.mean([m.cpu_usage for m in metrics]),
                'avg_throughput': np.mean([m.throughput for m in metrics])
            }
        
        return {
            'overall_stats': {
                'avg_execution_time': avg_execution_time,
                'avg_cpu_usage': avg_cpu_usage,
                'avg_memory_usage_gb': avg_memory_usage,
                'avg_throughput': avg_throughput,
                'total_operations': len(self.performance_history)
            },
            'operation_stats': operation_summary,
            'cache_stats': self.cache.get_statistics(),
            'hardware_stats': self.hardware_accelerator.get_hardware_status(),
            'current_profile': {
                'name': self.current_profile.profile_name,
                'cpu_threads': self.current_profile.cpu_threads,
                'use_gpu': self.current_profile.use_gpu,
                'optimization_level': self.current_profile.optimization_level.value
            }
        }
    
    def shutdown(self):
        """Shutdown performance optimizer."""
        self.monitoring_active = False
        if self.resource_monitor.is_alive():
            self.resource_monitor.join(timeout=5.0)
        
        self.logger.info("Performance optimizer shutdown")