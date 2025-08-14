"""
GPU Acceleration for HDC Robot Controller.

Provides CUDA-accelerated hyperdimensional computing operations
for high-performance robotics applications.
"""

import numpy as np
import threading
import time
from typing import List, Optional, Tuple, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Try to import GPU libraries
CUPY_AVAILABLE = False
NUMBA_CUDA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None

try:
    from numba import cuda
    import numba
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    cuda = None
    numba = None

from ..core.hypervector import HyperVector
from ..core.logging_system import get_logger


class GPUAccelerator:
    """GPU acceleration for HDC operations using CuPy and Numba CUDA."""
    
    def __init__(self, prefer_cupy: bool = True):
        """Initialize GPU accelerator.
        
        Args:
            prefer_cupy: Whether to prefer CuPy over Numba CUDA when both are available
        """
        self.logger = get_logger()
        self.prefer_cupy = prefer_cupy
        
        # Check GPU availability
        self.gpu_available = False
        self.backend = None
        self.device_count = 0
        
        self._initialize_gpu()
        
        # Performance cache
        self._bundle_cache = {}
        self._similarity_cache = {}
        self._cache_lock = threading.RLock()
        self._cache_max_size = 1000
        
        # Thread pool for async operations
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("GPU Accelerator initialized",
                        gpu_available=self.gpu_available,
                        backend=self.backend,
                        device_count=self.device_count)
    
    def _initialize_gpu(self):
        """Initialize GPU backend."""
        if self.prefer_cupy and CUPY_AVAILABLE:
            try:
                # Check if CUDA is available
                cp.cuda.runtime.getDeviceCount()
                self.gpu_available = True
                self.backend = "cupy"
                self.device_count = cp.cuda.runtime.getDeviceCount()
                
                # Set memory pool to improve performance
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=2**30)  # 1GB limit
                
                self.logger.info(f"CuPy backend initialized with {self.device_count} GPU(s)")
                return
            except:
                self.logger.warning("CuPy available but no CUDA devices found")
        
        if NUMBA_CUDA_AVAILABLE:
            try:
                cuda.detect()
                self.gpu_available = True
                self.backend = "numba_cuda"
                self.device_count = len(cuda.gpus)
                
                self.logger.info(f"Numba CUDA backend initialized with {self.device_count} GPU(s)")
                return
            except:
                self.logger.warning("Numba CUDA available but no CUDA devices found")
        
        self.logger.warning("No GPU acceleration available, falling back to CPU")
        self.gpu_available = False
        self.backend = "cpu"
    
    def is_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        return self.gpu_available
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get GPU device information."""
        info = {
            'gpu_available': self.gpu_available,
            'backend': self.backend,
            'device_count': self.device_count
        }
        
        if self.backend == "cupy" and CUPY_AVAILABLE:
            try:
                device = cp.cuda.Device(0)
                info.update({
                    'device_name': device.attributes['Name'],
                    'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
                    'total_memory': cp.cuda.runtime.memGetInfo()[1] // (1024**3),  # GB
                    'free_memory': cp.cuda.runtime.memGetInfo()[0] // (1024**3)   # GB
                })
            except:
                pass
        
        elif self.backend == "numba_cuda" and NUMBA_CUDA_AVAILABLE:
            try:
                device = cuda.get_current_device()
                info.update({
                    'device_name': device.name,
                    'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
                    'total_memory': device.total_memory // (1024**3)  # GB
                })
            except:
                pass
        
        return info
    
    def accelerated_bundle(self, vectors: List[HyperVector], 
                          use_cache: bool = True) -> HyperVector:
        """GPU-accelerated vector bundling using majority rule.
        
        Args:
            vectors: List of hypervectors to bundle
            use_cache: Whether to use caching for repeated operations
            
        Returns:
            Bundled hypervector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty vector list")
        
        # Check cache first
        if use_cache:
            cache_key = self._get_bundle_cache_key(vectors)
            with self._cache_lock:
                if cache_key in self._bundle_cache:
                    return self._bundle_cache[cache_key]
        
        dimension = vectors[0].dimension
        
        # Validate dimensions
        for v in vectors:
            if v.dimension != dimension:
                raise ValueError("All vectors must have same dimension")
        
        if not self.gpu_available:
            return self._cpu_bundle(vectors)
        
        try:
            if self.backend == "cupy":
                result = self._cupy_bundle(vectors)
            elif self.backend == "numba_cuda":
                result = self._numba_bundle(vectors)
            else:
                result = self._cpu_bundle(vectors)
            
            # Cache result
            if use_cache:
                with self._cache_lock:
                    if len(self._bundle_cache) >= self._cache_max_size:
                        # Remove oldest entry
                        oldest_key = next(iter(self._bundle_cache))
                        del self._bundle_cache[oldest_key]
                    
                    self._bundle_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.warning(f"GPU bundling failed, falling back to CPU: {str(e)}")
            return self._cpu_bundle(vectors)
    
    def accelerated_similarity_batch(self, query_vector: HyperVector,
                                   target_vectors: List[HyperVector],
                                   use_cache: bool = True) -> List[float]:
        """GPU-accelerated batch similarity computation.
        
        Args:
            query_vector: Query hypervector
            target_vectors: List of target vectors for similarity computation
            use_cache: Whether to use caching
            
        Returns:
            List of similarity scores
        """
        if not target_vectors:
            return []
        
        if not self.gpu_available:
            return self._cpu_similarity_batch(query_vector, target_vectors)
        
        try:
            if self.backend == "cupy":
                return self._cupy_similarity_batch(query_vector, target_vectors)
            elif self.backend == "numba_cuda":
                return self._numba_similarity_batch(query_vector, target_vectors)
            else:
                return self._cpu_similarity_batch(query_vector, target_vectors)
                
        except Exception as e:
            self.logger.warning(f"GPU similarity batch failed, falling back to CPU: {str(e)}")
            return self._cpu_similarity_batch(query_vector, target_vectors)
    
    def accelerated_bind_batch(self, vectors_a: List[HyperVector],
                             vectors_b: List[HyperVector]) -> List[HyperVector]:
        """GPU-accelerated batch binding operation.
        
        Args:
            vectors_a: First set of vectors
            vectors_b: Second set of vectors (must be same length as vectors_a)
            
        Returns:
            List of bound vectors
        """
        if len(vectors_a) != len(vectors_b):
            raise ValueError("Vector lists must have same length")
        
        if not vectors_a:
            return []
        
        if not self.gpu_available:
            return self._cpu_bind_batch(vectors_a, vectors_b)
        
        try:
            if self.backend == "cupy":
                return self._cupy_bind_batch(vectors_a, vectors_b)
            elif self.backend == "numba_cuda":
                return self._numba_bind_batch(vectors_a, vectors_b)
            else:
                return self._cpu_bind_batch(vectors_a, vectors_b)
                
        except Exception as e:
            self.logger.warning(f"GPU bind batch failed, falling back to CPU: {str(e)}")
            return self._cpu_bind_batch(vectors_a, vectors_b)
    
    def _cupy_bundle(self, vectors: List[HyperVector]) -> HyperVector:
        """CuPy implementation of vector bundling."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        
        # Convert to GPU arrays
        gpu_vectors = []
        for v in vectors:
            gpu_vectors.append(cp.array(v.data, dtype=cp.int32))
        
        # Stack and sum
        stacked = cp.stack(gpu_vectors, axis=0)
        summed = cp.sum(stacked, axis=0)
        
        # Apply majority rule
        result_data = cp.where(summed > 0, 1, -1).astype(cp.int8)
        
        # Convert back to CPU
        cpu_result = cp.asnumpy(result_data)
        
        return HyperVector(vectors[0].dimension, cpu_result)
    
    def _cupy_similarity_batch(self, query_vector: HyperVector,
                              target_vectors: List[HyperVector]) -> List[float]:
        """CuPy implementation of batch similarity computation."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        
        # Convert query vector to GPU
        query_gpu = cp.array(query_vector.data, dtype=cp.int32)
        
        # Convert target vectors to GPU
        target_gpu = []
        for v in target_vectors:
            target_gpu.append(cp.array(v.data, dtype=cp.int32))
        
        # Stack targets
        targets_stacked = cp.stack(target_gpu, axis=0)
        
        # Batch dot product
        dot_products = cp.dot(targets_stacked, query_gpu)
        
        # Normalize by dimension
        similarities = dot_products.astype(cp.float32) / query_vector.dimension
        
        # Convert to CPU
        return cp.asnumpy(similarities).tolist()
    
    def _cupy_bind_batch(self, vectors_a: List[HyperVector],
                        vectors_b: List[HyperVector]) -> List[HyperVector]:
        """CuPy implementation of batch binding."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        
        # Convert to GPU arrays
        gpu_a = cp.stack([cp.array(v.data, dtype=cp.int8) for v in vectors_a])
        gpu_b = cp.stack([cp.array(v.data, dtype=cp.int8) for v in vectors_b])
        
        # Element-wise multiplication (binding)
        result_gpu = gpu_a * gpu_b
        
        # Convert back to CPU and create HyperVectors
        result_cpu = cp.asnumpy(result_gpu)
        
        results = []
        for i in range(len(vectors_a)):
            results.append(HyperVector(vectors_a[i].dimension, result_cpu[i]))
        
        return results
    
    def _numba_bundle(self, vectors: List[HyperVector]) -> HyperVector:
        """Numba CUDA implementation of vector bundling."""
        if not NUMBA_CUDA_AVAILABLE:
            raise RuntimeError("Numba CUDA not available")
        
        dimension = vectors[0].dimension
        num_vectors = len(vectors)
        
        # Create input array
        input_data = np.zeros((num_vectors, dimension), dtype=np.int32)
        for i, v in enumerate(vectors):
            input_data[i] = v.data.astype(np.int32)
        
        # Allocate output
        output_data = np.zeros(dimension, dtype=np.int8)
        
        # Launch kernel
        threadsperblock = 256
        blockspergrid = (dimension + threadsperblock - 1) // threadsperblock
        
        bundle_kernel[blockspergrid, threadsperblock](
            input_data, output_data, num_vectors, dimension)
        
        return HyperVector(dimension, output_data)
    
    def _numba_similarity_batch(self, query_vector: HyperVector,
                               target_vectors: List[HyperVector]) -> List[float]:
        """Numba CUDA implementation of batch similarity."""
        if not NUMBA_CUDA_AVAILABLE:
            raise RuntimeError("Numba CUDA not available")
        
        dimension = query_vector.dimension
        num_targets = len(target_vectors)
        
        # Create input arrays
        query_data = query_vector.data.astype(np.int32)
        target_data = np.zeros((num_targets, dimension), dtype=np.int32)
        
        for i, v in enumerate(target_vectors):
            target_data[i] = v.data.astype(np.int32)
        
        # Allocate output
        similarities = np.zeros(num_targets, dtype=np.float32)
        
        # Launch kernel
        threadsperblock = 256
        blockspergrid = (num_targets + threadsperblock - 1) // threadsperblock
        
        similarity_kernel[blockspergrid, threadsperblock](
            query_data, target_data, similarities, num_targets, dimension)
        
        return similarities.tolist()
    
    def _numba_bind_batch(self, vectors_a: List[HyperVector],
                         vectors_b: List[HyperVector]) -> List[HyperVector]:
        """Numba CUDA implementation of batch binding."""
        if not NUMBA_CUDA_AVAILABLE:
            raise RuntimeError("Numba CUDA not available")
        
        num_vectors = len(vectors_a)
        dimension = vectors_a[0].dimension
        
        # Create input arrays
        data_a = np.zeros((num_vectors, dimension), dtype=np.int8)
        data_b = np.zeros((num_vectors, dimension), dtype=np.int8)
        
        for i in range(num_vectors):
            data_a[i] = vectors_a[i].data
            data_b[i] = vectors_b[i].data
        
        # Allocate output
        result_data = np.zeros((num_vectors, dimension), dtype=np.int8)
        
        # Launch kernel
        threadsperblock = 256
        blockspergrid = ((num_vectors * dimension) + threadsperblock - 1) // threadsperblock
        
        bind_kernel[blockspergrid, threadsperblock](
            data_a, data_b, result_data, num_vectors, dimension)
        
        # Convert to HyperVectors
        results = []
        for i in range(num_vectors):
            results.append(HyperVector(dimension, result_data[i]))
        
        return results
    
    def _cpu_bundle(self, vectors: List[HyperVector]) -> HyperVector:
        """CPU fallback for vector bundling."""
        return HyperVector.bundle_vectors(vectors)
    
    def _cpu_similarity_batch(self, query_vector: HyperVector,
                             target_vectors: List[HyperVector]) -> List[float]:
        """CPU fallback for batch similarity."""
        return [query_vector.similarity(target) for target in target_vectors]
    
    def _cpu_bind_batch(self, vectors_a: List[HyperVector],
                       vectors_b: List[HyperVector]) -> List[HyperVector]:
        """CPU fallback for batch binding."""
        return [a.bind(b) for a, b in zip(vectors_a, vectors_b)]
    
    def _get_bundle_cache_key(self, vectors: List[HyperVector]) -> str:
        """Generate cache key for bundle operation."""
        # Use hash of vector data for cache key
        combined_hash = 0
        for v in vectors:
            combined_hash ^= hash(v.data.tobytes())
        return str(combined_hash)
    
    def clear_cache(self):
        """Clear performance cache."""
        with self._cache_lock:
            self._bundle_cache.clear()
            self._similarity_cache.clear()
        
        self.logger.info("GPU accelerator cache cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with self._cache_lock:
            return {
                'gpu_available': self.gpu_available,
                'backend': self.backend,
                'device_count': self.device_count,
                'cache_size': len(self._bundle_cache) + len(self._similarity_cache),
                'cache_hit_rate': 0.0  # TODO: Implement hit rate tracking
            }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)


# Numba CUDA kernels (only available if Numba CUDA is installed)
if NUMBA_CUDA_AVAILABLE:
    @cuda.jit
    def bundle_kernel(input_data, output_data, num_vectors, dimension):
        """CUDA kernel for vector bundling."""
        idx = cuda.grid(1)
        
        if idx < dimension:
            total = 0
            for i in range(num_vectors):
                total += input_data[i, idx]
            
            output_data[idx] = 1 if total > 0 else -1
    
    @cuda.jit
    def similarity_kernel(query_data, target_data, similarities, num_targets, dimension):
        """CUDA kernel for batch similarity computation."""
        idx = cuda.grid(1)
        
        if idx < num_targets:
            dot_product = 0
            for i in range(dimension):
                dot_product += query_data[i] * target_data[idx, i]
            
            similarities[idx] = float(dot_product) / dimension
    
    @cuda.jit
    def bind_kernel(data_a, data_b, result_data, num_vectors, dimension):
        """CUDA kernel for batch binding operation."""
        idx = cuda.grid(1)
        
        total_elements = num_vectors * dimension
        if idx < total_elements:
            vector_idx = idx // dimension
            element_idx = idx % dimension
            
            result_data[vector_idx, element_idx] = data_a[vector_idx, element_idx] * data_b[vector_idx, element_idx]


class ParallelProcessor:
    """Multi-core CPU parallel processing for HDC operations."""
    
    def __init__(self, num_workers: Optional[int] = None):
        """Initialize parallel processor.
        
        Args:
            num_workers: Number of worker processes (defaults to CPU count)
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.logger = get_logger()
        
        self.logger.info(f"Parallel processor initialized with {self.num_workers} workers")
    
    def parallel_bundle(self, vectors: List[HyperVector], chunk_size: int = 100) -> HyperVector:
        """Parallel vector bundling using multiple CPU cores.
        
        Args:
            vectors: List of vectors to bundle
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            Bundled hypervector
        """
        if not vectors:
            raise ValueError("Cannot bundle empty vector list")
        
        if len(vectors) <= chunk_size:
            return HyperVector.bundle_vectors(vectors)
        
        # Split into chunks
        chunks = [vectors[i:i + chunk_size] for i in range(0, len(vectors), chunk_size)]
        
        # Process chunks in parallel
        with mp.Pool(self.num_workers) as pool:
            chunk_results = pool.map(self._bundle_chunk, chunks)
        
        # Bundle chunk results
        return HyperVector.bundle_vectors(chunk_results)
    
    def parallel_similarity_search(self, query_vector: HyperVector,
                                 target_vectors: List[HyperVector],
                                 top_k: int = 10,
                                 chunk_size: int = 1000) -> List[Tuple[int, float]]:
        """Parallel similarity search using multiple CPU cores.
        
        Args:
            query_vector: Query vector
            target_vectors: List of target vectors
            top_k: Number of top results to return
            chunk_size: Size of chunks for parallel processing
            
        Returns:
            List of (index, similarity) tuples sorted by similarity
        """
        if not target_vectors:
            return []
        
        # Split into chunks
        chunks = []
        for i in range(0, len(target_vectors), chunk_size):
            chunk = target_vectors[i:i + chunk_size]
            chunks.append((query_vector, chunk, i))  # Include start index
        
        # Process chunks in parallel
        with mp.Pool(self.num_workers) as pool:
            chunk_results = pool.map(self._similarity_chunk, chunks)
        
        # Combine and sort results
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        # Sort by similarity and return top k
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:top_k]
    
    @staticmethod
    def _bundle_chunk(chunk: List[HyperVector]) -> HyperVector:
        """Bundle a chunk of vectors."""
        return HyperVector.bundle_vectors(chunk)
    
    @staticmethod
    def _similarity_chunk(args: Tuple[HyperVector, List[HyperVector], int]) -> List[Tuple[int, float]]:
        """Compute similarities for a chunk of vectors."""
        query_vector, target_chunk, start_index = args
        
        results = []
        for i, target_vector in enumerate(target_chunk):
            similarity = query_vector.similarity(target_vector)
            results.append((start_index + i, similarity))
        
        return results


# Global GPU accelerator instance
_gpu_accelerator = None


def get_gpu_accelerator() -> GPUAccelerator:
    """Get or create global GPU accelerator instance."""
    global _gpu_accelerator
    if _gpu_accelerator is None:
        _gpu_accelerator = GPUAccelerator()
    return _gpu_accelerator