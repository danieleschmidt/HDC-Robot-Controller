"""
GPU Acceleration Engine for HDC Operations
CUDA-powered high-performance hyperdimensional computing.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import logging

try:
    import cupy as cp
    import cupyx.scipy.sparse as cupyx_sparse
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from numba import cuda, jit, types
    import numba
    NUMBA_CUDA_AVAILABLE = True
except ImportError:
    NUMBA_CUDA_AVAILABLE = False
    cuda = None

from ..core.hypervector import HyperVector

logger = logging.getLogger(__name__)


class GPUMemoryManager:
    """Efficient GPU memory management for HDC operations."""
    
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.memory_pool = None
        self.allocated_arrays = {}
        self.memory_stats = {
            'total_allocated': 0,
            'peak_allocated': 0,
            'allocations': 0,
            'deallocations': 0
        }
        
        if CUPY_AVAILABLE:
            try:
                cp.cuda.Device(device_id).use()
                self.memory_pool = cp.get_default_memory_pool()
                logger.info(f"GPU memory manager initialized on device {device_id}")
            except Exception as e:
                logger.error(f"GPU memory manager initialization failed: {e}")
                
    def allocate_array(self, shape: Tuple[int, ...], dtype=cp.int8, name: str = None) -> cp.ndarray:
        """Allocate GPU array with tracking."""
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for GPU operations")
            
        try:
            array = cp.zeros(shape, dtype=dtype)
            
            if name:
                self.allocated_arrays[name] = array
                
            # Update statistics
            array_size = array.nbytes
            self.memory_stats['total_allocated'] += array_size
            self.memory_stats['peak_allocated'] = max(
                self.memory_stats['peak_allocated'],
                self.memory_stats['total_allocated']
            )
            self.memory_stats['allocations'] += 1
            
            return array
            
        except Exception as e:
            logger.error(f"GPU array allocation failed: {e}")
            raise
            
    def deallocate_array(self, name: str):
        """Deallocate named GPU array."""
        if name in self.allocated_arrays:
            array = self.allocated_arrays[name]
            array_size = array.nbytes
            
            del self.allocated_arrays[name]
            del array
            
            self.memory_stats['total_allocated'] -= array_size
            self.memory_stats['deallocations'] += 1
            
    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        if not CUPY_AVAILABLE:
            return {'error': 'CuPy not available'}
            
        try:
            mempool = cp.get_default_memory_pool()
            device = cp.cuda.Device()
            
            return {
                'device_id': self.device_id,
                'total_memory': device.mem_info[1],  # Total memory
                'free_memory': device.mem_info[0],   # Free memory
                'used_memory': device.mem_info[1] - device.mem_info[0],
                'pool_used_bytes': mempool.used_bytes(),
                'pool_total_bytes': mempool.total_bytes(),
                'allocated_arrays': len(self.allocated_arrays),
                'stats': self.memory_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory info: {e}")
            return {'error': str(e)}
            
    def clear_memory_pool(self):
        """Clear GPU memory pool."""
        if CUPY_AVAILABLE and self.memory_pool:
            self.memory_pool.free_all_blocks()
            self.allocated_arrays.clear()
            self.memory_stats['total_allocated'] = 0


# CUDA kernels for HDC operations
if NUMBA_CUDA_AVAILABLE:
    
    @cuda.jit
    def cuda_bundle_kernel(vectors_in, weights, result_out, n_vectors, dimension):
        """CUDA kernel for weighted bundling of hypervectors."""
        idx = cuda.grid(1)
        
        if idx < dimension:
            weighted_sum = 0.0
            total_weight = 0.0
            
            for v in range(n_vectors):
                weighted_sum += vectors_in[v, idx] * weights[v]
                total_weight += weights[v]
                
            if total_weight > 0:
                result_out[idx] = 1 if weighted_sum > 0 else -1
            else:
                result_out[idx] = 0
                
    @cuda.jit
    def cuda_bind_kernel(vector_a, vector_b, result_out, dimension):
        """CUDA kernel for binding two hypervectors."""
        idx = cuda.grid(1)
        
        if idx < dimension:
            result_out[idx] = vector_a[idx] * vector_b[idx]
            
    @cuda.jit
    def cuda_similarity_kernel(vector_a, vector_b, partial_sums, dimension):
        """CUDA kernel for computing similarity between hypervectors."""
        idx = cuda.grid(1)
        
        if idx < dimension:
            partial_sums[idx] = vector_a[idx] * vector_b[idx]
            
    @cuda.jit
    def cuda_permute_kernel(vector_in, vector_out, shift, dimension):
        """CUDA kernel for permuting hypervector elements."""
        idx = cuda.grid(1)
        
        if idx < dimension:
            new_idx = (idx + shift) % dimension
            vector_out[new_idx] = vector_in[idx]
            
    @cuda.jit
    def cuda_threshold_kernel(vector_in, vector_out, threshold, dimension):
        """CUDA kernel for thresholding hypervector elements."""
        idx = cuda.grid(1)
        
        if idx < dimension:
            if vector_in[idx] > threshold:
                vector_out[idx] = 1
            elif vector_in[idx] < -threshold:
                vector_out[idx] = -1
            else:
                vector_out[idx] = 0


class CUDAHyperVector:
    """GPU-accelerated hypervector implementation using CUDA."""
    
    def __init__(self, dimension: int, data: Optional[cp.ndarray] = None):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available for CUDA operations")
            
        self.dimension = dimension
        
        if data is not None:
            if len(data) != dimension:
                raise ValueError("Data length must match dimension")
            self.data = cp.asarray(data, dtype=cp.int8)
        else:
            self.data = cp.zeros(dimension, dtype=cp.int8)
            
    @classmethod
    def from_hypervector(cls, hv: HyperVector) -> 'CUDAHyperVector':
        """Create CUDA hypervector from regular hypervector."""
        return cls(hv.dimension, cp.asarray(hv.data))
        
    def to_hypervector(self) -> HyperVector:
        """Convert to regular hypervector."""
        cpu_data = cp.asnumpy(self.data)
        return HyperVector(self.dimension, cpu_data)
        
    @classmethod
    def random(cls, dimension: int, seed: Optional[int] = None) -> 'CUDAHyperVector':
        """Create random CUDA hypervector."""
        if seed is not None:
            cp.random.seed(seed)
            
        data = cp.random.choice([-1, 1], size=dimension).astype(cp.int8)
        return cls(dimension, data)
        
    def bundle(self, other: 'CUDAHyperVector') -> 'CUDAHyperVector':
        """Bundle with another CUDA hypervector."""
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for bundling")
            
        result_data = cp.where(
            self.data + other.data > 0, 1, -1
        ).astype(cp.int8)
        
        return CUDAHyperVector(self.dimension, result_data)
        
    def bind(self, other: 'CUDAHyperVector') -> 'CUDAHyperVector':
        """Bind with another CUDA hypervector."""
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for binding")
            
        if NUMBA_CUDA_AVAILABLE:
            # Use CUDA kernel for binding
            result_data = cp.zeros(self.dimension, dtype=cp.int8)
            
            threads_per_block = 256
            blocks_per_grid = (self.dimension + threads_per_block - 1) // threads_per_block
            
            cuda_bind_kernel[blocks_per_grid, threads_per_block](
                self.data, other.data, result_data, self.dimension
            )
            
            return CUDAHyperVector(self.dimension, result_data)
        else:
            # Fallback to CuPy operation
            result_data = (self.data * other.data).astype(cp.int8)
            return CUDAHyperVector(self.dimension, result_data)
            
    def similarity(self, other: 'CUDAHyperVector') -> float:
        """Compute similarity with another CUDA hypervector."""
        if self.dimension != other.dimension:
            raise ValueError("Dimensions must match for similarity")
            
        if NUMBA_CUDA_AVAILABLE:
            # Use CUDA kernel for similarity computation
            partial_sums = cp.zeros(self.dimension, dtype=cp.int32)
            
            threads_per_block = 256
            blocks_per_grid = (self.dimension + threads_per_block - 1) // threads_per_block
            
            cuda_similarity_kernel[blocks_per_grid, threads_per_block](
                self.data, other.data, partial_sums, self.dimension
            )
            
            dot_product = cp.sum(partial_sums)
            return float(dot_product) / self.dimension
        else:
            # Fallback to CuPy operation
            dot_product = cp.dot(self.data, other.data)
            return float(dot_product) / self.dimension
            
    def permute(self, shift: int = 1) -> 'CUDAHyperVector':
        """Permute hypervector elements."""
        if NUMBA_CUDA_AVAILABLE:
            result_data = cp.zeros(self.dimension, dtype=cp.int8)
            
            threads_per_block = 256
            blocks_per_grid = (self.dimension + threads_per_block - 1) // threads_per_block
            
            cuda_permute_kernel[blocks_per_grid, threads_per_block](
                self.data, result_data, shift, self.dimension
            )
            
            return CUDAHyperVector(self.dimension, result_data)
        else:
            result_data = cp.roll(self.data, shift)
            return CUDAHyperVector(self.dimension, result_data)
            
    def add_noise(self, noise_ratio: float = 0.1) -> 'CUDAHyperVector':
        """Add noise to hypervector."""
        num_flips = int(self.dimension * noise_ratio)
        flip_indices = cp.random.choice(self.dimension, num_flips, replace=False)
        
        result_data = self.data.copy()
        result_data[flip_indices] *= -1
        
        return CUDAHyperVector(self.dimension, result_data)


class GPUAcceleratedHDC:
    """GPU-accelerated HDC operations manager."""
    
    def __init__(self, device_id: int = 0, enable_memory_management: bool = True):
        self.device_id = device_id
        self.enable_memory_management = enable_memory_management
        
        # Check GPU availability
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available - GPU acceleration disabled")
            
        try:
            cp.cuda.Device(device_id).use()
            self.device_info = cp.cuda.Device().attributes
            logger.info(f"GPU acceleration enabled on device {device_id}")
        except Exception as e:
            logger.error(f"GPU initialization failed: {e}")
            raise
            
        # Initialize memory manager
        if enable_memory_management:
            self.memory_manager = GPUMemoryManager(device_id)
        else:
            self.memory_manager = None
            
        # Performance metrics
        self.performance_metrics = {
            'operations_performed': 0,
            'total_gpu_time': 0.0,
            'average_speedup': 0.0,
            'memory_transfers': 0,
            'cache_hits': 0
        }
        
        # Operation cache for frequently used vectors
        self.vector_cache = {}
        self.cache_size_limit = 1000
        
    def bundle_vectors_gpu(self, 
                          vectors: List[Union[HyperVector, CUDAHyperVector]], 
                          weights: Optional[List[float]] = None) -> CUDAHyperVector:
        """GPU-accelerated vector bundling."""
        if not vectors:
            raise ValueError("No vectors provided for bundling")
            
        start_time = time.time()
        
        try:
            dimension = vectors[0].dimension
            
            # Convert to CUDA vectors if needed
            cuda_vectors = []
            for v in vectors:
                if isinstance(v, HyperVector):
                    cuda_v = CUDAHyperVector.from_hypervector(v)
                else:
                    cuda_v = v
                cuda_vectors.append(cuda_v)
                
            # Prepare weights
            if weights is None:
                weights = [1.0] * len(cuda_vectors)
            weights_gpu = cp.asarray(weights, dtype=cp.float32)
            
            if NUMBA_CUDA_AVAILABLE and len(cuda_vectors) > 1:
                # Use CUDA kernel for efficient bundling
                vectors_array = cp.stack([v.data for v in cuda_vectors])
                result_data = cp.zeros(dimension, dtype=cp.int8)
                
                threads_per_block = 256
                blocks_per_grid = (dimension + threads_per_block - 1) // threads_per_block
                
                cuda_bundle_kernel[blocks_per_grid, threads_per_block](
                    vectors_array, weights_gpu, result_data, len(cuda_vectors), dimension
                )
                
                result = CUDAHyperVector(dimension, result_data)
                
            else:
                # Fallback to CuPy operations
                weighted_sum = cp.zeros(dimension, dtype=cp.float32)
                
                for cuda_v, weight in zip(cuda_vectors, weights):
                    weighted_sum += cuda_v.data.astype(cp.float32) * weight
                    
                result_data = cp.where(weighted_sum > 0, 1, -1).astype(cp.int8)
                result = CUDAHyperVector(dimension, result_data)
                
            # Update performance metrics
            gpu_time = time.time() - start_time
            self.performance_metrics['operations_performed'] += 1
            self.performance_metrics['total_gpu_time'] += gpu_time
            
            return result
            
        except Exception as e:
            logger.error(f"GPU bundling failed: {e}")
            raise
            
    def batch_similarity_gpu(self, 
                           query_vectors: List[Union[HyperVector, CUDAHyperVector]],
                           reference_vectors: List[Union[HyperVector, CUDAHyperVector]]) -> cp.ndarray:
        """GPU-accelerated batch similarity computation."""
        start_time = time.time()
        
        try:
            # Convert to CUDA vectors
            query_cuda = []
            for v in query_vectors:
                if isinstance(v, HyperVector):
                    query_cuda.append(CUDAHyperVector.from_hypervector(v))
                else:
                    query_cuda.append(v)
                    
            ref_cuda = []
            for v in reference_vectors:
                if isinstance(v, HyperVector):
                    ref_cuda.append(CUDAHyperVector.from_hypervector(v))
                else:
                    ref_cuda.append(v)
                    
            # Create matrices for batch computation
            query_matrix = cp.stack([v.data for v in query_cuda])  # Shape: (n_queries, dimension)
            ref_matrix = cp.stack([v.data for v in ref_cuda])      # Shape: (n_refs, dimension)
            
            # Compute all similarities using matrix multiplication
            similarities = cp.dot(query_matrix, ref_matrix.T) / query_cuda[0].dimension
            
            # Update performance metrics
            gpu_time = time.time() - start_time
            self.performance_metrics['operations_performed'] += 1
            self.performance_metrics['total_gpu_time'] += gpu_time
            
            return similarities
            
        except Exception as e:
            logger.error(f"GPU batch similarity failed: {e}")
            raise
            
    def memory_query_gpu(self,
                        query_vector: Union[HyperVector, CUDAHyperVector],
                        memory_vectors: List[Union[HyperVector, CUDAHyperVector]],
                        top_k: int = 10,
                        threshold: float = 0.0) -> List[Tuple[int, float]]:
        """GPU-accelerated memory query operation."""
        start_time = time.time()
        
        try:
            # Convert query to CUDA
            if isinstance(query_vector, HyperVector):
                query_cuda = CUDAHyperVector.from_hypervector(query_vector)
            else:
                query_cuda = query_vector
                
            # Batch similarity computation
            similarities = self.batch_similarity_gpu([query_cuda], memory_vectors)
            similarities = similarities[0]  # Get first (and only) row
            
            # Apply threshold
            valid_indices = cp.where(similarities >= threshold)[0]
            valid_similarities = similarities[valid_indices]
            
            # Get top-k results
            if len(valid_similarities) > top_k:
                top_indices = cp.argpartition(valid_similarities, -top_k)[-top_k:]
                top_indices = top_indices[cp.argsort(valid_similarities[top_indices])[::-1]]
            else:
                top_indices = cp.argsort(valid_similarities)[::-1]
                
            # Convert to CPU and create results
            results = []
            for idx in top_indices:
                original_idx = int(valid_indices[idx])
                similarity = float(valid_similarities[idx])
                results.append((original_idx, similarity))
                
            # Update performance metrics
            gpu_time = time.time() - start_time
            self.performance_metrics['operations_performed'] += 1
            self.performance_metrics['total_gpu_time'] += gpu_time
            
            return results
            
        except Exception as e:
            logger.error(f"GPU memory query failed: {e}")
            raise
            
    def cache_vector(self, vector: Union[HyperVector, CUDAHyperVector], cache_key: str):
        """Cache frequently used vector in GPU memory."""
        try:
            if len(self.vector_cache) >= self.cache_size_limit:
                # Remove oldest cached vector
                oldest_key = next(iter(self.vector_cache))
                del self.vector_cache[oldest_key]
                
            if isinstance(vector, HyperVector):
                cached_vector = CUDAHyperVector.from_hypervector(vector)
            else:
                cached_vector = vector
                
            self.vector_cache[cache_key] = cached_vector
            
        except Exception as e:
            logger.error(f"Vector caching failed: {e}")
            
    def get_cached_vector(self, cache_key: str) -> Optional[CUDAHyperVector]:
        """Retrieve cached vector from GPU memory."""
        if cache_key in self.vector_cache:
            self.performance_metrics['cache_hits'] += 1
            return self.vector_cache[cache_key]
        return None
        
    def clear_cache(self):
        """Clear vector cache."""
        self.vector_cache.clear()
        if self.memory_manager:
            self.memory_manager.clear_memory_pool()
            
    def benchmark_operations(self, dimension: int = 10000, num_vectors: int = 1000) -> Dict[str, float]:
        """Benchmark GPU HDC operations."""
        logger.info(f"Benchmarking GPU operations with {num_vectors} vectors of dimension {dimension}")
        
        results = {}
        
        try:
            # Create test vectors
            test_vectors = [CUDAHyperVector.random(dimension) for _ in range(num_vectors)]
            
            # Benchmark bundling
            start_time = time.time()
            bundle_result = self.bundle_vectors_gpu(test_vectors[:100])
            results['bundle_time'] = time.time() - start_time
            
            # Benchmark binding
            start_time = time.time()
            for i in range(100):
                bind_result = test_vectors[i].bind(test_vectors[(i + 1) % num_vectors])
            results['bind_time'] = time.time() - start_time
            
            # Benchmark similarity
            start_time = time.time()
            similarities = self.batch_similarity_gpu(test_vectors[:50], test_vectors[50:100])
            results['similarity_time'] = time.time() - start_time
            
            # Benchmark memory query
            start_time = time.time()
            query_results = self.memory_query_gpu(test_vectors[0], test_vectors[1:101], top_k=10)
            results['memory_query_time'] = time.time() - start_time
            
            # Calculate throughput
            results['bundle_throughput'] = 100 / results['bundle_time']  # operations per second
            results['bind_throughput'] = 100 / results['bind_time']
            results['similarity_throughput'] = 2500 / results['similarity_time']  # 50x50 similarities
            results['memory_query_throughput'] = 1 / results['memory_query_time']
            
            logger.info("GPU benchmark completed successfully")
            
        except Exception as e:
            logger.error(f"GPU benchmark failed: {e}")
            results['error'] = str(e)
            
        return results
        
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        metrics = self.performance_metrics.copy()
        
        # Add GPU information
        if self.memory_manager:
            metrics['memory_info'] = self.memory_manager.get_memory_info()
            
        # Add device information
        metrics['device_info'] = {
            'device_id': self.device_id,
            'device_name': cp.cuda.Device().name.decode('utf-8'),
            'compute_capability': cp.cuda.Device().compute_capability,
            'memory_total': cp.cuda.Device().mem_info[1],
            'multiprocessor_count': self.device_info.get('MultiprocessorCount', 0),
        }
        
        # Calculate derived metrics
        if metrics['operations_performed'] > 0:
            metrics['average_operation_time'] = (
                metrics['total_gpu_time'] / metrics['operations_performed']
            )
            
        if self.performance_metrics['memory_transfers'] > 0:
            cache_hit_rate = (
                self.performance_metrics['cache_hits'] / 
                self.performance_metrics['memory_transfers']
            )
            metrics['cache_hit_rate'] = cache_hit_rate
            
        return metrics
        
    def optimize_memory_usage(self):
        """Optimize GPU memory usage."""
        try:
            # Clear unused cached vectors
            current_time = time.time()
            keys_to_remove = []
            
            for key in self.vector_cache:
                # Remove cached vectors not accessed recently (placeholder logic)
                # In practice, you'd track access times
                if len(keys_to_remove) < len(self.vector_cache) // 4:  # Remove 25%
                    keys_to_remove.append(key)
                    
            for key in keys_to_remove:
                del self.vector_cache[key]
                
            # Clear memory pool
            if self.memory_manager:
                self.memory_manager.clear_memory_pool()
                
            logger.info("GPU memory optimization completed")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")


# Utility functions for GPU acceleration
def create_gpu_hdc_engine(device_id: int = 0) -> Optional[GPUAcceleratedHDC]:
    """Create GPU-accelerated HDC engine if available."""
    try:
        if not CUPY_AVAILABLE:
            logger.warning("CuPy not available - GPU acceleration disabled")
            return None
            
        engine = GPUAcceleratedHDC(device_id)
        logger.info("GPU HDC engine created successfully")
        return engine
        
    except Exception as e:
        logger.error(f"Failed to create GPU HDC engine: {e}")
        return None


def gpu_accelerated_bundle(vectors: List[HyperVector], 
                          weights: Optional[List[float]] = None) -> HyperVector:
    """GPU-accelerated bundle operation with automatic fallback."""
    try:
        engine = create_gpu_hdc_engine()
        if engine is not None:
            cuda_result = engine.bundle_vectors_gpu(vectors, weights)
            return cuda_result.to_hypervector()
        else:
            # Fallback to CPU implementation
            if weights:
                weighted_vectors = list(zip(vectors, weights))
                return weighted_bundle(weighted_vectors)
            else:
                return HyperVector.bundle_vectors(vectors)
                
    except Exception as e:
        logger.error(f"GPU bundle operation failed, falling back to CPU: {e}")
        # Fallback to CPU implementation
        if weights:
            weighted_vectors = list(zip(vectors, weights))
            return weighted_bundle(weighted_vectors)
        else:
            return HyperVector.bundle_vectors(vectors)


def benchmark_gpu_vs_cpu(dimension: int = 10000, num_vectors: int = 1000) -> Dict[str, Any]:
    """Benchmark GPU vs CPU performance for HDC operations."""
    results = {'dimension': dimension, 'num_vectors': num_vectors}
    
    # Create test data
    test_vectors = [HyperVector.random(dimension) for _ in range(num_vectors)]
    
    try:
        # CPU benchmark
        cpu_start = time.time()
        cpu_bundle = HyperVector.bundle_vectors(test_vectors[:100])
        cpu_time = time.time() - cpu_start
        
        results['cpu_bundle_time'] = cpu_time
        results['cpu_bundle_throughput'] = 100 / cpu_time
        
        # GPU benchmark
        gpu_engine = create_gpu_hdc_engine()
        if gpu_engine:
            gpu_start = time.time()
            gpu_bundle = gpu_engine.bundle_vectors_gpu(test_vectors[:100])
            gpu_time = time.time() - gpu_start
            
            results['gpu_bundle_time'] = gpu_time
            results['gpu_bundle_throughput'] = 100 / gpu_time
            results['speedup'] = cpu_time / gpu_time if gpu_time > 0 else 0
            
            # GPU memory and device info
            results['gpu_metrics'] = gpu_engine.get_performance_metrics()
        else:
            results['gpu_error'] = 'GPU engine not available'
            
    except Exception as e:
        results['benchmark_error'] = str(e)
        
    return results