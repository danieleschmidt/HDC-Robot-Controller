"""
Performance optimization module for HDC Robot Controller.
Provides automatic optimization, profiling, and scaling capabilities.
"""

import time
import threading
import multiprocessing
import psutil
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import functools
import pickle
import os
import logging
from queue import Queue, Empty

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from numba import jit, cuda, prange
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

from ..core.hypervector import HyperVector
from ..core.error_handling import robust_hdc_operation


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    duration: float
    memory_usage: float
    cpu_usage: float
    throughput: float
    error_rate: float
    timestamp: float


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    use_gpu: bool = CUPY_AVAILABLE
    use_numba: bool = NUMBA_AVAILABLE
    use_multiprocessing: bool = True
    max_workers: int = multiprocessing.cpu_count()
    batch_size: int = 100
    cache_size: int = 10000
    profiling_enabled: bool = True
    auto_optimization: bool = True


class PerformanceProfiler:
    """Profiles performance of HDC operations."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.logger = logging.getLogger(__name__)
    
    def profile_operation(self, operation_name: str):
        """Decorator for profiling operations."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                try:
                    result = func(*args, **kwargs)
                    error_rate = 0.0
                except Exception as e:
                    error_rate = 1.0
                    raise
                finally:
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    
                    duration = end_time - start_time
                    memory_delta = end_memory - start_memory
                    cpu_usage = psutil.Process().cpu_percent()
                    
                    # Calculate throughput (operations per second)
                    throughput = 1.0 / duration if duration > 0 else 0.0
                    
                    metrics = PerformanceMetrics(
                        operation_name=operation_name,
                        duration=duration,
                        memory_usage=memory_delta,
                        cpu_usage=cpu_usage,
                        throughput=throughput,
                        error_rate=error_rate,
                        timestamp=end_time
                    )
                    
                    self.record_metrics(metrics)
                
                return result
            return wrapper
        return decorator
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.metrics_history[metrics.operation_name].append(metrics)
    
    def get_average_metrics(self, operation_name: str, window: int = 100) -> Optional[Dict[str, float]]:
        """Get average metrics for an operation."""
        if operation_name not in self.metrics_history:
            return None
        
        recent_metrics = list(self.metrics_history[operation_name])[-window:]
        if not recent_metrics:
            return None
        
        return {
            "avg_duration": np.mean([m.duration for m in recent_metrics]),
            "avg_memory": np.mean([m.memory_usage for m in recent_metrics]),
            "avg_cpu": np.mean([m.cpu_usage for m in recent_metrics]),
            "avg_throughput": np.mean([m.throughput for m in recent_metrics]),
            "error_rate": np.mean([m.error_rate for m in recent_metrics]),
            "sample_count": len(recent_metrics)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        report = {}
        
        for operation_name in self.metrics_history:
            metrics = self.get_average_metrics(operation_name)
            if metrics:
                report[operation_name] = metrics
        
        return report


class CudaAccelerator:
    """CUDA-based acceleration for HDC operations."""
    
    def __init__(self):
        self.available = CUPY_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        if self.available:
            try:
                # Test CUDA availability
                cp.cuda.Device(0).use()
                self.logger.info("CUDA acceleration available")
            except Exception as e:
                self.available = False
                self.logger.warning(f"CUDA not available: {e}")
        else:
            self.logger.info("CuPy not installed - CUDA acceleration disabled")
    
    def bundle_vectors_gpu(self, vectors: List[HyperVector]) -> HyperVector:
        """Bundle vectors using GPU acceleration."""
        if not self.available or not vectors:
            return self.bundle_vectors_cpu(vectors)
        
        try:
            dimension = vectors[0].dimension
            num_vectors = len(vectors)
            
            # Transfer to GPU
            gpu_vectors = []
            for v in vectors:
                gpu_v = cp.array(v.data, dtype=cp.int8)
                gpu_vectors.append(gpu_v)
            
            # Stack vectors for batch processing
            stacked = cp.stack(gpu_vectors, axis=0)
            
            # Sum across vectors
            summed = cp.sum(stacked, axis=0, dtype=cp.int32)
            
            # Apply majority rule
            result_gpu = cp.where(summed > 0, 1, -1).astype(cp.int8)
            
            # Transfer back to CPU
            result_cpu = cp.asnumpy(result_gpu)
            
            return HyperVector(dimension, result_cpu)
            
        except Exception as e:
            self.logger.warning(f"GPU bundling failed, falling back to CPU: {e}")
            return self.bundle_vectors_cpu(vectors)
    
    def bundle_vectors_cpu(self, vectors: List[HyperVector]) -> HyperVector:
        """CPU fallback for vector bundling."""
        return HyperVector.bundle_vectors(vectors)
    
    def similarity_matrix_gpu(self, vectors: List[HyperVector]) -> np.ndarray:
        """Compute similarity matrix using GPU."""
        if not self.available or not vectors:
            return self.similarity_matrix_cpu(vectors)
        
        try:
            dimension = vectors[0].dimension
            num_vectors = len(vectors)
            
            # Transfer to GPU
            gpu_matrix = cp.zeros((num_vectors, dimension), dtype=cp.int8)
            for i, v in enumerate(vectors):
                gpu_matrix[i] = cp.array(v.data)
            
            # Compute similarity matrix: A * A^T / dimension
            similarity_matrix = cp.dot(gpu_matrix, gpu_matrix.T) / dimension
            
            # Transfer back to CPU
            return cp.asnumpy(similarity_matrix)
            
        except Exception as e:
            self.logger.warning(f"GPU similarity matrix failed, falling back to CPU: {e}")
            return self.similarity_matrix_cpu(vectors)
    
    def similarity_matrix_cpu(self, vectors: List[HyperVector]) -> np.ndarray:
        """CPU fallback for similarity matrix."""
        n = len(vectors)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                sim = vectors[i].similarity(vectors[j])
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        return matrix
    
    def batch_operations_gpu(self, vectors: List[HyperVector], 
                           operation: str, **kwargs) -> List[HyperVector]:
        """Perform batch operations on GPU."""
        if not self.available or not vectors:
            return self.batch_operations_cpu(vectors, operation, **kwargs)
        
        try:
            dimension = vectors[0].dimension
            num_vectors = len(vectors)
            
            # Transfer to GPU
            gpu_matrix = cp.zeros((num_vectors, dimension), dtype=cp.int8)
            for i, v in enumerate(vectors):
                gpu_matrix[i] = cp.array(v.data)
            
            if operation == "invert":
                result_gpu = -gpu_matrix
            elif operation == "permute":
                shift = kwargs.get("shift", 1)
                result_gpu = cp.roll(gpu_matrix, shift, axis=1)
            elif operation == "add_noise":
                noise_ratio = kwargs.get("noise_ratio", 0.1)
                mask = cp.random.random((num_vectors, dimension)) < noise_ratio
                result_gpu = cp.where(mask, -gpu_matrix, gpu_matrix)
            else:
                result_gpu = gpu_matrix
            
            # Ensure bipolar
            result_gpu = cp.where(result_gpu > 0, 1, -1).astype(cp.int8)
            
            # Transfer back to CPU
            result_cpu = cp.asnumpy(result_gpu)
            
            return [HyperVector(dimension, result_cpu[i]) for i in range(num_vectors)]
            
        except Exception as e:
            self.logger.warning(f"GPU batch operations failed, falling back to CPU: {e}")
            return self.batch_operations_cpu(vectors, operation, **kwargs)
    
    def batch_operations_cpu(self, vectors: List[HyperVector], 
                           operation: str, **kwargs) -> List[HyperVector]:
        """CPU fallback for batch operations."""
        results = []
        
        for v in vectors:
            if operation == "invert":
                results.append(v.invert())
            elif operation == "permute":
                shift = kwargs.get("shift", 1)
                results.append(v.permute(shift))
            elif operation == "add_noise":
                noise_ratio = kwargs.get("noise_ratio", 0.1)
                results.append(v.add_noise(noise_ratio))
            else:
                results.append(v)
        
        return results


class NumbaAccelerator:
    """Numba-based JIT compilation for performance."""
    
    def __init__(self):
        self.available = NUMBA_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        if self.available:
            self.logger.info("Numba JIT acceleration available")
            self._compile_kernels()
        else:
            self.logger.info("Numba not available - JIT acceleration disabled")
    
    def _compile_kernels(self):
        """Pre-compile frequently used kernels."""
        if not self.available:
            return
        
        @jit(nopython=True, parallel=True)
        def bundle_vectors_numba(vectors_array, result):
            num_vectors, dimension = vectors_array.shape
            for i in prange(dimension):
                total = 0
                for j in range(num_vectors):
                    total += vectors_array[j, i]
                result[i] = 1 if total > 0 else -1
        
        @jit(nopython=True, parallel=True)
        def bind_vectors_numba(a, b, result):
            for i in prange(len(a)):
                result[i] = a[i] * b[i]
        
        @jit(nopython=True, parallel=True)
        def similarity_numba(a, b):
            total = 0
            for i in prange(len(a)):
                total += a[i] * b[i]
            return total / len(a)
        
        @jit(nopython=True, parallel=True)
        def permute_numba(input_vec, result, shift):
            dimension = len(input_vec)
            for i in prange(dimension):
                src_idx = (i - shift) % dimension
                result[i] = input_vec[src_idx]
        
        # Store compiled functions
        self.bundle_vectors_numba = bundle_vectors_numba
        self.bind_vectors_numba = bind_vectors_numba
        self.similarity_numba = similarity_numba
        self.permute_numba = permute_numba
    
    def bundle_vectors_jit(self, vectors: List[HyperVector]) -> HyperVector:
        """JIT-accelerated vector bundling."""
        if not self.available or not vectors:
            return HyperVector.bundle_vectors(vectors)
        
        try:
            dimension = vectors[0].dimension
            num_vectors = len(vectors)
            
            # Convert to numpy array
            vectors_array = np.zeros((num_vectors, dimension), dtype=np.int8)
            for i, v in enumerate(vectors):
                vectors_array[i] = v.data
            
            result = np.zeros(dimension, dtype=np.int8)
            self.bundle_vectors_numba(vectors_array, result)
            
            return HyperVector(dimension, result)
            
        except Exception as e:
            self.logger.warning(f"Numba bundling failed, falling back: {e}")
            return HyperVector.bundle_vectors(vectors)
    
    def bind_vectors_jit(self, a: HyperVector, b: HyperVector) -> HyperVector:
        """JIT-accelerated vector binding."""
        if not self.available:
            return a.bind(b)
        
        try:
            result = np.zeros(a.dimension, dtype=np.int8)
            self.bind_vectors_numba(a.data, b.data, result)
            return HyperVector(a.dimension, result)
            
        except Exception as e:
            self.logger.warning(f"Numba binding failed, falling back: {e}")
            return a.bind(b)
    
    def similarity_jit(self, a: HyperVector, b: HyperVector) -> float:
        """JIT-accelerated similarity computation."""
        if not self.available:
            return a.similarity(b)
        
        try:
            return float(self.similarity_numba(a.data, b.data))
        except Exception as e:
            self.logger.warning(f"Numba similarity failed, falling back: {e}")
            return a.similarity(b)


class ParallelProcessor:
    """Parallel processing for HDC operations."""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or multiprocessing.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        self.logger = logging.getLogger(__name__)
    
    def parallel_similarities(self, query: HyperVector, 
                            database: List[HyperVector]) -> List[float]:
        """Compute similarities in parallel."""
        def compute_sim(target):
            return query.similarity(target)
        
        if len(database) < 100:  # Use threads for small datasets
            futures = [self.thread_pool.submit(compute_sim, vec) for vec in database]
        else:  # Use processes for large datasets
            futures = [self.process_pool.submit(compute_sim, vec) for vec in database]
        
        return [future.result() for future in futures]
    
    def parallel_operations(self, vectors: List[HyperVector], 
                          operation_func: Callable, 
                          chunk_size: int = 100) -> List[Any]:
        """Apply operation to vectors in parallel chunks."""
        chunks = [vectors[i:i+chunk_size] for i in range(0, len(vectors), chunk_size)]
        
        def process_chunk(chunk):
            return [operation_func(v) for v in chunk]
        
        if len(chunks) < 10:
            futures = [self.thread_pool.submit(process_chunk, chunk) for chunk in chunks]
        else:
            futures = [self.process_pool.submit(process_chunk, chunk) for chunk in chunks]
        
        results = []
        for future in futures:
            results.extend(future.result())
        
        return results
    
    def cleanup(self):
        """Cleanup thread/process pools."""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)


class MemoryOptimizer:
    """Optimizes memory usage for HDC operations."""
    
    def __init__(self, cache_size: int = 10000):
        self.cache_size = cache_size
        self.cache = {}
        self.access_counts = defaultdict(int)
        self.logger = logging.getLogger(__name__)
    
    def cached_operation(self, operation_name: str, cache_key: str):
        """Decorator for caching operation results."""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                full_key = f"{operation_name}_{cache_key}"
                
                if full_key in self.cache:
                    self.access_counts[full_key] += 1
                    return self.cache[full_key]
                
                result = func(*args, **kwargs)
                
                # Cache management
                if len(self.cache) >= self.cache_size:
                    self._evict_least_used()
                
                self.cache[full_key] = result
                self.access_counts[full_key] = 1
                
                return result
            return wrapper
        return decorator
    
    def _evict_least_used(self):
        """Evict least recently used items from cache."""
        if not self.cache:
            return
        
        # Remove 10% of least used items
        sorted_items = sorted(self.access_counts.items(), key=lambda x: x[1])
        num_to_remove = max(1, len(sorted_items) // 10)
        
        for key, _ in sorted_items[:num_to_remove]:
            if key in self.cache:
                del self.cache[key]
            del self.access_counts[key]
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        import sys
        
        cache_memory = sum(sys.getsizeof(v) for v in self.cache.values())
        
        return {
            "cache_size": len(self.cache),
            "cache_memory_mb": cache_memory / (1024 * 1024),
            "cache_hit_rate": self._calculate_hit_rate(),
            "most_accessed": max(self.access_counts.items(), key=lambda x: x[1]) if self.access_counts else None
        }
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if not self.access_counts:
            return 0.0
        
        total_accesses = sum(self.access_counts.values())
        cache_hits = sum(1 for count in self.access_counts.values() if count > 1)
        
        return cache_hits / total_accesses if total_accesses > 0 else 0.0


class AdaptiveOptimizer:
    """Adaptive optimization that learns from performance patterns."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.profiler = PerformanceProfiler()
        self.cuda_accelerator = CudaAccelerator() if config.use_gpu else None
        self.numba_accelerator = NumbaAccelerator() if config.use_numba else None
        self.parallel_processor = ParallelProcessor(config.max_workers) if config.use_multiprocessing else None
        self.memory_optimizer = MemoryOptimizer(config.cache_size)
        
        self.optimization_strategies = {
            "bundle_vectors": self._optimize_bundle_vectors,
            "bind_vectors": self._optimize_bind_vectors,
            "similarity_computation": self._optimize_similarity,
            "batch_operations": self._optimize_batch_operations
        }
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Adaptive optimizer initialized")
    
    def optimize_operation(self, operation_name: str, *args, **kwargs):
        """Automatically optimize operation based on learned patterns."""
        if not self.config.auto_optimization:
            return self._fallback_operation(operation_name, *args, **kwargs)
        
        optimizer_func = self.optimization_strategies.get(operation_name)
        if optimizer_func:
            return optimizer_func(*args, **kwargs)
        else:
            return self._fallback_operation(operation_name, *args, **kwargs)
    
    def _optimize_bundle_vectors(self, vectors: List[HyperVector]) -> HyperVector:
        """Optimize vector bundling based on data size and hardware."""
        num_vectors = len(vectors)
        dimension = vectors[0].dimension if vectors else 0
        
        # Decision tree for optimization strategy
        if num_vectors == 0:
            return HyperVector.zero(dimension)
        elif num_vectors == 1:
            return vectors[0]
        elif num_vectors < 10 and dimension < 1000:
            # Small case - use simple CPU
            return HyperVector.bundle_vectors(vectors)
        elif self.cuda_accelerator and self.cuda_accelerator.available and dimension > 5000:
            # Large dimension - use GPU
            return self.cuda_accelerator.bundle_vectors_gpu(vectors)
        elif self.numba_accelerator and self.numba_accelerator.available:
            # Medium case - use JIT
            return self.numba_accelerator.bundle_vectors_jit(vectors)
        else:
            return HyperVector.bundle_vectors(vectors)
    
    def _optimize_bind_vectors(self, a: HyperVector, b: HyperVector) -> HyperVector:
        """Optimize vector binding."""
        if self.numba_accelerator and self.numba_accelerator.available and a.dimension > 1000:
            return self.numba_accelerator.bind_vectors_jit(a, b)
        else:
            return a.bind(b)
    
    def _optimize_similarity(self, a: HyperVector, b: HyperVector) -> float:
        """Optimize similarity computation."""
        cache_key = f"{id(a)}_{id(b)}"
        
        @self.memory_optimizer.cached_operation("similarity", cache_key)
        def cached_similarity():
            if self.numba_accelerator and self.numba_accelerator.available and a.dimension > 1000:
                return self.numba_accelerator.similarity_jit(a, b)
            else:
                return a.similarity(b)
        
        return cached_similarity()
    
    def _optimize_batch_operations(self, vectors: List[HyperVector], 
                                 operation: str, **kwargs) -> List[HyperVector]:
        """Optimize batch operations."""
        num_vectors = len(vectors)
        dimension = vectors[0].dimension if vectors else 0
        
        if num_vectors == 0:
            return []
        elif self.cuda_accelerator and self.cuda_accelerator.available and num_vectors > 50:
            # Use GPU for large batches
            return self.cuda_accelerator.batch_operations_gpu(vectors, operation, **kwargs)
        elif self.parallel_processor and num_vectors > 20:
            # Use parallel processing for medium batches
            operation_func = self._get_operation_function(operation, **kwargs)
            return self.parallel_processor.parallel_operations(vectors, operation_func)
        else:
            # Use simple sequential processing
            return self._sequential_batch_operations(vectors, operation, **kwargs)
    
    def _get_operation_function(self, operation: str, **kwargs) -> Callable:
        """Get operation function for parallel processing."""
        if operation == "invert":
            return lambda v: v.invert()
        elif operation == "permute":
            shift = kwargs.get("shift", 1)
            return lambda v: v.permute(shift)
        elif operation == "add_noise":
            noise_ratio = kwargs.get("noise_ratio", 0.1)
            return lambda v: v.add_noise(noise_ratio)
        else:
            return lambda v: v
    
    def _sequential_batch_operations(self, vectors: List[HyperVector], 
                                   operation: str, **kwargs) -> List[HyperVector]:
        """Sequential batch operations."""
        operation_func = self._get_operation_function(operation, **kwargs)
        return [operation_func(v) for v in vectors]
    
    def _fallback_operation(self, operation_name: str, *args, **kwargs):
        """Fallback to standard operations."""
        self.logger.warning(f"No optimization available for {operation_name}")
        return None
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        report = {
            "performance_metrics": self.profiler.get_performance_report(),
            "memory_stats": self.memory_optimizer.get_memory_stats(),
            "hardware_availability": {
                "cuda_available": self.cuda_accelerator.available if self.cuda_accelerator else False,
                "numba_available": self.numba_accelerator.available if self.numba_accelerator else False,
                "parallel_workers": self.parallel_processor.max_workers if self.parallel_processor else 0
            },
            "system_resources": {
                "cpu_count": multiprocessing.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent
            }
        }
        
        return report
    
    def cleanup(self):
        """Cleanup resources."""
        if self.parallel_processor:
            self.parallel_processor.cleanup()


# Global optimizer instance
_global_optimizer = None

def get_global_optimizer(config: Optional[OptimizationConfig] = None) -> AdaptiveOptimizer:
    """Get global optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AdaptiveOptimizer(config or OptimizationConfig())
    return _global_optimizer


def optimized_operation(operation_name: str):
    """Decorator for automatically optimizing operations."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            optimizer = get_global_optimizer()
            
            # Try optimized version first
            if optimizer.config.auto_optimization:
                try:
                    result = optimizer.optimize_operation(operation_name, *args, **kwargs)
                    if result is not None:
                        return result
                except Exception as e:
                    optimizer.logger.warning(f"Optimization failed for {operation_name}: {e}")
            
            # Fallback to original function
            return func(*args, **kwargs)
        
        return wrapper
    return decorator