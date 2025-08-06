#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

extern "C" {

/**
 * CUDA kernel for high-performance hypervector bundling
 */
__global__ void bundle_kernel(
    const int8_t* vectors,
    int8_t* result,
    int num_vectors,
    int dimension,
    int num_threads_per_vector
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int vector_id = tid / num_threads_per_vector;
    int elem_id = tid % num_threads_per_vector;
    
    if (vector_id >= num_vectors || elem_id >= dimension) {
        return;
    }
    
    // Each thread processes one element across all vectors
    int sum = 0;
    for (int v = 0; v < num_vectors; v++) {
        sum += vectors[v * dimension + elem_id];
    }
    
    // Apply majority rule
    result[elem_id] = (sum > 0) ? 1 : -1;
}

/**
 * Optimized bundling with shared memory
 */
__global__ void bundle_kernel_shared(
    const int8_t* vectors,
    int8_t* result,
    int num_vectors,
    int dimension
) {
    extern __shared__ int shared_sums[];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    if (tid < dimension) {
        // Initialize shared memory
        shared_sums[local_tid] = 0;
        
        // Sum across all vectors for this dimension
        for (int v = 0; v < num_vectors; v++) {
            shared_sums[local_tid] += vectors[v * dimension + tid];
        }
        
        __syncthreads();
        
        // Apply majority rule and write result
        result[tid] = (shared_sums[local_tid] > 0) ? 1 : -1;
    }
}

/**
 * Multi-vector bundling with reduction
 */
__global__ void bundle_vectors_kernel(
    const int8_t* input_vectors,
    int8_t* output_vector,
    int num_vectors,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        int sum = 0;
        
        // Unroll for better performance with small vector counts
        if (num_vectors <= 8) {
            #pragma unroll
            for (int v = 0; v < num_vectors; v++) {
                sum += input_vectors[v * dimension + idx];
            }
        } else {
            for (int v = 0; v < num_vectors; v++) {
                sum += input_vectors[v * dimension + idx];
            }
        }
        
        // Majority rule thresholding
        output_vector[idx] = (sum > 0) ? 1 : -1;
    }
}

/**
 * Batch bundling for multiple operations
 */
__global__ void batch_bundle_kernel(
    const int8_t* input_vectors,
    int8_t* output_vectors,
    int num_operations,
    int vectors_per_operation,
    int dimension
) {
    int op_id = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (op_id >= num_operations || idx >= dimension) {
        return;
    }
    
    int input_offset = op_id * vectors_per_operation * dimension;
    int output_offset = op_id * dimension;
    
    int sum = 0;
    for (int v = 0; v < vectors_per_operation; v++) {
        sum += input_vectors[input_offset + v * dimension + idx];
    }
    
    output_vectors[output_offset + idx] = (sum > 0) ? 1 : -1;
}

/**
 * Weighted bundling kernel
 */
__global__ void weighted_bundle_kernel(
    const int8_t* vectors,
    const float* weights,
    int8_t* result,
    int num_vectors,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        float weighted_sum = 0.0f;
        
        for (int v = 0; v < num_vectors; v++) {
            weighted_sum += vectors[v * dimension + idx] * weights[v];
        }
        
        result[idx] = (weighted_sum > 0.0f) ? 1 : -1;
    }
}

/**
 * Host function to launch bundle kernel
 */
extern "C" void cuda_bundle_vectors(
    const int8_t* h_vectors,
    int8_t* h_result,
    int num_vectors,
    int dimension
) {
    int8_t *d_vectors, *d_result;
    size_t vector_size = num_vectors * dimension * sizeof(int8_t);
    size_t result_size = dimension * sizeof(int8_t);
    
    // Allocate GPU memory
    cudaMalloc(&d_vectors, vector_size);
    cudaMalloc(&d_result, result_size);
    
    // Copy input data to GPU
    cudaMemcpy(d_vectors, h_vectors, vector_size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    int threads_per_block = 256;
    int blocks = (dimension + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    bundle_vectors_kernel<<<blocks, threads_per_block>>>(
        d_vectors, d_result, num_vectors, dimension
    );
    
    // Check for kernel launch errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        // Handle error
        cudaFree(d_vectors);
        cudaFree(d_result);
        return;
    }
    
    // Copy result back to host
    cudaMemcpy(h_result, d_result, result_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_vectors);
    cudaFree(d_result);
    
    // Synchronize to ensure completion
    cudaDeviceSynchronize();
}

/**
 * Host function for weighted bundling
 */
extern "C" void cuda_weighted_bundle(
    const int8_t* h_vectors,
    const float* h_weights,
    int8_t* h_result,
    int num_vectors,
    int dimension
) {
    int8_t *d_vectors, *d_result;
    float *d_weights;
    
    size_t vector_size = num_vectors * dimension * sizeof(int8_t);
    size_t result_size = dimension * sizeof(int8_t);
    size_t weights_size = num_vectors * sizeof(float);
    
    // Allocate GPU memory
    cudaMalloc(&d_vectors, vector_size);
    cudaMalloc(&d_weights, weights_size);
    cudaMalloc(&d_result, result_size);
    
    // Copy input data to GPU
    cudaMemcpy(d_vectors, h_vectors, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, weights_size, cudaMemcpyHostToDevice);
    
    // Configure and launch kernel
    int threads_per_block = 256;
    int blocks = (dimension + threads_per_block - 1) / threads_per_block;
    
    weighted_bundle_kernel<<<blocks, threads_per_block>>>(
        d_vectors, d_weights, d_result, num_vectors, dimension
    );
    
    // Copy result and cleanup
    cudaMemcpy(h_result, d_result, result_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_vectors);
    cudaFree(d_weights);
    cudaFree(d_result);
    cudaDeviceSynchronize();
}

/**
 * Optimized batch bundling
 */
extern "C" void cuda_batch_bundle(
    const int8_t* h_input_vectors,
    int8_t* h_output_vectors,
    int num_operations,
    int vectors_per_operation,
    int dimension
) {
    int8_t *d_input, *d_output;
    
    size_t input_size = num_operations * vectors_per_operation * dimension * sizeof(int8_t);
    size_t output_size = num_operations * dimension * sizeof(int8_t);
    
    // Allocate GPU memory
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_output, output_size);
    
    // Copy input data
    cudaMemcpy(d_input, h_input_vectors, input_size, cudaMemcpyHostToDevice);
    
    // Configure 2D kernel launch
    dim3 threads_per_block(256);
    dim3 blocks((dimension + threads_per_block.x - 1) / threads_per_block.x, num_operations);
    
    // Launch batch kernel
    batch_bundle_kernel<<<blocks, threads_per_block>>>(
        d_input, d_output, num_operations, vectors_per_operation, dimension
    );
    
    // Copy results and cleanup
    cudaMemcpy(h_output_vectors, d_output, output_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaDeviceSynchronize();
}

/**
 * Memory pool for efficient GPU memory management
 */
struct CudaMemoryPool {
    void* pool_ptr;
    size_t pool_size;
    size_t allocated_size;
    bool initialized;
};

static CudaMemoryPool g_memory_pool = {nullptr, 0, 0, false};

extern "C" void cuda_init_memory_pool(size_t pool_size_mb) {
    if (g_memory_pool.initialized) {
        return;
    }
    
    g_memory_pool.pool_size = pool_size_mb * 1024 * 1024;
    cudaMalloc(&g_memory_pool.pool_ptr, g_memory_pool.pool_size);
    g_memory_pool.allocated_size = 0;
    g_memory_pool.initialized = true;
}

extern "C" void cuda_cleanup_memory_pool() {
    if (g_memory_pool.initialized) {
        cudaFree(g_memory_pool.pool_ptr);
        g_memory_pool.initialized = false;
    }
}

/**
 * Stream-based asynchronous operations
 */
extern "C" void cuda_bundle_vectors_async(
    const int8_t* h_vectors,
    int8_t* h_result,
    int num_vectors,
    int dimension,
    cudaStream_t stream
) {
    int8_t *d_vectors, *d_result;
    size_t vector_size = num_vectors * dimension * sizeof(int8_t);
    size_t result_size = dimension * sizeof(int8_t);
    
    // Use pinned memory for faster transfers
    cudaMallocHost((void**)&d_vectors, vector_size);
    cudaMallocHost((void**)&d_result, result_size);
    
    // Asynchronous memory copy and kernel execution
    cudaMemcpyAsync(d_vectors, h_vectors, vector_size, cudaMemcpyHostToDevice, stream);
    
    int threads_per_block = 256;
    int blocks = (dimension + threads_per_block - 1) / threads_per_block;
    
    bundle_vectors_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_vectors, d_result, num_vectors, dimension
    );
    
    cudaMemcpyAsync(h_result, d_result, result_size, cudaMemcpyDeviceToHost, stream);
    
    // Note: Caller should synchronize the stream
}

} // extern "C"

#endif // WITH_CUDA