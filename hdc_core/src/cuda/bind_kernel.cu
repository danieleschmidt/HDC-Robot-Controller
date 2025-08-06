#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

extern "C" {

/**
 * CUDA kernel for element-wise binding (multiplication)
 */
__global__ void bind_kernel(
    const int8_t* vector_a,
    const int8_t* vector_b,
    int8_t* result,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        result[idx] = vector_a[idx] * vector_b[idx];
    }
}

/**
 * Vectorized binding using int32 operations
 */
__global__ void bind_kernel_vectorized(
    const int32_t* vector_a,
    const int32_t* vector_b,
    int32_t* result,
    int dimension_div4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension_div4) {
        // Process 4 elements at once
        result[idx] = vector_a[idx] * vector_b[idx];
    }
}

/**
 * Circular convolution binding kernel
 */
__global__ void circular_bind_kernel(
    const int8_t* vector_a,
    const int8_t* vector_b,
    int8_t* result,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        int sum = 0;
        
        // Circular convolution
        for (int j = 0; j < dimension; j++) {
            int shifted_idx = (idx - j + dimension) % dimension;
            sum += vector_a[j] * vector_b[shifted_idx];
        }
        
        result[idx] = (sum > 0) ? 1 : -1;
    }
}

/**
 * Batch binding kernel for multiple operations
 */
__global__ void batch_bind_kernel(
    const int8_t* vectors_a,
    const int8_t* vectors_b,
    int8_t* results,
    int num_operations,
    int dimension
) {
    int op_id = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (op_id >= num_operations || idx >= dimension) {
        return;
    }
    
    int offset = op_id * dimension;
    results[offset + idx] = vectors_a[offset + idx] * vectors_b[offset + idx];
}

/**
 * Optimized binding with shared memory for small dimensions
 */
__global__ void bind_kernel_shared(
    const int8_t* vector_a,
    const int8_t* vector_b,
    int8_t* result,
    int dimension
) {
    extern __shared__ int8_t shared_mem[];
    int8_t* shared_a = shared_mem;
    int8_t* shared_b = shared_mem + blockDim.x;
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    if (tid < dimension) {
        // Load into shared memory
        shared_a[local_tid] = vector_a[tid];
        shared_b[local_tid] = vector_b[tid];
        
        __syncthreads();
        
        // Perform binding
        result[tid] = shared_a[local_tid] * shared_b[local_tid];
    }
}

/**
 * Multi-dimensional binding (tensor product)
 */
__global__ void tensor_bind_kernel(
    const int8_t* vector_a,
    const int8_t* vector_b,
    int8_t* result,
    int dim_a,
    int dim_b
) {
    int idx_a = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_b = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx_a < dim_a && idx_b < dim_b) {
        int result_idx = idx_a * dim_b + idx_b;
        result[result_idx] = vector_a[idx_a] * vector_b[idx_b];
    }
}

/**
 * Host function for element-wise binding
 */
extern "C" void cuda_bind_vectors(
    const int8_t* h_vector_a,
    const int8_t* h_vector_b,
    int8_t* h_result,
    int dimension
) {
    int8_t *d_vector_a, *d_vector_b, *d_result;
    size_t vector_size = dimension * sizeof(int8_t);
    
    // Allocate GPU memory
    cudaMalloc(&d_vector_a, vector_size);
    cudaMalloc(&d_vector_b, vector_size);
    cudaMalloc(&d_result, vector_size);
    
    // Copy input vectors to GPU
    cudaMemcpy(d_vector_a, h_vector_a, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, h_vector_b, vector_size, cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    int threads_per_block = 256;
    int blocks = (dimension + threads_per_block - 1) / threads_per_block;
    
    // Launch kernel
    bind_kernel<<<blocks, threads_per_block>>>(
        d_vector_a, d_vector_b, d_result, dimension
    );
    
    // Check for errors
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        cudaFree(d_vector_a);
        cudaFree(d_vector_b);
        cudaFree(d_result);
        return;
    }
    
    // Copy result back to host
    cudaMemcpy(h_result, d_result, vector_size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
    cudaFree(d_result);
    cudaDeviceSynchronize();
}

/**
 * Host function for circular binding
 */
extern "C" void cuda_circular_bind(
    const int8_t* h_vector_a,
    const int8_t* h_vector_b,
    int8_t* h_result,
    int dimension
) {
    int8_t *d_vector_a, *d_vector_b, *d_result;
    size_t vector_size = dimension * sizeof(int8_t);
    
    cudaMalloc(&d_vector_a, vector_size);
    cudaMalloc(&d_vector_b, vector_size);
    cudaMalloc(&d_result, vector_size);
    
    cudaMemcpy(d_vector_a, h_vector_a, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, h_vector_b, vector_size, cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (dimension + threads_per_block - 1) / threads_per_block;
    
    circular_bind_kernel<<<blocks, threads_per_block>>>(
        d_vector_a, d_vector_b, d_result, dimension
    );
    
    cudaMemcpy(h_result, d_result, vector_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
    cudaFree(d_result);
    cudaDeviceSynchronize();
}

/**
 * Host function for batch binding
 */
extern "C" void cuda_batch_bind(
    const int8_t* h_vectors_a,
    const int8_t* h_vectors_b,
    int8_t* h_results,
    int num_operations,
    int dimension
) {
    int8_t *d_vectors_a, *d_vectors_b, *d_results;
    size_t total_size = num_operations * dimension * sizeof(int8_t);
    
    cudaMalloc(&d_vectors_a, total_size);
    cudaMalloc(&d_vectors_b, total_size);
    cudaMalloc(&d_results, total_size);
    
    cudaMemcpy(d_vectors_a, h_vectors_a, total_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectors_b, h_vectors_b, total_size, cudaMemcpyHostToDevice);
    
    // 2D kernel launch for batch operations
    dim3 threads_per_block(256);
    dim3 blocks((dimension + threads_per_block.x - 1) / threads_per_block.x, num_operations);
    
    batch_bind_kernel<<<blocks, threads_per_block>>>(
        d_vectors_a, d_vectors_b, d_results, num_operations, dimension
    );
    
    cudaMemcpy(h_results, d_results, total_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_vectors_a);
    cudaFree(d_vectors_b);
    cudaFree(d_results);
    cudaDeviceSynchronize();
}

/**
 * Optimized vectorized binding for aligned data
 */
extern "C" void cuda_bind_vectors_optimized(
    const int8_t* h_vector_a,
    const int8_t* h_vector_b,
    int8_t* h_result,
    int dimension
) {
    // Check if dimension is divisible by 4 for vectorization
    bool can_vectorize = (dimension % 4 == 0);
    
    if (can_vectorize) {
        int32_t *d_vector_a, *d_vector_b, *d_result;
        size_t vector_size = dimension * sizeof(int8_t);
        size_t vector_size_int32 = (dimension / 4) * sizeof(int32_t);
        
        cudaMalloc(&d_vector_a, vector_size);
        cudaMalloc(&d_vector_b, vector_size);
        cudaMalloc(&d_result, vector_size);
        
        cudaMemcpy(d_vector_a, h_vector_a, vector_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_vector_b, h_vector_b, vector_size, cudaMemcpyHostToDevice);
        
        int threads_per_block = 256;
        int blocks = ((dimension / 4) + threads_per_block - 1) / threads_per_block;
        
        bind_kernel_vectorized<<<blocks, threads_per_block>>>(
            d_vector_a, d_vector_b, d_result, dimension / 4
        );
        
        cudaMemcpy(h_result, d_result, vector_size, cudaMemcpyDeviceToHost);
        
        cudaFree(d_vector_a);
        cudaFree(d_vector_b);
        cudaFree(d_result);
    } else {
        // Fall back to regular binding
        cuda_bind_vectors(h_vector_a, h_vector_b, h_result, dimension);
    }
    
    cudaDeviceSynchronize();
}

/**
 * Asynchronous binding with streams
 */
extern "C" void cuda_bind_vectors_async(
    const int8_t* h_vector_a,
    const int8_t* h_vector_b,
    int8_t* h_result,
    int dimension,
    cudaStream_t stream
) {
    int8_t *d_vector_a, *d_vector_b, *d_result;
    size_t vector_size = dimension * sizeof(int8_t);
    
    // Allocate pinned memory for faster transfers
    cudaMallocHost((void**)&d_vector_a, vector_size);
    cudaMallocHost((void**)&d_vector_b, vector_size);
    cudaMallocHost((void**)&d_result, vector_size);
    
    // Asynchronous operations
    cudaMemcpyAsync(d_vector_a, h_vector_a, vector_size, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_vector_b, h_vector_b, vector_size, cudaMemcpyHostToDevice, stream);
    
    int threads_per_block = 256;
    int blocks = (dimension + threads_per_block - 1) / threads_per_block;
    
    bind_kernel<<<blocks, threads_per_block, 0, stream>>>(
        d_vector_a, d_vector_b, d_result, dimension
    );
    
    cudaMemcpyAsync(h_result, d_result, vector_size, cudaMemcpyDeviceToHost, stream);
    
    // Note: Caller should synchronize the stream
}

/**
 * Tensor product binding for creating higher-dimensional representations
 */
extern "C" void cuda_tensor_bind(
    const int8_t* h_vector_a,
    const int8_t* h_vector_b,
    int8_t* h_result,
    int dim_a,
    int dim_b
) {
    int8_t *d_vector_a, *d_vector_b, *d_result;
    size_t size_a = dim_a * sizeof(int8_t);
    size_t size_b = dim_b * sizeof(int8_t);
    size_t result_size = dim_a * dim_b * sizeof(int8_t);
    
    cudaMalloc(&d_vector_a, size_a);
    cudaMalloc(&d_vector_b, size_b);
    cudaMalloc(&d_result, result_size);
    
    cudaMemcpy(d_vector_a, h_vector_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, h_vector_b, size_b, cudaMemcpyHostToDevice);
    
    // 2D thread blocks for tensor product
    dim3 threads_per_block(16, 16);
    dim3 blocks((dim_a + threads_per_block.x - 1) / threads_per_block.x,
                (dim_b + threads_per_block.y - 1) / threads_per_block.y);
    
    tensor_bind_kernel<<<blocks, threads_per_block>>>(
        d_vector_a, d_vector_b, d_result, dim_a, dim_b
    );
    
    cudaMemcpy(h_result, d_result, result_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
    cudaFree(d_result);
    cudaDeviceSynchronize();
}

} // extern "C"

#endif // WITH_CUDA