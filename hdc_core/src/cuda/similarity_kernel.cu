#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <float.h>

namespace cg = cooperative_groups;

extern "C" {

/**
 * CUDA kernel for computing similarity (dot product) between two vectors
 */
__global__ void similarity_kernel(
    const int8_t* vector_a,
    const int8_t* vector_b,
    float* result,
    int dimension
) {
    __shared__ float shared_sums[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    shared_sums[local_tid] = 0.0f;
    
    // Each thread computes partial dot product
    if (tid < dimension) {
        shared_sums[local_tid] = static_cast<float>(vector_a[tid] * vector_b[tid]);
    }
    
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            shared_sums[local_tid] += shared_sums[local_tid + stride];
        }
        __syncthreads();
    }
    
    // First thread writes block sum to global memory
    if (local_tid == 0) {
        atomicAdd(result, shared_sums[0]);
    }
}

/**
 * Optimized similarity with warp-level primitives
 */
__global__ void similarity_kernel_warp_optimized(
    const int8_t* vector_a,
    const int8_t* vector_b,
    float* result,
    int dimension
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_sum = 0.0f;
    
    // Each thread processes multiple elements
    for (int i = tid; i < dimension; i += blockDim.x * gridDim.x) {
        local_sum += static_cast<float>(vector_a[i] * vector_b[i]);
    }
    
    // Warp-level reduction
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());
    local_sum = cg::reduce(tile32, local_sum, cg::plus<float>());
    
    // First thread in warp adds to global result
    if (tile32.thread_rank() == 0) {
        atomicAdd(result, local_sum);
    }
}

/**
 * Batch similarity computation for multiple vector pairs
 */
__global__ void batch_similarity_kernel(
    const int8_t* vectors_a,
    const int8_t* vectors_b,
    float* results,
    int num_pairs,
    int dimension
) {
    int pair_id = blockIdx.y;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pair_id >= num_pairs) return;
    
    __shared__ float shared_sums[256];
    int local_tid = threadIdx.x;
    
    shared_sums[local_tid] = 0.0f;
    
    int vector_offset = pair_id * dimension;
    if (tid < dimension) {
        shared_sums[local_tid] = static_cast<float>(
            vectors_a[vector_offset + tid] * vectors_b[vector_offset + tid]
        );
    }
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            shared_sums[local_tid] += shared_sums[local_tid + stride];
        }
        __syncthreads();
    }
    
    if (local_tid == 0) {
        atomicAdd(&results[pair_id], shared_sums[0]);
    }
}

/**
 * One-to-many similarity computation (query against database)
 */
__global__ void query_similarity_kernel(
    const int8_t* query_vector,
    const int8_t* database_vectors,
    float* similarities,
    int num_vectors,
    int dimension
) {
    int vector_id = blockIdx.x;
    int tid = threadIdx.x;
    
    if (vector_id >= num_vectors) return;
    
    __shared__ float shared_sums[256];
    shared_sums[tid] = 0.0f;
    
    int vector_offset = vector_id * dimension;
    
    // Each thread processes multiple elements
    for (int i = tid; i < dimension; i += blockDim.x) {
        shared_sums[tid] += static_cast<float>(
            query_vector[i] * database_vectors[vector_offset + i]
        );
    }
    
    __syncthreads();
    
    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_sums[tid] += shared_sums[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        similarities[vector_id] = shared_sums[0] / dimension; // Normalized similarity
    }
}

/**
 * Advanced similarity with early termination
 */
__global__ void similarity_early_termination_kernel(
    const int8_t* query_vector,
    const int8_t* database_vectors,
    float* similarities,
    float threshold,
    int* valid_indices,
    int num_vectors,
    int dimension
) {
    int vector_id = blockIdx.x;
    int tid = threadIdx.x;
    
    if (vector_id >= num_vectors) return;
    
    __shared__ float shared_sums[256];
    __shared__ bool early_terminate;
    
    if (tid == 0) early_terminate = false;
    
    shared_sums[tid] = 0.0f;
    __syncthreads();
    
    int vector_offset = vector_id * dimension;
    float running_sum = 0.0f;
    
    // Process elements with early termination check
    for (int i = tid; i < dimension && !early_terminate; i += blockDim.x) {
        float contrib = static_cast<float>(
            query_vector[i] * database_vectors[vector_offset + i]
        );
        running_sum += contrib;
        
        // Check for early termination every 32 elements
        if ((i % 32) == 0) {
            float partial_sim = running_sum / (i + 1);
            if (partial_sim < threshold - 0.1f) { // Conservative threshold
                if (tid == 0) early_terminate = true;
                __syncthreads();
            }
        }
    }
    
    if (!early_terminate) {
        shared_sums[tid] = running_sum;
        __syncthreads();
        
        // Reduction
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_sums[tid] += shared_sums[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            float sim = shared_sums[0] / dimension;
            similarities[vector_id] = sim;
            valid_indices[vector_id] = (sim >= threshold) ? 1 : 0;
        }
    } else {
        if (tid == 0) {
            similarities[vector_id] = -1.0f; // Mark as invalid
            valid_indices[vector_id] = 0;
        }
    }
}

/**
 * Hamming distance kernel
 */
__global__ void hamming_distance_kernel(
    const int8_t* vector_a,
    const int8_t* vector_b,
    int* result,
    int dimension
) {
    __shared__ int shared_counts[256];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    shared_counts[local_tid] = 0;
    
    if (tid < dimension) {
        shared_counts[local_tid] = (vector_a[tid] != vector_b[tid]) ? 1 : 0;
    }
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride) {
            shared_counts[local_tid] += shared_counts[local_tid + stride];
        }
        __syncthreads();
    }
    
    if (local_tid == 0) {
        atomicAdd(result, shared_counts[0]);
    }
}

/**
 * Top-K similarity search kernel
 */
__global__ void topk_similarity_kernel(
    const int8_t* query_vector,
    const int8_t* database_vectors,
    float* top_similarities,
    int* top_indices,
    int num_vectors,
    int dimension,
    int k
) {
    int vector_id = blockIdx.x;
    if (vector_id >= num_vectors) return;
    
    // Compute similarity for this vector
    float similarity = 0.0f;
    int vector_offset = vector_id * dimension;
    
    for (int i = threadIdx.x; i < dimension; i += blockDim.x) {
        similarity += static_cast<float>(
            query_vector[i] * database_vectors[vector_offset + i]
        );
    }
    
    // Warp-level reduction
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cg::this_thread_block());
    similarity = cg::reduce(tile32, similarity, cg::plus<float>());
    
    if (tile32.thread_rank() == 0) {
        similarity /= dimension; // Normalize
        
        // Insert into top-K list (simple insertion sort for small K)
        for (int i = 0; i < k; i++) {
            if (similarity > top_similarities[i]) {
                // Shift elements down
                for (int j = k - 1; j > i; j--) {
                    top_similarities[j] = top_similarities[j - 1];
                    top_indices[j] = top_indices[j - 1];
                }
                // Insert new element
                top_similarities[i] = similarity;
                top_indices[i] = vector_id;
                break;
            }
        }
    }
}

/**
 * Host function for similarity computation
 */
extern "C" float cuda_similarity(
    const int8_t* h_vector_a,
    const int8_t* h_vector_b,
    int dimension
) {
    int8_t *d_vector_a, *d_vector_b;
    float *d_result, h_result = 0.0f;
    
    size_t vector_size = dimension * sizeof(int8_t);
    
    // Allocate GPU memory
    cudaMalloc(&d_vector_a, vector_size);
    cudaMalloc(&d_vector_b, vector_size);
    cudaMalloc(&d_result, sizeof(float));
    
    // Copy input data
    cudaMemcpy(d_vector_a, h_vector_a, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, h_vector_b, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (dimension + threads_per_block - 1) / threads_per_block;
    
    similarity_kernel_warp_optimized<<<blocks, threads_per_block>>>(
        d_vector_a, d_vector_b, d_result, dimension
    );
    
    // Copy result back
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
    cudaFree(d_result);
    cudaDeviceSynchronize();
    
    return h_result / dimension; // Normalized similarity
}

/**
 * Host function for batch similarity computation
 */
extern "C" void cuda_batch_similarity(
    const int8_t* h_vectors_a,
    const int8_t* h_vectors_b,
    float* h_results,
    int num_pairs,
    int dimension
) {
    int8_t *d_vectors_a, *d_vectors_b;
    float *d_results;
    
    size_t total_vector_size = num_pairs * dimension * sizeof(int8_t);
    size_t results_size = num_pairs * sizeof(float);
    
    cudaMalloc(&d_vectors_a, total_vector_size);
    cudaMalloc(&d_vectors_b, total_vector_size);
    cudaMalloc(&d_results, results_size);
    
    cudaMemcpy(d_vectors_a, h_vectors_a, total_vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectors_b, h_vectors_b, total_vector_size, cudaMemcpyHostToDevice);
    cudaMemset(d_results, 0, results_size);
    
    // 2D kernel launch
    dim3 threads_per_block(256);
    dim3 blocks((dimension + threads_per_block.x - 1) / threads_per_block.x, num_pairs);
    
    batch_similarity_kernel<<<blocks, threads_per_block>>>(
        d_vectors_a, d_vectors_b, d_results, num_pairs, dimension
    );
    
    // Normalize results
    dim3 norm_blocks((num_pairs + 255) / 256);
    dim3 norm_threads(256);
    
    // Simple normalization kernel
    auto normalize_kernel = [] __device__ (float* results, int num_pairs, int dimension) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num_pairs) {
            results[idx] /= dimension;
        }
    };
    
    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);
    
    // Normalize on host for simplicity
    for (int i = 0; i < num_pairs; i++) {
        h_results[i] /= dimension;
    }
    
    cudaFree(d_vectors_a);
    cudaFree(d_vectors_b);
    cudaFree(d_results);
    cudaDeviceSynchronize();
}

/**
 * Host function for query similarity (one-to-many)
 */
extern "C" void cuda_query_similarity(
    const int8_t* h_query,
    const int8_t* h_database,
    float* h_similarities,
    int num_vectors,
    int dimension
) {
    int8_t *d_query, *d_database;
    float *d_similarities;
    
    size_t query_size = dimension * sizeof(int8_t);
    size_t database_size = num_vectors * dimension * sizeof(int8_t);
    size_t results_size = num_vectors * sizeof(float);
    
    cudaMalloc(&d_query, query_size);
    cudaMalloc(&d_database, database_size);
    cudaMalloc(&d_similarities, results_size);
    
    cudaMemcpy(d_query, h_query, query_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_database, h_database, database_size, cudaMemcpyHostToDevice);
    
    // Launch kernel with one block per database vector
    int threads_per_block = 256;
    
    query_similarity_kernel<<<num_vectors, threads_per_block>>>(
        d_query, d_database, d_similarities, num_vectors, dimension
    );
    
    cudaMemcpy(h_similarities, d_similarities, results_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_query);
    cudaFree(d_database);
    cudaFree(d_similarities);
    cudaDeviceSynchronize();
}

/**
 * Host function for Hamming distance
 */
extern "C" int cuda_hamming_distance(
    const int8_t* h_vector_a,
    const int8_t* h_vector_b,
    int dimension
) {
    int8_t *d_vector_a, *d_vector_b;
    int *d_result, h_result = 0;
    
    size_t vector_size = dimension * sizeof(int8_t);
    
    cudaMalloc(&d_vector_a, vector_size);
    cudaMalloc(&d_vector_b, vector_size);
    cudaMalloc(&d_result, sizeof(int));
    
    cudaMemcpy(d_vector_a, h_vector_a, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vector_b, h_vector_b, vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);
    
    int threads_per_block = 256;
    int blocks = (dimension + threads_per_block - 1) / threads_per_block;
    
    hamming_distance_kernel<<<blocks, threads_per_block>>>(
        d_vector_a, d_vector_b, d_result, dimension
    );
    
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_vector_a);
    cudaFree(d_vector_b);
    cudaFree(d_result);
    cudaDeviceSynchronize();
    
    return h_result;
}

} // extern "C"

#endif // WITH_CUDA