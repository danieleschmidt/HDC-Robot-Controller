/**
 * CUDA-accelerated operations for HDC Robot Controller
 * Provides GPU acceleration for hyperdimensional computing operations
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Constants
#define THREADS_PER_BLOCK 256
#define MAX_BLOCKS 65535
#define WARP_SIZE 32

extern "C" {

/**
 * CUDA kernel for bundling multiple hypervectors using majority rule
 */
__global__ void bundle_hypervectors_kernel(
    const int8_t* vectors,
    int8_t* result,
    int num_vectors,
    int dimension,
    int total_threads
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        int sum = 0;
        
        // Sum across all vectors for this dimension
        for (int v = 0; v < num_vectors; v++) {
            sum += vectors[v * dimension + idx];
        }
        
        // Apply majority rule
        result[idx] = (sum > 0) ? 1 : -1;
    }
}

/**
 * CUDA kernel for binding two hypervectors (element-wise multiplication)
 */
__global__ void bind_hypervectors_kernel(
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
 * CUDA kernel for permuting (rotating) a hypervector
 */
__global__ void permute_hypervector_kernel(
    const int8_t* input,
    int8_t* result,
    int dimension,
    int shift
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        int src_idx = (idx - shift + dimension) % dimension;
        result[idx] = input[src_idx];
    }
}

/**
 * CUDA kernel for computing similarity between two hypervectors
 */
__global__ void similarity_kernel(
    const int8_t* vector_a,
    const int8_t* vector_b,
    int* partial_sums,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    __shared__ int sdata[THREADS_PER_BLOCK];
    
    // Each thread computes dot product for its elements
    int sum = 0;
    if (idx < dimension) {
        sum = vector_a[idx] * vector_b[idx];
    }
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

/**
 * CUDA kernel for generating random hypervector
 */
__global__ void generate_random_kernel(
    int8_t* result,
    int dimension,
    unsigned long long seed,
    int offset = 0
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        curandState state;
        curand_init(seed, idx + offset, 0, &state);
        
        float random_val = curand_uniform(&state);
        result[idx] = (random_val < 0.5f) ? -1 : 1;
    }
}

/**
 * CUDA kernel for weighted bundling of hypervectors
 */
__global__ void weighted_bundle_kernel(
    const int8_t* const* vectors,
    const float* weights,
    int8_t* result,
    int num_vectors,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        float weighted_sum = 0.0f;
        
        for (int v = 0; v < num_vectors; v++) {
            weighted_sum += vectors[v][idx] * weights[v];
        }
        
        result[idx] = (weighted_sum > 0.0f) ? 1 : -1;
    }
}

/**
 * CUDA kernel for batch similarity computation
 */
__global__ void batch_similarity_kernel(
    const int8_t* query_vector,
    const int8_t* database_vectors,
    float* similarities,
    int num_database_vectors,
    int dimension
) {
    int vector_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (vector_idx < num_database_vectors) {
        __shared__ int sdata[THREADS_PER_BLOCK];
        
        int sum = 0;
        for (int i = dim_idx; i < dimension; i += blockDim.x) {
            sum += query_vector[i] * database_vectors[vector_idx * dimension + i];
        }
        
        sdata[threadIdx.x] = sum;
        __syncthreads();
        
        // Reduce within block
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            similarities[vector_idx] = (float)sdata[0] / dimension;
        }
    }
}

/**
 * CUDA kernel for noise injection
 */
__global__ void add_noise_kernel(
    const int8_t* input,
    int8_t* result,
    int dimension,
    float noise_ratio,
    unsigned long long seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        
        int8_t value = input[idx];
        
        if (curand_uniform(&state) < noise_ratio) {
            value = -value;  // Flip bit
        }
        
        result[idx] = value;
    }
}

/**
 * CUDA kernel for sequence encoding with position binding
 */
__global__ void encode_sequence_kernel(
    const int8_t* const* sequence_vectors,
    const int8_t* const* position_vectors,
    int8_t* result,
    int sequence_length,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        int sum = 0;
        
        for (int pos = 0; pos < sequence_length; pos++) {
            // Bind sequence element with position
            int8_t bound_value = sequence_vectors[pos][idx] * position_vectors[pos][idx];
            sum += bound_value;
        }
        
        result[idx] = (sum > 0) ? 1 : -1;
    }
}

/**
 * CUDA kernel for n-gram encoding
 */
__global__ void encode_ngram_kernel(
    const int8_t* sequence,
    int8_t* result,
    int sequence_length,
    int dimension,
    int n,
    const int8_t* const* position_vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        int total_sum = 0;
        
        // Generate all n-grams
        for (int start = 0; start <= sequence_length - n; start++) {
            int ngram_sum = 0;
            
            // Encode this n-gram
            for (int i = 0; i < n; i++) {
                int seq_pos = start + i;
                int8_t element = sequence[seq_pos * dimension + idx];
                int8_t position = position_vectors[i][idx];
                ngram_sum += element * position;
            }
            
            // Add n-gram to total
            total_sum += (ngram_sum > 0) ? 1 : -1;
        }
        
        result[idx] = (total_sum > 0) ? 1 : -1;
    }
}

/**
 * CUDA kernel for circular convolution binding
 */
__global__ void circular_convolution_kernel(
    const int8_t* vector_a,
    const int8_t* vector_b,
    int8_t* result,
    int dimension
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < dimension) {
        int sum = 0;
        
        for (int j = 0; j < dimension; j++) {
            int k = (idx - j + dimension) % dimension;
            sum += vector_a[j] * vector_b[k];
        }
        
        result[idx] = (sum > 0) ? 1 : -1;
    }
}

/**
 * CUDA kernel for cleanup operation using basis vectors
 */
__global__ void cleanup_kernel(
    const int8_t* input,
    const int8_t* const* basis_vectors,
    int8_t* result,
    int dimension,
    int num_basis_vectors,
    float* similarities
) {
    int basis_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (basis_idx < num_basis_vectors) {
        __shared__ int sdata[THREADS_PER_BLOCK];
        
        int sum = 0;
        for (int i = dim_idx; i < dimension; i += blockDim.x) {
            sum += input[i] * basis_vectors[basis_idx][i];
        }
        
        sdata[threadIdx.x] = sum;
        __syncthreads();
        
        // Reduce within block
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            }
            __syncthreads();
        }
        
        if (threadIdx.x == 0) {
            similarities[basis_idx] = (float)sdata[0] / dimension;
        }
    }
}

/**
 * CUDA kernel for spatial encoding
 */
__global__ void spatial_encoding_kernel(
    float* coordinates,
    int8_t* result,
    int num_points,
    int dimension,
    float resolution,
    unsigned long long seed
) {
    int point_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (point_idx < num_points && dim_idx < dimension) {
        float x = coordinates[point_idx * 3 + 0];
        float y = coordinates[point_idx * 3 + 1];
        float z = coordinates[point_idx * 3 + 2];
        
        // Discretize coordinates
        int grid_x = (int)(x / resolution);
        int grid_y = (int)(y / resolution);
        int grid_z = (int)(z / resolution);
        
        // Create deterministic hash for this position
        unsigned long long pos_hash = seed;
        pos_hash = pos_hash * 31 + grid_x;
        pos_hash = pos_hash * 31 + grid_y;
        pos_hash = pos_hash * 31 + grid_z;
        
        curandState state;
        curand_init(pos_hash, dim_idx, 0, &state);
        
        result[point_idx * dimension + dim_idx] = (curand_uniform(&state) < 0.5f) ? -1 : 1;
    }
}

/**
 * Multi-stream batch processing kernel
 */
__global__ void batch_process_kernel(
    const int8_t* input_vectors,
    int8_t* output_vectors,
    int num_vectors,
    int dimension,
    int operation_type
) {
    int vector_idx = blockIdx.y;
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (vector_idx < num_vectors && dim_idx < dimension) {
        int input_idx = vector_idx * dimension + dim_idx;
        int8_t value = input_vectors[input_idx];
        
        switch (operation_type) {
            case 0: // Identity
                output_vectors[input_idx] = value;
                break;
            case 1: // Invert
                output_vectors[input_idx] = -value;
                break;
            case 2: // Threshold (already bipolar)
                output_vectors[input_idx] = (value > 0) ? 1 : -1;
                break;
            default:
                output_vectors[input_idx] = value;
        }
    }
}

/**
 * Memory-optimized large vector operations
 */
__global__ void large_vector_operation_kernel(
    const int8_t* vector_a,
    const int8_t* vector_b,
    int8_t* result,
    int dimension,
    int operation_type
) {
    // Use grid-stride loop for large vectors
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < dimension; 
         idx += blockDim.x * gridDim.x) {
        
        int8_t a = vector_a[idx];
        int8_t b = vector_b[idx];
        int8_t res;
        
        switch (operation_type) {
            case 0: // Bundle (majority rule for 2 vectors)
                res = (a + b > 0) ? 1 : -1;
                break;
            case 1: // Bind
                res = a * b;
                break;
            case 2: // XOR
                res = (a == b) ? 1 : -1;
                break;
            default:
                res = a;
        }
        
        result[idx] = res;
    }
}

} // extern "C"

// C++ wrapper functions for easier integration

class CudaHDCOperations {
public:
    static bool bundle_vectors(const std::vector<int8_t*>& device_vectors,
                              int8_t* device_result,
                              int num_vectors,
                              int dimension,
                              cudaStream_t stream = 0) {
        
        // Flatten input vectors for kernel
        size_t total_size = num_vectors * dimension * sizeof(int8_t);
        int8_t* flat_vectors;
        CUDA_CHECK(cudaMalloc(&flat_vectors, total_size));
        
        // Copy vectors to flattened array
        for (int i = 0; i < num_vectors; i++) {
            CUDA_CHECK(cudaMemcpyAsync(
                flat_vectors + i * dimension,
                device_vectors[i],
                dimension * sizeof(int8_t),
                cudaMemcpyDeviceToDevice,
                stream
            ));
        }
        
        // Launch kernel
        int blocks = (dimension + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        blocks = min(blocks, MAX_BLOCKS);
        
        bundle_hypervectors_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            flat_vectors, device_result, num_vectors, dimension, dimension
        );
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(flat_vectors));
        
        return true;
    }
    
    static bool bind_vectors(const int8_t* device_vector_a,
                            const int8_t* device_vector_b,
                            int8_t* device_result,
                            int dimension,
                            cudaStream_t stream = 0) {
        
        int blocks = (dimension + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        blocks = min(blocks, MAX_BLOCKS);
        
        bind_hypervectors_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            device_vector_a, device_vector_b, device_result, dimension
        );
        
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    
    static float compute_similarity(const int8_t* device_vector_a,
                                   const int8_t* device_vector_b,
                                   int dimension,
                                   cudaStream_t stream = 0) {
        
        int blocks = (dimension + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        blocks = min(blocks, MAX_BLOCKS);
        
        // Allocate temporary storage for partial sums
        int* device_partial_sums;
        CUDA_CHECK(cudaMalloc(&device_partial_sums, blocks * sizeof(int)));
        
        // Launch similarity kernel
        similarity_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            device_vector_a, device_vector_b, device_partial_sums, dimension
        );
        
        // Reduce partial sums on CPU (could be optimized with CUB)
        thrust::device_vector<int> partial_sums_vec(device_partial_sums, device_partial_sums + blocks);
        int total_sum = thrust::reduce(partial_sums_vec.begin(), partial_sums_vec.end());
        
        CUDA_CHECK(cudaFree(device_partial_sums));
        
        return (float)total_sum / dimension;
    }
    
    static bool generate_random_vector(int8_t* device_result,
                                      int dimension,
                                      unsigned long long seed,
                                      cudaStream_t stream = 0) {
        
        int blocks = (dimension + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        blocks = min(blocks, MAX_BLOCKS);
        
        generate_random_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            device_result, dimension, seed
        );
        
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    
    static bool batch_similarity(const int8_t* device_query,
                                const int8_t* device_database,
                                float* device_similarities,
                                int num_database_vectors,
                                int dimension,
                                cudaStream_t stream = 0) {
        
        dim3 grid_size(num_database_vectors, 1, 1);
        dim3 block_size(min(THREADS_PER_BLOCK, dimension), 1, 1);
        
        batch_similarity_kernel<<<grid_size, block_size, 0, stream>>>(
            device_query, device_database, device_similarities,
            num_database_vectors, dimension
        );
        
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
    
    static bool permute_vector(const int8_t* device_input,
                              int8_t* device_result,
                              int dimension,
                              int shift,
                              cudaStream_t stream = 0) {
        
        int blocks = (dimension + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        blocks = min(blocks, MAX_BLOCKS);
        
        permute_hypervector_kernel<<<blocks, THREADS_PER_BLOCK, 0, stream>>>(
            device_input, device_result, dimension, shift
        );
        
        CUDA_CHECK(cudaGetLastError());
        return true;
    }
};

// Performance optimization utilities
class CudaMemoryManager {
private:
    std::vector<void*> allocated_pointers_;
    size_t total_allocated_;
    
public:
    CudaMemoryManager() : total_allocated_(0) {}
    
    ~CudaMemoryManager() {
        cleanup();
    }
    
    void* allocate(size_t size) {
        void* ptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        allocated_pointers_.push_back(ptr);
        total_allocated_ += size;
        return ptr;
    }
    
    void deallocate(void* ptr) {
        auto it = std::find(allocated_pointers_.begin(), allocated_pointers_.end(), ptr);
        if (it != allocated_pointers_.end()) {
            allocated_pointers_.erase(it);
            CUDA_CHECK(cudaFree(ptr));
        }
    }
    
    void cleanup() {
        for (void* ptr : allocated_pointers_) {
            cudaFree(ptr);
        }
        allocated_pointers_.clear();
        total_allocated_ = 0;
    }
    
    size_t get_total_allocated() const { return total_allocated_; }
};

// Stream management for concurrent operations
class CudaStreamManager {
private:
    std::vector<cudaStream_t> streams_;
    int current_stream_;
    
public:
    CudaStreamManager(int num_streams = 4) : current_stream_(0) {
        streams_.resize(num_streams);
        for (int i = 0; i < num_streams; i++) {
            CUDA_CHECK(cudaStreamCreate(&streams_[i]));
        }
    }
    
    ~CudaStreamManager() {
        for (cudaStream_t stream : streams_) {
            cudaStreamDestroy(stream);
        }
    }
    
    cudaStream_t get_next_stream() {
        cudaStream_t stream = streams_[current_stream_];
        current_stream_ = (current_stream_ + 1) % streams_.size();
        return stream;
    }
    
    void synchronize_all() {
        for (cudaStream_t stream : streams_) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
};