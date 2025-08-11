#include "operations.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <unordered_map>
#include <immintrin.h>  // AVX/SIMD support
#include <thread>
#include <future>
#include <execution>

namespace hdc {

HyperVector Operations::majority_bundle(const std::vector<HyperVector>& vectors) {
    if (vectors.empty()) {
        throw std::invalid_argument("Cannot bundle empty vector set");
    }
    
    // Use SIMD-optimized bundling for large vector sets
    if (vectors.size() > 8 && vectors[0].dimension() % 8 == 0) {
        return simd_bundle_vectors(vectors);
    }
    
    return HyperVector::bundle_vectors(vectors);
}

HyperVector Operations::weighted_bundle(const std::vector<HyperVector>& vectors, 
                                       const std::vector<double>& weights) {
    if (vectors.size() != weights.size()) {
        throw std::invalid_argument("Vector and weight counts must match");
    }
    
    std::vector<std::pair<HyperVector, double>> weighted_pairs;
    for (size_t i = 0; i < vectors.size(); ++i) {
        weighted_pairs.emplace_back(vectors[i], weights[i]);
    }
    
    return hdc::weighted_bundle(weighted_pairs);
}

HyperVector Operations::elementwise_bind(const HyperVector& a, const HyperVector& b) {
    return a.bind(b);
}

HyperVector Operations::circular_bind(const HyperVector& a, const HyperVector& b) {
    // For circular convolution binding (more complex operation)
    const int dim = a.dimension();
    HyperVector result(dim);
    
    for (int i = 0; i < dim; ++i) {
        int sum = 0;
        for (int j = 0; j < dim; ++j) {
            sum += a[j] * b[(i - j + dim) % dim];
        }
        result[i] = (sum > 0) ? 1 : -1;
    }
    
    return result;
}

HyperVector Operations::left_rotate(const HyperVector& v, int positions) {
    return v.permute(positions);
}

HyperVector Operations::right_rotate(const HyperVector& v, int positions) {
    return v.permute(-positions);
}

double Operations::cosine_similarity(const HyperVector& a, const HyperVector& b) {
    if (a.dimension() != b.dimension()) {
        throw std::invalid_argument("Vector dimensions must match");
    }
    
    // Use SIMD-accelerated similarity for large vectors
    if (a.dimension() >= 1024 && a.dimension() % 8 == 0) {
        return simd_cosine_similarity(a, b);
    }
    
    return a.similarity(b);
}

double Operations::hamming_similarity(const HyperVector& a, const HyperVector& b) {
    return 1.0 - a.hamming_distance(b);
}

double Operations::jaccard_similarity(const HyperVector& a, const HyperVector& b) {
    if (a.dimension() != b.dimension()) {
        throw std::invalid_argument("Dimensions must match");
    }
    
    int intersection = 0, union_count = 0;
    
    for (int i = 0; i < a.dimension(); ++i) {
        if (a[i] == 1 || b[i] == 1) {
            union_count++;
            if (a[i] == 1 && b[i] == 1) {
                intersection++;
            }
        }
    }
    
    return union_count > 0 ? static_cast<double>(intersection) / union_count : 0.0;
}

HyperVector Operations::cleanup(const HyperVector& v, double threshold) {
    HyperVector result = v;
    result.threshold();
    return result;
}

HyperVector Operations::soft_threshold(const HyperVector& v, double threshold) {
    // For bipolar vectors, soft thresholding is not directly applicable
    // We'll implement it as probabilistic thresholding
    HyperVector result(v.dimension());
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    for (int i = 0; i < v.dimension(); ++i) {
        double prob = 0.5 + (v[i] > 0 ? threshold : -threshold);
        result[i] = (dist(rng) < prob) ? 1 : -1;
    }
    
    return result;
}

HyperVector Operations::add_noise(const HyperVector& v, double noise_ratio, uint32_t seed) {
    if (noise_ratio < 0.0 || noise_ratio > 1.0) {
        throw std::invalid_argument("Noise ratio must be between 0 and 1");
    }
    
    std::mt19937 rng(seed != 0 ? seed : std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    HyperVector result = v;
    for (int i = 0; i < v.dimension(); ++i) {
        if (dist(rng) < noise_ratio) {
            result[i] = -result[i]; // Flip bit
        }
    }
    
    return result;
}

HyperVector Operations::flip_bits(const HyperVector& v, int num_flips, uint32_t seed) {
    if (num_flips < 0 || num_flips > v.dimension()) {
        throw std::invalid_argument("Number of flips must be between 0 and dimension");
    }
    
    std::mt19937 rng(seed != 0 ? seed : std::random_device{}());
    HyperVector result = v;
    
    // Random unique indices to flip
    std::vector<int> indices(v.dimension());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    for (int i = 0; i < num_flips; ++i) {
        result[indices[i]] = -result[indices[i]];
    }
    
    return result;
}

HyperVector Operations::encode_sequence(const std::vector<HyperVector>& sequence,
                                      bool use_positions) {
    if (use_positions) {
        return create_sequence(sequence);
    } else {
        return HyperVector::bundle_vectors(sequence);
    }
}

HyperVector Operations::decode_sequence_element(const HyperVector& sequence_hv,
                                              const std::vector<HyperVector>& candidates,
                                              int position) {
    // Create position vector
    HyperVector pos_vec = HyperVector::random(sequence_hv.dimension(), 
                                            static_cast<uint32_t>(position + 1));
    
    // Unbind position to get element estimate
    HyperVector unbound = sequence_hv.bind(pos_vec);
    
    // Find best match among candidates
    double best_sim = -1.0;
    HyperVector best_match = candidates[0];
    
    for (const auto& candidate : candidates) {
        double sim = unbound.similarity(candidate);
        if (sim > best_sim) {
            best_sim = sim;
            best_match = candidate;
        }
    }
    
    return best_match;
}

HyperVector Operations::set_union(const std::vector<HyperVector>& vectors) {
    return HyperVector::bundle_vectors(vectors);
}

HyperVector Operations::set_intersection(const std::vector<HyperVector>& vectors, 
                                       double threshold) {
    if (vectors.empty()) {
        throw std::invalid_argument("Cannot intersect empty vector set");
    }
    
    const int dim = vectors[0].dimension();
    std::vector<int> sums(dim, 0);
    
    // Sum all vectors
    for (const auto& vec : vectors) {
        for (int i = 0; i < dim; ++i) {
            if (vec[i] > 0) sums[i]++;
        }
    }
    
    const int min_votes = static_cast<int>(threshold * vectors.size());
    HyperVector result(dim);
    
    for (int i = 0; i < dim; ++i) {
        result[i] = (sums[i] >= min_votes) ? 1 : -1;
    }
    
    return result;
}

double Operations::entropy(const HyperVector& v) {
    int positive_count = 0;
    for (int i = 0; i < v.dimension(); ++i) {
        if (v[i] > 0) positive_count++;
    }
    
    double p1 = static_cast<double>(positive_count) / v.dimension();
    double p0 = 1.0 - p1;
    
    if (p1 == 0.0 || p0 == 0.0) return 0.0;
    
    return -(p1 * std::log2(p1) + p0 * std::log2(p0));
}

std::vector<int> Operations::get_active_dimensions(const HyperVector& v) {
    std::vector<int> active_dims;
    for (int i = 0; i < v.dimension(); ++i) {
        if (v[i] > 0) {
            active_dims.push_back(i);
        }
    }
    return active_dims;
}

double Operations::sparsity(const HyperVector& v) {
    int positive_count = get_active_dimensions(v).size();
    return static_cast<double>(positive_count) / v.dimension();
}

double Operations::euclidean_distance(const HyperVector& a, const HyperVector& b) {
    if (a.dimension() != b.dimension()) {
        throw std::invalid_argument("Dimensions must match");
    }
    
    double sum_sq = 0.0;
    for (int i = 0; i < a.dimension(); ++i) {
        double diff = a[i] - b[i];
        sum_sq += diff * diff;
    }
    
    return std::sqrt(sum_sq);
}

double Operations::manhattan_distance(const HyperVector& a, const HyperVector& b) {
    if (a.dimension() != b.dimension()) {
        throw std::invalid_argument("Dimensions must match");
    }
    
    double sum = 0.0;
    for (int i = 0; i < a.dimension(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    
    return sum;
}

double Operations::chebyshev_distance(const HyperVector& a, const HyperVector& b) {
    if (a.dimension() != b.dimension()) {
        throw std::invalid_argument("Dimensions must match");
    }
    
    double max_diff = 0.0;
    for (int i = 0; i < a.dimension(); ++i) {
        double diff = std::abs(a[i] - b[i]);
        max_diff = std::max(max_diff, diff);
    }
    
    return max_diff;
}

// PositionVectors implementation
PositionVectors::PositionVectors(int dimension, int max_positions) 
    : dimension_(dimension), rng_(std::random_device{}()) {
    precompute_positions(max_positions);
}

HyperVector PositionVectors::get_position_vector(int position) const {
    if (position < 0) {
        throw std::invalid_argument("Position must be non-negative");
    }
    
    if (position < static_cast<int>(position_vectors_.size())) {
        return position_vectors_[position];
    }
    
    // Generate on-the-fly if not precomputed
    return HyperVector::random(dimension_, static_cast<uint32_t>(position + 1));
}

void PositionVectors::precompute_positions(int count) {
    position_vectors_.clear();
    position_vectors_.reserve(count);
    
    for (int i = 0; i < count; ++i) {
        position_vectors_.push_back(HyperVector::random(dimension_, 
                                                       static_cast<uint32_t>(i + 1)));
    }
}

// BasisVectors implementation
BasisVectors::BasisVectors(int dimension) 
    : dimension_(dimension), rng_(std::random_device{}()) {
    
    // Pre-generate basis vectors for common ranges
    integer_basis_.reserve(2000);
    for (int i = 0; i < 2000; ++i) {
        integer_basis_.push_back(HyperVector::random(dimension_, 
                                                   static_cast<uint32_t>(i + 10000)));
    }
    
    float_basis_.reserve(200);
    for (int i = 0; i < 200; ++i) {
        float_basis_.push_back(HyperVector::random(dimension_, 
                                                 static_cast<uint32_t>(i + 20000)));
    }
}

HyperVector BasisVectors::encode_integer(int value, int min_val, int max_val) {
    if (value < min_val || value > max_val) {
        throw std::invalid_argument("Value outside specified range");
    }
    
    int index = value - min_val;
    if (index < static_cast<int>(integer_basis_.size())) {
        return integer_basis_[index];
    }
    
    return HyperVector::random(dimension_, static_cast<uint32_t>(value + 10000));
}

HyperVector BasisVectors::encode_float(double value, double min_val, double max_val, 
                                      int precision) {
    if (value < min_val || value > max_val) {
        throw std::invalid_argument("Value outside specified range");
    }
    
    // Discretize to precision levels
    int discrete_val = static_cast<int>((value - min_val) / (max_val - min_val) * precision);
    discrete_val = std::max(0, std::min(precision - 1, discrete_val));
    
    if (discrete_val < static_cast<int>(float_basis_.size())) {
        return float_basis_[discrete_val];
    }
    
    return HyperVector::random(dimension_, static_cast<uint32_t>(discrete_val + 20000));
}

HyperVector BasisVectors::encode_category(const std::string& category) {
    auto it = category_vectors_.find(category);
    if (it != category_vectors_.end()) {
        return it->second;
    }
    
    // Generate new category vector
    HyperVector cat_vec = HyperVector::random(dimension_, 
                                            static_cast<uint32_t>(
                                                std::hash<std::string>{}(category)));
    category_vectors_[category] = cat_vec;
    return cat_vec;
}

void BasisVectors::register_categories(const std::vector<std::string>& categories) {
    for (const auto& category : categories) {
        encode_category(category); // This will add it to the map
    }
}

HyperVector BasisVectors::encode_2d_position(double x, double y, double resolution) {
    int grid_x = static_cast<int>(x / resolution);
    int grid_y = static_cast<int>(y / resolution);
    
    HyperVector x_vec = encode_integer(grid_x, -1000, 1000);
    HyperVector y_vec = encode_integer(grid_y, -1000, 1000);
    
    return x_vec.bind(y_vec.permute(1)); // Permute to differentiate x and y
}

HyperVector BasisVectors::encode_3d_position(double x, double y, double z, double resolution) {
    int grid_x = static_cast<int>(x / resolution);
    int grid_y = static_cast<int>(y / resolution);
    int grid_z = static_cast<int>(z / resolution);
    
    HyperVector x_vec = encode_integer(grid_x, -1000, 1000);
    HyperVector y_vec = encode_integer(grid_y, -1000, 1000);
    HyperVector z_vec = encode_integer(grid_z, -1000, 1000);
    
    return x_vec.bind(y_vec.permute(1)).bind(z_vec.permute(2));
}

HyperVector BasisVectors::encode_angle(double angle_rad) {
    // Map angle to [0, 360) degrees for discretization
    double angle_deg = angle_rad * 180.0 / M_PI;
    angle_deg = fmod(angle_deg + 360.0, 360.0); // Ensure positive
    
    int discrete_angle = static_cast<int>(angle_deg);
    return encode_integer(discrete_angle, 0, 359);
}

HyperVector BasisVectors::encode_quaternion(double w, double x, double y, double z) {
    // Normalize quaternion first
    double norm = std::sqrt(w*w + x*x + y*y + z*z);
    if (norm > 0) {
        w /= norm; x /= norm; y /= norm; z /= norm;
    }
    
    // Encode each component
    HyperVector w_vec = encode_float(w, -1.0, 1.0, 200);
    HyperVector x_vec = encode_float(x, -1.0, 1.0, 200);
    HyperVector y_vec = encode_float(y, -1.0, 1.0, 200);
    HyperVector z_vec = encode_float(z, -1.0, 1.0, 200);
    
    // Bind with permutations to differentiate components
    return w_vec.bind(x_vec.permute(1)).bind(y_vec.permute(2)).bind(z_vec.permute(3));
}

// SIMD-optimized operations for performance enhancement
HyperVector Operations::simd_bundle_vectors(const std::vector<HyperVector>& vectors) {
    if (vectors.empty()) {
        throw std::invalid_argument("Cannot bundle empty vector set");
    }
    
    const int dim = vectors[0].dimension();
    const int simd_width = 8; // AVX uses 8 32-bit integers
    const int simd_iterations = dim / simd_width;
    
    HyperVector result(dim);
    std::vector<int32_t> sums(dim, 0);
    
    // SIMD bundling for aligned portions
    for (int v = 0; v < vectors.size(); ++v) {
        const auto& vec_data = vectors[v].data();
        
        #pragma omp simd
        for (int i = 0; i < simd_iterations; ++i) {
            __m256i vec_chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&vec_data[i * simd_width]));
            __m256i sum_chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sums[i * simd_width]));
            __m256i result_chunk = _mm256_add_epi32(vec_chunk, sum_chunk);
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(&sums[i * simd_width]), result_chunk);
        }
        
        // Handle remaining elements
        for (int i = simd_iterations * simd_width; i < dim; ++i) {
            sums[i] += vec_data[i];
        }
    }
    
    // Apply majority rule with SIMD
    const int threshold = vectors.size() / 2;
    auto result_data = result.data_mutable();
    
    #pragma omp simd
    for (int i = 0; i < simd_iterations; ++i) {
        __m256i sum_chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&sums[i * simd_width]));
        __m256i thresh_vec = _mm256_set1_epi32(threshold);
        __m256i mask = _mm256_cmpgt_epi32(sum_chunk, thresh_vec);
        __m256i ones = _mm256_set1_epi32(1);
        __m256i neg_ones = _mm256_set1_epi32(-1);
        __m256i result_chunk = _mm256_blendv_epi8(neg_ones, ones, mask);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(&result_data[i * simd_width]), result_chunk);
    }
    
    // Handle remaining elements
    for (int i = simd_iterations * simd_width; i < dim; ++i) {
        result_data[i] = (sums[i] > threshold) ? 1 : -1;
    }
    
    return result;
}

double Operations::simd_cosine_similarity(const HyperVector& a, const HyperVector& b) {
    const int dim = a.dimension();
    const int simd_width = 8;
    const int simd_iterations = dim / simd_width;
    
    const auto& a_data = a.data();
    const auto& b_data = b.data();
    
    __m256 dot_product = _mm256_setzero_ps();
    __m256 norm_a = _mm256_setzero_ps();
    __m256 norm_b = _mm256_setzero_ps();
    
    // SIMD dot product computation
    for (int i = 0; i < simd_iterations; ++i) {
        __m256 a_chunk = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a_data[i * simd_width])));
        __m256 b_chunk = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b_data[i * simd_width])));
        
        dot_product = _mm256_fmadd_ps(a_chunk, b_chunk, dot_product);
        norm_a = _mm256_fmadd_ps(a_chunk, a_chunk, norm_a);
        norm_b = _mm256_fmadd_ps(b_chunk, b_chunk, norm_b);
    }
    
    // Horizontal sum using SIMD
    float dot_sum = 0.0f, norm_a_sum = 0.0f, norm_b_sum = 0.0f;\n    \n    float dot_array[8], norm_a_array[8], norm_b_array[8];
    _mm256_storeu_ps(dot_array, dot_product);
    _mm256_storeu_ps(norm_a_array, norm_a);
    _mm256_storeu_ps(norm_b_array, norm_b);
    \n    for (int i = 0; i < 8; ++i) {
        dot_sum += dot_array[i];
        norm_a_sum += norm_a_array[i];
        norm_b_sum += norm_b_array[i];
    }
    
    // Handle remaining elements
    for (int i = simd_iterations * simd_width; i < dim; ++i) {
        dot_sum += a_data[i] * b_data[i];
        norm_a_sum += a_data[i] * a_data[i];
        norm_b_sum += b_data[i] * b_data[i];
    }
    
    double norm_product = std::sqrt(norm_a_sum) * std::sqrt(norm_b_sum);
    return norm_product > 0.0 ? dot_sum / norm_product : 0.0;
}

// Parallel processing operations for large-scale HDC
HyperVector Operations::parallel_bundle(const std::vector<HyperVector>& vectors, int num_threads) {
    if (vectors.empty()) {
        throw std::invalid_argument("Cannot bundle empty vector set");
    }
    
    const int dim = vectors[0].dimension();
    const int chunk_size = std::max(1, static_cast<int>(vectors.size()) / num_threads);
    
    std::vector<std::future<std::vector<int>>> futures;
    
    for (int t = 0; t < num_threads; ++t) {
        int start_idx = t * chunk_size;
        int end_idx = std::min(start_idx + chunk_size, static_cast<int>(vectors.size()));
        
        if (start_idx >= vectors.size()) break;
        
        futures.emplace_back(std::async(std::launch::async, [&vectors, start_idx, end_idx, dim]() {
            std::vector<int> local_sums(dim, 0);
            for (int v = start_idx; v < end_idx; ++v) {
                const auto& vec_data = vectors[v].data();
                for (int i = 0; i < dim; ++i) {
                    local_sums[i] += vec_data[i];
                }
            }
            return local_sums;
        }));
    }
    
    // Combine results from all threads
    std::vector<int> total_sums(dim, 0);
    for (auto& future : futures) {
        auto local_sums = future.get();
        for (int i = 0; i < dim; ++i) {
            total_sums[i] += local_sums[i];
        }
    }
    
    // Apply majority rule
    const int threshold = vectors.size() / 2;
    HyperVector result(dim);
    auto result_data = result.data_mutable();
    
    std::transform(std::execution::par_unseq, total_sums.begin(), total_sums.end(), 
                   result_data, [threshold](int sum) { return sum > threshold ? 1 : -1; });
    
    return result;
}

// Memory-efficient operations for large hypervectors
class HDCMemoryPool {
private:
    static constexpr size_t POOL_SIZE = 1024 * 1024 * 10; // 10MB pool
    static std::unique_ptr<int32_t[]> memory_pool_;
    static std::atomic<size_t> pool_offset_;
    static std::mutex pool_mutex_;
    
public:
    static int32_t* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        size_t current_offset = pool_offset_.load();
        
        if (current_offset + size > POOL_SIZE) {
            pool_offset_ = 0; // Reset pool (simple circular buffer)
            current_offset = 0;
        }
        
        if (!memory_pool_) {
            memory_pool_ = std::make_unique<int32_t[]>(POOL_SIZE);
        }
        
        pool_offset_ += size;
        return &memory_pool_[current_offset];
    }
    
    static void reset() {
        std::lock_guard<std::mutex> lock(pool_mutex_);
        pool_offset_ = 0;
    }
};

std::unique_ptr<int32_t[]> HDCMemoryPool::memory_pool_;
std::atomic<size_t> HDCMemoryPool::pool_offset_{0};
std::mutex HDCMemoryPool::pool_mutex_;

} // namespace hdc