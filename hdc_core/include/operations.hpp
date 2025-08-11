#pragma once

#include "hypervector.hpp"
#include <vector>
#include <functional>

namespace hdc {

/**
 * Core HDC operations and utilities
 */
class Operations {
public:
    // Bundling operations
    static HyperVector majority_bundle(const std::vector<HyperVector>& vectors);
    static HyperVector weighted_bundle(const std::vector<HyperVector>& vectors, 
                                     const std::vector<double>& weights);
    
    // Binding operations  
    static HyperVector elementwise_bind(const HyperVector& a, const HyperVector& b);
    static HyperVector circular_bind(const HyperVector& a, const HyperVector& b);
    
    // Permutation operations
    static HyperVector left_rotate(const HyperVector& v, int positions);
    static HyperVector right_rotate(const HyperVector& v, int positions);
    
    // Similarity measurements
    static double cosine_similarity(const HyperVector& a, const HyperVector& b);
    static double hamming_similarity(const HyperVector& a, const HyperVector& b);
    static double jaccard_similarity(const HyperVector& a, const HyperVector& b);
    
    // Cleanup and thresholding
    static HyperVector cleanup(const HyperVector& v, double threshold = 0.0);
    static HyperVector soft_threshold(const HyperVector& v, double threshold = 0.1);
    
    // Noise operations
    static HyperVector add_noise(const HyperVector& v, double noise_ratio = 0.1, uint32_t seed = 0);
    static HyperVector flip_bits(const HyperVector& v, int num_flips, uint32_t seed = 0);
    
    // Sequence operations
    static HyperVector encode_sequence(const std::vector<HyperVector>& sequence,
                                     bool use_positions = true);
    static HyperVector decode_sequence_element(const HyperVector& sequence_hv,
                                             const std::vector<HyperVector>& candidates,
                                             int position);
    
    // Set operations
    static HyperVector set_union(const std::vector<HyperVector>& vectors);
    static HyperVector set_intersection(const std::vector<HyperVector>& vectors, 
                                       double threshold = 0.5);
    
    // Statistical operations
    static double entropy(const HyperVector& v);
    static std::vector<int> get_active_dimensions(const HyperVector& v);
    static double sparsity(const HyperVector& v);
    
    // Compression operations
    static HyperVector compress(const HyperVector& v, int target_dimension);
    static HyperVector decompress(const HyperVector& v, int target_dimension);
    
    // Distance metrics
    static double euclidean_distance(const HyperVector& a, const HyperVector& b);
    static double manhattan_distance(const HyperVector& a, const HyperVector& b);
    static double chebyshev_distance(const HyperVector& a, const HyperVector& b);
    
    // SIMD-optimized operations for performance enhancement
    static HyperVector simd_bundle_vectors(const std::vector<HyperVector>& vectors);
    static double simd_cosine_similarity(const HyperVector& a, const HyperVector& b);
    
    // Parallel processing operations for scalability
    static HyperVector parallel_bundle(const std::vector<HyperVector>& vectors, int num_threads = 4);
};

/**
 * HDC operations with GPU acceleration when available
 */
#ifdef WITH_CUDA
class CudaOperations {
public:
    static HyperVector bundle_cuda(const std::vector<HyperVector>& vectors);
    static HyperVector bind_cuda(const HyperVector& a, const HyperVector& b);
    static double similarity_cuda(const HyperVector& a, const HyperVector& b);
    static std::vector<double> batch_similarity_cuda(const HyperVector& query,
                                                    const std::vector<HyperVector>& candidates);
    
    // Memory management
    static void initialize_cuda();
    static void cleanup_cuda();
    static bool is_cuda_available();
};
#endif

/**
 * Position vectors for sequence encoding
 */
class PositionVectors {
public:
    explicit PositionVectors(int dimension = HyperVector::DEFAULT_DIMENSION, 
                           int max_positions = 1000);
    
    HyperVector get_position_vector(int position) const;
    void precompute_positions(int count);
    
private:
    int dimension_;
    std::vector<HyperVector> position_vectors_;
    std::mt19937 rng_;
};

/**
 * Basis vectors for encoding different data types
 */
class BasisVectors {
public:
    explicit BasisVectors(int dimension = HyperVector::DEFAULT_DIMENSION);
    
    // Numeric encoding
    HyperVector encode_integer(int value, int min_val = -1000, int max_val = 1000);
    HyperVector encode_float(double value, double min_val = -10.0, double max_val = 10.0, 
                           int precision = 100);
    
    // Categorical encoding
    HyperVector encode_category(const std::string& category);
    void register_categories(const std::vector<std::string>& categories);
    
    // Spatial encoding
    HyperVector encode_2d_position(double x, double y, double resolution = 0.1);
    HyperVector encode_3d_position(double x, double y, double z, double resolution = 0.1);
    
    // Angular encoding
    HyperVector encode_angle(double angle_rad);
    HyperVector encode_quaternion(double w, double x, double y, double z);
    
private:
    int dimension_;
    std::mt19937 rng_;
    std::unordered_map<std::string, HyperVector> category_vectors_;
    std::vector<HyperVector> integer_basis_;
    std::vector<HyperVector> float_basis_;
};

} // namespace hdc