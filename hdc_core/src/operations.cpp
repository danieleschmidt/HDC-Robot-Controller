#include "operations.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <unordered_map>

namespace hdc {

HyperVector Operations::majority_bundle(const std::vector<HyperVector>& vectors) {
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
    return a.similarity(b); // Already implements normalized dot product
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

} // namespace hdc