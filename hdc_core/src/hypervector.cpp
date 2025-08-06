#include "hypervector.hpp"
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>
#include <cmath>

namespace hdc {

std::mt19937 HyperVector::rng_(std::random_device{}());

HyperVector::HyperVector(int dimension) : data_(dimension, 0) {
    if (dimension <= 0) {
        throw std::invalid_argument("Dimension must be positive");
    }
}

HyperVector::HyperVector(const std::vector<int8_t>& data) : data_(data) {
    if (data.empty()) {
        throw std::invalid_argument("Data cannot be empty");
    }
}

HyperVector HyperVector::bundle(const HyperVector& other) const {
    if (dimension() != other.dimension()) {
        throw std::invalid_argument("Dimensions must match for bundling");
    }
    
    HyperVector result(dimension());
    for (int i = 0; i < dimension(); ++i) {
        int sum = data_[i] + other.data_[i];
        result.data_[i] = (sum > 0) ? 1 : -1;
    }
    
    return result;
}

HyperVector HyperVector::bind(const HyperVector& other) const {
    if (dimension() != other.dimension()) {
        throw std::invalid_argument("Dimensions must match for binding");
    }
    
    HyperVector result(dimension());
    for (int i = 0; i < dimension(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    
    return result;
}

HyperVector HyperVector::permute(int shift) const {
    HyperVector result(dimension());
    const int dim = dimension();
    
    // Handle negative shifts
    shift = ((shift % dim) + dim) % dim;
    
    for (int i = 0; i < dim; ++i) {
        result.data_[(i + shift) % dim] = data_[i];
    }
    
    return result;
}

HyperVector HyperVector::invert() const {
    HyperVector result(dimension());
    for (int i = 0; i < dimension(); ++i) {
        result.data_[i] = -data_[i];
    }
    return result;
}

double HyperVector::similarity(const HyperVector& other) const {
    if (dimension() != other.dimension()) {
        throw std::invalid_argument("Dimensions must match for similarity");
    }
    
    int dot_product = 0;
    for (int i = 0; i < dimension(); ++i) {
        dot_product += data_[i] * other.data_[i];
    }
    
    return static_cast<double>(dot_product) / dimension();
}

double HyperVector::hamming_distance(const HyperVector& other) const {
    if (dimension() != other.dimension()) {
        throw std::invalid_argument("Dimensions must match for Hamming distance");
    }
    
    int differences = 0;
    for (int i = 0; i < dimension(); ++i) {
        if (data_[i] != other.data_[i]) {
            differences++;
        }
    }
    
    return static_cast<double>(differences) / dimension();
}

HyperVector HyperVector::random(int dimension, uint32_t seed) {
    if (seed != 0) {
        rng_.seed(seed);
    }
    
    HyperVector result(dimension);
    std::uniform_int_distribution<int> dist(0, 1);
    
    for (int i = 0; i < dimension; ++i) {
        result.data_[i] = (dist(rng_) == 0) ? -1 : 1;
    }
    
    return result;
}

HyperVector HyperVector::zero(int dimension) {
    return HyperVector(dimension); // Already initialized to zeros
}

HyperVector HyperVector::bundle_vectors(const std::vector<HyperVector>& vectors) {
    if (vectors.empty()) {
        throw std::invalid_argument("Cannot bundle empty vector list");
    }
    
    const int dim = vectors[0].dimension();
    std::vector<int> sums(dim, 0);
    
    // Sum all vectors
    for (const auto& vec : vectors) {
        if (vec.dimension() != dim) {
            throw std::invalid_argument("All vectors must have same dimension");
        }
        
        for (int i = 0; i < dim; ++i) {
            sums[i] += vec.data_[i];
        }
    }
    
    // Apply majority rule
    HyperVector result(dim);
    for (int i = 0; i < dim; ++i) {
        result.data_[i] = (sums[i] > 0) ? 1 : -1;
    }
    
    return result;
}

void HyperVector::threshold() {
    for (auto& val : data_) {
        val = (val > 0) ? 1 : -1;
    }
}

void HyperVector::normalize() {
    // For bipolar vectors, normalization is just thresholding
    threshold();
}

bool HyperVector::is_zero_vector() const {
    return std::all_of(data_.begin(), data_.end(), [](int8_t val) { return val == 0; });
}

std::vector<uint8_t> HyperVector::to_bytes() const {
    std::vector<uint8_t> bytes;
    bytes.reserve((dimension() + 7) / 8); // Ceiling division
    
    for (int i = 0; i < dimension(); i += 8) {
        uint8_t byte = 0;
        for (int j = 0; j < 8 && (i + j) < dimension(); ++j) {
            if (data_[i + j] > 0) {
                byte |= (1 << j);
            }
        }
        bytes.push_back(byte);
    }
    
    return bytes;
}

void HyperVector::from_bytes(const std::vector<uint8_t>& bytes) {
    const int expected_bytes = (dimension() + 7) / 8;
    if (static_cast<int>(bytes.size()) != expected_bytes) {
        throw std::invalid_argument("Byte array size doesn't match vector dimension");
    }
    
    for (int i = 0; i < dimension(); ++i) {
        const int byte_idx = i / 8;
        const int bit_idx = i % 8;
        data_[i] = (bytes[byte_idx] & (1 << bit_idx)) ? 1 : -1;
    }
}

std::string HyperVector::to_string(int max_elements) const {
    std::ostringstream oss;
    oss << "HyperVector(dim=" << dimension() << ", [";
    
    const int show_count = std::min(max_elements, dimension());
    for (int i = 0; i < show_count; ++i) {
        oss << static_cast<int>(data_[i]);
        if (i < show_count - 1) oss << ", ";
    }
    
    if (show_count < dimension()) {
        oss << ", ...";
    }
    
    oss << "])";
    return oss.str();
}

bool HyperVector::operator==(const HyperVector& other) const {
    return data_ == other.data_;
}

bool HyperVector::operator!=(const HyperVector& other) const {
    return !(*this == other);
}

HyperVector weighted_bundle(const std::vector<std::pair<HyperVector, double>>& weighted_vectors) {
    if (weighted_vectors.empty()) {
        throw std::invalid_argument("Cannot bundle empty weighted vector list");
    }
    
    const int dim = weighted_vectors[0].first.dimension();
    std::vector<double> sums(dim, 0.0);
    double total_weight = 0.0;
    
    // Weighted sum
    for (const auto& [vec, weight] : weighted_vectors) {
        if (vec.dimension() != dim) {
            throw std::invalid_argument("All vectors must have same dimension");
        }
        
        for (int i = 0; i < dim; ++i) {
            sums[i] += vec.data()[i] * weight;
        }
        total_weight += weight;
    }
    
    // Normalize and threshold
    HyperVector result(dim);
    for (int i = 0; i < dim; ++i) {
        result[i] = (sums[i] > 0) ? 1 : -1;
    }
    
    return result;
}

HyperVector create_sequence(const std::vector<HyperVector>& sequence) {
    if (sequence.empty()) {
        throw std::invalid_argument("Cannot create sequence from empty vector list");
    }
    
    const int dim = sequence[0].dimension();
    HyperVector result = HyperVector::zero(dim);
    
    for (size_t i = 0; i < sequence.size(); ++i) {
        // Create position vector (simple rotation)
        HyperVector pos_vec = HyperVector::random(dim, static_cast<uint32_t>(i + 1));
        
        // Bind element with position and bundle into result
        HyperVector bound = sequence[i].bind(pos_vec);
        result = result.bundle(bound);
    }
    
    return result;
}

HyperVector create_ngram(const std::vector<HyperVector>& sequence, int n) {
    if (static_cast<int>(sequence.size()) < n) {
        throw std::invalid_argument("Sequence too short for n-gram");
    }
    
    std::vector<HyperVector> ngrams;
    for (size_t i = 0; i <= sequence.size() - n; ++i) {
        std::vector<HyperVector> gram(sequence.begin() + i, sequence.begin() + i + n);
        ngrams.push_back(create_sequence(gram));
    }
    
    return HyperVector::bundle_vectors(ngrams);
}

} // namespace hdc