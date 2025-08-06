#pragma once

#include <vector>
#include <cstdint>
#include <random>
#include <memory>
#include <string>

namespace hdc {

/**
 * Core hyperdimensional vector class for HDC operations
 * Uses bipolar representation (-1, +1) for efficiency
 */
class HyperVector {
public:
    static constexpr int DEFAULT_DIMENSION = 10000;
    
    // Constructors
    explicit HyperVector(int dimension = DEFAULT_DIMENSION);
    HyperVector(const std::vector<int8_t>& data);
    HyperVector(const HyperVector& other) = default;
    HyperVector(HyperVector&& other) = default;
    
    // Assignment operators
    HyperVector& operator=(const HyperVector& other) = default;
    HyperVector& operator=(HyperVector&& other) = default;
    
    // Core HDC operations
    HyperVector bundle(const HyperVector& other) const;
    HyperVector bind(const HyperVector& other) const;
    HyperVector permute(int shift = 1) const;
    HyperVector invert() const;
    
    // Similarity and distance
    double similarity(const HyperVector& other) const;
    double hamming_distance(const HyperVector& other) const;
    
    // Static factory methods
    static HyperVector random(int dimension = DEFAULT_DIMENSION, uint32_t seed = 0);
    static HyperVector zero(int dimension = DEFAULT_DIMENSION);
    static HyperVector bundle_vectors(const std::vector<HyperVector>& vectors);
    
    // Utility methods
    void threshold();
    void normalize();
    bool is_zero_vector() const;
    
    // Accessors
    int dimension() const { return static_cast<int>(data_.size()); }
    const std::vector<int8_t>& data() const { return data_; }
    std::vector<int8_t>& data() { return data_; }
    
    // Element access
    int8_t operator[](int index) const { return data_[index]; }
    int8_t& operator[](int index) { return data_[index]; }
    
    // Serialization
    std::vector<uint8_t> to_bytes() const;
    void from_bytes(const std::vector<uint8_t>& bytes);
    
    // String representation
    std::string to_string(int max_elements = 20) const;
    
    // Comparison operators
    bool operator==(const HyperVector& other) const;
    bool operator!=(const HyperVector& other) const;
    
private:
    std::vector<int8_t> data_;
    static std::mt19937 rng_;
};

/**
 * Bundle multiple vectors with weighted contributions
 */
HyperVector weighted_bundle(const std::vector<std::pair<HyperVector, double>>& weighted_vectors);

/**
 * Create sequence hypervector by binding vectors with positions
 */
HyperVector create_sequence(const std::vector<HyperVector>& sequence);

/**
 * N-gram encoding for temporal sequences
 */
HyperVector create_ngram(const std::vector<HyperVector>& sequence, int n = 3);

} // namespace hdc