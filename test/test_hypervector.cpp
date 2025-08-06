#include <gtest/gtest.h>
#include "hypervector.hpp"
#include "operations.hpp"
#include <vector>
#include <random>
#include <chrono>

using namespace hdc;

class HyperVectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up test fixtures
        dimension = 1000;
        hv1 = HyperVector::random(dimension, 42);
        hv2 = HyperVector::random(dimension, 43);
        hv_zero = HyperVector::zero(dimension);
    }
    
    int dimension;
    HyperVector hv1, hv2, hv_zero;
};

// Basic Construction Tests
TEST_F(HyperVectorTest, ConstructionTest) {
    EXPECT_EQ(hv1.dimension(), dimension);
    EXPECT_EQ(hv2.dimension(), dimension);
    EXPECT_EQ(hv_zero.dimension(), dimension);
    
    // Test zero vector
    EXPECT_TRUE(hv_zero.is_zero_vector());
    EXPECT_FALSE(hv1.is_zero_vector());
}

TEST_F(HyperVectorTest, InvalidDimensionTest) {
    EXPECT_THROW(HyperVector(-1), std::invalid_argument);
    EXPECT_THROW(HyperVector(0), std::invalid_argument);
}

// Bundle Operation Tests
TEST_F(HyperVectorTest, BundleTest) {
    HyperVector result = hv1.bundle(hv2);
    EXPECT_EQ(result.dimension(), dimension);
    
    // Bundle should be different from both inputs
    EXPECT_LT(result.similarity(hv1), 1.0);
    EXPECT_LT(result.similarity(hv2), 1.0);
}

TEST_F(HyperVectorTest, BundleAssociativityTest) {
    HyperVector hv3 = HyperVector::random(dimension, 44);
    
    HyperVector result1 = hv1.bundle(hv2).bundle(hv3);
    HyperVector result2 = hv1.bundle(hv2.bundle(hv3));
    
    // Bundling should be approximately associative
    double similarity = result1.similarity(result2);
    EXPECT_GT(similarity, 0.8); // High similarity expected
}

TEST_F(HyperVectorTest, BundleCommutativityTest) {
    HyperVector result1 = hv1.bundle(hv2);
    HyperVector result2 = hv2.bundle(hv1);
    
    // Bundling should be commutative
    EXPECT_TRUE(result1 == result2);
}

TEST_F(HyperVectorTest, BundleDimensionMismatchTest) {
    HyperVector hv_small = HyperVector::random(500, 45);
    EXPECT_THROW(hv1.bundle(hv_small), std::invalid_argument);
}

// Bind Operation Tests
TEST_F(HyperVectorTest, BindTest) {
    HyperVector result = hv1.bind(hv2);
    EXPECT_EQ(result.dimension(), dimension);
    
    // Bind should produce different vector
    EXPECT_LT(result.similarity(hv1), 1.0);
    EXPECT_LT(result.similarity(hv2), 1.0);
}

TEST_F(HyperVectorTest, BindInverseTest) {
    HyperVector bound = hv1.bind(hv2);
    HyperVector unbound = bound.bind(hv2); // Unbind by binding with same vector
    
    // Should be similar to original (but not identical due to thresholding)
    double similarity = hv1.similarity(unbound);
    EXPECT_GT(similarity, 0.7);
}

TEST_F(HyperVectorTest, BindCommutativityTest) {
    HyperVector result1 = hv1.bind(hv2);
    HyperVector result2 = hv2.bind(hv1);
    
    // Binding should be commutative
    EXPECT_TRUE(result1 == result2);
}

TEST_F(HyperVectorTest, BindDistributivityTest) {
    HyperVector hv3 = HyperVector::random(dimension, 44);
    
    // Test distributivity: A * (B + C) ≈ (A * B) + (A * C)
    HyperVector bundled = hv2.bundle(hv3);
    HyperVector left_side = hv1.bind(bundled);
    
    HyperVector bound1 = hv1.bind(hv2);
    HyperVector bound2 = hv1.bind(hv3);
    HyperVector right_side = bound1.bundle(bound2);
    
    double similarity = left_side.similarity(right_side);
    EXPECT_GT(similarity, 0.5); // Should be reasonably similar
}

// Permutation Tests
TEST_F(HyperVectorTest, PermuteTest) {
    HyperVector permuted = hv1.permute(10);
    EXPECT_EQ(permuted.dimension(), dimension);
    
    // Permuted vector should have same sparsity but different structure
    EXPECT_NE(permuted, hv1);
    
    // Permuting back should give original
    HyperVector back = permuted.permute(-10);
    EXPECT_TRUE(back == hv1);
}

TEST_F(HyperVectorTest, PermuteZeroTest) {
    HyperVector result = hv1.permute(0);
    EXPECT_TRUE(result == hv1);
}

TEST_F(HyperVectorTest, PermuteWrapAroundTest) {
    HyperVector result1 = hv1.permute(dimension);
    HyperVector result2 = hv1.permute(0);
    EXPECT_TRUE(result1 == result2);
}

// Inversion Tests
TEST_F(HyperVectorTest, InvertTest) {
    HyperVector inverted = hv1.invert();
    EXPECT_EQ(inverted.dimension(), dimension);
    
    // Self-similarity should be -1
    double similarity = hv1.similarity(inverted);
    EXPECT_NEAR(similarity, -1.0, 0.01);
    
    // Double inversion should give original
    HyperVector double_inverted = inverted.invert();
    EXPECT_TRUE(double_inverted == hv1);
}

// Similarity Tests
TEST_F(HyperVectorTest, SimilarityTest) {
    // Self-similarity should be 1
    EXPECT_NEAR(hv1.similarity(hv1), 1.0, 0.01);
    
    // Similarity should be symmetric
    double sim1 = hv1.similarity(hv2);
    double sim2 = hv2.similarity(hv1);
    EXPECT_NEAR(sim1, sim2, 0.01);
    
    // Similarity should be in [-1, 1]
    EXPECT_GE(sim1, -1.0);
    EXPECT_LE(sim1, 1.0);
}

TEST_F(HyperVectorTest, SimilarityDimensionMismatchTest) {
    HyperVector hv_small = HyperVector::random(500, 45);
    EXPECT_THROW(hv1.similarity(hv_small), std::invalid_argument);
}

// Hamming Distance Tests
TEST_F(HyperVectorTest, HammingDistanceTest) {
    // Self-distance should be 0
    EXPECT_NEAR(hv1.hamming_distance(hv1), 0.0, 0.01);
    
    // Distance should be symmetric
    double dist1 = hv1.hamming_distance(hv2);
    double dist2 = hv2.hamming_distance(hv1);
    EXPECT_NEAR(dist1, dist2, 0.01);
    
    // Distance should be in [0, 1]
    EXPECT_GE(dist1, 0.0);
    EXPECT_LE(dist1, 1.0);
}

// Bundle Vectors Tests
TEST_F(HyperVectorTest, BundleVectorsTest) {
    std::vector<HyperVector> vectors = {hv1, hv2};
    HyperVector result = HyperVector::bundle_vectors(vectors);
    
    EXPECT_EQ(result.dimension(), dimension);
    
    // Should be similar to individual bundling
    HyperVector manual_bundle = hv1.bundle(hv2);
    EXPECT_TRUE(result == manual_bundle);
}

TEST_F(HyperVectorTest, BundleVectorsEmptyTest) {
    std::vector<HyperVector> empty_vectors;
    EXPECT_THROW(HyperVector::bundle_vectors(empty_vectors), std::invalid_argument);
}

TEST_F(HyperVectorTest, BundleVectorsDimensionMismatchTest) {
    HyperVector hv_small = HyperVector::random(500, 45);
    std::vector<HyperVector> vectors = {hv1, hv_small};
    EXPECT_THROW(HyperVector::bundle_vectors(vectors), std::invalid_argument);
}

// Serialization Tests
TEST_F(HyperVectorTest, SerializationTest) {
    auto bytes = hv1.to_bytes();
    EXPECT_FALSE(bytes.empty());
    
    HyperVector recovered(dimension);
    recovered.from_bytes(bytes);
    
    EXPECT_TRUE(recovered == hv1);
}

TEST_F(HyperVectorTest, SerializationDimensionMismatchTest) {
    auto bytes = hv1.to_bytes();
    HyperVector hv_different(dimension / 2);
    EXPECT_THROW(hv_different.from_bytes(bytes), std::invalid_argument);
}

// Operations Class Tests
TEST_F(HyperVectorTest, OperationsTest) {
    // Test majority bundle
    std::vector<HyperVector> vectors = {hv1, hv2};
    HyperVector result1 = Operations::majority_bundle(vectors);
    HyperVector result2 = HyperVector::bundle_vectors(vectors);
    EXPECT_TRUE(result1 == result2);
    
    // Test weighted bundle
    std::vector<double> weights = {0.7, 0.3};
    HyperVector weighted_result = Operations::weighted_bundle(vectors, weights);
    EXPECT_EQ(weighted_result.dimension(), dimension);
    
    // Test element-wise bind
    HyperVector bind_result = Operations::elementwise_bind(hv1, hv2);
    HyperVector manual_bind = hv1.bind(hv2);
    EXPECT_TRUE(bind_result == manual_bind);
}

// Noise Tests
TEST_F(HyperVectorTest, NoiseTest) {
    double noise_ratio = 0.1;
    HyperVector noisy = Operations::add_noise(hv1, noise_ratio, 123);
    
    // Should still be similar but not identical
    double similarity = hv1.similarity(noisy);
    EXPECT_GT(similarity, 0.8);  // Should be quite similar
    EXPECT_LT(similarity, 1.0);  // But not identical
    
    // Test with different noise ratios
    HyperVector very_noisy = Operations::add_noise(hv1, 0.5, 124);
    double low_similarity = hv1.similarity(very_noisy);
    EXPECT_LT(low_similarity, similarity); // More noise = less similar
}

TEST_F(HyperVectorTest, FlipBitsTest) {
    int num_flips = 100;
    HyperVector flipped = Operations::flip_bits(hv1, num_flips, 125);
    
    // Should have exactly num_flips differences
    double hamming_dist = hv1.hamming_distance(flipped);
    double expected_ratio = static_cast<double>(num_flips) / dimension;
    EXPECT_NEAR(hamming_dist, expected_ratio, 0.01);
}

// Sequence Encoding Tests
TEST_F(HyperVectorTest, SequenceEncodingTest) {
    std::vector<HyperVector> sequence = {hv1, hv2};
    HyperVector seq_encoded = create_sequence(sequence);
    
    EXPECT_EQ(seq_encoded.dimension(), dimension);
    
    // Sequence encoding should be different from simple bundling
    HyperVector simple_bundle = hv1.bundle(hv2);
    double similarity = seq_encoded.similarity(simple_bundle);
    EXPECT_LT(similarity, 0.9); // Should be different
}

TEST_F(HyperVectorTest, NGramEncodingTest) {
    std::vector<HyperVector> sequence = {hv1, hv2, HyperVector::random(dimension, 46)};
    HyperVector ngram = create_ngram(sequence, 2);
    
    EXPECT_EQ(ngram.dimension(), dimension);
}

TEST_F(HyperVectorTest, NGramTooShortTest) {
    std::vector<HyperVector> short_sequence = {hv1};
    EXPECT_THROW(create_ngram(short_sequence, 3), std::invalid_argument);
}

// Performance Tests
TEST_F(HyperVectorTest, PerformanceTest) {
    const int num_operations = 1000;
    auto start = std::chrono::high_resolution_clock::now();
    
    // Perform many operations
    HyperVector result = hv1;
    for (int i = 0; i < num_operations; ++i) {
        result = result.bundle(hv2);
        result = result.bind(hv1);
        result = result.permute(1);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    std::cout << "Performance test: " << num_operations << " operations took " 
              << duration.count() << " microseconds" << std::endl;
    
    // Should complete within reasonable time (< 100ms for 1000 ops)
    EXPECT_LT(duration.count(), 100000);
}

// Statistical Properties Tests
TEST_F(HyperVectorTest, StatisticalPropertiesTest) {
    // Test entropy
    double entropy = Operations::entropy(hv1);
    EXPECT_GT(entropy, 0.9); // Should be close to 1 for random vectors
    
    // Test sparsity
    double sparsity = Operations::sparsity(hv1);
    EXPECT_GT(sparsity, 0.4);
    EXPECT_LT(sparsity, 0.6); // Should be around 0.5 for random bipolar vectors
    
    // Test active dimensions
    auto active_dims = Operations::get_active_dimensions(hv1);
    EXPECT_EQ(active_dims.size(), static_cast<size_t>(dimension * sparsity));
}

// Distance Metrics Tests
TEST_F(HyperVectorTest, DistanceMetricsTest) {
    double euclidean = Operations::euclidean_distance(hv1, hv2);
    double manhattan = Operations::manhattan_distance(hv1, hv2);
    double chebyshev = Operations::chebyshev_distance(hv1, hv2);
    
    EXPECT_GT(euclidean, 0.0);
    EXPECT_GT(manhattan, 0.0);
    EXPECT_GT(chebyshev, 0.0);
    
    // Self-distances should be zero
    EXPECT_NEAR(Operations::euclidean_distance(hv1, hv1), 0.0, 0.01);
    EXPECT_NEAR(Operations::manhattan_distance(hv1, hv1), 0.0, 0.01);
    EXPECT_NEAR(Operations::chebyshev_distance(hv1, hv1), 0.0, 0.01);
}

// Weighted Bundle Tests
TEST_F(HyperVectorTest, WeightedBundleTest) {
    std::vector<std::pair<HyperVector, double>> weighted_vectors = {
        {hv1, 0.8},
        {hv2, 0.2}
    };
    
    HyperVector result = weighted_bundle(weighted_vectors);
    EXPECT_EQ(result.dimension(), dimension);
    
    // Should be more similar to hv1 (higher weight)
    double sim1 = result.similarity(hv1);
    double sim2 = result.similarity(hv2);
    EXPECT_GT(sim1, sim2);
}

// Random Vector Properties Tests
TEST_F(HyperVectorTest, RandomVectorPropertiesTest) {
    const int num_vectors = 100;
    std::vector<HyperVector> random_vectors;
    
    // Generate many random vectors
    for (int i = 0; i < num_vectors; ++i) {
        random_vectors.push_back(HyperVector::random(dimension, i + 1000));
    }
    
    // Calculate pairwise similarities
    std::vector<double> similarities;
    for (int i = 0; i < num_vectors; ++i) {
        for (int j = i + 1; j < num_vectors; ++j) {
            similarities.push_back(random_vectors[i].similarity(random_vectors[j]));
        }
    }
    
    // Calculate mean and variance
    double mean = std::accumulate(similarities.begin(), similarities.end(), 0.0) / similarities.size();
    double variance = 0.0;
    for (double sim : similarities) {
        variance += (sim - mean) * (sim - mean);
    }
    variance /= similarities.size();
    
    // Random vectors should have mean similarity near 0
    EXPECT_NEAR(mean, 0.0, 0.1);
    
    // Standard deviation should be roughly 1/sqrt(dimension)
    double expected_std = 1.0 / std::sqrt(dimension);
    double actual_std = std::sqrt(variance);
    EXPECT_NEAR(actual_std, expected_std, expected_std * 0.5);
    
    std::cout << "Random vector statistics: mean=" << mean 
              << ", std=" << actual_std << ", expected_std=" << expected_std << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}