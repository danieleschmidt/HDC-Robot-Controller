#include <gtest/gtest.h>
#include "operations.hpp"
#include "memory.hpp"
#include <vector>
#include <random>
#include <chrono>

using namespace hdc;

class OperationsTest : public ::testing::Test {
protected:
    void SetUp() override {
        dimension = 1000;
        hv1 = HyperVector::random(dimension, 42);
        hv2 = HyperVector::random(dimension, 43);
        hv3 = HyperVector::random(dimension, 44);
    }
    
    int dimension;
    HyperVector hv1, hv2, hv3;
};

// BasisVectors Tests
TEST_F(OperationsTest, BasisVectorIntegerEncodingTest) {
    BasisVectors basis(dimension);
    
    // Test integer encoding
    HyperVector int_vec1 = basis.encode_integer(42);
    HyperVector int_vec2 = basis.encode_integer(43);
    HyperVector int_vec3 = basis.encode_integer(42); // Same as first
    
    EXPECT_EQ(int_vec1.dimension(), dimension);
    EXPECT_TRUE(int_vec1 == int_vec3); // Same integers should give same vectors
    EXPECT_FALSE(int_vec1 == int_vec2); // Different integers should give different vectors
}

TEST_F(OperationsTest, BasisVectorFloatEncodingTest) {
    BasisVectors basis(dimension);
    
    HyperVector float_vec1 = basis.encode_float(3.14, 0.0, 10.0, 100);
    HyperVector float_vec2 = basis.encode_float(3.15, 0.0, 10.0, 100);
    HyperVector float_vec3 = basis.encode_float(3.14, 0.0, 10.0, 100);
    
    EXPECT_EQ(float_vec1.dimension(), dimension);
    EXPECT_TRUE(float_vec1 == float_vec3); // Same values should give same vectors
    // Different values might give same vectors due to discretization
}

TEST_F(OperationsTest, BasisVectorCategoryEncodingTest) {
    BasisVectors basis(dimension);
    
    HyperVector cat1 = basis.encode_category("robot");
    HyperVector cat2 = basis.encode_category("human");
    HyperVector cat3 = basis.encode_category("robot");
    
    EXPECT_EQ(cat1.dimension(), dimension);
    EXPECT_TRUE(cat1 == cat3);  // Same category should give same vector
    EXPECT_FALSE(cat1 == cat2); // Different categories should give different vectors
}

TEST_F(OperationsTest, BasisVectorSpatialEncodingTest) {
    BasisVectors basis(dimension);
    
    HyperVector pos1 = basis.encode_2d_position(1.0, 2.0, 0.1);
    HyperVector pos2 = basis.encode_2d_position(1.0, 2.0, 0.1);
    HyperVector pos3 = basis.encode_2d_position(1.1, 2.1, 0.1);
    
    EXPECT_TRUE(pos1 == pos2); // Same position should give same vector
    EXPECT_FALSE(pos1 == pos3); // Different positions should give different vectors
}

TEST_F(OperationsTest, BasisVector3DSpatialEncodingTest) {
    BasisVectors basis(dimension);
    
    HyperVector pos_3d = basis.encode_3d_position(1.0, 2.0, 3.0, 0.1);
    EXPECT_EQ(pos_3d.dimension(), dimension);
    
    // Test that different coordinates give different results
    HyperVector pos_3d_diff = basis.encode_3d_position(1.1, 2.0, 3.0, 0.1);
    EXPECT_FALSE(pos_3d == pos_3d_diff);
}

TEST_F(OperationsTest, BasisVectorAngleEncodingTest) {
    BasisVectors basis(dimension);
    
    HyperVector angle1 = basis.encode_angle(M_PI / 4);  // 45 degrees
    HyperVector angle2 = basis.encode_angle(M_PI / 2);  // 90 degrees
    HyperVector angle3 = basis.encode_angle(M_PI / 4);  // 45 degrees again
    
    EXPECT_TRUE(angle1 == angle3);
    EXPECT_FALSE(angle1 == angle2);
}

TEST_F(OperationsTest, BasisVectorQuaternionEncodingTest) {
    BasisVectors basis(dimension);
    
    // Unit quaternion representing no rotation
    HyperVector quat1 = basis.encode_quaternion(1.0, 0.0, 0.0, 0.0);
    // 180-degree rotation around x-axis
    HyperVector quat2 = basis.encode_quaternion(0.0, 1.0, 0.0, 0.0);
    
    EXPECT_EQ(quat1.dimension(), dimension);
    EXPECT_FALSE(quat1 == quat2);
}

// PositionVectors Tests
TEST_F(OperationsTest, PositionVectorsTest) {
    PositionVectors pos_vecs(dimension, 100);
    
    HyperVector pos0 = pos_vecs.get_position_vector(0);
    HyperVector pos1 = pos_vecs.get_position_vector(1);
    HyperVector pos0_again = pos_vecs.get_position_vector(0);
    
    EXPECT_EQ(pos0.dimension(), dimension);
    EXPECT_TRUE(pos0 == pos0_again); // Same position should give same vector
    EXPECT_FALSE(pos0 == pos1);      // Different positions should give different vectors
}

TEST_F(OperationsTest, PositionVectorsLargeIndexTest) {
    PositionVectors pos_vecs(dimension, 10); // Small precomputed size
    
    // This should work even though index > precomputed size
    HyperVector pos_large = pos_vecs.get_position_vector(50);
    EXPECT_EQ(pos_large.dimension(), dimension);
}

// Advanced Operations Tests
TEST_F(OperationsTest, CircularBindTest) {
    HyperVector result = Operations::circular_bind(hv1, hv2);
    EXPECT_EQ(result.dimension(), dimension);
    
    // Should be different from element-wise bind
    HyperVector elementwise_result = Operations::elementwise_bind(hv1, hv2);
    EXPECT_FALSE(result == elementwise_result);
}

TEST_F(OperationsTest, RotationTest) {
    int shift = 10;
    HyperVector left_rotated = Operations::left_rotate(hv1, shift);
    HyperVector right_rotated = Operations::right_rotate(hv1, shift);
    
    EXPECT_EQ(left_rotated.dimension(), dimension);
    EXPECT_EQ(right_rotated.dimension(), dimension);
    
    // Left and right rotation should be inverses
    HyperVector back_to_original = Operations::right_rotate(left_rotated, shift);
    EXPECT_TRUE(back_to_original == hv1);
}

TEST_F(OperationsTest, SimilarityMetricsTest) {
    double cosine_sim = Operations::cosine_similarity(hv1, hv2);
    double hamming_sim = Operations::hamming_similarity(hv1, hv2);
    double jaccard_sim = Operations::jaccard_similarity(hv1, hv2);
    
    // All similarities should be in valid ranges
    EXPECT_GE(cosine_sim, -1.0);
    EXPECT_LE(cosine_sim, 1.0);
    EXPECT_GE(hamming_sim, 0.0);
    EXPECT_LE(hamming_sim, 1.0);
    EXPECT_GE(jaccard_sim, 0.0);
    EXPECT_LE(jaccard_sim, 1.0);
    
    // Self-similarity tests
    EXPECT_NEAR(Operations::cosine_similarity(hv1, hv1), 1.0, 0.01);
    EXPECT_NEAR(Operations::hamming_similarity(hv1, hv1), 1.0, 0.01);
}

TEST_F(OperationsTest, CleanupOperationsTest) {
    HyperVector noisy = hv1;
    // Add some "noise" by setting some elements to zero
    for (int i = 0; i < 100; ++i) {
        noisy[i] = 0;
    }
    
    HyperVector cleaned = Operations::cleanup(noisy);
    EXPECT_EQ(cleaned.dimension(), dimension);
    
    // All elements should be bipolar after cleanup
    for (int i = 0; i < dimension; ++i) {
        EXPECT_TRUE(cleaned[i] == 1 || cleaned[i] == -1);
    }
}

TEST_F(OperationsTest, SetOperationsTest) {
    std::vector<HyperVector> vectors = {hv1, hv2, hv3};
    
    HyperVector set_union = Operations::set_union(vectors);
    HyperVector set_intersection = Operations::set_intersection(vectors, 0.7);
    
    EXPECT_EQ(set_union.dimension(), dimension);
    EXPECT_EQ(set_intersection.dimension(), dimension);
    
    // Union should be same as bundling for HDC
    HyperVector manual_bundle = hv1.bundle(hv2).bundle(hv3);
    EXPECT_TRUE(set_union == manual_bundle);
}

TEST_F(OperationsTest, SequenceOperationsTest) {
    std::vector<HyperVector> sequence = {hv1, hv2, hv3};
    
    HyperVector with_positions = Operations::encode_sequence(sequence, true);
    HyperVector without_positions = Operations::encode_sequence(sequence, false);
    
    EXPECT_EQ(with_positions.dimension(), dimension);
    EXPECT_EQ(without_positions.dimension(), dimension);
    
    // Should be different encoding methods
    double similarity = with_positions.similarity(without_positions);
    EXPECT_LT(similarity, 0.9);
}

TEST_F(OperationsTest, SequenceDecodingTest) {
    std::vector<HyperVector> sequence = {hv1, hv2, hv3};
    std::vector<HyperVector> candidates = {hv1, hv2, hv3};
    
    HyperVector encoded = Operations::encode_sequence(sequence);
    
    // Try to decode first element
    HyperVector decoded = Operations::decode_sequence_element(encoded, candidates, 0);
    
    // Should be one of the candidates
    bool found_match = false;
    for (const auto& candidate : candidates) {
        if (decoded == candidate) {
            found_match = true;
            break;
        }
    }
    EXPECT_TRUE(found_match);
}

// Memory Tests
TEST_F(OperationsTest, AssociativeMemoryTest) {
    AssociativeMemory memory(dimension, 0.7);
    
    memory.store("vector1", hv1, 0.9);
    memory.store("vector2", hv2, 0.8);
    
    EXPECT_TRUE(memory.contains("vector1"));
    EXPECT_TRUE(memory.contains("vector2"));
    EXPECT_FALSE(memory.contains("nonexistent"));
    EXPECT_EQ(memory.size(), 2);
}

TEST_F(OperationsTest, AssociativeMemoryQueryTest) {
    AssociativeMemory memory(dimension, 0.7);
    
    memory.store("vector1", hv1, 0.9);
    memory.store("vector2", hv2, 0.8);
    memory.store("vector3", hv3, 0.7);
    
    // Query with exact match
    auto results = memory.query(hv1, 3);
    EXPECT_GE(results.size(), 1);
    EXPECT_EQ(results[0].label, "vector1"); // Best match should be exact match
    EXPECT_NEAR(results[0].similarity, 1.0, 0.01);
    
    // Query with threshold
    auto threshold_results = memory.query_threshold(hv1, 0.9);
    EXPECT_GE(threshold_results.size(), 1); // At least the exact match
}

TEST_F(OperationsTest, AssociativeMemoryUpdateTest) {
    AssociativeMemory memory(dimension, 0.7);
    
    memory.store("test", hv1, 0.5);
    double initial_confidence = memory.get_confidence("test");
    
    memory.store_with_update("test", hv2, 0.1); // Small learning rate
    double updated_confidence = memory.get_confidence("test");
    
    EXPECT_GT(updated_confidence, initial_confidence);
}

TEST_F(OperationsTest, EpisodicMemoryTest) {
    EpisodicMemory episodic(dimension, 100);
    
    std::vector<HyperVector> episode1 = {hv1, hv2};
    std::vector<HyperVector> episode2 = {hv2, hv3};
    std::vector<std::string> labels = {"action1", "action2"};
    
    episodic.store_episode(episode1, labels, 0.9);
    episodic.store_episode(episode2, labels, 0.8);
    
    EXPECT_EQ(episodic.size(), 2);
    
    // Query similar episodes
    auto similar = episodic.query_similar_episodes(episode1, 0.5);
    EXPECT_GE(similar.size(), 1); // Should find at least the exact match
}

TEST_F(OperationsTest, WorkingMemoryTest) {
    WorkingMemory working(dimension, 10);
    
    EXPECT_TRUE(working.empty());
    EXPECT_EQ(working.size(), 0);
    
    working.push(hv1, "context1");
    working.push(hv2, "context2");
    
    EXPECT_FALSE(working.empty());
    EXPECT_EQ(working.size(), 2);
    
    HyperVector popped = working.pop();
    EXPECT_TRUE(popped == hv2); // LIFO behavior
    EXPECT_EQ(working.size(), 1);
}

TEST_F(OperationsTest, WorkingMemoryPatternDetectionTest) {
    WorkingMemory working(dimension, 10);
    
    // Create a pattern
    working.push(hv1);
    working.push(hv2);
    working.push(hv3);
    
    auto patterns = working.detect_patterns(2); // 2-grams
    EXPECT_EQ(patterns.size(), 2); // Should detect 2 patterns of length 2
    
    // Test pattern recognition
    std::vector<HyperVector> test_pattern = {hv1, hv2};
    EXPECT_TRUE(working.has_pattern(test_pattern));
    
    std::vector<HyperVector> non_pattern = {hv3, hv1};
    EXPECT_FALSE(working.has_pattern(non_pattern));
}

TEST_F(OperationsTest, HierarchicalMemoryTest) {
    HierarchicalMemory hierarchical(dimension);
    
    // Store experience
    hierarchical.store_experience(hv1, hv2, "experience1", 0.9);
    
    EXPECT_EQ(hierarchical.get_associative_memory().size(), 1);
    EXPECT_EQ(hierarchical.get_episodic_memory().size(), 1);
    EXPECT_EQ(hierarchical.get_working_memory().size(), 1);
    
    // Query experience
    HyperVector query = hv1.bind(hv2);
    auto results = hierarchical.query_experience(query, 5);
    EXPECT_GE(results.size(), 1);
}

// Performance Tests
TEST_F(OperationsTest, BundlePerformanceTest) {
    const int num_vectors = 100;
    std::vector<HyperVector> vectors;
    
    for (int i = 0; i < num_vectors; ++i) {
        vectors.push_back(HyperVector::random(dimension, i + 500));
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    HyperVector result = Operations::majority_bundle(vectors);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Bundle performance: " << num_vectors << " vectors took " 
              << duration.count() << " microseconds" << std::endl;
    
    EXPECT_EQ(result.dimension(), dimension);
    EXPECT_LT(duration.count(), 50000); // Should complete within 50ms
}

TEST_F(OperationsTest, MemoryPerformanceTest) {
    AssociativeMemory memory(dimension, 0.7);
    const int num_entries = 1000;
    
    // Store many entries
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_entries; ++i) {
        HyperVector hv = HyperVector::random(dimension, i + 1000);
        memory.store("entry_" + std::to_string(i), hv);
    }
    auto store_end = std::chrono::high_resolution_clock::now();
    
    // Query entries
    HyperVector query = HyperVector::random(dimension, 9999);
    auto results = memory.query(query, 10);
    auto query_end = std::chrono::high_resolution_clock::now();
    
    auto store_duration = std::chrono::duration_cast<std::chrono::microseconds>(store_end - start);
    auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(query_end - store_end);
    
    std::cout << "Memory performance: " << num_entries << " stores took " 
              << store_duration.count() << " μs, query took " 
              << query_duration.count() << " μs" << std::endl;
    
    EXPECT_EQ(memory.size(), num_entries);
    EXPECT_EQ(results.size(), 10);
}

// Edge Cases and Error Handling
TEST_F(OperationsTest, ErrorHandlingTest) {
    // Test various error conditions
    EXPECT_THROW(BasisVectors(-1), std::invalid_argument);
    
    AssociativeMemory memory(dimension);
    EXPECT_THROW(memory.store("", hv1), std::invalid_argument);
    EXPECT_THROW(memory.remove("nonexistent"), std::invalid_argument);
    
    EpisodicMemory episodic(dimension);
    std::vector<HyperVector> empty_sequence;
    EXPECT_THROW(episodic.store_episode(empty_sequence), std::invalid_argument);
    
    WorkingMemory working(dimension, 1);
    EXPECT_THROW(working.pop(), std::runtime_error); // Empty memory
}

TEST_F(OperationsTest, MemorySerializationTest) {
    AssociativeMemory memory(dimension);
    memory.store("test1", hv1, 0.9);
    memory.store("test2", hv2, 0.8);
    
    std::string filename = "/tmp/test_memory.bin";
    
    // Save and load
    memory.save_to_file(filename);
    
    AssociativeMemory loaded_memory(dimension);
    loaded_memory.load_from_file(filename);
    
    EXPECT_EQ(loaded_memory.size(), 2);
    EXPECT_TRUE(loaded_memory.contains("test1"));
    EXPECT_TRUE(loaded_memory.contains("test2"));
    EXPECT_NEAR(loaded_memory.get_confidence("test1"), 0.9, 0.01);
    
    // Cleanup
    std::remove(filename.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}