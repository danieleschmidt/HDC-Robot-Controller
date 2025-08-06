#pragma once

#include "hypervector.hpp"
#include <map>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>

namespace hdc {

struct QueryResult {
    std::string key;
    double similarity;
    HyperVector vector;
};

struct MemoryStats {
    size_t total_items{0};
    size_t total_accesses{0};
    double average_accesses{0.0};
    std::string most_accessed;
    std::string least_accessed;
    size_t max_accesses{0};
    size_t min_accesses{0};
};

/**
 * Associative memory for storing and retrieving hyperdimensional vectors
 * Supports similarity-based querying and automatic cleanup
 */
class AssociativeMemory {
public:
    AssociativeMemory(int dimension = 10000, double cleanup_threshold = 0.1);
    
    // Basic storage operations
    void store(const std::string& key, const HyperVector& value);
    HyperVector retrieve(const std::string& key) const;
    bool contains(const std::string& key) const;
    void remove(const std::string& key);
    
    // Similarity-based querying
    std::vector<QueryResult> query(const HyperVector& query, 
                                  int max_results = 10,
                                  double min_similarity = 0.0) const;
    HyperVector best_match(const HyperVector& query) const;
    double similarity_to_best_match(const HyperVector& query) const;
    
    // Memory management
    std::vector<std::string> get_all_keys() const;
    size_t size() const;
    void clear();
    
    // Advanced operations
    void update_vector(const std::string& key, const HyperVector& new_value);
    HyperVector interpolate(const std::string& key1, const std::string& key2, 
                           double weight1 = 0.5) const;
    
    // Batch operations
    void batch_store(const std::map<std::string, HyperVector>& items);
    std::map<std::string, HyperVector> batch_retrieve(const std::vector<std::string>& keys) const;
    
    // Memory merging and statistics
    void merge_memories(const AssociativeMemory& other, double weight = 0.5);
    MemoryStats get_statistics() const;
    
    // Persistence
    void save_to_file(const std::string& filename) const;
    void load_from_file(const std::string& filename);

private:
    void initialize_cleanup_memory();
    void cleanup_memory();
    
    int dimension_;
    double cleanup_threshold_;
    
    // Main memory storage
    std::map<std::string, HyperVector> memory_;
    
    // Access statistics for cleanup
    std::map<std::string, int> access_count_;
    std::map<std::string, std::chrono::steady_clock::time_point> last_access_;
    
    // Cleanup parameters
    static constexpr size_t max_memory_size_ = 100000;
    std::vector<HyperVector> cleanup_basis_;
};

}  // namespace hdc