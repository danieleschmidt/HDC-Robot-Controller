#pragma once

#include "hypervector.hpp"
#include <unordered_map>
#include <string>
#include <vector>
#include <memory>
#include <mutex>

namespace hdc {

/**
 * Associative memory for storing and retrieving hypervectors
 */
class AssociativeMemory {
public:
    struct MemoryEntry {
        HyperVector vector;
        std::string label;
        double confidence;
        uint64_t access_count;
        std::chrono::steady_clock::time_point last_access;
        
        MemoryEntry(const HyperVector& v, const std::string& l, double c = 1.0)
            : vector(v), label(l), confidence(c), access_count(0),
              last_access(std::chrono::steady_clock::now()) {}
    };
    
    struct QueryResult {
        std::string label;
        HyperVector vector;
        double similarity;
        double confidence;
        
        QueryResult(const std::string& l, const HyperVector& v, double s, double c)
            : label(l), vector(v), similarity(s), confidence(c) {}
    };
    
    explicit AssociativeMemory(int dimension = HyperVector::DEFAULT_DIMENSION,
                             double similarity_threshold = 0.7);
    
    // Storage operations
    void store(const std::string& label, const HyperVector& vector, double confidence = 1.0);
    void store_with_update(const std::string& label, const HyperVector& vector, 
                          double learning_rate = 0.1);
    
    // Retrieval operations
    std::vector<QueryResult> query(const HyperVector& query_vector, 
                                  int max_results = 10) const;
    QueryResult query_best(const HyperVector& query_vector) const;
    std::vector<QueryResult> query_threshold(const HyperVector& query_vector,
                                           double threshold) const;
    
    // Memory management
    bool contains(const std::string& label) const;
    void remove(const std::string& label);
    void clear();
    size_t size() const { return memory_.size(); }
    
    // Statistics
    std::vector<std::string> get_labels() const;
    double get_confidence(const std::string& label) const;
    void update_confidence(const std::string& label, double confidence);
    
    // Consolidation and cleanup
    void consolidate_similar(double similarity_threshold = 0.95);
    void decay_confidence(double decay_rate = 0.01);
    void remove_low_confidence(double min_confidence = 0.1);
    
    // Serialization
    void save_to_file(const std::string& filename) const;
    void load_from_file(const std::string& filename);
    
    // Thread safety
    void lock() { mutex_.lock(); }
    void unlock() { mutex_.unlock(); }
    
private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::unique_ptr<MemoryEntry>> memory_;
    int dimension_;
    double similarity_threshold_;
};

/**
 * Episodic memory for storing temporal sequences
 */
class EpisodicMemory {
public:
    struct Episode {
        std::vector<HyperVector> sequence;
        std::vector<std::string> labels;
        std::chrono::steady_clock::time_point timestamp;
        double importance;
        
        Episode(const std::vector<HyperVector>& seq, 
                const std::vector<std::string>& lab = {}, double imp = 1.0)
            : sequence(seq), labels(lab), 
              timestamp(std::chrono::steady_clock::now()), importance(imp) {}
    };
    
    explicit EpisodicMemory(int dimension = HyperVector::DEFAULT_DIMENSION,
                          size_t max_episodes = 1000);
    
    // Episode storage
    void store_episode(const std::vector<HyperVector>& sequence,
                      const std::vector<std::string>& labels = {},
                      double importance = 1.0);
    
    // Episode retrieval
    std::vector<Episode> query_similar_episodes(const std::vector<HyperVector>& query_sequence,
                                               double similarity_threshold = 0.7) const;
    std::vector<Episode> query_by_pattern(const HyperVector& pattern,
                                         int max_results = 10) const;
    
    // Temporal queries
    std::vector<Episode> get_recent_episodes(int count = 10) const;
    std::vector<Episode> get_episodes_in_range(
        std::chrono::steady_clock::time_point start,
        std::chrono::steady_clock::time_point end) const;
    
    // Memory consolidation
    void consolidate_episodes(double similarity_threshold = 0.9);
    void decay_importance(double decay_rate = 0.02);
    void prune_old_episodes(std::chrono::hours max_age = std::chrono::hours(24 * 7));
    
    // Statistics
    size_t size() const { return episodes_.size(); }
    double total_importance() const;
    
private:
    std::vector<Episode> episodes_;
    int dimension_;
    size_t max_episodes_;
    mutable std::mutex mutex_;
};

/**
 * Working memory for real-time processing
 */
class WorkingMemory {
public:
    explicit WorkingMemory(int dimension = HyperVector::DEFAULT_DIMENSION,
                         size_t capacity = 100);
    
    // Buffer management
    void push(const HyperVector& vector, const std::string& context = "");
    HyperVector pop();
    HyperVector peek() const;
    
    // Context operations
    HyperVector get_context() const;
    void update_context(const HyperVector& new_context, double blend_factor = 0.5);
    
    // Pattern detection
    std::vector<HyperVector> detect_patterns(int pattern_length = 3) const;
    bool has_pattern(const std::vector<HyperVector>& pattern) const;
    
    // State management
    void clear();
    bool empty() const { return buffer_.empty(); }
    bool full() const { return buffer_.size() >= capacity_; }
    size_t size() const { return buffer_.size(); }
    
private:
    std::deque<std::pair<HyperVector, std::string>> buffer_;
    size_t capacity_;
    HyperVector context_;
    mutable std::mutex mutex_;
};

/**
 * Hierarchical memory combining different memory types
 */
class HierarchicalMemory {
public:
    explicit HierarchicalMemory(int dimension = HyperVector::DEFAULT_DIMENSION);
    
    // Memory access
    AssociativeMemory& get_associative_memory() { return associative_memory_; }
    EpisodicMemory& get_episodic_memory() { return episodic_memory_; }
    WorkingMemory& get_working_memory() { return working_memory_; }
    
    // Unified operations
    void store_experience(const HyperVector& perception, const HyperVector& action,
                         const std::string& label, double confidence = 1.0);
    
    std::vector<AssociativeMemory::QueryResult> query_experience(
        const HyperVector& query, int max_results = 10) const;
    
    // Memory consolidation
    void consolidate_all(double similarity_threshold = 0.9);
    void cleanup_all();
    
    // Statistics
    size_t total_size() const;
    void print_stats() const;
    
private:
    AssociativeMemory associative_memory_;
    EpisodicMemory episodic_memory_;
    WorkingMemory working_memory_;
};

} // namespace hdc