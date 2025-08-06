#include "associative_memory.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace hdc {

AssociativeMemory::AssociativeMemory(int dimension, double cleanup_threshold)
    : dimension_(dimension), cleanup_threshold_(cleanup_threshold) {
    initialize_cleanup_memory();
}

void AssociativeMemory::store(const std::string& key, const HyperVector& value) {
    memory_[key] = value;
    
    // Update usage statistics
    access_count_[key] = 0;
    last_access_[key] = std::chrono::steady_clock::now();
    
    // Trigger cleanup if memory is getting large
    if (memory_.size() > max_memory_size_) {
        cleanup_memory();
    }
}

HyperVector AssociativeMemory::retrieve(const std::string& key) const {
    auto it = memory_.find(key);
    if (it != memory_.end()) {
        // Update access statistics (const_cast is safe here for statistics)
        const_cast<AssociativeMemory*>(this)->access_count_[key]++;
        const_cast<AssociativeMemory*>(this)->last_access_[key] = std::chrono::steady_clock::now();
        
        return it->second;
    }
    
    return HyperVector::zero(dimension_);
}

bool AssociativeMemory::contains(const std::string& key) const {
    return memory_.find(key) != memory_.end();
}

void AssociativeMemory::remove(const std::string& key) {
    memory_.erase(key);
    access_count_.erase(key);
    last_access_.erase(key);
}

std::vector<QueryResult> AssociativeMemory::query(const HyperVector& query, 
                                                 int max_results,
                                                 double min_similarity) const {
    std::vector<QueryResult> results;
    
    for (const auto& [key, stored_vector] : memory_) {
        double similarity = query.similarity(stored_vector);
        
        if (similarity >= min_similarity) {
            QueryResult result;
            result.key = key;
            result.similarity = similarity;
            result.vector = stored_vector;
            results.push_back(result);
        }
    }
    
    // Sort by similarity (descending)
    std::sort(results.begin(), results.end(), 
              [](const QueryResult& a, const QueryResult& b) {
                  return a.similarity > b.similarity;
              });
    
    // Limit results
    if (max_results > 0 && results.size() > static_cast<size_t>(max_results)) {
        results.resize(max_results);
    }
    
    return results;
}

HyperVector AssociativeMemory::best_match(const HyperVector& query) const {
    auto results = this->query(query, 1, 0.0);
    
    if (!results.empty()) {
        // Update access statistics for best match
        const std::string& key = results[0].key;
        const_cast<AssociativeMemory*>(this)->access_count_[key]++;
        const_cast<AssociativeMemory*>(this)->last_access_[key] = std::chrono::steady_clock::now();
        
        return results[0].vector;
    }
    
    return HyperVector::zero(dimension_);
}

double AssociativeMemory::similarity_to_best_match(const HyperVector& query) const {
    auto results = this->query(query, 1, 0.0);
    
    if (!results.empty()) {
        return results[0].similarity;
    }
    
    return 0.0;
}

std::vector<std::string> AssociativeMemory::get_all_keys() const {
    std::vector<std::string> keys;
    keys.reserve(memory_.size());
    
    for (const auto& [key, _] : memory_) {
        keys.push_back(key);
    }
    
    return keys;
}

size_t AssociativeMemory::size() const {
    return memory_.size();
}

void AssociativeMemory::clear() {
    memory_.clear();
    access_count_.clear();
    last_access_.clear();
}

void AssociativeMemory::update_vector(const std::string& key, const HyperVector& new_value) {
    auto it = memory_.find(key);
    if (it != memory_.end()) {
        it->second = new_value;
        access_count_[key]++;
        last_access_[key] = std::chrono::steady_clock::now();
    } else {
        store(key, new_value);
    }
}

HyperVector AssociativeMemory::interpolate(const std::string& key1, const std::string& key2, 
                                         double weight1) const {
    auto it1 = memory_.find(key1);
    auto it2 = memory_.find(key2);
    
    if (it1 == memory_.end() || it2 == memory_.end()) {
        throw std::invalid_argument("One or both keys not found for interpolation");
    }
    
    double weight2 = 1.0 - weight1;
    
    std::vector<std::pair<HyperVector, double>> weighted_vectors = {
        {it1->second, weight1},
        {it2->second, weight2}
    };
    
    return weighted_bundle(weighted_vectors);
}

void AssociativeMemory::batch_store(const std::map<std::string, HyperVector>& items) {
    for (const auto& [key, value] : items) {
        store(key, value);
    }
}

std::map<std::string, HyperVector> AssociativeMemory::batch_retrieve(
    const std::vector<std::string>& keys) const {
    std::map<std::string, HyperVector> results;
    
    for (const auto& key : keys) {
        auto vector = retrieve(key);
        if (!vector.is_zero_vector()) {
            results[key] = vector;
        }
    }
    
    return results;
}

void AssociativeMemory::merge_memories(const AssociativeMemory& other, double weight) {
    for (const auto& [key, other_vector] : other.memory_) {
        auto it = memory_.find(key);
        
        if (it != memory_.end()) {
            // Merge existing vector
            std::vector<std::pair<HyperVector, double>> weighted_vectors = {
                {it->second, 1.0 - weight},
                {other_vector, weight}
            };
            it->second = weighted_bundle(weighted_vectors);
        } else {
            // Add new vector
            memory_[key] = other_vector;
        }
        
        // Update access statistics
        access_count_[key] = 0;
        last_access_[key] = std::chrono::steady_clock::now();
    }
}

MemoryStats AssociativeMemory::get_statistics() const {
    MemoryStats stats;
    stats.total_items = memory_.size();
    stats.total_accesses = 0;
    
    for (const auto& [key, count] : access_count_) {
        stats.total_accesses += count;
    }
    
    stats.average_accesses = stats.total_items > 0 ? 
        static_cast<double>(stats.total_accesses) / stats.total_items : 0.0;
    
    // Find most and least accessed items
    if (!access_count_.empty()) {
        auto max_it = std::max_element(access_count_.begin(), access_count_.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        stats.most_accessed = max_it->first;
        stats.max_accesses = max_it->second;
        
        auto min_it = std::min_element(access_count_.begin(), access_count_.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        stats.least_accessed = min_it->first;
        stats.min_accesses = min_it->second;
    }
    
    return stats;
}

void AssociativeMemory::save_to_file(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write header
    size_t num_items = memory_.size();
    file.write(reinterpret_cast<const char*>(&num_items), sizeof(num_items));
    file.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
    
    // Write each item
    for (const auto& [key, vector] : memory_) {
        // Write key length and key
        size_t key_len = key.size();
        file.write(reinterpret_cast<const char*>(&key_len), sizeof(key_len));
        file.write(key.data(), key_len);
        
        // Write vector data
        auto vector_bytes = vector.to_bytes();
        size_t vector_size = vector_bytes.size();
        file.write(reinterpret_cast<const char*>(&vector_size), sizeof(vector_size));
        file.write(reinterpret_cast<const char*>(vector_bytes.data()), vector_size);
    }
    
    file.close();
}

void AssociativeMemory::load_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Read header
    size_t num_items;
    int file_dimension;
    file.read(reinterpret_cast<char*>(&num_items), sizeof(num_items));
    file.read(reinterpret_cast<char*>(&file_dimension), sizeof(file_dimension));
    
    if (file_dimension != dimension_) {
        throw std::runtime_error("Dimension mismatch in file: " + std::to_string(file_dimension) + 
                                " vs " + std::to_string(dimension_));
    }
    
    // Clear existing memory
    clear();
    
    // Read each item
    for (size_t i = 0; i < num_items; ++i) {
        // Read key
        size_t key_len;
        file.read(reinterpret_cast<char*>(&key_len), sizeof(key_len));
        
        std::string key(key_len, '\0');
        file.read(&key[0], key_len);
        
        // Read vector
        size_t vector_size;
        file.read(reinterpret_cast<char*>(&vector_size), sizeof(vector_size));
        
        std::vector<uint8_t> vector_bytes(vector_size);
        file.read(reinterpret_cast<char*>(vector_bytes.data()), vector_size);
        
        // Reconstruct vector
        HyperVector vector(dimension_);
        vector.from_bytes(vector_bytes);
        
        // Store in memory
        memory_[key] = vector;
        access_count_[key] = 0;
        last_access_[key] = std::chrono::steady_clock::now();
    }
    
    file.close();
}

void AssociativeMemory::initialize_cleanup_memory() {
    // Create basis vectors for cleanup operations
    for (int i = 0; i < 100; ++i) {
        cleanup_basis_.push_back(HyperVector::random(dimension_, i + 100000));
    }
}

void AssociativeMemory::cleanup_memory() {
    if (memory_.size() <= max_memory_size_ / 2) {
        return;  // No need to cleanup yet
    }
    
    auto now = std::chrono::steady_clock::now();
    std::vector<std::string> candidates_for_removal;
    
    // Find items to remove based on access patterns
    for (const auto& [key, last_time] : last_access_) {
        auto time_since_access = std::chrono::duration_cast<std::chrono::hours>(
            now - last_time).count();
        
        int access_count = access_count_.at(key);
        
        // Remove items that haven't been accessed recently and have low access count
        if (time_since_access > 24 && access_count < 5) {  // 24 hours, less than 5 accesses
            candidates_for_removal.push_back(key);
        }
    }
    
    // Remove candidates
    for (const auto& key : candidates_for_removal) {
        remove(key);
        
        // Stop if we've freed enough space
        if (memory_.size() <= max_memory_size_ * 0.8) {
            break;
        }
    }
}

}  // namespace hdc