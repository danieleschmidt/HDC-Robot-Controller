#include "memory.hpp"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <chrono>

namespace hdc {

// AssociativeMemory Implementation
AssociativeMemory::AssociativeMemory(int dimension, double similarity_threshold)
    : dimension_(dimension), similarity_threshold_(similarity_threshold) {
    if (dimension <= 0) {
        throw std::invalid_argument("Dimension must be positive");
    }
    if (similarity_threshold < -1.0 || similarity_threshold > 1.0) {
        throw std::invalid_argument("Similarity threshold must be between -1 and 1");
    }
}

void AssociativeMemory::store(const std::string& label, const HyperVector& vector, double confidence) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (label.empty()) {
        throw std::invalid_argument("Label cannot be empty");
    }
    
    if (vector.dimension() != dimension_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    
    if (confidence < 0.0 || confidence > 1.0) {
        throw std::invalid_argument("Confidence must be between 0 and 1");
    }
    
    memory_[label] = std::make_unique<MemoryEntry>(vector, label, confidence);
}

void AssociativeMemory::store_with_update(const std::string& label, const HyperVector& vector, 
                                         double learning_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (learning_rate < 0.0 || learning_rate > 1.0) {
        throw std::invalid_argument("Learning rate must be between 0 and 1");
    }
    
    auto it = memory_.find(label);
    if (it != memory_.end()) {
        // Update existing entry with exponential moving average
        auto& existing = it->second;
        for (int i = 0; i < dimension_; ++i) {
            double current = existing->vector[i];
            double new_val = current * (1.0 - learning_rate) + vector[i] * learning_rate;
            existing->vector[i] = (new_val > 0) ? 1 : -1;
        }
        existing->confidence = std::min(1.0, existing->confidence + learning_rate * 0.1);
        existing->access_count++;
        existing->last_access = std::chrono::steady_clock::now();
    } else {
        store(label, vector, 0.5); // Start with medium confidence for new entries
    }
}

std::vector<AssociativeMemory::QueryResult> AssociativeMemory::query(
    const HyperVector& query_vector, int max_results) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (query_vector.dimension() != dimension_) {
        throw std::invalid_argument("Query vector dimension mismatch");
    }
    
    if (max_results <= 0) {
        throw std::invalid_argument("Max results must be positive");
    }
    
    std::vector<QueryResult> results;
    results.reserve(memory_.size());
    
    for (const auto& [label, entry] : memory_) {
        double sim = query_vector.similarity(entry->vector);
        results.emplace_back(label, entry->vector, sim, entry->confidence);
        
        // Update access statistics
        entry->access_count++;
        entry->last_access = std::chrono::steady_clock::now();
    }
    
    // Sort by similarity (descending)
    std::sort(results.begin(), results.end(), 
              [](const QueryResult& a, const QueryResult& b) {
                  return a.similarity > b.similarity;
              });
    
    // Return top results
    if (static_cast<int>(results.size()) > max_results) {
        results.resize(max_results);
    }
    
    return results;
}

AssociativeMemory::QueryResult AssociativeMemory::query_best(const HyperVector& query_vector) const {
    auto results = query(query_vector, 1);
    if (results.empty()) {
        throw std::runtime_error("No entries in memory to query");
    }
    return results[0];
}

std::vector<AssociativeMemory::QueryResult> AssociativeMemory::query_threshold(
    const HyperVector& query_vector, double threshold) const {
    
    auto all_results = query(query_vector, static_cast<int>(memory_.size()));
    
    std::vector<QueryResult> filtered_results;
    for (const auto& result : all_results) {
        if (result.similarity >= threshold) {
            filtered_results.push_back(result);
        } else {
            break; // Results are sorted, so we can stop here
        }
    }
    
    return filtered_results;
}

bool AssociativeMemory::contains(const std::string& label) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return memory_.find(label) != memory_.end();
}

void AssociativeMemory::remove(const std::string& label) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = memory_.find(label);
    if (it == memory_.end()) {
        throw std::invalid_argument("Label not found in memory");
    }
    
    memory_.erase(it);
}

void AssociativeMemory::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    memory_.clear();
}

std::vector<std::string> AssociativeMemory::get_labels() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> labels;
    labels.reserve(memory_.size());
    
    for (const auto& [label, entry] : memory_) {
        labels.push_back(label);
    }
    
    return labels;
}

double AssociativeMemory::get_confidence(const std::string& label) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = memory_.find(label);
    if (it == memory_.end()) {
        throw std::invalid_argument("Label not found in memory");
    }
    
    return it->second->confidence;
}

void AssociativeMemory::update_confidence(const std::string& label, double confidence) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (confidence < 0.0 || confidence > 1.0) {
        throw std::invalid_argument("Confidence must be between 0 and 1");
    }
    
    auto it = memory_.find(label);
    if (it == memory_.end()) {
        throw std::invalid_argument("Label not found in memory");
    }
    
    it->second->confidence = confidence;
}

void AssociativeMemory::consolidate_similar(double similarity_threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::string> to_remove;
    std::unordered_map<std::string, HyperVector> consolidated;
    
    // Find similar entries and mark for consolidation
    for (auto it1 = memory_.begin(); it1 != memory_.end(); ++it1) {
        if (std::find(to_remove.begin(), to_remove.end(), it1->first) != to_remove.end()) {
            continue; // Already marked for removal
        }
        
        std::vector<std::pair<HyperVector, double>> similar_vectors;
        similar_vectors.emplace_back(it1->second->vector, it1->second->confidence);
        
        for (auto it2 = std::next(it1); it2 != memory_.end(); ++it2) {
            if (std::find(to_remove.begin(), to_remove.end(), it2->first) != to_remove.end()) {
                continue; // Already marked for removal
            }
            
            double sim = it1->second->vector.similarity(it2->second->vector);
            if (sim >= similarity_threshold) {
                similar_vectors.emplace_back(it2->second->vector, it2->second->confidence);
                to_remove.push_back(it2->first);
            }
        }
        
        if (similar_vectors.size() > 1) {
            // Consolidate similar vectors
            HyperVector consolidated_vector = weighted_bundle(similar_vectors);
            consolidated[it1->first] = consolidated_vector;
        }
    }
    
    // Remove marked entries
    for (const auto& label : to_remove) {
        memory_.erase(label);
    }
    
    // Update consolidated entries
    for (const auto& [label, vector] : consolidated) {
        auto it = memory_.find(label);
        if (it != memory_.end()) {
            it->second->vector = vector;
            it->second->confidence = std::min(1.0, it->second->confidence * 1.1); // Boost confidence
        }
    }
}

void AssociativeMemory::decay_confidence(double decay_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (decay_rate < 0.0 || decay_rate > 1.0) {
        throw std::invalid_argument("Decay rate must be between 0 and 1");
    }
    
    for (auto& [label, entry] : memory_) {
        entry->confidence *= (1.0 - decay_rate);
    }
}

void AssociativeMemory::remove_low_confidence(double min_confidence) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = memory_.begin();
    while (it != memory_.end()) {
        if (it->second->confidence < min_confidence) {
            it = memory_.erase(it);
        } else {
            ++it;
        }
    }
}

void AssociativeMemory::save_to_file(const std::string& filename) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Write header
    file.write(reinterpret_cast<const char*>(&dimension_), sizeof(dimension_));
    
    size_t num_entries = memory_.size();
    file.write(reinterpret_cast<const char*>(&num_entries), sizeof(num_entries));
    
    // Write entries
    for (const auto& [label, entry] : memory_) {
        // Write label length and label
        size_t label_len = label.length();
        file.write(reinterpret_cast<const char*>(&label_len), sizeof(label_len));
        file.write(label.c_str(), label_len);
        
        // Write confidence
        file.write(reinterpret_cast<const char*>(&entry->confidence), sizeof(entry->confidence));
        
        // Write vector data
        auto bytes = entry->vector.to_bytes();
        size_t bytes_len = bytes.size();
        file.write(reinterpret_cast<const char*>(&bytes_len), sizeof(bytes_len));
        file.write(reinterpret_cast<const char*>(bytes.data()), bytes_len);
    }
    
    if (!file) {
        throw std::runtime_error("Error writing to file: " + filename);
    }
}

void AssociativeMemory::load_from_file(const std::string& filename) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    // Read header
    int file_dimension;
    file.read(reinterpret_cast<char*>(&file_dimension), sizeof(file_dimension));
    
    if (file_dimension != dimension_) {
        throw std::runtime_error("Dimension mismatch in loaded file");
    }
    
    size_t num_entries;
    file.read(reinterpret_cast<char*>(&num_entries), sizeof(num_entries));
    
    memory_.clear();
    
    // Read entries
    for (size_t i = 0; i < num_entries; ++i) {
        // Read label
        size_t label_len;
        file.read(reinterpret_cast<char*>(&label_len), sizeof(label_len));
        
        std::string label(label_len, '\0');
        file.read(&label[0], label_len);
        
        // Read confidence
        double confidence;
        file.read(reinterpret_cast<char*>(&confidence), sizeof(confidence));
        
        // Read vector
        size_t bytes_len;
        file.read(reinterpret_cast<char*>(&bytes_len), sizeof(bytes_len));
        
        std::vector<uint8_t> bytes(bytes_len);
        file.read(reinterpret_cast<char*>(bytes.data()), bytes_len);
        
        HyperVector vector(dimension_);
        vector.from_bytes(bytes);
        
        memory_[label] = std::make_unique<MemoryEntry>(vector, label, confidence);
    }
    
    if (!file && !file.eof()) {
        throw std::runtime_error("Error reading from file: " + filename);
    }
}

// EpisodicMemory Implementation
EpisodicMemory::EpisodicMemory(int dimension, size_t max_episodes)
    : dimension_(dimension), max_episodes_(max_episodes) {
    if (dimension <= 0) {
        throw std::invalid_argument("Dimension must be positive");
    }
    if (max_episodes == 0) {
        throw std::invalid_argument("Max episodes must be positive");
    }
    
    episodes_.reserve(max_episodes);
}

void EpisodicMemory::store_episode(const std::vector<HyperVector>& sequence,
                                  const std::vector<std::string>& labels,
                                  double importance) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (sequence.empty()) {
        throw std::invalid_argument("Episode sequence cannot be empty");
    }
    
    if (importance < 0.0 || importance > 1.0) {
        throw std::invalid_argument("Importance must be between 0 and 1");
    }
    
    // Validate dimensions
    for (const auto& vector : sequence) {
        if (vector.dimension() != dimension_) {
            throw std::invalid_argument("Vector dimension mismatch in sequence");
        }
    }
    
    // Add new episode
    episodes_.emplace_back(sequence, labels, importance);
    
    // Maintain size limit
    if (episodes_.size() > max_episodes_) {
        // Remove oldest episodes with lowest importance
        std::sort(episodes_.begin(), episodes_.end(),
                  [](const Episode& a, const Episode& b) {
                      if (a.importance != b.importance) {
                          return a.importance < b.importance;
                      }
                      return a.timestamp < b.timestamp;
                  });
        
        episodes_.erase(episodes_.begin());
    }
}

std::vector<EpisodicMemory::Episode> EpisodicMemory::query_similar_episodes(
    const std::vector<HyperVector>& query_sequence, double similarity_threshold) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (query_sequence.empty()) {
        throw std::invalid_argument("Query sequence cannot be empty");
    }
    
    // Encode query sequence
    HyperVector query_hv = create_sequence(query_sequence);
    
    std::vector<std::pair<Episode, double>> episode_similarities;
    
    for (const auto& episode : episodes_) {
        HyperVector episode_hv = create_sequence(episode.sequence);
        double similarity = query_hv.similarity(episode_hv);
        
        if (similarity >= similarity_threshold) {
            episode_similarities.emplace_back(episode, similarity);
        }
    }
    
    // Sort by similarity
    std::sort(episode_similarities.begin(), episode_similarities.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
    
    std::vector<Episode> results;
    for (const auto& [episode, sim] : episode_similarities) {
        results.push_back(episode);
    }
    
    return results;
}

std::vector<EpisodicMemory::Episode> EpisodicMemory::query_by_pattern(
    const HyperVector& pattern, int max_results) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<std::pair<Episode, double>> pattern_matches;
    
    for (const auto& episode : episodes_) {
        double max_similarity = 0.0;
        
        // Check similarity against all vectors in episode
        for (const auto& vector : episode.sequence) {
            double sim = pattern.similarity(vector);
            max_similarity = std::max(max_similarity, sim);
        }
        
        pattern_matches.emplace_back(episode, max_similarity);
    }
    
    // Sort by best similarity
    std::sort(pattern_matches.begin(), pattern_matches.end(),
              [](const auto& a, const auto& b) {
                  return a.second > b.second;
              });
    
    std::vector<Episode> results;
    for (int i = 0; i < std::min(max_results, static_cast<int>(pattern_matches.size())); ++i) {
        results.push_back(pattern_matches[i].first);
    }
    
    return results;
}

std::vector<EpisodicMemory::Episode> EpisodicMemory::get_recent_episodes(int count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<Episode> sorted_episodes = episodes_;
    std::sort(sorted_episodes.begin(), sorted_episodes.end(),
              [](const Episode& a, const Episode& b) {
                  return a.timestamp > b.timestamp;
              });
    
    if (count > static_cast<int>(sorted_episodes.size())) {
        count = static_cast<int>(sorted_episodes.size());
    }
    
    return std::vector<Episode>(sorted_episodes.begin(), sorted_episodes.begin() + count);
}

std::vector<EpisodicMemory::Episode> EpisodicMemory::get_episodes_in_range(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end) const {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<Episode> results;
    for (const auto& episode : episodes_) {
        if (episode.timestamp >= start && episode.timestamp <= end) {
            results.push_back(episode);
        }
    }
    
    return results;
}

void EpisodicMemory::consolidate_episodes(double similarity_threshold) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<bool> to_remove(episodes_.size(), false);
    
    for (size_t i = 0; i < episodes_.size(); ++i) {
        if (to_remove[i]) continue;
        
        HyperVector episode_i = create_sequence(episodes_[i].sequence);
        
        for (size_t j = i + 1; j < episodes_.size(); ++j) {
            if (to_remove[j]) continue;
            
            HyperVector episode_j = create_sequence(episodes_[j].sequence);
            double similarity = episode_i.similarity(episode_j);
            
            if (similarity >= similarity_threshold) {
                // Keep the more important episode
                if (episodes_[i].importance >= episodes_[j].importance) {
                    to_remove[j] = true;
                    episodes_[i].importance = std::min(1.0, episodes_[i].importance + 0.1);
                } else {
                    to_remove[i] = true;
                    episodes_[j].importance = std::min(1.0, episodes_[j].importance + 0.1);
                    break;
                }
            }
        }
    }
    
    // Remove marked episodes
    auto new_end = std::remove_if(episodes_.begin(), episodes_.end(),
                                  [&](const Episode& ep) {
                                      size_t idx = &ep - &episodes_[0];
                                      return to_remove[idx];
                                  });
    episodes_.erase(new_end, episodes_.end());
}

void EpisodicMemory::decay_importance(double decay_rate) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (decay_rate < 0.0 || decay_rate > 1.0) {
        throw std::invalid_argument("Decay rate must be between 0 and 1");
    }
    
    for (auto& episode : episodes_) {
        episode.importance *= (1.0 - decay_rate);
    }
}

void EpisodicMemory::prune_old_episodes(std::chrono::hours max_age) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto cutoff_time = std::chrono::steady_clock::now() - max_age;
    
    auto new_end = std::remove_if(episodes_.begin(), episodes_.end(),
                                  [cutoff_time](const Episode& ep) {
                                      return ep.timestamp < cutoff_time;
                                  });
    episodes_.erase(new_end, episodes_.end());
}

double EpisodicMemory::total_importance() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    double total = 0.0;
    for (const auto& episode : episodes_) {
        total += episode.importance;
    }
    return total;
}

// WorkingMemory Implementation
WorkingMemory::WorkingMemory(int dimension, size_t capacity)
    : capacity_(capacity), context_(HyperVector::zero(dimension)) {
    if (capacity == 0) {
        throw std::invalid_argument("Capacity must be positive");
    }
}

void WorkingMemory::push(const HyperVector& vector, const std::string& context) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (full()) {
        buffer_.pop_front(); // Remove oldest item
    }
    
    buffer_.emplace_back(vector, context);
    
    // Update running context
    if (buffer_.size() == 1) {
        context_ = vector;
    } else {
        context_ = context_.bundle(vector);
    }
}

HyperVector WorkingMemory::pop() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (empty()) {
        throw std::runtime_error("Cannot pop from empty working memory");
    }
    
    auto result = buffer_.back().first;
    buffer_.pop_back();
    
    // Recalculate context
    if (buffer_.empty()) {
        context_ = HyperVector::zero(context_.dimension());
    } else {
        std::vector<HyperVector> vectors;
        for (const auto& [vec, ctx] : buffer_) {
            vectors.push_back(vec);
        }
        context_ = HyperVector::bundle_vectors(vectors);
    }
    
    return result;
}

HyperVector WorkingMemory::peek() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (empty()) {
        throw std::runtime_error("Cannot peek empty working memory");
    }
    
    return buffer_.back().first;
}

HyperVector WorkingMemory::get_context() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return context_;
}

void WorkingMemory::update_context(const HyperVector& new_context, double blend_factor) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (blend_factor < 0.0 || blend_factor > 1.0) {
        throw std::invalid_argument("Blend factor must be between 0 and 1");
    }
    
    // Weighted bundle of old and new context
    std::vector<std::pair<HyperVector, double>> weighted_contexts = {
        {context_, 1.0 - blend_factor},
        {new_context, blend_factor}
    };
    
    context_ = weighted_bundle(weighted_contexts);
}

std::vector<HyperVector> WorkingMemory::detect_patterns(int pattern_length) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (pattern_length <= 0) {
        throw std::invalid_argument("Pattern length must be positive");
    }
    
    if (static_cast<int>(buffer_.size()) < pattern_length) {
        return {}; // Not enough data for patterns
    }
    
    std::vector<HyperVector> patterns;
    
    for (size_t i = 0; i <= buffer_.size() - pattern_length; ++i) {
        std::vector<HyperVector> pattern_vectors;
        for (int j = 0; j < pattern_length; ++j) {
            pattern_vectors.push_back(buffer_[i + j].first);
        }
        patterns.push_back(create_sequence(pattern_vectors));
    }
    
    return patterns;
}

bool WorkingMemory::has_pattern(const std::vector<HyperVector>& pattern) const {
    if (pattern.empty()) {
        return false;
    }
    
    HyperVector target_pattern = create_sequence(pattern);
    auto detected_patterns = detect_patterns(static_cast<int>(pattern.size()));
    
    const double similarity_threshold = 0.8;
    for (const auto& detected : detected_patterns) {
        if (target_pattern.similarity(detected) >= similarity_threshold) {
            return true;
        }
    }
    
    return false;
}

void WorkingMemory::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
    context_ = HyperVector::zero(context_.dimension());
}

// HierarchicalMemory Implementation
HierarchicalMemory::HierarchicalMemory(int dimension)
    : associative_memory_(dimension), 
      episodic_memory_(dimension),
      working_memory_(dimension) {}

void HierarchicalMemory::store_experience(const HyperVector& perception, const HyperVector& action,
                                        const std::string& label, double confidence) {
    // Store in associative memory
    HyperVector experience = perception.bind(action);
    associative_memory_.store(label, experience, confidence);
    
    // Store in episodic memory as single-step episode
    episodic_memory_.store_episode({perception, action}, {label});
    
    // Add to working memory
    working_memory_.push(experience, label);
}

std::vector<AssociativeMemory::QueryResult> HierarchicalMemory::query_experience(
    const HyperVector& query, int max_results) const {
    return associative_memory_.query(query, max_results);
}

void HierarchicalMemory::consolidate_all(double similarity_threshold) {
    associative_memory_.consolidate_similar(similarity_threshold);
    episodic_memory_.consolidate_episodes(similarity_threshold);
}

void HierarchicalMemory::cleanup_all() {
    associative_memory_.decay_confidence(0.01);
    associative_memory_.remove_low_confidence(0.1);
    
    episodic_memory_.decay_importance(0.02);
    episodic_memory_.prune_old_episodes(std::chrono::hours(24 * 7));
}

size_t HierarchicalMemory::total_size() const {
    return associative_memory_.size() + episodic_memory_.size() + working_memory_.size();
}

void HierarchicalMemory::print_stats() const {
    std::cout << "Hierarchical Memory Statistics:" << std::endl;
    std::cout << "  Associative Memory: " << associative_memory_.size() << " entries" << std::endl;
    std::cout << "  Episodic Memory: " << episodic_memory_.size() << " episodes" << std::endl;
    std::cout << "  Working Memory: " << working_memory_.size() << " items" << std::endl;
    std::cout << "  Total Size: " << total_size() << std::endl;
}

} // namespace hdc