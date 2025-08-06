#pragma once

#include <vector>
#include <memory>
#include <map>
#include <string>
#include <algorithm>
#include <numeric>

#include "hypervector.hpp"

namespace hdc_robot_controller {

// Forward declaration
struct DemonstrationSample;

class BehaviorLearner {
public:
    BehaviorLearner(int dimension = 10000, double similarity_threshold = 0.85)
        : dimension_(dimension), similarity_threshold_(similarity_threshold) {
        initialize_learning_parameters();
    }
    
    // Learn behavior from demonstration samples
    hdc::HyperVector learn_from_demonstration(const std::vector<DemonstrationSample>& samples) {
        if (samples.empty()) {
            throw std::invalid_argument("Cannot learn from empty demonstration");
        }
        
        // Create state-action pairs
        std::vector<hdc::HyperVector> state_action_pairs;
        
        for (const auto& sample : samples) {
            // Bind perception with action to create state-action pair
            auto state_action = sample.perception.bind(sample.action);
            
            // Weight by reward if available
            if (sample.reward > 0.5) {  // Only use positive examples
                state_action_pairs.push_back(state_action);
            }
        }
        
        if (state_action_pairs.empty()) {
            throw std::runtime_error("No positive examples in demonstration");
        }
        
        // Bundle all state-action pairs to create behavior hypervector
        auto behavior_hv = hdc::HyperVector::bundle_vectors(state_action_pairs);
        
        // Apply cleanup operations
        behavior_hv.normalize();
        
        return behavior_hv;
    }
    
    // Query behavior for given perception
    hdc::HyperVector query_behavior(const hdc::HyperVector& behavior, 
                                   const hdc::HyperVector& perception) {
        // Unbind perception from behavior to get action
        auto action_hv = behavior.bind(perception);
        
        // Clean up the result
        action_hv.normalize();
        
        return action_hv;
    }
    
    // Adapt existing behavior with new samples
    hdc::HyperVector adapt_behavior(const hdc::HyperVector& existing_behavior,
                                   const std::vector<DemonstrationSample>& new_samples) {
        if (new_samples.empty()) {
            return existing_behavior;
        }
        
        // Learn from new samples
        auto new_behavior_component = learn_from_demonstration(new_samples);
        
        // Weighted combination: favor existing knowledge but incorporate new learning
        double existing_weight = 0.8;
        double new_weight = 0.2;
        
        std::vector<std::pair<hdc::HyperVector, double>> weighted_behaviors = {
            {existing_behavior, existing_weight},
            {new_behavior_component, new_weight}
        };
        
        return hdc::weighted_bundle(weighted_behaviors);
    }
    
    // Online learning update
    void online_update(const std::vector<DemonstrationSample>& recent_samples) {
        if (recent_samples.empty()) {
            return;
        }
        
        // Update experience buffer
        for (const auto& sample : recent_samples) {
            experience_buffer_.push_back(sample);
            
            // Maintain buffer size
            if (experience_buffer_.size() > max_experience_buffer_size_) {
                experience_buffer_.erase(experience_buffer_.begin());
            }
        }
        
        // Perform consolidation if we have enough samples
        if (experience_buffer_.size() >= consolidation_threshold_) {
            consolidate_experience();
        }
    }
    
    // Few-shot learning: adapt behavior with minimal examples
    hdc::HyperVector few_shot_adaptation(const hdc::HyperVector& base_behavior,
                                        const std::vector<DemonstrationSample>& few_shot_samples) {
        if (few_shot_samples.empty()) {
            return base_behavior;
        }
        
        // Create adaptation vector from few-shot samples
        std::vector<hdc::HyperVector> adaptation_pairs;
        
        for (const auto& sample : few_shot_samples) {
            // Create perception-action association
            auto association = sample.perception.bind(sample.action);
            adaptation_pairs.push_back(association);
        }
        
        auto adaptation_hv = hdc::HyperVector::bundle_vectors(adaptation_pairs);
        
        // Combine with base behavior using low weight for adaptation
        double base_weight = 0.9;
        double adaptation_weight = 0.1;
        
        std::vector<std::pair<hdc::HyperVector, double>> weighted_combination = {
            {base_behavior, base_weight},
            {adaptation_hv, adaptation_weight}
        };
        
        return hdc::weighted_bundle(weighted_combination);
    }
    
    // Evaluate behavior quality against demonstration
    double evaluate_behavior(const hdc::HyperVector& behavior,
                           const std::vector<DemonstrationSample>& test_samples) {
        if (test_samples.empty()) {
            return 0.0;
        }
        
        double total_similarity = 0.0;
        int valid_samples = 0;
        
        for (const auto& sample : test_samples) {
            auto predicted_action = query_behavior(behavior, sample.perception);
            double similarity = sample.action.similarity(predicted_action);
            
            total_similarity += similarity;
            valid_samples++;
        }
        
        return valid_samples > 0 ? total_similarity / valid_samples : 0.0;
    }
    
    // Create behavior composition from multiple sub-behaviors
    hdc::HyperVector compose_behaviors(const std::vector<std::pair<hdc::HyperVector, double>>& behaviors) {
        if (behaviors.empty()) {
            return hdc::HyperVector::zero(dimension_);
        }
        
        return hdc::weighted_bundle(behaviors);
    }
    
    // Decompose behavior into components (approximate)
    std::vector<hdc::HyperVector> decompose_behavior(const hdc::HyperVector& behavior,
                                                    const std::vector<hdc::HyperVector>& component_basis) {
        std::vector<hdc::HyperVector> components;
        
        for (const auto& basis_vector : component_basis) {
            // Project behavior onto basis vector
            double similarity = behavior.similarity(basis_vector);
            
            if (similarity > similarity_threshold_) {
                // Extract component
                auto component = behavior.bind(basis_vector);
                components.push_back(component);
            }
        }
        
        return components;
    }
    
    // Novelty detection
    double compute_novelty(const hdc::HyperVector& sample) {
        if (experience_buffer_.empty()) {
            return 1.0;  // Everything is novel initially
        }
        
        double max_similarity = -1.0;
        
        // Find maximum similarity to existing experience
        for (const auto& exp_sample : experience_buffer_) {
            auto exp_state_action = exp_sample.perception.bind(exp_sample.action);
            double similarity = sample.similarity(exp_state_action);
            
            if (similarity > max_similarity) {
                max_similarity = similarity;
            }
        }
        
        // Novelty is inverse of maximum similarity
        return 1.0 - max_similarity;
    }
    
    // Transfer learning between behaviors
    hdc::HyperVector transfer_learning(const hdc::HyperVector& source_behavior,
                                      const hdc::HyperVector& target_context,
                                      double transfer_strength = 0.3) {
        // Create transferred behavior by binding source with target context
        auto transferred = source_behavior.bind(target_context);
        
        // Normalize and apply transfer strength
        transferred.normalize();
        
        // Weighted combination with zero (to apply strength)
        std::vector<std::pair<hdc::HyperVector, double>> weighted_transfer = {
            {transferred, transfer_strength},
            {hdc::HyperVector::zero(dimension_), 1.0 - transfer_strength}
        };
        
        return hdc::weighted_bundle(weighted_transfer);
    }
    
    // Meta-learning: learn to learn faster
    void meta_learning_update(const std::vector<std::vector<DemonstrationSample>>& task_samples) {
        if (task_samples.size() < 2) {
            return;  // Need multiple tasks for meta-learning
        }
        
        // Extract common patterns across tasks
        std::vector<hdc::HyperVector> task_representations;
        
        for (const auto& task : task_samples) {
            if (!task.empty()) {
                auto task_behavior = learn_from_demonstration(task);
                task_representations.push_back(task_behavior);
            }
        }
        
        if (task_representations.size() >= 2) {
            // Create meta-learning representation
            meta_learning_prior_ = hdc::HyperVector::bundle_vectors(task_representations);
            meta_learning_prior_.normalize();
        }
    }

private:
    void initialize_learning_parameters() {
        // Learning hyperparameters
        max_experience_buffer_size_ = 10000;
        consolidation_threshold_ = 100;
        learning_rate_ = 0.1;
        
        // Initialize meta-learning prior
        meta_learning_prior_ = hdc::HyperVector::zero(dimension_);
    }
    
    void consolidate_experience() {
        if (experience_buffer_.size() < consolidation_threshold_) {
            return;
        }
        
        // Group experiences by similarity
        std::vector<std::vector<DemonstrationSample>> experience_clusters;
        
        for (const auto& sample : experience_buffer_) {
            bool added_to_cluster = false;
            
            // Try to add to existing cluster
            for (auto& cluster : experience_clusters) {
                if (!cluster.empty()) {
                    double similarity = sample.perception.similarity(cluster[0].perception);
                    
                    if (similarity > 0.7) {  // Similarity threshold for clustering
                        cluster.push_back(sample);
                        added_to_cluster = true;
                        break;
                    }
                }
            }
            
            // Create new cluster if needed
            if (!added_to_cluster) {
                experience_clusters.push_back({sample});
            }
        }
        
        // Consolidate each cluster
        consolidated_behaviors_.clear();
        for (size_t i = 0; i < experience_clusters.size(); ++i) {
            if (!experience_clusters[i].empty()) {
                auto cluster_behavior = learn_from_demonstration(experience_clusters[i]);
                consolidated_behaviors_["cluster_" + std::to_string(i)] = cluster_behavior;
            }
        }
        
        // Clear experience buffer to make room for new experiences
        experience_buffer_.clear();
    }
    
    int dimension_;
    double similarity_threshold_;
    double learning_rate_;
    
    // Experience management
    std::vector<DemonstrationSample> experience_buffer_;
    size_t max_experience_buffer_size_;
    size_t consolidation_threshold_;
    
    // Consolidated knowledge
    std::map<std::string, hdc::HyperVector> consolidated_behaviors_;
    
    // Meta-learning
    hdc::HyperVector meta_learning_prior_;
};

}  // namespace hdc_robot_controller