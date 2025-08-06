#pragma once

#include <nav_msgs/msg/odometry.hpp>
#include <memory>
#include <vector>
#include <map>
#include <string>

#include "hypervector.hpp"

namespace hdc_robot_controller {

struct ControlResult {
    double linear_x{0.0};
    double linear_y{0.0};
    double angular_z{0.0};
    double confidence{0.0};
    std::string status{"ok"};
};

class FaultTolerantController {
public:
    FaultTolerantController(int dimension = 10000, double similarity_threshold = 0.8)
        : dimension_(dimension), similarity_threshold_(similarity_threshold) {
        initialize_control_memory();
    }
    
    ControlResult compute_control(
        const hdc::HyperVector& target_behavior,
        const hdc::HyperVector& current_perception,
        const nav_msgs::msg::Odometry::SharedPtr& current_odom) {
        
        ControlResult result;
        
        try {
            // Bind behavior with current perception to create context
            auto context_hv = target_behavior.bind(current_perception);
            
            // Add odometry information if available
            if (current_odom) {
                auto odom_hv = encode_odometry_state(*current_odom);
                context_hv = context_hv.bundle(odom_hv);
            }
            
            // Query control memory for best match
            auto best_match = find_best_control_match(context_hv);
            
            // Decode control commands from best match
            result = decode_control_commands(best_match);
            
            // Calculate confidence based on similarity
            result.confidence = context_hv.similarity(best_match);
            
            // Apply adaptive control based on confidence
            if (result.confidence < similarity_threshold_) {
                result = apply_conservative_control(result);
                result.status = "low_confidence";
            }
            
        } catch (const std::exception& e) {
            result.confidence = 0.0;
            result.status = "error";
            result.linear_x = 0.0;
            result.linear_y = 0.0;
            result.angular_z = 0.0;
        }
        
        return result;
    }
    
    void learn_control_association(
        const hdc::HyperVector& context,
        const ControlResult& control_action) {
        
        // Encode control action as hypervector
        auto control_hv = encode_control_action(control_action);
        
        // Store association in control memory
        control_memory_[context] = control_hv;
    }
    
    void update_sensor_weights(const std::map<std::string, double>& sensor_health) {
        sensor_weights_ = sensor_health;
    }

private:
    void initialize_control_memory() {
        // Create basic control patterns
        
        // Stop pattern
        auto stop_context = create_context_pattern("stop");
        auto stop_control = encode_control_action({0.0, 0.0, 0.0, 1.0, "stop"});
        control_memory_[stop_context] = stop_control;
        
        // Forward motion patterns
        for (int speed = 1; speed <= 5; ++speed) {
            double velocity = speed * 0.2;  // 0.2, 0.4, 0.6, 0.8, 1.0 m/s
            
            auto forward_context = create_context_pattern("forward_" + std::to_string(speed));
            auto forward_control = encode_control_action({velocity, 0.0, 0.0, 0.9, "forward"});
            control_memory_[forward_context] = forward_control;
        }
        
        // Turning patterns
        for (int turn_rate = 1; turn_rate <= 3; ++turn_rate) {
            double angular_vel = turn_rate * 0.3;  // 0.3, 0.6, 0.9 rad/s
            
            auto turn_left_context = create_context_pattern("turn_left_" + std::to_string(turn_rate));
            auto turn_left_control = encode_control_action({0.0, 0.0, angular_vel, 0.85, "turn_left"});
            control_memory_[turn_left_context] = turn_left_control;
            
            auto turn_right_context = create_context_pattern("turn_right_" + std::to_string(turn_rate));
            auto turn_right_control = encode_control_action({0.0, 0.0, -angular_vel, 0.85, "turn_right"});
            control_memory_[turn_right_context] = turn_right_control;
        }
        
        // Obstacle avoidance patterns
        auto avoid_context = create_context_pattern("avoid_obstacle");
        auto avoid_control = encode_control_action({0.1, 0.0, 0.8, 0.8, "avoid"});
        control_memory_[avoid_context] = avoid_control;
        
        // Complex maneuvers
        auto curve_left_context = create_context_pattern("curve_left");
        auto curve_left_control = encode_control_action({0.5, 0.0, 0.3, 0.8, "curve_left"});
        control_memory_[curve_left_context] = curve_left_control;
        
        auto curve_right_context = create_context_pattern("curve_right");
        auto curve_right_control = encode_control_action({0.5, 0.0, -0.3, 0.8, "curve_right"});
        control_memory_[curve_right_context] = curve_right_control;
    }
    
    hdc::HyperVector create_context_pattern(const std::string& pattern_name) {
        // Create a unique pattern for each context
        auto hash = std::hash<std::string>{}(pattern_name);
        return hdc::HyperVector::random(dimension_, hash % 100000 + 50000);
    }
    
    hdc::HyperVector encode_control_action(const ControlResult& action) {
        std::vector<hdc::HyperVector> components;
        
        // Discretize and encode each control component
        auto lin_x_hv = discretize_control_value(action.linear_x, -2.0, 2.0, 80);
        auto lin_y_hv = discretize_control_value(action.linear_y, -1.0, 1.0, 40);
        auto ang_z_hv = discretize_control_value(action.angular_z, -2.0, 2.0, 80);
        
        // Bind with component identifiers
        components.push_back(get_control_component("linear_x").bind(lin_x_hv));
        components.push_back(get_control_component("linear_y").bind(lin_y_hv));
        components.push_back(get_control_component("angular_z").bind(ang_z_hv));
        
        return hdc::HyperVector::bundle_vectors(components);
    }
    
    hdc::HyperVector encode_odometry_state(const nav_msgs::msg::Odometry& odom) {
        std::vector<hdc::HyperVector> components;
        
        // Encode velocity information
        auto vel_x = discretize_control_value(odom.twist.twist.linear.x, -3.0, 3.0, 60);
        auto vel_y = discretize_control_value(odom.twist.twist.linear.y, -3.0, 3.0, 60);
        auto ang_vel = discretize_control_value(odom.twist.twist.angular.z, -3.0, 3.0, 60);
        
        components.push_back(get_control_component("current_vel_x").bind(vel_x));
        components.push_back(get_control_component("current_vel_y").bind(vel_y));
        components.push_back(get_control_component("current_ang_vel").bind(ang_vel));
        
        return hdc::HyperVector::bundle_vectors(components);
    }
    
    hdc::HyperVector find_best_control_match(const hdc::HyperVector& context) {
        double best_similarity = -1.0;
        hdc::HyperVector best_match = hdc::HyperVector::zero(dimension_);
        
        for (const auto& [stored_context, stored_control] : control_memory_) {
            double similarity = context.similarity(stored_context);
            
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_match = stored_control;
            }
        }
        
        return best_match;
    }
    
    ControlResult decode_control_commands(const hdc::HyperVector& control_hv) {
        ControlResult result;
        
        // Create component vectors for decoding
        auto lin_x_component = get_control_component("linear_x");
        auto lin_y_component = get_control_component("linear_y");
        auto ang_z_component = get_control_component("angular_z");
        
        // Unbind components (approximate decoding)
        auto lin_x_bound = control_hv.bind(lin_x_component);
        auto lin_y_bound = control_hv.bind(lin_y_component);
        auto ang_z_bound = control_hv.bind(ang_z_component);
        
        // Decode by finding best matches in discretized space
        result.linear_x = decode_control_value(lin_x_bound, -2.0, 2.0, 80);
        result.linear_y = decode_control_value(lin_y_bound, -1.0, 1.0, 40);
        result.angular_z = decode_control_value(ang_z_bound, -2.0, 2.0, 80);
        
        return result;
    }
    
    double decode_control_value(const hdc::HyperVector& encoded_value, 
                               double min_val, double max_val, int levels) {
        double best_similarity = -1.0;
        int best_level = 0;
        
        // Find best matching discretized level
        for (int level = 0; level < levels; ++level) {
            auto level_hv = hdc::HyperVector::random(dimension_, level + 60000);
            double similarity = encoded_value.similarity(level_hv);
            
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_level = level;
            }
        }
        
        // Convert back to continuous value
        return min_val + (best_level * (max_val - min_val) / (levels - 1));
    }
    
    hdc::HyperVector discretize_control_value(double value, double min_val, double max_val, int levels) {
        if (value < min_val) value = min_val;
        if (value > max_val) value = max_val;
        
        int level = static_cast<int>((value - min_val) / (max_val - min_val) * (levels - 1));
        level = std::min(levels - 1, std::max(0, level));
        
        return hdc::HyperVector::random(dimension_, level + 60000);
    }
    
    hdc::HyperVector get_control_component(const std::string& name) {
        auto hash = std::hash<std::string>{}(name);
        return hdc::HyperVector::random(dimension_, hash % 100000 + 70000);
    }
    
    ControlResult apply_conservative_control(const ControlResult& original) {
        ControlResult conservative = original;
        
        // Reduce velocities for low confidence
        conservative.linear_x *= 0.5;
        conservative.linear_y *= 0.5;
        conservative.angular_z *= 0.7;
        
        // Ensure we don't exceed conservative limits
        const double max_conservative_linear = 0.3;
        const double max_conservative_angular = 0.5;
        
        conservative.linear_x = std::max(-max_conservative_linear, 
                                       std::min(max_conservative_linear, conservative.linear_x));
        conservative.linear_y = std::max(-max_conservative_linear, 
                                       std::min(max_conservative_linear, conservative.linear_y));
        conservative.angular_z = std::max(-max_conservative_angular, 
                                        std::min(max_conservative_angular, conservative.angular_z));
        
        return conservative;
    }
    
    int dimension_;
    double similarity_threshold_;
    
    // Control memory: context -> control action
    std::map<hdc::HyperVector, hdc::HyperVector> control_memory_;
    
    // Sensor health weights
    std::map<std::string, double> sensor_weights_;
};

}  // namespace hdc_robot_controller