#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int8_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <memory>
#include <chrono>
#include <vector>
#include <fstream>

#include "behavior_learner.hpp"
#include "hypervector.hpp"

namespace hdc_robot_controller {

struct DemonstrationSample {
    hdc::HyperVector perception;
    hdc::HyperVector action;
    rclcpp::Time timestamp;
    double reward{0.0};
};

class LearningNode : public rclcpp::Node {
public:
    LearningNode() : Node("hdc_learning_node") {
        // Initialize behavior learner
        learner_ = std::make_unique<BehaviorLearner>(10000, 0.85);
        
        // Subscriber for perception hypervectors
        perception_sub_ = this->create_subscription<std_msgs::msg::Int8MultiArray>(
            "/hdc/perception/hypervector", 10,
            std::bind(&LearningNode::perception_callback, this, std::placeholders::_1));
            
        // Subscriber for action commands
        action_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel", 10,
            std::bind(&LearningNode::action_callback, this, std::placeholders::_1));
            
        // Subscriber for learning commands
        learning_cmd_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/hdc/learning/command", 10,
            std::bind(&LearningNode::learning_command_callback, this, std::placeholders::_1));
            
        // Subscriber for recording control
        record_sub_ = this->create_subscription<std_msgs::msg::Bool>(
            "/hdc/learning/record", 10,
            std::bind(&LearningNode::record_callback, this, std::placeholders::_1));
        
        // Publisher for learned behaviors
        behavior_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/hdc/behavior/learned", 10);
            
        // Publisher for learning status
        status_pub_ = this->create_publisher<std_msgs::msg::String>(
            "/hdc/learning/status", 10);
        
        // Timer for learning updates
        learning_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),  // 10Hz
            std::bind(&LearningNode::learning_update, this));
            
        // Initialize state
        recording_ = false;
        current_behavior_name_ = "";
        demonstration_count_ = 0;
        
        // Load existing behaviors
        load_learned_behaviors();
        
        RCLCPP_INFO(this->get_logger(), "HDC Learning Node initialized");
        publish_status("Learning node ready - use /hdc/learning/command to control");
    }

private:
    void perception_callback(const std_msgs::msg::Int8MultiArray::SharedPtr msg) {
        if (msg->data.size() != 10000) {
            return;
        }
        
        // Convert to HyperVector
        std::vector<int8_t> hv_data(msg->data.begin(), msg->data.end());
        current_perception_ = std::make_shared<hdc::HyperVector>(hv_data);
        last_perception_time_ = this->get_clock()->now();
    }
    
    void action_callback(const geometry_msgs::msg::Twist::SharedPtr msg) {
        // Encode action as hypervector
        current_action_ = std::make_shared<hdc::HyperVector>(encode_action(*msg));
        last_action_time_ = this->get_clock()->now();
    }
    
    void learning_command_callback(const std_msgs::msg::String::SharedPtr msg) {
        std::string command = msg->data;
        
        if (command.substr(0, 11) == "start_demo:") {
            // Extract behavior name
            current_behavior_name_ = command.substr(11);
            start_demonstration(current_behavior_name_);
        }
        else if (command == "stop_demo") {
            stop_demonstration();
        }
        else if (command.substr(0, 6) == "learn:") {
            // Learn from recorded demonstrations
            std::string behavior_name = command.substr(6);
            learn_behavior(behavior_name);
        }
        else if (command.substr(0, 5) == "save:") {
            std::string behavior_name = command.substr(5);
            save_behavior(behavior_name);
        }
        else if (command.substr(0, 5) == "load:") {
            std::string behavior_name = command.substr(5);
            load_behavior(behavior_name);
        }
        else if (command == "clear_demos") {
            clear_demonstrations();
        }
        else if (command == "list_behaviors") {
            list_learned_behaviors();
        }
        else {
            RCLCPP_WARN(this->get_logger(), "Unknown learning command: %s", command.c_str());
        }
    }
    
    void record_callback(const std_msgs::msg::Bool::SharedPtr msg) {
        if (msg->data && !current_behavior_name_.empty()) {
            recording_ = true;
            RCLCPP_INFO(this->get_logger(), "Started recording for behavior: %s", 
                       current_behavior_name_.c_str());
        } else {
            recording_ = false;
            RCLCPP_INFO(this->get_logger(), "Stopped recording");
        }
    }
    
    void learning_update() {
        // Record demonstration samples if actively recording
        if (recording_ && current_perception_ && current_action_) {
            auto now = this->get_clock()->now();
            
            // Check if data is fresh (within 200ms)
            if ((now - last_perception_time_).seconds() < 0.2 && 
                (now - last_action_time_).seconds() < 0.2) {
                
                DemonstrationSample sample;
                sample.perception = *current_perception_;
                sample.action = *current_action_;
                sample.timestamp = now;
                sample.reward = 1.0;  // Default positive reward
                
                demonstration_buffer_.push_back(sample);
                
                // Limit buffer size
                if (demonstration_buffer_.size() > max_buffer_size_) {
                    demonstration_buffer_.erase(demonstration_buffer_.begin());
                }
                
                // Publish status every 50 samples
                if (demonstration_buffer_.size() % 50 == 0) {
                    publish_status("Recording: " + std::to_string(demonstration_buffer_.size()) + 
                                 " samples for behavior '" + current_behavior_name_ + "'");
                }
            }
        }
        
        // Perform continual learning updates
        if (!demonstration_buffer_.empty() && 
            learning_update_counter_++ % 50 == 0) {  // Every 5 seconds
            
            continual_learning_update();
        }
    }
    
    void start_demonstration(const std::string& behavior_name) {
        if (behavior_name.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Empty behavior name");
            return;
        }
        
        current_behavior_name_ = behavior_name;
        demonstration_buffer_.clear();
        demonstration_count_++;
        
        publish_status("Ready to record demonstration for: " + behavior_name);
        RCLCPP_INFO(this->get_logger(), "Starting demonstration recording for: %s", 
                   behavior_name.c_str());
    }
    
    void stop_demonstration() {
        recording_ = false;
        
        if (!demonstration_buffer_.empty()) {
            // Store demonstration
            stored_demonstrations_[current_behavior_name_] = demonstration_buffer_;
            
            publish_status("Recorded " + std::to_string(demonstration_buffer_.size()) + 
                         " samples for behavior '" + current_behavior_name_ + "'");
            
            RCLCPP_INFO(this->get_logger(), "Stopped recording. Collected %zu samples for %s",
                       demonstration_buffer_.size(), current_behavior_name_.c_str());
        }
    }
    
    void learn_behavior(const std::string& behavior_name) {
        auto demo_it = stored_demonstrations_.find(behavior_name);
        if (demo_it == stored_demonstrations_.end()) {
            RCLCPP_ERROR(this->get_logger(), "No demonstrations found for behavior: %s", 
                        behavior_name.c_str());
            return;
        }
        
        const auto& samples = demo_it->second;
        if (samples.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Empty demonstration for behavior: %s", 
                        behavior_name.c_str());
            return;
        }
        
        try {
            // Learn behavior using HDC
            auto learned_behavior = learner_->learn_from_demonstration(samples);
            
            // Store learned behavior
            learned_behaviors_[behavior_name] = learned_behavior;
            
            // Evaluate learning quality
            double quality = evaluate_learning_quality(behavior_name, samples);
            
            publish_status("Learned behavior '" + behavior_name + "' from " + 
                         std::to_string(samples.size()) + " samples (quality: " + 
                         std::to_string(quality) + ")");
            
            // Announce new behavior
            std_msgs::msg::String behavior_msg;
            behavior_msg.data = behavior_name;
            behavior_pub_->publish(behavior_msg);
            
            RCLCPP_INFO(this->get_logger(), 
                       "Successfully learned behavior '%s' from %zu samples (quality: %.3f)",
                       behavior_name.c_str(), samples.size(), quality);
                       
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to learn behavior '%s': %s", 
                        behavior_name.c_str(), e.what());
        }
    }
    
    void continual_learning_update() {
        if (demonstration_buffer_.size() < 10) {
            return;  // Need minimum samples
        }
        
        try {
            // Perform online learning with recent samples
            std::vector<DemonstrationSample> recent_samples(
                demonstration_buffer_.end() - std::min(demonstration_buffer_.size(), size_t{20}),
                demonstration_buffer_.end()
            );
            
            learner_->online_update(recent_samples);
            
            // Update existing behaviors with new experience
            for (auto& [name, behavior] : learned_behaviors_) {
                if (name == current_behavior_name_) {
                    behavior = learner_->adapt_behavior(behavior, recent_samples);
                }
            }
            
        } catch (const std::exception& e) {
            RCLCPP_WARN(this->get_logger(), "Continual learning update failed: %s", e.what());
        }
    }
    
    double evaluate_learning_quality(const std::string& behavior_name, 
                                   const std::vector<DemonstrationSample>& samples) {
        auto behavior_it = learned_behaviors_.find(behavior_name);
        if (behavior_it == learned_behaviors_.end()) {
            return 0.0;
        }
        
        const auto& learned_behavior = behavior_it->second;
        double total_similarity = 0.0;
        int valid_samples = 0;
        
        // Check how well the learned behavior matches the demonstration
        for (const auto& sample : samples) {
            auto expected_action = learner_->query_behavior(learned_behavior, sample.perception);
            double similarity = sample.action.similarity(expected_action);
            
            total_similarity += similarity;
            valid_samples++;
        }
        
        return valid_samples > 0 ? total_similarity / valid_samples : 0.0;
    }
    
    hdc::HyperVector encode_action(const geometry_msgs::msg::Twist& action) {
        std::vector<hdc::HyperVector> components;
        
        // Discretize action components
        auto lin_x = discretize_value(action.linear.x, -2.0, 2.0, 80);
        auto lin_y = discretize_value(action.linear.y, -1.0, 1.0, 40);
        auto ang_z = discretize_value(action.angular.z, -2.0, 2.0, 80);
        
        // Bind with component identifiers
        components.push_back(get_action_component("linear_x").bind(lin_x));
        components.push_back(get_action_component("linear_y").bind(lin_y));
        components.push_back(get_action_component("angular_z").bind(ang_z));
        
        return hdc::HyperVector::bundle_vectors(components);
    }
    
    hdc::HyperVector discretize_value(double value, double min_val, double max_val, int levels) {
        if (value < min_val) value = min_val;
        if (value > max_val) value = max_val;
        
        int level = static_cast<int>((value - min_val) / (max_val - min_val) * (levels - 1));
        level = std::min(levels - 1, std::max(0, level));
        
        return hdc::HyperVector::random(10000, level + 80000);
    }
    
    hdc::HyperVector get_action_component(const std::string& name) {
        auto hash = std::hash<std::string>{}(name);
        return hdc::HyperVector::random(10000, hash % 100000 + 90000);
    }
    
    void save_behavior(const std::string& behavior_name) {
        auto behavior_it = learned_behaviors_.find(behavior_name);
        if (behavior_it == learned_behaviors_.end()) {
            RCLCPP_ERROR(this->get_logger(), "Behavior not found: %s", behavior_name.c_str());
            return;
        }
        
        try {
            std::string filename = "behaviors/" + behavior_name + ".hv";
            std::ofstream file(filename, std::ios::binary);
            
            if (!file.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Cannot open file for writing: %s", filename.c_str());
                return;
            }
            
            // Save hypervector data
            auto data = behavior_it->second.to_bytes();
            file.write(reinterpret_cast<const char*>(data.data()), data.size());
            file.close();
            
            publish_status("Saved behavior '" + behavior_name + "' to " + filename);
            RCLCPP_INFO(this->get_logger(), "Saved behavior to %s", filename.c_str());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to save behavior '%s': %s", 
                        behavior_name.c_str(), e.what());
        }
    }
    
    void load_behavior(const std::string& behavior_name) {
        try {
            std::string filename = "behaviors/" + behavior_name + ".hv";
            std::ifstream file(filename, std::ios::binary);
            
            if (!file.is_open()) {
                RCLCPP_ERROR(this->get_logger(), "Cannot open file for reading: %s", filename.c_str());
                return;
            }
            
            // Read hypervector data
            std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                                     std::istreambuf_iterator<char>());
            file.close();
            
            // Reconstruct hypervector
            hdc::HyperVector loaded_behavior(10000);
            loaded_behavior.from_bytes(data);
            
            learned_behaviors_[behavior_name] = loaded_behavior;
            
            publish_status("Loaded behavior '" + behavior_name + "' from " + filename);
            RCLCPP_INFO(this->get_logger(), "Loaded behavior from %s", filename.c_str());
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Failed to load behavior '%s': %s", 
                        behavior_name.c_str(), e.what());
        }
    }
    
    void load_learned_behaviors() {
        // Try to load any existing behavior files
        // This is a simplified version - in practice you'd scan the behaviors directory
        std::vector<std::string> behavior_files = {
            "pick_and_place", "navigate_corridor", "follow_wall", 
            "avoid_obstacles", "dock_station"
        };
        
        for (const auto& behavior : behavior_files) {
            load_behavior(behavior);
        }
    }
    
    void clear_demonstrations() {
        stored_demonstrations_.clear();
        demonstration_buffer_.clear();
        current_behavior_name_ = "";
        recording_ = false;
        
        publish_status("Cleared all demonstrations");
        RCLCPP_INFO(this->get_logger(), "Cleared all stored demonstrations");
    }
    
    void list_learned_behaviors() {
        std::string behavior_list = "Learned behaviors: ";
        for (const auto& [name, _] : learned_behaviors_) {
            behavior_list += name + ", ";
        }
        
        if (behavior_list.size() > 20) {
            behavior_list = behavior_list.substr(0, behavior_list.size() - 2);  // Remove last ", "
        } else {
            behavior_list += "(none)";
        }
        
        publish_status(behavior_list);
        RCLCPP_INFO(this->get_logger(), "%s", behavior_list.c_str());
    }
    
    void publish_status(const std::string& status) {
        std_msgs::msg::String status_msg;
        status_msg.data = status;
        status_pub_->publish(status_msg);
    }

    // Core components
    std::unique_ptr<BehaviorLearner> learner_;
    
    // ROS interfaces
    rclcpp::Subscription<std_msgs::msg::Int8MultiArray>::SharedPtr perception_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr action_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr learning_cmd_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr record_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr behavior_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
    
    rclcpp::TimerBase::SharedPtr learning_timer_;
    
    // Learning state
    bool recording_;
    std::string current_behavior_name_;
    
    std::shared_ptr<hdc::HyperVector> current_perception_;
    std::shared_ptr<hdc::HyperVector> current_action_;
    
    rclcpp::Time last_perception_time_;
    rclcpp::Time last_action_time_;
    
    // Demonstration data
    std::vector<DemonstrationSample> demonstration_buffer_;
    std::map<std::string, std::vector<DemonstrationSample>> stored_demonstrations_;
    std::map<std::string, hdc::HyperVector> learned_behaviors_;
    
    // Configuration
    static constexpr size_t max_buffer_size_ = 10000;
    size_t demonstration_count_;
    size_t learning_update_counter_{0};
};

}  // namespace hdc_robot_controller

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<hdc_robot_controller::LearningNode>();
    
    RCLCPP_INFO(node->get_logger(), "Starting HDC Learning Node");
    
    // Create behaviors directory if it doesn't exist
    system("mkdir -p behaviors");
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(node->get_logger(), "Node crashed: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}