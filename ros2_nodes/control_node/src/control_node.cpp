#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/int8_multi_array.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/string.hpp>
#include <memory>
#include <chrono>
#include <string>

#include "fault_tolerant_controller.hpp"
#include "hypervector.hpp"

namespace hdc_robot_controller {

class ControlNode : public rclcpp::Node {
public:
    ControlNode() : Node("hdc_control_node") {
        // Initialize fault-tolerant controller
        controller_ = std::make_unique<FaultTolerantController>(10000, 0.85);
        
        // Subscriber for fused perception hypervectors
        perception_sub_ = this->create_subscription<std_msgs::msg::Int8MultiArray>(
            "/hdc/perception/hypervector", 10,
            std::bind(&ControlNode::perception_callback, this, std::placeholders::_1));
            
        // Subscriber for behavior commands
        behavior_sub_ = this->create_subscription<std_msgs::msg::String>(
            "/hdc/behavior/command", 10,
            std::bind(&ControlNode::behavior_callback, this, std::placeholders::_1));
            
        // Subscriber for current odometry (for state feedback)
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&ControlNode::odom_callback, this, std::placeholders::_1));
        
        // Publishers for control commands
        cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        joint_cmd_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_commands", 10);
        
        // Publisher for control diagnostics
        diagnostics_pub_ = this->create_publisher<std_msgs::msg::String>("/hdc/control/diagnostics", 10);
        
        // Control loop timer (50Hz for real-time control)
        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(20),  
            std::bind(&ControlNode::control_loop, this));
            
        // Initialize default behaviors
        initialize_behaviors();
        
        // State variables
        current_behavior_ = "idle";
        control_active_ = false;
        confidence_threshold_ = 0.6;
        safety_mode_ = false;
        
        RCLCPP_INFO(this->get_logger(), "HDC Control Node initialized");
    }

private:
    void perception_callback(const std_msgs::msg::Int8MultiArray::SharedPtr msg) {
        if (msg->data.size() != 10000) {
            RCLCPP_WARN(this->get_logger(), "Invalid hypervector dimension: %zu", msg->data.size());
            return;
        }
        
        // Convert ROS message to HyperVector
        std::vector<int8_t> hv_data(msg->data.begin(), msg->data.end());
        current_perception_ = std::make_shared<hdc::HyperVector>(hv_data);
        
        // Update timestamp
        last_perception_time_ = this->get_clock()->now();
    }
    
    void behavior_callback(const std_msgs::msg::String::SharedPtr msg) {
        std::string new_behavior = msg->data;
        
        if (behaviors_.find(new_behavior) != behaviors_.end()) {
            current_behavior_ = new_behavior;
            control_active_ = true;
            RCLCPP_INFO(this->get_logger(), "Switching to behavior: %s", new_behavior.c_str());
        } else if (new_behavior == "stop" || new_behavior == "idle") {
            current_behavior_ = "idle";
            control_active_ = false;
            publish_zero_commands();
            RCLCPP_INFO(this->get_logger(), "Stopping control");
        } else {
            RCLCPP_WARN(this->get_logger(), "Unknown behavior: %s", new_behavior.c_str());
        }
    }
    
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_odom_ = msg;
    }
    
    void control_loop() {
        if (!control_active_ || current_behavior_ == "idle") {
            return;
        }
        
        // Check if we have recent perception data
        auto now = this->get_clock()->now();
        if (!current_perception_ || 
            (now - last_perception_time_).seconds() > 0.1) {  // 100ms timeout
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                "No recent perception data - entering safety mode");
            enter_safety_mode();
            return;
        }
        
        try {
            // Get target behavior hypervector
            auto behavior_it = behaviors_.find(current_behavior_);
            if (behavior_it == behaviors_.end()) {
                RCLCPP_ERROR(this->get_logger(), "Behavior not found: %s", current_behavior_.c_str());
                return;
            }
            
            // Query associative memory for best action
            auto control_result = controller_->compute_control(
                behavior_it->second, 
                *current_perception_,
                current_odom_
            );
            
            // Check confidence
            if (control_result.confidence < confidence_threshold_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                    "Low confidence control: %.3f (threshold: %.3f)", 
                                    control_result.confidence, confidence_threshold_);
                
                if (control_result.confidence < 0.3) {
                    enter_safety_mode();
                    return;
                }
            }
            
            // Apply safety constraints
            auto safe_commands = apply_safety_constraints(control_result);
            
            // Publish control commands
            publish_commands(safe_commands);
            
            // Publish diagnostics
            publish_diagnostics(control_result);
            
            // Exit safety mode if we were in it
            if (safety_mode_) {
                safety_mode_ = false;
                RCLCPP_INFO(this->get_logger(), "Exiting safety mode");
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Control loop error: %s", e.what());
            enter_safety_mode();
        }
    }
    
    void initialize_behaviors() {
        // Create basic behavior hypervectors
        
        // Idle behavior - zero motion
        behaviors_["idle"] = hdc::HyperVector::zero(10000);
        
        // Forward motion behavior
        auto forward_components = create_motion_components(1.0, 0.0, 0.0);  // linear.x = 1.0
        behaviors_["move_forward"] = hdc::HyperVector::bundle_vectors(forward_components);
        
        // Backward motion
        auto backward_components = create_motion_components(-0.5, 0.0, 0.0);
        behaviors_["move_backward"] = hdc::HyperVector::bundle_vectors(backward_components);
        
        // Turn left
        auto turn_left_components = create_motion_components(0.0, 0.0, 1.0);  // angular.z = 1.0
        behaviors_["turn_left"] = hdc::HyperVector::bundle_vectors(turn_left_components);
        
        // Turn right  
        auto turn_right_components = create_motion_components(0.0, 0.0, -1.0);
        behaviors_["turn_right"] = hdc::HyperVector::bundle_vectors(turn_right_components);
        
        // Obstacle avoidance
        auto avoid_components = create_motion_components(0.2, 0.0, 0.5);
        behaviors_["avoid_obstacle"] = hdc::HyperVector::bundle_vectors(avoid_components);
        
        // Follow wall
        auto follow_components = create_motion_components(0.5, 0.3, 0.1);
        behaviors_["follow_wall"] = hdc::HyperVector::bundle_vectors(follow_components);
        
        RCLCPP_INFO(this->get_logger(), "Initialized %zu behaviors", behaviors_.size());
    }
    
    std::vector<hdc::HyperVector> create_motion_components(double linear_x, double linear_y, double angular_z) {
        std::vector<hdc::HyperVector> components;
        
        // Discretize motion commands
        auto lin_x_hv = discretize_value(linear_x, -2.0, 2.0, 40);
        auto lin_y_hv = discretize_value(linear_y, -1.0, 1.0, 20);
        auto ang_z_hv = discretize_value(angular_z, -2.0, 2.0, 40);
        
        // Bind with component identifiers
        auto cmd_linear_x = get_component_vector("cmd_linear_x").bind(lin_x_hv);
        auto cmd_linear_y = get_component_vector("cmd_linear_y").bind(lin_y_hv);
        auto cmd_angular_z = get_component_vector("cmd_angular_z").bind(ang_z_hv);
        
        components.push_back(cmd_linear_x);
        components.push_back(cmd_linear_y);
        components.push_back(cmd_angular_z);
        
        return components;
    }
    
    hdc::HyperVector discretize_value(double value, double min_val, double max_val, int levels) {
        if (value < min_val) value = min_val;
        if (value > max_val) value = max_val;
        
        int level = static_cast<int>((value - min_val) / (max_val - min_val) * (levels - 1));
        level = std::min(levels - 1, std::max(0, level));
        
        return hdc::HyperVector::random(10000, level + 20000);
    }
    
    hdc::HyperVector get_component_vector(const std::string& name) {
        auto hash = std::hash<std::string>{}(name);
        return hdc::HyperVector::random(10000, hash % 100000 + 30000);
    }
    
    void enter_safety_mode() {
        if (!safety_mode_) {
            RCLCPP_WARN(this->get_logger(), "Entering safety mode - stopping robot");
            safety_mode_ = true;
        }
        publish_zero_commands();
    }
    
    void publish_zero_commands() {
        // Stop all motion
        geometry_msgs::msg::Twist stop_cmd;
        stop_cmd.linear.x = 0.0;
        stop_cmd.linear.y = 0.0;
        stop_cmd.linear.z = 0.0;
        stop_cmd.angular.x = 0.0;
        stop_cmd.angular.y = 0.0;
        stop_cmd.angular.z = 0.0;
        
        cmd_vel_pub_->publish(stop_cmd);
    }
    
    ControlResult apply_safety_constraints(const ControlResult& original) {
        ControlResult safe_result = original;
        
        // Limit maximum velocities
        const double max_linear = 1.5;   // m/s
        const double max_angular = 1.0;  // rad/s
        
        if (std::abs(safe_result.linear_x) > max_linear) {
            safe_result.linear_x = std::copysign(max_linear, safe_result.linear_x);
        }
        if (std::abs(safe_result.linear_y) > max_linear) {
            safe_result.linear_y = std::copysign(max_linear, safe_result.linear_y);
        }
        if (std::abs(safe_result.angular_z) > max_angular) {
            safe_result.angular_z = std::copysign(max_angular, safe_result.angular_z);
        }
        
        // Apply confidence-based scaling
        double confidence_scale = std::max(0.1, safe_result.confidence);
        safe_result.linear_x *= confidence_scale;
        safe_result.linear_y *= confidence_scale;
        safe_result.angular_z *= confidence_scale;
        
        return safe_result;
    }
    
    void publish_commands(const ControlResult& control_result) {
        // Publish velocity command
        geometry_msgs::msg::Twist cmd_vel;
        cmd_vel.linear.x = control_result.linear_x;
        cmd_vel.linear.y = control_result.linear_y;
        cmd_vel.linear.z = 0.0;
        cmd_vel.angular.x = 0.0;
        cmd_vel.angular.y = 0.0;
        cmd_vel.angular.z = control_result.angular_z;
        
        cmd_vel_pub_->publish(cmd_vel);
        
        // Update statistics
        control_count_++;
    }
    
    void publish_diagnostics(const ControlResult& control_result) {
        if (control_count_ % 50 == 0) {  // Every 1 second at 50Hz
            std::string diagnostics = "HDC Control - Behavior: " + current_behavior_ + 
                                    ", Confidence: " + std::to_string(control_result.confidence) +
                                    ", Safety: " + (safety_mode_ ? "ON" : "OFF");
            
            std_msgs::msg::String diag_msg;
            diag_msg.data = diagnostics;
            diagnostics_pub_->publish(diag_msg);
        }
    }

    // Core components
    std::unique_ptr<FaultTolerantController> controller_;
    
    // ROS interfaces
    rclcpp::Subscription<std_msgs::msg::Int8MultiArray>::SharedPtr perception_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr behavior_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_cmd_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr diagnostics_pub_;
    
    rclcpp::TimerBase::SharedPtr control_timer_;
    
    // State
    std::map<std::string, hdc::HyperVector> behaviors_;
    std::shared_ptr<hdc::HyperVector> current_perception_;
    nav_msgs::msg::Odometry::SharedPtr current_odom_;
    
    std::string current_behavior_;
    bool control_active_;
    bool safety_mode_;
    double confidence_threshold_;
    
    rclcpp::Time last_perception_time_;
    size_t control_count_{0};
};

}  // namespace hdc_robot_controller

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<hdc_robot_controller::ControlNode>();
    
    RCLCPP_INFO(node->get_logger(), "Starting HDC Control Node");
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(node->get_logger(), "Node crashed: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}