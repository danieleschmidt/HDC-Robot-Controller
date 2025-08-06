#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "multimodal_encoder.hpp"
#include "hypervector.hpp"

namespace hdc_robot_controller {

class PerceptionNode : public rclcpp::Node {
public:
    PerceptionNode() : Node("hdc_perception_node") {
        // Initialize HDC encoder
        encoder_ = std::make_unique<MultimodalEncoder>(10000);
        
        // Subscribers for sensor data
        lidar_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
            "/scan", 10,
            std::bind(&PerceptionNode::lidar_callback, this, std::placeholders::_1));
            
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10,
            std::bind(&PerceptionNode::image_callback, this, std::placeholders::_1));
            
        imu_sub_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu/data", 10,
            std::bind(&PerceptionNode::imu_callback, this, std::placeholders::_1));
            
        joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
            "/joint_states", 10,
            std::bind(&PerceptionNode::joint_callback, this, std::placeholders::_1));
            
        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10,
            std::bind(&PerceptionNode::odom_callback, this, std::placeholders::_1));
        
        // Publisher for fused hypervector perception
        hypervector_pub_ = this->create_publisher<std_msgs::msg::Int8MultiArray>(
            "/hdc/perception/hypervector", 10);
            
        // Timer for periodic fusion
        fusion_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50),  // 20Hz
            std::bind(&PerceptionNode::fusion_callback, this));
            
        // Initialize sensor availability flags
        lidar_available_ = false;
        camera_available_ = false;
        imu_available_ = false;
        joints_available_ = false;
        odom_available_ = false;
        
        // Initialize sensor health monitoring
        sensor_health_.insert({"lidar", 1.0});
        sensor_health_.insert({"camera", 1.0});
        sensor_health_.insert({"imu", 1.0});
        sensor_health_.insert({"joints", 1.0});
        sensor_health_.insert({"odometry", 1.0});
        
        RCLCPP_INFO(this->get_logger(), "HDC Perception Node initialized");
    }

private:
    void lidar_callback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
        lidar_data_ = *msg;
        lidar_available_ = true;
        update_sensor_health("lidar", msg->header.stamp);
    }
    
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            image_data_ = cv_ptr->image.clone();
            camera_available_ = true;
            update_sensor_health("camera", msg->header.stamp);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge exception: %s", e.what());
            sensor_health_["camera"] = 0.0;
        }
    }
    
    void imu_callback(const sensor_msgs::msg::Imu::SharedPtr msg) {
        imu_data_ = *msg;
        imu_available_ = true;
        update_sensor_health("imu", msg->header.stamp);
    }
    
    void joint_callback(const sensor_msgs::msg::JointState::SharedPtr msg) {
        joint_data_ = *msg;
        joints_available_ = true;
        update_sensor_health("joints", msg->header.stamp);
    }
    
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        odom_data_ = *msg;
        odom_available_ = true;
        update_sensor_health("odometry", msg->header.stamp);
    }
    
    void update_sensor_health(const std::string& sensor_name, const builtin_interfaces::msg::Time& timestamp) {
        auto now = this->get_clock()->now();
        auto msg_time = rclcpp::Time(timestamp);
        auto age = (now - msg_time).seconds();
        
        // Degrade health based on message age
        double health = std::max(0.0, 1.0 - (age / 1.0));  // 1 second timeout
        sensor_health_[sensor_name] = health;
        
        if (health < 0.5) {
            RCLCPP_WARN(this->get_logger(), "Sensor %s health degraded: %.2f", 
                       sensor_name.c_str(), health);
        }
    }
    
    void fusion_callback() {
        try {
            // Encode available sensor data
            std::vector<hdc::HyperVector> sensor_vectors;
            std::vector<double> sensor_weights;
            
            // Encode LIDAR data if available
            if (lidar_available_ && sensor_health_["lidar"] > 0.1) {
                auto lidar_hv = encoder_->encode_lidar(lidar_data_);
                sensor_vectors.push_back(lidar_hv);
                sensor_weights.push_back(sensor_health_["lidar"]);
            }
            
            // Encode camera data if available
            if (camera_available_ && sensor_health_["camera"] > 0.1) {
                auto camera_hv = encoder_->encode_image(image_data_);
                sensor_vectors.push_back(camera_hv);
                sensor_weights.push_back(sensor_health_["camera"]);
            }
            
            // Encode IMU data if available
            if (imu_available_ && sensor_health_["imu"] > 0.1) {
                auto imu_hv = encoder_->encode_imu(imu_data_);
                sensor_vectors.push_back(imu_hv);
                sensor_weights.push_back(sensor_health_["imu"]);
            }
            
            // Encode joint states if available
            if (joints_available_ && sensor_health_["joints"] > 0.1) {
                auto joints_hv = encoder_->encode_joints(joint_data_);
                sensor_vectors.push_back(joints_hv);
                sensor_weights.push_back(sensor_health_["joints"]);
            }
            
            // Encode odometry if available
            if (odom_available_ && sensor_health_["odometry"] > 0.1) {
                auto odom_hv = encoder_->encode_odometry(odom_data_);
                sensor_vectors.push_back(odom_hv);
                sensor_weights.push_back(sensor_health_["odometry"]);
            }
            
            // Fuse sensors if we have any data
            if (!sensor_vectors.empty()) {
                // Create weighted pairs
                std::vector<std::pair<hdc::HyperVector, double>> weighted_pairs;
                for (size_t i = 0; i < sensor_vectors.size(); ++i) {
                    weighted_pairs.emplace_back(sensor_vectors[i], sensor_weights[i]);
                }
                
                // Weighted bundle
                auto fused_hv = hdc::weighted_bundle(weighted_pairs);
                
                // Convert to ROS message
                auto msg = std_msgs::msg::Int8MultiArray();
                msg.data.resize(fused_hv.dimension());
                for (int i = 0; i < fused_hv.dimension(); ++i) {
                    msg.data[i] = fused_hv[i];
                }
                
                // Add metadata
                msg.layout.dim.resize(1);
                msg.layout.dim[0].label = "hypervector";
                msg.layout.dim[0].size = fused_hv.dimension();
                msg.layout.dim[0].stride = 1;
                
                hypervector_pub_->publish(msg);
                
                // Log fusion statistics
                if (++fusion_count_ % 100 == 0) {  // Every 5 seconds at 20Hz
                    RCLCPP_INFO(this->get_logger(), 
                               "Fused %zu sensors: LIDAR=%.2f, CAM=%.2f, IMU=%.2f, JOINTS=%.2f, ODOM=%.2f",
                               sensor_vectors.size(),
                               sensor_health_["lidar"], sensor_health_["camera"], 
                               sensor_health_["imu"], sensor_health_["joints"], 
                               sensor_health_["odometry"]);
                }
            } else {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                                    "No healthy sensors available for fusion");
            }
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Fusion error: %s", e.what());
        }
    }

    // Subscriptions
    rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr lidar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    
    // Publishers
    rclcpp::Publisher<std_msgs::msg::Int8MultiArray>::SharedPtr hypervector_pub_;
    
    // Timer
    rclcpp::TimerBase::SharedPtr fusion_timer_;
    
    // HDC encoder
    std::unique_ptr<MultimodalEncoder> encoder_;
    
    // Sensor data
    sensor_msgs::msg::LaserScan lidar_data_;
    cv::Mat image_data_;
    sensor_msgs::msg::Imu imu_data_;
    sensor_msgs::msg::JointState joint_data_;
    nav_msgs::msg::Odometry odom_data_;
    
    // Sensor availability flags
    bool lidar_available_;
    bool camera_available_;
    bool imu_available_;
    bool joints_available_;
    bool odom_available_;
    
    // Sensor health monitoring
    std::map<std::string, double> sensor_health_;
    
    // Statistics
    size_t fusion_count_{0};
};

}  // namespace hdc_robot_controller

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<hdc_robot_controller::PerceptionNode>();
    
    RCLCPP_INFO(node->get_logger(), "Starting HDC Perception Node");
    
    try {
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_FATAL(node->get_logger(), "Node crashed: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}