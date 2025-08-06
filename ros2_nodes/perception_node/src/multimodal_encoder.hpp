#pragma once

#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <memory>
#include <vector>
#include <cmath>

#include "hypervector.hpp"

namespace hdc_robot_controller {

class MultimodalEncoder {
public:
    explicit MultimodalEncoder(int dimension = 10000) : dimension_(dimension) {
        initialize_basis_vectors();
    }
    
    // Encode LIDAR data into hyperdimensional space
    hdc::HyperVector encode_lidar(const sensor_msgs::msg::LaserScan& scan) {
        const int num_sectors = 36;  // 10-degree sectors
        const double sector_angle = 2.0 * M_PI / num_sectors;
        
        std::vector<hdc::HyperVector> sector_vectors;
        
        for (int sector = 0; sector < num_sectors; ++sector) {
            double min_range = std::numeric_limits<double>::max();
            int count = 0;
            
            // Find minimum range in this sector
            int start_idx = sector * scan.ranges.size() / num_sectors;
            int end_idx = (sector + 1) * scan.ranges.size() / num_sectors;
            
            for (int i = start_idx; i < end_idx && i < static_cast<int>(scan.ranges.size()); ++i) {
                if (std::isfinite(scan.ranges[i]) && 
                    scan.ranges[i] >= scan.range_min && 
                    scan.ranges[i] <= scan.range_max) {
                    min_range = std::min(min_range, static_cast<double>(scan.ranges[i]));
                    count++;
                }
            }
            
            if (count > 0) {
                // Discretize range
                int range_level = static_cast<int>(std::min(9.0, min_range));
                
                // Create sector vector: bind direction with range
                auto direction_hv = get_direction_vector(sector);
                auto range_hv = get_range_vector(range_level);
                auto sector_hv = direction_hv.bind(range_hv);
                
                sector_vectors.push_back(sector_hv);
            }
        }
        
        if (sector_vectors.empty()) {
            return hdc::HyperVector::zero(dimension_);
        }
        
        return hdc::HyperVector::bundle_vectors(sector_vectors);
    }
    
    // Encode camera image into hyperdimensional space
    hdc::HyperVector encode_image(const cv::Mat& image) {
        if (image.empty()) {
            return hdc::HyperVector::zero(dimension_);
        }
        
        // Convert to grayscale if needed
        cv::Mat gray;
        if (image.channels() == 3) {
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        
        // Resize to standard size
        cv::Mat resized;
        cv::resize(gray, resized, cv::Size(64, 48));
        
        // Extract features using simple grid-based approach
        const int grid_rows = 8;
        const int grid_cols = 8;
        const int cell_height = resized.rows / grid_rows;
        const int cell_width = resized.cols / grid_cols;
        
        std::vector<hdc::HyperVector> cell_vectors;
        
        for (int r = 0; r < grid_rows; ++r) {
            for (int c = 0; c < grid_cols; ++c) {
                cv::Rect cell_rect(c * cell_width, r * cell_height, cell_width, cell_height);
                cv::Mat cell = resized(cell_rect);
                
                // Calculate average intensity
                double avg_intensity = cv::mean(cell)[0];
                int intensity_level = static_cast<int>(avg_intensity / 32);  // 0-7 levels
                intensity_level = std::min(7, std::max(0, intensity_level));
                
                // Bind spatial position with intensity
                auto pos_hv = get_spatial_vector(r, c, grid_rows, grid_cols);
                auto intensity_hv = get_intensity_vector(intensity_level);
                auto cell_hv = pos_hv.bind(intensity_hv);
                
                cell_vectors.push_back(cell_hv);
            }
        }
        
        return hdc::HyperVector::bundle_vectors(cell_vectors);
    }
    
    // Encode IMU data
    hdc::HyperVector encode_imu(const sensor_msgs::msg::Imu& imu) {
        std::vector<hdc::HyperVector> components;
        
        // Encode angular velocity
        auto omega_x = discretize_and_encode(imu.angular_velocity.x, -10.0, 10.0, 20);
        auto omega_y = discretize_and_encode(imu.angular_velocity.y, -10.0, 10.0, 20);
        auto omega_z = discretize_and_encode(imu.angular_velocity.z, -10.0, 10.0, 20);
        
        // Encode linear acceleration
        auto acc_x = discretize_and_encode(imu.linear_acceleration.x, -20.0, 20.0, 40);
        auto acc_y = discretize_and_encode(imu.linear_acceleration.y, -20.0, 20.0, 40);
        auto acc_z = discretize_and_encode(imu.linear_acceleration.z, -20.0, 20.0, 40);
        
        // Bind with component identifiers
        components.push_back(get_component_vector("omega_x").bind(omega_x));
        components.push_back(get_component_vector("omega_y").bind(omega_y));
        components.push_back(get_component_vector("omega_z").bind(omega_z));
        components.push_back(get_component_vector("acc_x").bind(acc_x));
        components.push_back(get_component_vector("acc_y").bind(acc_y));
        components.push_back(get_component_vector("acc_z").bind(acc_z));
        
        return hdc::HyperVector::bundle_vectors(components);
    }
    
    // Encode joint states
    hdc::HyperVector encode_joints(const sensor_msgs::msg::JointState& joints) {
        std::vector<hdc::HyperVector> joint_vectors;
        
        for (size_t i = 0; i < joints.name.size() && i < joints.position.size(); ++i) {
            // Get joint identifier
            auto joint_id = get_joint_vector(joints.name[i]);
            
            // Encode position
            auto position = discretize_and_encode(joints.position[i], -M_PI, M_PI, 72);  // 5-degree resolution
            
            // Encode velocity if available
            hdc::HyperVector velocity_hv = hdc::HyperVector::zero(dimension_);
            if (i < joints.velocity.size()) {
                velocity_hv = discretize_and_encode(joints.velocity[i], -5.0, 5.0, 50);
            }
            
            // Combine joint information
            auto joint_state = joint_id.bind(position);
            if (!velocity_hv.is_zero_vector()) {
                auto vel_component = get_component_vector("velocity").bind(velocity_hv);
                joint_state = joint_state.bundle(vel_component);
            }
            
            joint_vectors.push_back(joint_state);
        }
        
        if (joint_vectors.empty()) {
            return hdc::HyperVector::zero(dimension_);
        }
        
        return hdc::HyperVector::bundle_vectors(joint_vectors);
    }
    
    // Encode odometry
    hdc::HyperVector encode_odometry(const nav_msgs::msg::Odometry& odom) {
        std::vector<hdc::HyperVector> components;
        
        // Encode position
        auto pos_x = discretize_and_encode(odom.pose.pose.position.x, -100.0, 100.0, 1000);
        auto pos_y = discretize_and_encode(odom.pose.pose.position.y, -100.0, 100.0, 1000);
        auto pos_z = discretize_and_encode(odom.pose.pose.position.z, -10.0, 10.0, 100);
        
        // Encode linear velocity
        auto vel_x = discretize_and_encode(odom.twist.twist.linear.x, -5.0, 5.0, 100);
        auto vel_y = discretize_and_encode(odom.twist.twist.linear.y, -5.0, 5.0, 100);
        
        // Encode angular velocity
        auto ang_vel = discretize_and_encode(odom.twist.twist.angular.z, -2.0, 2.0, 80);
        
        // Bind with component identifiers
        components.push_back(get_component_vector("pos_x").bind(pos_x));
        components.push_back(get_component_vector("pos_y").bind(pos_y));
        components.push_back(get_component_vector("pos_z").bind(pos_z));
        components.push_back(get_component_vector("vel_x").bind(vel_x));
        components.push_back(get_component_vector("vel_y").bind(vel_y));
        components.push_back(get_component_vector("ang_vel").bind(ang_vel));
        
        return hdc::HyperVector::bundle_vectors(components);
    }

private:
    void initialize_basis_vectors() {
        // Initialize cached basis vectors for common encodings
        for (int i = 0; i < 100; ++i) {
            direction_vectors_.push_back(hdc::HyperVector::random(dimension_, i + 1000));
            range_vectors_.push_back(hdc::HyperVector::random(dimension_, i + 2000));
            intensity_vectors_.push_back(hdc::HyperVector::random(dimension_, i + 3000));
        }
    }
    
    hdc::HyperVector get_direction_vector(int sector) {
        if (sector < static_cast<int>(direction_vectors_.size())) {
            return direction_vectors_[sector];
        }
        return hdc::HyperVector::random(dimension_, sector + 1000);
    }
    
    hdc::HyperVector get_range_vector(int range_level) {
        if (range_level < static_cast<int>(range_vectors_.size())) {
            return range_vectors_[range_level];
        }
        return hdc::HyperVector::random(dimension_, range_level + 2000);
    }
    
    hdc::HyperVector get_intensity_vector(int level) {
        if (level < static_cast<int>(intensity_vectors_.size())) {
            return intensity_vectors_[level];
        }
        return hdc::HyperVector::random(dimension_, level + 3000);
    }
    
    hdc::HyperVector get_spatial_vector(int row, int col, int max_rows, int max_cols) {
        int spatial_id = row * max_cols + col;
        return hdc::HyperVector::random(dimension_, spatial_id + 4000);
    }
    
    hdc::HyperVector get_component_vector(const std::string& name) {
        auto hash = std::hash<std::string>{}(name);
        return hdc::HyperVector::random(dimension_, hash % 1000000 + 5000);
    }
    
    hdc::HyperVector get_joint_vector(const std::string& joint_name) {
        auto hash = std::hash<std::string>{}(joint_name);
        return hdc::HyperVector::random(dimension_, hash % 1000000 + 6000);
    }
    
    hdc::HyperVector discretize_and_encode(double value, double min_val, double max_val, int levels) {
        if (value < min_val) value = min_val;
        if (value > max_val) value = max_val;
        
        int discrete_level = static_cast<int>((value - min_val) / (max_val - min_val) * (levels - 1));
        discrete_level = std::min(levels - 1, std::max(0, discrete_level));
        
        return hdc::HyperVector::random(dimension_, discrete_level + 10000);
    }
    
    int dimension_;
    std::vector<hdc::HyperVector> direction_vectors_;
    std::vector<hdc::HyperVector> range_vectors_;
    std::vector<hdc::HyperVector> intensity_vectors_;
};

}  // namespace hdc_robot_controller