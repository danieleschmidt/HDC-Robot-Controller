#include "spatial_encoder.hpp"
#include <cmath>
#include <algorithm>

namespace hdc {

SpatialEncoder::SpatialEncoder(int dimension, double resolution) 
    : dimension_(dimension), resolution_(resolution) {
    initialize_basis_vectors();
}

HyperVector SpatialEncoder::encode_2d_point(double x, double y) const {
    // Discretize coordinates
    int grid_x = static_cast<int>(std::round(x / resolution_));
    int grid_y = static_cast<int>(std::round(y / resolution_));
    
    // Clamp to reasonable bounds
    grid_x = std::max(-1000, std::min(1000, grid_x));
    grid_y = std::max(-1000, std::min(1000, grid_y));
    
    // Get basis vectors for x and y coordinates
    auto x_vector = get_coordinate_vector(grid_x, 0);  // X-axis
    auto y_vector = get_coordinate_vector(grid_y, 1);  // Y-axis
    
    // Bind x and y coordinates
    return x_vector.bind(y_vector);
}

HyperVector SpatialEncoder::encode_3d_point(double x, double y, double z) const {
    // Discretize coordinates
    int grid_x = static_cast<int>(std::round(x / resolution_));
    int grid_y = static_cast<int>(std::round(y / resolution_));
    int grid_z = static_cast<int>(std::round(z / resolution_));
    
    // Clamp to reasonable bounds
    grid_x = std::max(-1000, std::min(1000, grid_x));
    grid_y = std::max(-1000, std::min(1000, grid_y));
    grid_z = std::max(-1000, std::min(1000, grid_z));
    
    // Get basis vectors for each coordinate
    auto x_vector = get_coordinate_vector(grid_x, 0);
    auto y_vector = get_coordinate_vector(grid_y, 1);
    auto z_vector = get_coordinate_vector(grid_z, 2);
    
    // Bind all three coordinates
    return x_vector.bind(y_vector).bind(z_vector);
}

HyperVector SpatialEncoder::encode_polar(double radius, double angle) const {
    // Discretize radius and angle
    int grid_radius = static_cast<int>(std::round(radius / resolution_));
    int grid_angle = static_cast<int>(std::round(angle * 180.0 / M_PI));  // Convert to degrees
    
    // Clamp values
    grid_radius = std::max(0, std::min(1000, grid_radius));
    grid_angle = ((grid_angle % 360) + 360) % 360;  // Normalize to [0, 360)
    
    // Get basis vectors
    auto radius_vector = get_coordinate_vector(grid_radius, 3);  // Radius axis
    auto angle_vector = get_coordinate_vector(grid_angle, 4);    // Angle axis
    
    return radius_vector.bind(angle_vector);
}

HyperVector SpatialEncoder::encode_grid_cell(int row, int col, int max_rows, int max_cols) const {
    // Validate inputs
    if (row < 0 || row >= max_rows || col < 0 || col >= max_cols) {
        return HyperVector::zero(dimension_);
    }
    
    // Create unique identifier for this grid cell
    int cell_id = row * max_cols + col;
    
    // Use deterministic random generation based on cell ID
    return HyperVector::random(dimension_, cell_id + 200000);
}

HyperVector SpatialEncoder::encode_spatial_region(const std::vector<std::pair<double, double>>& points) const {
    if (points.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    std::vector<HyperVector> point_vectors;
    point_vectors.reserve(points.size());
    
    for (const auto& [x, y] : points) {
        point_vectors.push_back(encode_2d_point(x, y));
    }
    
    // Bundle all points to represent the region
    return HyperVector::bundle_vectors(point_vectors);
}

HyperVector SpatialEncoder::encode_trajectory(const std::vector<std::pair<double, double>>& trajectory) const {
    if (trajectory.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    std::vector<HyperVector> trajectory_vectors;
    trajectory_vectors.reserve(trajectory.size());
    
    for (size_t i = 0; i < trajectory.size(); ++i) {
        // Encode position
        auto position_hv = encode_2d_point(trajectory[i].first, trajectory[i].second);
        
        // Encode temporal position
        auto time_hv = get_coordinate_vector(static_cast<int>(i), 5);  // Time axis
        
        // Bind position with time
        auto timestamped_position = position_hv.bind(time_hv);
        trajectory_vectors.push_back(timestamped_position);
    }
    
    return HyperVector::bundle_vectors(trajectory_vectors);
}

HyperVector SpatialEncoder::encode_bounding_box(double min_x, double min_y, 
                                              double max_x, double max_y) const {
    auto min_point = encode_2d_point(min_x, min_y);
    auto max_point = encode_2d_point(max_x, max_y);
    
    // Bind min and max points with identifiers
    auto min_hv = get_coordinate_vector(0, 6).bind(min_point);  // "min" identifier
    auto max_hv = get_coordinate_vector(1, 6).bind(max_point);  // "max" identifier
    
    return min_hv.bundle(max_hv);
}

HyperVector SpatialEncoder::encode_orientation(double roll, double pitch, double yaw) const {
    // Convert angles to degrees and discretize
    int roll_deg = static_cast<int>(std::round(roll * 180.0 / M_PI));
    int pitch_deg = static_cast<int>(std::round(pitch * 180.0 / M_PI));
    int yaw_deg = static_cast<int>(std::round(yaw * 180.0 / M_PI));
    
    // Normalize angles to [0, 360)
    roll_deg = ((roll_deg % 360) + 360) % 360;
    pitch_deg = ((pitch_deg % 360) + 360) % 360;
    yaw_deg = ((yaw_deg % 360) + 360) % 360;
    
    // Get basis vectors for each angle
    auto roll_vector = get_coordinate_vector(roll_deg, 7);   // Roll axis
    auto pitch_vector = get_coordinate_vector(pitch_deg, 8); // Pitch axis
    auto yaw_vector = get_coordinate_vector(yaw_deg, 9);     // Yaw axis
    
    return roll_vector.bind(pitch_vector).bind(yaw_vector);
}

std::pair<double, double> SpatialEncoder::decode_2d_point(const HyperVector& encoded) const {
    double best_x = 0.0, best_y = 0.0;
    double best_similarity = -1.0;
    
    // Search in a reasonable grid around origin
    for (int x = -50; x <= 50; x += 2) {  // Coarse search first
        for (int y = -50; y <= 50; y += 2) {
            auto candidate = encode_2d_point(x * resolution_, y * resolution_);
            double similarity = encoded.similarity(candidate);
            
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_x = x * resolution_;
                best_y = y * resolution_;
            }
        }
    }
    
    // Fine search around best candidate
    int fine_search_range = 5;
    double fine_resolution = resolution_ / 5.0;
    
    for (int dx = -fine_search_range; dx <= fine_search_range; ++dx) {
        for (int dy = -fine_search_range; dy <= fine_search_range; ++dy) {
            double x = best_x + dx * fine_resolution;
            double y = best_y + dy * fine_resolution;
            
            auto candidate = encode_2d_point(x, y);
            double similarity = encoded.similarity(candidate);
            
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_x = x;
                best_y = y;
            }
        }
    }
    
    return {best_x, best_y};
}

double SpatialEncoder::spatial_similarity(const HyperVector& pos1, const HyperVector& pos2) const {
    return pos1.similarity(pos2);
}

HyperVector SpatialEncoder::interpolate_positions(const HyperVector& pos1, const HyperVector& pos2,
                                                double weight) const {
    std::vector<std::pair<HyperVector, double>> weighted_positions = {
        {pos1, 1.0 - weight},
        {pos2, weight}
    };
    
    return weighted_bundle(weighted_positions);
}

HyperVector SpatialEncoder::encode_relative_position(const HyperVector& reference, 
                                                   const HyperVector& target) const {
    // Use binding to create relative position encoding
    return reference.bind(target);
}

HyperVector SpatialEncoder::transform_position(const HyperVector& position, 
                                             double dx, double dy, double rotation) const {
    // Decode original position (approximate)
    auto [orig_x, orig_y] = decode_2d_point(position);
    
    // Apply transformation
    double cos_rot = std::cos(rotation);
    double sin_rot = std::sin(rotation);
    
    double new_x = orig_x + dx * cos_rot - dy * sin_rot;
    double new_y = orig_y + dx * sin_rot + dy * cos_rot;
    
    // Encode transformed position
    return encode_2d_point(new_x, new_y);
}

void SpatialEncoder::initialize_basis_vectors() {
    // Pre-compute some commonly used basis vectors
    // This could be expanded for better performance
    
    // Reserve space for coordinate vectors cache
    coordinate_cache_.reserve(1000);
}

HyperVector SpatialEncoder::get_coordinate_vector(int coordinate, int axis) const {
    // Create unique hash for this coordinate-axis pair
    size_t hash_key = static_cast<size_t>(coordinate + axis * 10000);
    
    auto it = coordinate_cache_.find(hash_key);
    if (it != coordinate_cache_.end()) {
        return it->second;
    }
    
    // Create new coordinate vector
    uint32_t seed = static_cast<uint32_t>(hash_key + 300000);
    auto coord_vector = HyperVector::random(dimension_, seed);
    
    // Cache it
    coordinate_cache_[hash_key] = coord_vector;
    
    return coord_vector;
}

}  // namespace hdc