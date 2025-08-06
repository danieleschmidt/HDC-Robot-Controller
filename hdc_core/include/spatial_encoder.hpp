#pragma once

#include "hypervector.hpp"
#include <vector>
#include <utility>
#include <map>

namespace hdc {

/**
 * Spatial encoder for converting spatial data to hyperdimensional vectors
 * Supports 2D/3D points, trajectories, regions, and spatial transformations
 */
class SpatialEncoder {
public:
    explicit SpatialEncoder(int dimension = 10000, double resolution = 0.1);
    
    // Basic spatial encoding
    HyperVector encode_2d_point(double x, double y) const;
    HyperVector encode_3d_point(double x, double y, double z) const;
    HyperVector encode_polar(double radius, double angle) const;
    
    // Grid-based encoding
    HyperVector encode_grid_cell(int row, int col, int max_rows, int max_cols) const;
    
    // Complex spatial structures
    HyperVector encode_spatial_region(const std::vector<std::pair<double, double>>& points) const;
    HyperVector encode_trajectory(const std::vector<std::pair<double, double>>& trajectory) const;
    HyperVector encode_bounding_box(double min_x, double min_y, double max_x, double max_y) const;
    
    // Orientation encoding
    HyperVector encode_orientation(double roll, double pitch, double yaw) const;
    
    // Decoding (approximate)
    std::pair<double, double> decode_2d_point(const HyperVector& encoded) const;
    
    // Spatial operations
    double spatial_similarity(const HyperVector& pos1, const HyperVector& pos2) const;
    HyperVector interpolate_positions(const HyperVector& pos1, const HyperVector& pos2, 
                                    double weight = 0.5) const;
    HyperVector encode_relative_position(const HyperVector& reference, const HyperVector& target) const;
    HyperVector transform_position(const HyperVector& position, double dx, double dy, 
                                 double rotation = 0.0) const;
    
    // Configuration
    double get_resolution() const { return resolution_; }
    void set_resolution(double resolution) { resolution_ = resolution; }

private:
    void initialize_basis_vectors();
    HyperVector get_coordinate_vector(int coordinate, int axis) const;
    
    int dimension_;
    double resolution_;
    
    // Cache for coordinate vectors
    mutable std::map<size_t, HyperVector> coordinate_cache_;
};

}  // namespace hdc