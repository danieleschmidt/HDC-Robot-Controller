#pragma once

#include "hypervector.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace hdc {

/**
 * Visual encoder for converting image data to hyperdimensional vectors
 * Supports patch-based encoding, feature extraction, and spatial relationships
 */
class VisualEncoder {
public:
    explicit VisualEncoder(int dimension = 10000, int patch_size = 8, int vocabulary_size = 1000);
    
    // Patch-based encoding
    HyperVector encode_image_patches(const cv::Mat& image, int stride = 4) const;
    HyperVector encode_image_grid(const cv::Mat& image, int grid_rows = 8, int grid_cols = 8) const;
    
    // Feature-based encoding
    HyperVector encode_edge_features(const cv::Mat& image) const;
    HyperVector encode_color_histogram(const cv::Mat& image, int num_bins = 32) const;
    HyperVector encode_texture_features(const cv::Mat& image) const;
    HyperVector encode_shape_features(const cv::Mat& image) const;
    
    // Configuration
    int get_patch_size() const { return patch_size_; }
    void set_patch_size(int size) { patch_size_ = size; }
    
    int get_vocabulary_size() const { return vocabulary_size_; }
    void set_vocabulary_size(int size) { 
        vocabulary_size_ = size;
        initialize_visual_vocabulary();
    }

private:
    void initialize_visual_vocabulary();
    
    // Low-level encoding functions
    HyperVector encode_image_patch(const cv::Mat& patch) const;
    HyperVector extract_cell_features(const cv::Mat& cell) const;
    HyperVector encode_spatial_position(int x, int y, int width, int height) const;
    HyperVector encode_grid_position(int row, int col, int max_rows, int max_cols) const;
    HyperVector encode_edge_orientation(int orientation_bin, double magnitude) const;
    HyperVector encode_histogram_bin(int channel, int bin, double value) const;
    HyperVector encode_texture_pattern(int pattern_id, double strength) const;
    HyperVector encode_shape_feature(const std::string& feature_name, double value, 
                                   double min_val, double max_val) const;
    HyperVector encode_intensity_feature(double intensity, double min_val, double max_val) const;
    HyperVector get_feature_vector(const std::string& feature_name) const;
    
    // Utility functions
    cv::Mat compute_lbp(const cv::Mat& image) const;
    
    int dimension_;
    int patch_size_;
    int vocabulary_size_;
    
    // Visual vocabulary for patch encoding
    std::vector<HyperVector> visual_vocabulary_;
    std::vector<HyperVector> feature_basis_;
};

}  // namespace hdc