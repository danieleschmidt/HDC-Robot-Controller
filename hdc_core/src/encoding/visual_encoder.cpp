#include "visual_encoder.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace hdc {

VisualEncoder::VisualEncoder(int dimension, int patch_size, int vocabulary_size)
    : dimension_(dimension), patch_size_(patch_size), vocabulary_size_(vocabulary_size) {
    initialize_visual_vocabulary();
}

HyperVector VisualEncoder::encode_image_patches(const cv::Mat& image, int stride) const {
    if (image.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    // Convert to grayscale if necessary
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    std::vector<HyperVector> patch_vectors;
    
    // Extract patches
    for (int y = 0; y <= gray.rows - patch_size_; y += stride) {
        for (int x = 0; x <= gray.cols - patch_size_; x += stride) {
            cv::Rect patch_rect(x, y, patch_size_, patch_size_);
            cv::Mat patch = gray(patch_rect);
            
            // Encode patch
            auto patch_hv = encode_image_patch(patch);
            
            // Add spatial information
            auto spatial_hv = encode_spatial_position(x, y, gray.cols, gray.rows);
            auto spatially_bound_patch = patch_hv.bind(spatial_hv);
            
            patch_vectors.push_back(spatially_bound_patch);
        }
    }
    
    if (patch_vectors.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    return HyperVector::bundle_vectors(patch_vectors);
}

HyperVector VisualEncoder::encode_image_grid(const cv::Mat& image, int grid_rows, int grid_cols) const {
    if (image.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    std::vector<HyperVector> grid_vectors;
    
    int cell_width = gray.cols / grid_cols;
    int cell_height = gray.rows / grid_rows;
    
    for (int row = 0; row < grid_rows; ++row) {
        for (int col = 0; col < grid_cols; ++col) {
            // Define cell boundaries
            int x = col * cell_width;
            int y = row * cell_height;
            int w = (col == grid_cols - 1) ? gray.cols - x : cell_width;
            int h = (row == grid_rows - 1) ? gray.rows - y : cell_height;
            
            cv::Rect cell_rect(x, y, w, h);
            cv::Mat cell = gray(cell_rect);
            
            // Extract features from cell
            auto features_hv = extract_cell_features(cell);
            
            // Add grid position information
            auto grid_pos_hv = encode_grid_position(row, col, grid_rows, grid_cols);
            auto positioned_features = features_hv.bind(grid_pos_hv);
            
            grid_vectors.push_back(positioned_features);
        }
    }
    
    return HyperVector::bundle_vectors(grid_vectors);
}

HyperVector VisualEncoder::encode_edge_features(const cv::Mat& image) const {
    if (image.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Apply Sobel edge detection
    cv::Mat grad_x, grad_y;
    cv::Sobel(gray, grad_x, CV_64F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_64F, 0, 1, 3);
    
    // Compute gradient magnitude and direction
    cv::Mat magnitude, direction;
    cv::cartToPolar(grad_x, grad_y, magnitude, direction);
    
    // Discretize edge orientations into bins
    std::vector<HyperVector> edge_vectors;
    const int num_orientations = 8;  // 0°, 45°, 90°, 135°, etc.
    
    for (int y = 0; y < magnitude.rows; ++y) {
        for (int x = 0; x < magnitude.cols; ++x) {
            double mag = magnitude.at<double>(y, x);
            double dir = direction.at<double>(y, x);
            
            // Only consider significant edges
            if (mag > 30.0) {  // Threshold for edge strength
                // Discretize direction
                int orientation_bin = static_cast<int>((dir * num_orientations) / (2 * M_PI));
                orientation_bin = orientation_bin % num_orientations;
                
                // Encode edge
                auto edge_hv = encode_edge_orientation(orientation_bin, mag);
                auto spatial_hv = encode_spatial_position(x, y, gray.cols, gray.rows);
                auto spatially_bound_edge = edge_hv.bind(spatial_hv);
                
                edge_vectors.push_back(spatially_bound_edge);
            }
        }
    }
    
    if (edge_vectors.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    return HyperVector::bundle_vectors(edge_vectors);
}

HyperVector VisualEncoder::encode_color_histogram(const cv::Mat& image, int num_bins) const {
    if (image.empty() || image.channels() != 3) {
        return HyperVector::zero(dimension_);
    }
    
    // Split into BGR channels
    std::vector<cv::Mat> bgr_channels;
    cv::split(image, bgr_channels);
    
    std::vector<HyperVector> channel_vectors;
    
    for (int channel = 0; channel < 3; ++channel) {
        // Compute histogram
        cv::Mat hist;
        int hist_size = num_bins;
        float range[] = {0, 256};
        const float* hist_range = {range};
        
        cv::calcHist(&bgr_channels[channel], 1, 0, cv::Mat(), hist, 1, &hist_size, &hist_range);
        
        // Normalize histogram
        cv::normalize(hist, hist, 0, 1, cv::NORM_L1);
        
        // Encode histogram bins
        std::vector<HyperVector> bin_vectors;
        for (int bin = 0; bin < num_bins; ++bin) {
            double bin_value = hist.at<float>(bin);
            
            if (bin_value > 0.01) {  // Only encode significant bins
                auto bin_hv = encode_histogram_bin(channel, bin, bin_value);
                bin_vectors.push_back(bin_hv);
            }
        }
        
        if (!bin_vectors.empty()) {
            auto channel_hv = HyperVector::bundle_vectors(bin_vectors);
            channel_vectors.push_back(channel_hv);
        }
    }
    
    if (channel_vectors.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    return HyperVector::bundle_vectors(channel_vectors);
}

HyperVector VisualEncoder::encode_texture_features(const cv::Mat& image) const {
    if (image.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    std::vector<HyperVector> texture_vectors;
    
    // Local Binary Pattern (simplified version)
    cv::Mat lbp = compute_lbp(gray);
    
    // Divide image into regions for texture analysis
    int num_regions = 4;
    int region_width = gray.cols / num_regions;
    int region_height = gray.rows / num_regions;
    
    for (int ry = 0; ry < num_regions; ++ry) {
        for (int rx = 0; rx < num_regions; ++rx) {
            cv::Rect region_rect(rx * region_width, ry * region_height,
                               region_width, region_height);
            cv::Mat region_lbp = lbp(region_rect);
            
            // Compute histogram of LBP values
            cv::Mat lbp_hist;
            int hist_size = 256;
            float range[] = {0, 256};
            const float* hist_range = {range};
            
            cv::calcHist(&region_lbp, 1, 0, cv::Mat(), lbp_hist, 1, &hist_size, &hist_range);
            cv::normalize(lbp_hist, lbp_hist, 0, 1, cv::NORM_L1);
            
            // Encode texture pattern
            std::vector<HyperVector> pattern_vectors;
            for (int i = 0; i < hist_size; ++i) {
                double bin_value = lbp_hist.at<float>(i);
                if (bin_value > 0.01) {
                    auto pattern_hv = encode_texture_pattern(i, bin_value);
                    pattern_vectors.push_back(pattern_hv);
                }
            }
            
            if (!pattern_vectors.empty()) {
                auto region_texture = HyperVector::bundle_vectors(pattern_vectors);
                auto region_pos = encode_grid_position(ry, rx, num_regions, num_regions);
                auto positioned_texture = region_texture.bind(region_pos);
                
                texture_vectors.push_back(positioned_texture);
            }
        }
    }
    
    if (texture_vectors.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    return HyperVector::bundle_vectors(texture_vectors);
}

HyperVector VisualEncoder::encode_shape_features(const cv::Mat& image) const {
    if (image.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }
    
    // Apply threshold to get binary image
    cv::Mat binary;
    cv::threshold(gray, binary, 128, 255, cv::THRESH_BINARY);
    
    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::vector<HyperVector> shape_vectors;
    
    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;  // Skip very small contours
        
        // Compute shape descriptors
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        cv::Moments moments = cv::moments(contour);
        
        if (area < 100) continue;  // Skip very small shapes
        
        // Compute shape features
        double aspect_ratio = 0.0;
        double solidity = 0.0;
        
        cv::RotatedRect bounding_rect = cv::minAreaRect(contour);
        if (bounding_rect.size.height > 0) {
            aspect_ratio = bounding_rect.size.width / bounding_rect.size.height;
        }
        
        std::vector<cv::Point> hull;
        cv::convexHull(contour, hull);
        double hull_area = cv::contourArea(hull);
        if (hull_area > 0) {
            solidity = area / hull_area;
        }
        
        // Encode shape features
        auto area_hv = encode_shape_feature("area", area, 0, 10000);
        auto perimeter_hv = encode_shape_feature("perimeter", perimeter, 0, 1000);
        auto aspect_hv = encode_shape_feature("aspect_ratio", aspect_ratio, 0, 5);
        auto solidity_hv = encode_shape_feature("solidity", solidity, 0, 1);
        
        // Bundle shape features
        std::vector<HyperVector> feature_vectors = {
            area_hv, perimeter_hv, aspect_hv, solidity_hv
        };
        auto shape_hv = HyperVector::bundle_vectors(feature_vectors);
        
        // Add centroid information
        if (moments.m00 > 0) {
            double cx = moments.m10 / moments.m00;
            double cy = moments.m01 / moments.m00;
            auto centroid_hv = encode_spatial_position(static_cast<int>(cx), static_cast<int>(cy),
                                                     gray.cols, gray.rows);
            shape_hv = shape_hv.bind(centroid_hv);
        }
        
        shape_vectors.push_back(shape_hv);
    }
    
    if (shape_vectors.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    return HyperVector::bundle_vectors(shape_vectors);
}

void VisualEncoder::initialize_visual_vocabulary() {
    // Pre-compute visual vocabulary vectors
    for (int i = 0; i < vocabulary_size_; ++i) {
        visual_vocabulary_.push_back(HyperVector::random(dimension_, i + 400000));
    }
    
    // Initialize feature basis vectors
    for (int i = 0; i < 100; ++i) {
        feature_basis_.push_back(HyperVector::random(dimension_, i + 500000));
    }
}

HyperVector VisualEncoder::encode_image_patch(const cv::Mat& patch) const {
    if (patch.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    // Compute simple features: mean intensity and standard deviation
    cv::Scalar mean_scalar, std_scalar;
    cv::meanStdDev(patch, mean_scalar, std_scalar);
    
    double mean_intensity = mean_scalar[0];
    double std_intensity = std_scalar[0];
    
    // Discretize features
    int mean_bin = static_cast<int>(mean_intensity / 32);  // 8 bins for mean
    int std_bin = static_cast<int>(std_intensity / 16);    // 16 bins for std
    
    mean_bin = std::min(7, std::max(0, mean_bin));
    std_bin = std::min(15, std::max(0, std_bin));
    
    // Get vocabulary vectors
    int vocab_index = mean_bin * 16 + std_bin;
    vocab_index = std::min(vocabulary_size_ - 1, vocab_index);
    
    return visual_vocabulary_[vocab_index];
}

HyperVector VisualEncoder::extract_cell_features(const cv::Mat& cell) const {
    if (cell.empty()) {
        return HyperVector::zero(dimension_);
    }
    
    std::vector<HyperVector> feature_vectors;
    
    // Intensity statistics
    cv::Scalar mean_scalar, std_scalar;
    cv::meanStdDev(cell, mean_scalar, std_scalar);
    
    auto mean_hv = encode_intensity_feature(mean_scalar[0], 0, 255);
    auto std_hv = encode_intensity_feature(std_scalar[0], 0, 100);
    
    feature_vectors.push_back(mean_hv);
    feature_vectors.push_back(std_hv);
    
    // Edge density (simplified)
    cv::Mat edges;
    cv::Canny(cell, edges, 50, 150);
    double edge_density = cv::sum(edges)[0] / (cell.rows * cell.cols * 255.0);
    auto edge_hv = encode_intensity_feature(edge_density, 0, 1);
    feature_vectors.push_back(edge_hv);
    
    return HyperVector::bundle_vectors(feature_vectors);
}

HyperVector VisualEncoder::encode_spatial_position(int x, int y, int width, int height) const {
    // Normalize coordinates to [0, 1]
    double norm_x = static_cast<double>(x) / width;
    double norm_y = static_cast<double>(y) / height;
    
    // Discretize
    int grid_x = static_cast<int>(norm_x * 32);  // 32x32 spatial grid
    int grid_y = static_cast<int>(norm_y * 32);
    
    grid_x = std::min(31, std::max(0, grid_x));
    grid_y = std::min(31, std::max(0, grid_y));
    
    return HyperVector::random(dimension_, grid_x * 32 + grid_y + 600000);
}

HyperVector VisualEncoder::encode_grid_position(int row, int col, int max_rows, int max_cols) const {
    int position_id = row * max_cols + col;
    return HyperVector::random(dimension_, position_id + 700000);
}

HyperVector VisualEncoder::encode_edge_orientation(int orientation_bin, double magnitude) const {
    auto orientation_hv = HyperVector::random(dimension_, orientation_bin + 800000);
    
    // Discretize magnitude
    int mag_bin = static_cast<int>(magnitude / 10.0);  // Bins of 10 units
    mag_bin = std::min(25, std::max(0, mag_bin));
    
    auto magnitude_hv = HyperVector::random(dimension_, mag_bin + 810000);
    
    return orientation_hv.bind(magnitude_hv);
}

HyperVector VisualEncoder::encode_histogram_bin(int channel, int bin, double value) const {
    auto channel_hv = HyperVector::random(dimension_, channel + 900000);
    auto bin_hv = HyperVector::random(dimension_, bin + 910000);
    
    // Discretize value
    int value_bin = static_cast<int>(value * 100);  // 100 levels
    value_bin = std::min(99, std::max(0, value_bin));
    auto value_hv = HyperVector::random(dimension_, value_bin + 920000);
    
    return channel_hv.bind(bin_hv).bind(value_hv);
}

HyperVector VisualEncoder::encode_texture_pattern(int pattern_id, double strength) const {
    auto pattern_hv = HyperVector::random(dimension_, pattern_id + 930000);
    
    // Discretize strength
    int strength_bin = static_cast<int>(strength * 50);  // 50 levels
    strength_bin = std::min(49, std::max(0, strength_bin));
    auto strength_hv = HyperVector::random(dimension_, strength_bin + 940000);
    
    return pattern_hv.bind(strength_hv);
}

HyperVector VisualEncoder::encode_shape_feature(const std::string& feature_name, 
                                              double value, double min_val, double max_val) const {
    auto feature_hv = get_feature_vector(feature_name);
    
    // Discretize value
    double normalized = (value - min_val) / (max_val - min_val);
    normalized = std::max(0.0, std::min(1.0, normalized));
    
    int value_bin = static_cast<int>(normalized * 99);  // 100 levels
    auto value_hv = HyperVector::random(dimension_, value_bin + 950000);
    
    return feature_hv.bind(value_hv);
}

HyperVector VisualEncoder::encode_intensity_feature(double intensity, double min_val, double max_val) const {
    double normalized = (intensity - min_val) / (max_val - min_val);
    normalized = std::max(0.0, std::min(1.0, normalized));
    
    int intensity_bin = static_cast<int>(normalized * 63);  // 64 levels
    return HyperVector::random(dimension_, intensity_bin + 960000);
}

HyperVector VisualEncoder::get_feature_vector(const std::string& feature_name) const {
    auto hash = std::hash<std::string>{}(feature_name);
    return HyperVector::random(dimension_, hash % 100000 + 970000);
}

cv::Mat VisualEncoder::compute_lbp(const cv::Mat& image) const {
    cv::Mat lbp = cv::Mat::zeros(image.size(), CV_8UC1);
    
    for (int y = 1; y < image.rows - 1; ++y) {
        for (int x = 1; x < image.cols - 1; ++x) {
            uint8_t center = image.at<uint8_t>(y, x);
            uint8_t code = 0;
            
            // 8-neighbor LBP
            if (image.at<uint8_t>(y - 1, x - 1) >= center) code |= 1;
            if (image.at<uint8_t>(y - 1, x) >= center) code |= 2;
            if (image.at<uint8_t>(y - 1, x + 1) >= center) code |= 4;
            if (image.at<uint8_t>(y, x + 1) >= center) code |= 8;
            if (image.at<uint8_t>(y + 1, x + 1) >= center) code |= 16;
            if (image.at<uint8_t>(y + 1, x) >= center) code |= 32;
            if (image.at<uint8_t>(y + 1, x - 1) >= center) code |= 64;
            if (image.at<uint8_t>(y, x - 1) >= center) code |= 128;
            
            lbp.at<uint8_t>(y, x) = code;
        }
    }
    
    return lbp;
}

}  // namespace hdc