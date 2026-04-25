#include "visualization.hpp"
#include <iostream>
#include <cmath>

cv::Vec3b component_id_to_color(int component_id) {
    if (component_id == 0) {
        return cv::Vec3b(0, 0, 0);  // background = black
    }
    float hue_degrees = fmod(component_id * 137.508f, 360.0f);
    cv::Mat hsv_pixel(1, 1, CV_8UC3,cv::Scalar(static_cast<uint8_t>(hue_degrees / 2), 220, 200));
    cv::Mat bgr_pixel;
    cv::cvtColor(hsv_pixel, bgr_pixel, cv::COLOR_HSV2BGR);
    return bgr_pixel.at<cv::Vec3b>(0, 0);
}

void save_label_visualization(const cv::Mat& label_map, const std::string& output_path, int max_dimension) {
    int image_height = label_map.rows;
    int image_width  = label_map.cols;

    cv::Mat visualization(image_height, image_width, CV_8UC3, cv::Scalar(0, 0, 0));
    for (int row = 0; row < image_height; row++) {
        for (int col = 0; col < image_width; col++) {
            int component_id = label_map.at<int>(row, col);
            if (component_id > 0)
                visualization.at<cv::Vec3b>(row, col) = component_id_to_color(component_id);
        }
    }

    int largest_dim = std::max(image_width, image_height);
    if (largest_dim > max_dimension) {
        double scale = static_cast<double>(max_dimension) / largest_dim;
        int out_width = static_cast<int>(image_width  * scale);
        int out_height = static_cast<int>(image_height * scale);

        cv::Mat downsampled;
        // INTER_NEAREST avoids blending colors across component boundaries
        cv::resize(visualization, downsampled, cv::Size(out_width, out_height), 0, 0, cv::INTER_NEAREST);

        cv::imwrite(output_path, downsampled);
        std::cout << "Saved: " << output_path
                  << " (downsampled from " << image_width << "x" << image_height
                  << " to " << out_width << "x" << out_height << ")\n";
    } else {
        cv::imwrite(output_path, visualization);
        std::cout << "Saved: " << output_path << "\n";
    }
}
