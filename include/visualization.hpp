#pragma once
#include <string>
#include <opencv2/opencv.hpp>

// Map a component ID to a distinct BGR color.
// Uses the golden-angle trick so sequential IDs get maximally spread hues.
cv::Vec3b component_id_to_color(int component_id);

// Render a CV_32S label map as a color image and save it to disk.
// If either dimension exceeds max_dimension, the output is downsampled to fit
void save_label_visualization(const cv::Mat& label_map, const std::string& output_path, int max_dimension = 12288);
