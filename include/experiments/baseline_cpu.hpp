#pragma once
#include <opencv2/opencv.hpp>

// Experiment 1 — Baseline CPU
// Runs cv::connectedComponents on the full image in one shot.
// No tiling. This is the reference result and the performance baseline.
cv::Mat run_baseline_cpu(const cv::Mat& binary_mask);
