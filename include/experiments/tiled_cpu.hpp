#pragma once
#include <opencv2/opencv.hpp>

// Experiment 2 — Tiled CPU
// Phase 1 (CPU): tile the image, run cv::connectedComponents on each tile.
// Phase 2 (CPU): merge labels at tile seams using Union-Find.
// Phase 3 (CPU): relabel every pixel with its final global component ID.
cv::Mat run_tiled_cpu(const cv::Mat& binary_mask, int tile_size = 1024);
