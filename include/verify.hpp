#pragma once
#include <opencv2/opencv.hpp>

// Checks that two label maps represent the same segmentation.
// Two maps are equivalent if for every pair of foreground pixels (p, q):
// map_a[p] == map_a[q]  iff  map_b[p] == map_b[q]
// This is stricter than comparing component counts — it catches cases where
// the same number of components exist but their boundaries differ.
// Returns true if equivalent, false and prints the first mismatch if not.
bool verify_label_equivalence(const cv::Mat& map_a, const cv::Mat& map_b);
