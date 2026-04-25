#include "verify.hpp"
#include <iostream>
#include <unordered_map>

bool verify_label_equivalence(const cv::Mat& map_a, const cv::Mat& map_b) {
    if (map_a.size() != map_b.size()) {
        std::cerr << "Verify FAILED: maps have different sizes\n";
        return false;
    }

    int image_height = map_a.rows;
    int image_width  = map_a.cols;

    // Build a bijection: label in map_a <-> label in map_b.
    // If a label in one map consistently maps to the same label in the other,
    // the segmentations are equivalent (just with different ID assignments).
    std::unordered_map<int, int> a_to_b;
    std::unordered_map<int, int> b_to_a;

    for (int row = 0; row < image_height; row++) {
        for (int col = 0; col < image_width; col++) {
            int label_a = map_a.at<int>(row, col);
            int label_b = map_b.at<int>(row, col);

            // Background must agree on both sides
            if ((label_a == 0) != (label_b == 0)) {
                std::cerr << "Verify FAILED at (" << col << "," << row << "): "
                          << "one map has background, the other has foreground\n";
                return false;
            }
            if (label_a == 0) continue;

            // Check a -> b mapping
            auto a_entry = a_to_b.find(label_a);
            if (a_entry == a_to_b.end()) {
                a_to_b[label_a] = label_b;
            } else if (a_entry->second != label_b) {
                std::cerr << "Verify FAILED at (" << col << "," << row << "): "
                          << "map_a label " << label_a << " previously mapped to "
                          << a_entry->second << " but now maps to " << label_b << "\n";
                return false;
            }

            // Check b -> a mapping (ensures the bijection is one-to-one)
            auto b_entry = b_to_a.find(label_b);
            if (b_entry == b_to_a.end()) {
                b_to_a[label_b] = label_a;
            } else if (b_entry->second != label_a) {
                std::cerr << "Verify FAILED at (" << col << "," << row << "): "
                          << "map_b label " << label_b << " previously mapped to "
                          << b_entry->second << " but now maps to " << label_a << "\n";
                return false;
            }
        }
    }

    std::cout << "Verify PASSED: both maps are equivalent ("
              << a_to_b.size() << " components matched)\n";
    return true;
}
