#include "experiments/baseline_cpu.hpp"
#include "timer.hpp"
#include <iostream>

cv::Mat run_baseline_cpu(const cv::Mat& binary_mask) {
    std::cout << "=== Experiment: Baseline CPU ===\n";

    Timer timer;
    timer.start();
    cv::Mat label_map;
    int num_labels_with_background = cv::connectedComponents(binary_mask, label_map, 8, CV_32S);
    double ccl_ms = timer.elapsed_ms();

    int num_components = num_labels_with_background - 1;
    double total_pixels = binary_mask.rows * binary_mask.cols;
    double throughput_mpix_s = (total_pixels / 1e6) / (ccl_ms / 1e3);

    std::cout << "Components found: " << num_components << "\n";
    std::cout << "Time - CCL: " << ccl_ms << " ms\n";
    std::cout << "Throughput: " << throughput_mpix_s << " MPixels/s\n\n";

    return label_map;
}
