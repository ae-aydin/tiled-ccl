#include <cstdlib>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "visualization.hpp"
#include "verify.hpp"
#include "experiments/baseline_cpu.hpp"
#include "experiments/tiled_cpu.hpp"
#include "experiments/tiled_gpu.hpp"

// Usage: ./ccl [image_path] [experiment] [tile_size] [-o]
//   experiment:  "baseline" | "tiled" | "tiled_gpu" | "verify" | "verify_gpu" (default: tiled)
//   tile_size:   tile size in pixels (default: 1024, only used by tiled/verify)
//   -o:          save colored label visualization to disk

int main(int argc, char* argv[]) {
    std::string image_path;
    std::string experiment = "tiled_gpu";
    int tile_size = 1024;
    bool save_output = false;

    if (argc > 1) image_path = argv[1];
    if (argc > 2) experiment = argv[2];
    if (argc > 3) tile_size = std::atoi(argv[3]);
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "-o") save_output = true;

    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Failed to load: " << image_path << "\n";
        return 1;
    }

    size_t memory_bytes = static_cast<size_t>(image.rows) * image.cols * image.elemSize();
    std::cout << "Image: " << image.cols << " x " << image.rows << " | in-memory: " << memory_bytes / (1024 * 1024) << " MB\n\n";

    cv::Mat binary_mask;
    cv::threshold(image, binary_mask, 0, 255, cv::THRESH_BINARY);

    if (experiment == "baseline") {
        cv::Mat label_map = run_baseline_cpu(binary_mask);
        if (save_output) save_label_visualization(label_map, "output_baseline.png");

    } else if (experiment == "tiled") {
        cv::Mat label_map = run_tiled_cpu(binary_mask, tile_size);
        if (save_output) save_label_visualization(label_map, "output_tiled.png");

    } else if (experiment == "tiled_gpu") {
        cv::Mat label_map = run_tiled_gpu_ccl(binary_mask, tile_size);
        if (save_output) save_label_visualization(label_map, "output_tiled_gpu.png");

    } else if (experiment == "verify") {
        cv::Mat baseline_map = run_baseline_cpu(binary_mask);
        cv::Mat tiled_map = run_tiled_cpu(binary_mask, tile_size);
        verify_label_equivalence(baseline_map, tiled_map);

    } else if (experiment == "verify_gpu") {
        cv::Mat baseline_map = run_baseline_cpu(binary_mask);
        cv::Mat gpu_map  = run_tiled_gpu_ccl(binary_mask, tile_size);
        verify_label_equivalence(baseline_map, gpu_map);

    } else {
        std::cerr << "Unknown experiment: " << experiment << "\n";
        std::cerr << "Options: baseline, tiled, tiled_gpu, verify, verify_gpu\n";
        return 1;
    }

    return 0;
}
