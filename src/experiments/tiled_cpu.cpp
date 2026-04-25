#include "experiments/tiled_cpu.hpp"
#include "tiling.hpp"
#include "union_find.hpp"
#include "timer.hpp"
#include <iostream>

cv::Mat run_tiled_cpu(const cv::Mat& binary_mask, int tile_size) {
    std::cout << "=== Experiment: Tiled CPU ===\n";
    Timer total_timer;
    total_timer.start();

    std::cout << "--- Phase 1: Per-tile CCL (tile size: " << tile_size << ") ---\n";
    int max_global_label = 0;
    int total_local_labels = 0;

    Timer phase_timer;
    phase_timer.start();
    std::vector<TileInfo> all_tiles = extract_tiles(binary_mask, tile_size, max_global_label, total_local_labels);
    double phase1_ms = phase_timer.elapsed_ms();

    std::cout << "--- Phase 2: Seam Merge (CPU Union-Find) ---\n";
    UnionFind union_find(max_global_label + 1);
    int num_tile_cols = (binary_mask.cols + tile_size - 1) / tile_size;
    int num_tile_rows = (binary_mask.rows + tile_size - 1) / tile_size;

    phase_timer.start();
    int total_merges = merge_seams(all_tiles, union_find, num_tile_cols, num_tile_rows);
    double phase2_ms = phase_timer.elapsed_ms();
    std::cout << "Merges performed: " << total_merges << "\n\n";

    std::cout << "--- Phase 3: Relabel ---\n";
    int num_final_components = 0;

    phase_timer.start();
    cv::Mat label_map = build_label_map(all_tiles, union_find, binary_mask.rows, binary_mask.cols, max_global_label, num_final_components);
    double phase3_ms = phase_timer.elapsed_ms();

    double total_ms = total_timer.elapsed_ms();
    double total_pixels = binary_mask.rows * binary_mask.cols;
    double throughput_mpix_s = (total_pixels / 1e6) / (total_ms / 1e3);

    std::cout << "Local labels (before merge): " << total_local_labels << "\n";
    std::cout << "Final components: " << num_final_components << "\n";
    std::cout << "Merged across seams: " << total_local_labels - num_final_components << "\n\n";

    std::cout << "Time - Phase 1 (tile CCL): " << phase1_ms << " ms\n";
    std::cout << "Time - Phase 2 (seam merge): " << phase2_ms << " ms\n";
    std::cout << "Time - Phase 3 (relabel): " << phase3_ms << " ms\n";
    std::cout << "Time - Total: " << total_ms  << " ms\n";
    std::cout << "Throughput: " << throughput_mpix_s << " MPixels/s\n\n";

    return label_map;
}
