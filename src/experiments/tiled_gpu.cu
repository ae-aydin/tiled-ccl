#include "experiments/tiled_gpu.hpp"
#include "tiling.hpp"
#include "union_find.hpp"
#include "timer.hpp"
#include <iostream>
#include <memory>
#include <vector>


struct TileID {
    int tile_row, tile_col;
    int origin_x, origin_y;
    int tile_width, tile_height;
    size_t storage_offset;
};

cv::Mat run_tiled_gpu_ccl(const cv::Mat& binary_mask, int tile_size) {
    std::cout << "=== Experiment: Tiled GPU CCL ===\n";

    int image_width = binary_mask.cols;
    int image_height = binary_mask.rows;
    int num_tile_cols = (image_width  + tile_size - 1) / tile_size;
    int num_tile_rows = (image_height + tile_size - 1) / tile_size;
    int total_tiles = num_tile_cols * num_tile_rows;

    // Allocate GPU resources before the timer — cudaMallocHost (pinned memory)
    // is a one-time OS cost amortized over all images in a real pipeline.
    GPUTileBuffers gpu_bufs[2] = {
        AllocGPUTileBuffers(tile_size, tile_size),
        AllocGPUTileBuffers(tile_size, tile_size)
    };
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    Timer total_timer, phase_timer;
    total_timer.start();
    phase_timer.start();

    std::cout << "--- Phase 0: scan + alloc ---\n";
    std::cout << "Tile grid: " << num_tile_cols << " x " << num_tile_rows << " = " << total_tiles << " tiles\n\n";

    // Pass 1: scan for empty/non-empty tiles.
    // Empty tiles are fully populated here; non-empty tiles queued for the GPU pipeline.
    // Exact allocation of label_storage (only non-empty pixels) cuts Phase 3 memory pressure.
    std::vector<TileInfo> all_tiles(total_tiles);
    std::vector<TileID> non_empty_list;
    non_empty_list.reserve(total_tiles);
    size_t total_storage_elems = 0;

    for (int tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        for (int tile_col = 0; tile_col < num_tile_cols; tile_col++) {
            int origin_x = tile_col * tile_size;
            int origin_y = tile_row * tile_size;
            int tile_width  = std::min(tile_size, image_width  - origin_x);
            int tile_height = std::min(tile_size, image_height - origin_y);

            TileInfo& tile = all_tiles[tile_row * num_tile_cols + tile_col];
            tile.tile_col = tile_col; tile.tile_row = tile_row;
            tile.origin_x = origin_x; tile.origin_y = origin_y;
            tile.width = tile_width; tile.height = tile_height;

            bool is_empty = true;
            for (int row = origin_y; row < origin_y + tile_height && is_empty; row++) {
                const uint8_t* ptr = binary_mask.ptr(row) + origin_x;
                for (int col = 0; col < tile_width; col++)
                    if (ptr[col]) { is_empty = false; break; }
            }

            if (is_empty) {
                tile.num_local_labels = 1;
                tile.global_label_offset = 0;
                tile.left_edge_labels.assign(tile_height, 0);
                tile.right_edge_labels.assign(tile_height, 0);
                tile.top_edge_labels.assign(tile_width,   0);
                tile.bottom_edge_labels.assign(tile_width, 0);
            } else {
                non_empty_list.push_back({tile_row, tile_col, origin_x, origin_y, tile_width, tile_height, total_storage_elems});
                total_storage_elems += (size_t)tile_width * tile_height;
            }
        }
    }

    // Parallel pre-faulting, warm up the memory now so the OS doesn't lag later
    std::unique_ptr<int32_t[]> label_storage(new int32_t[total_storage_elems ? total_storage_elems : 1]);
    {
        int32_t* p = label_storage.get();
        int64_t n_pages = ((int64_t)total_storage_elems * sizeof(int32_t) + 4095) / 4096;
        #pragma omp parallel for schedule(static)
        for (int64_t pg = 0; pg < n_pages; pg++)
            p[pg * (4096 / sizeof(int32_t))] = 0;
    }

    double phase0_ms = phase_timer.elapsed_ms();

    // Phase 1: double-buffered GPU CCL.
    // GPU tile t runs while CPU compact-processes tile t-1.
    std::cout << "--- Phase 1: Per-tile GPU CCL (tile size: " << tile_size << ") ---\n";
    phase_timer.start();

    int running_global_offset = 0;
    int total_local_labels = 0;
    int total_non_empty = (int)non_empty_list.size();
    int ne_pending_offset[2] = {0, 0};

    if (total_non_empty > 0) {
        const TileID& id = non_empty_list[0];
        LabelTileGPU(binary_mask.ptr(id.origin_y) + id.origin_x, binary_mask.step[0], id.tile_height, id.tile_width, gpu_bufs[0], streams[0]);
    }

    for (int t = 1; t <= total_non_empty; t++) {
        int slot = t % 2;
        int prev_slot = 1 - slot;

        if (t < total_non_empty) {
            const TileID& id = non_empty_list[t];
            LabelTileGPU(binary_mask.ptr(id.origin_y) + id.origin_x, binary_mask.step[0], id.tile_height, id.tile_width, gpu_bufs[slot], streams[slot]);
        }

        cudaStreamSynchronize(streams[prev_slot]);

        const TileID& p_id = non_empty_list[t - 1];
        TileInfo& tile = all_tiles[p_id.tile_row * num_tile_cols + p_id.tile_col];
        tile.global_label_offset = ne_pending_offset[prev_slot];

        int num_fg = *gpu_bufs[prev_slot].h_counter;
        cv::Mat src_mat(p_id.tile_height, p_id.tile_width, CV_32S, gpu_bufs[prev_slot].host_labels, gpu_bufs[prev_slot].max_cols * sizeof(int32_t));
        int32_t* dst_ptr = label_storage.get() + p_id.storage_offset;
        cv::Mat dst_mat(p_id.tile_height, p_id.tile_width, CV_32S, dst_ptr, p_id.tile_width * sizeof(int32_t));
        src_mat.copyTo(dst_mat);
        tile.local_label_mat = dst_mat;
        tile.num_local_labels = num_fg + 1;
        extract_tile_edges(tile, ne_pending_offset[prev_slot]);
        running_global_offset += num_fg;
        total_local_labels += num_fg;

        if (t < total_non_empty) ne_pending_offset[slot] = running_global_offset;
    }

    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    FreeGPUTileBuffers(gpu_bufs[0]);
    FreeGPUTileBuffers(gpu_bufs[1]);

    double phase1_ms = phase_timer.elapsed_ms();
    int max_global_label = running_global_offset;

    std::cout << "\nTotal local labels before merge: " << total_local_labels << "\n";
    std::cout << "Global label range: [1, " << max_global_label << "]\n\n";

    std::cout << "--- Phase 2: Seam Merge (CPU Union-Find) ---\n";
    UnionFind union_find(max_global_label + 1);
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

    double total_pixels = (double)binary_mask.rows * binary_mask.cols;
    double throughput_mpix_s = (total_pixels / 1e6) / (total_ms / 1e3);

    std::cout << "Local labels (before merge): " << total_local_labels << "\n";
    std::cout << "Final components: " << num_final_components << "\n";
    std::cout << "Merged across seams: " << total_local_labels - num_final_components << "\n\n";

    std::cout << "Time - Phase 0 (scan + alloc): " << phase0_ms << " ms\n";
    std::cout << "Time - Phase 1 (GPU tile CCL): " << phase1_ms << " ms\n";
    std::cout << "Time - Phase 2 (seam merge): " << phase2_ms << " ms\n";
    std::cout << "Time - Phase 3 (relabel):  " << phase3_ms << " ms\n";
    std::cout << "Time - Total: " << total_ms << " ms\n";
    std::cout << "Throughput: " << throughput_mpix_s << " MPixels/s\n\n";

    return label_map;
}
