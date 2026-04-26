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

    // --- Phase 0: Pre-scan ---
    //
    // The GPU double-buffer loop in Phase 1 needs to know all non-empty tiles upfront
    // before launching any GPU work — it must know tile[t+1] before tile[t] finishes.
    // So we scan the full image first to separate empty tiles from non-empty ones.
    //
    // all_tiles: metadata for every tile (empty and non-empty), indexed by grid position.
    //            Used by Phase 2 (seam merge) and Phase 3 (relabel), which need to
    //            visit all tiles in order.
    //
    // non_empty_list: ordered list of only the tiles that contain foreground pixels.
    //                 This is the work queue the GPU pipeline consumes in Phase 1.
    //
    // total_storage_elems: cumulative pixel count across all non-empty tiles.
    //                      Used to allocate label_storage to the exact size needed —
    //                      no wasted space for empty tiles, which are the majority in
    //                      sparse pathology images.
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

            // Scan this tile's pixels — exit as soon as any foreground pixel is found.
            bool is_empty = true;
            for (int row = origin_y; row < origin_y + tile_height && is_empty; row++) {
                const uint8_t* ptr = binary_mask.ptr(row) + origin_x;
                for (int col = 0; col < tile_width; col++)
                    if (ptr[col]) { is_empty = false; break; }
            }

            if (is_empty) {
                // Empty tile: no foreground pixels, so no components and all edge labels
                // are 0 (background). Fully resolved here — GPU never sees this tile.
                tile.num_local_labels = 1;
                tile.global_label_offset = 0;
                tile.left_edge_labels.assign(tile_height, 0);
                tile.right_edge_labels.assign(tile_height, 0);
                tile.top_edge_labels.assign(tile_width,   0);
                tile.bottom_edge_labels.assign(tile_width, 0);
            } else {
                // Non-empty tile: record its position and where its labels will live
                // inside label_storage. storage_offset is the flat index of the first
                // pixel of this tile's result block within the shared label_storage buffer.
                non_empty_list.push_back({tile_row, tile_col, origin_x, origin_y, tile_width, tile_height, total_storage_elems});
                total_storage_elems += (size_t)tile_width * tile_height;
            }
        }
    }

    // label_storage: a single flat int32 buffer that holds the per-pixel label results
    // for all non-empty tiles packed end-to-end. Each tile's block starts at its
    // storage_offset. Phase 1 writes into it; Phase 3 reads from it via local_label_mat
    // (a cv::Mat view into this buffer, so no copies needed at read time).
    //
    // Pre-faulting: modern OSes allocate memory lazily — new[] only reserves virtual address
    // space; physical RAM pages are assigned on first write (a page fault). Without this
    // loop, page faults would fire during Phase 1's copyTo calls, stalling the CPU side
    // of the double-buffer pipeline mid-flight. Touching one element per 4 KB page now,
    // in parallel with OpenMP, pays that cost upfront and in bulk.
    std::unique_ptr<int32_t[]> label_storage(new int32_t[total_storage_elems ? total_storage_elems : 1]);
    {
        int32_t* p = label_storage.get();
        int64_t n_pages = ((int64_t)total_storage_elems * sizeof(int32_t) + 4095) / 4096;
        #pragma omp parallel for schedule(static)
        for (int64_t pg = 0; pg < n_pages; pg++)
            p[pg * (4096 / sizeof(int32_t))] = 0;
    }

    double phase0_ms = phase_timer.elapsed_ms();

    // --- Phase 1: Double-buffered GPU CCL ---
    //
    // Processes non-empty tiles through the GPU pipeline using two buffer slots (0 and 1)
    // and two CUDA streams, so the GPU works on tile t while the CPU processes tile t-1.
    //
    // Buffer slots alternate: slot = t % 2, prev_slot = 1 - slot.
    // Each slot holds one tile's worth of pinned host memory (host_labels) and device memory.
    //
    // Each iteration:
    //   1. Launch GPU work for tile t on streams[slot]       (async, returns immediately)
    //   2. Wait for tile t-1 to finish on streams[prev_slot] (cudaStreamSynchronize)
    //   3. Read tile t-1 results from gpu_bufs[prev_slot] and store into label_storage
    //
    // Step 1 and step 3 overlap in time: while the CPU copies tile t-1 results,
    // the GPU is already computing tile t on the other stream.
    //
    // running_global_offset: counts total foreground labels assigned so far across all
    //   processed tiles. Each new tile's local labels (1..N) are shifted by this offset
    //   to make them globally unique across the whole image.
    //
    // ne_pending_offset[slot]: the running_global_offset value that was current when
    //   tile t was launched. Stored per-slot so we can correctly assign global_label_offset
    //   to tile t-1 after its GPU work completes (by which point running_global_offset
    //   has already advanced for tile t).
    std::cout << "--- Phase 1: Per-tile GPU CCL (tile size: " << tile_size << ") ---\n";
    phase_timer.start();

    int running_global_offset = 0;
    int total_local_labels = 0;
    int total_non_empty = (int)non_empty_list.size();
    int ne_pending_offset[2] = {0, 0};

    // Seed the pipeline: launch tile 0 before entering the loop so there is always
    // a tile in-flight on prev_slot when the loop body tries to collect results.
    if (total_non_empty > 0) {
        const TileID& id = non_empty_list[0];
        LabelTileGPU(binary_mask.ptr(id.origin_y) + id.origin_x, binary_mask.step[0], id.tile_height, id.tile_width, gpu_bufs[0], streams[0]);
    }

    for (int t = 1; t <= total_non_empty; t++) {
        int slot  = t % 2;       // buffer slot for tile t
        int prev_slot = 1 - slot;    // buffer slot for tile t-1 (whose results we collect)

        // Launch tile t on the GPU (async) — this runs concurrently with the CPU work below.
        if (t < total_non_empty) {
            const TileID& id = non_empty_list[t];
            LabelTileGPU(binary_mask.ptr(id.origin_y) + id.origin_x, binary_mask.step[0], id.tile_height, id.tile_width, gpu_bufs[slot], streams[slot]);
        }

        // Wait for tile t-1 (on prev_slot) to fully complete — H2D transfer, kernel,
        // compaction, and D2H of results back to host_labels and h_counter.
        cudaStreamSynchronize(streams[prev_slot]);

        // --- Collect results for tile t-1 ---
        const TileID& p_id = non_empty_list[t - 1];
        TileInfo& tile = all_tiles[p_id.tile_row * num_tile_cols + p_id.tile_col];

        // global_label_offset: shifts this tile's local labels (1..N) into the global
        // label space so no two tiles share a label ID. Recorded at the time tile t-1
        // was launched (ne_pending_offset[prev_slot]), not now, because running_global_offset
        // may have already advanced for tile t.
        tile.global_label_offset = ne_pending_offset[prev_slot];

        // h_counter holds the number of foreground components found in this tile.
        // host_labels holds the compacted per-pixel label result (local IDs 1..num_fg).
        int num_fg = *gpu_bufs[prev_slot].h_counter;

        // Wrap the pinned host_labels buffer as a cv::Mat (no copy) and copy its
        // contents into label_storage at this tile's reserved block. This is the only
        // copy — after this, local_label_mat points directly into label_storage so
        // Phase 3 can read it without any further allocation or copying.
        cv::Mat src_mat(p_id.tile_height, p_id.tile_width, CV_32S, gpu_bufs[prev_slot].host_labels, gpu_bufs[prev_slot].max_cols * sizeof(int32_t));
        int32_t* dst_ptr = label_storage.get() + p_id.storage_offset;
        cv::Mat dst_mat(p_id.tile_height, p_id.tile_width, CV_32S, dst_ptr, p_id.tile_width * sizeof(int32_t));
        src_mat.copyTo(dst_mat);
        tile.local_label_mat = dst_mat;
        tile.num_local_labels = num_fg + 1; // +1 for background (label 0)

        // Extract the four edge rows/columns of this tile's label result.
        // These edge labels (shifted to global IDs) are what Phase 2 compares across
        // adjacent tile boundaries to decide which components to merge.
        extract_tile_edges(tile, ne_pending_offset[prev_slot]);
        running_global_offset += num_fg;
        total_local_labels += num_fg;

        // Record the current global offset for tile t so we can assign it correctly
        // once tile t's GPU work completes (next iteration's prev_slot collection).
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
