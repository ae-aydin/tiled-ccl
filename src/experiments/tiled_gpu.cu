#include "experiments/tiled_gpu.hpp"
#include "tiling.hpp"
#include "union_find.hpp"
#include "timer.hpp"
#include <iostream>
#include <memory>
#include <vector>


// Lightweight descriptor for a non-empty tile, used as the GPU work queue in Phase 1.
// storage_offset is the index into the flat label_storage buffer where this tile's results go.
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

    // Allocate GPU buffers before the timer. AllocGPUTileBuffers calls cudaMallocHost which
    // allocates pinned (page-locked) memory on the CPU side. Pinned memory allows the GPU to
    // read/write it directly over PCIe without going through the OS virtual memory system,
    // making host<->device transfers faster. It is expensive to allocate, so we do it once here.
    GPUTileBuffers gpu_bufs[2] = {
        AllocGPUTileBuffers(tile_size, tile_size),
        AllocGPUTileBuffers(tile_size, tile_size)
    };
    // A CUDA stream is an ordered queue of GPU operations. Operations in the same stream
    // run sequentially; operations in different streams can run concurrently.
    // Two streams let us overlap GPU work on tile t+1 with CPU work on tile t.
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
    // before launching any GPU work, because it must know tile[t+1] before tile[t] finishes.
    // So we scan the full image first to separate empty tiles from non-empty ones.
    //
    // all_tiles: metadata for every tile (empty and non-empty), indexed by grid position.
    // Used by Phase 2 (seam merge) and Phase 3 (relabel), which need to visit all tiles.
    //
    // non_empty_list: ordered list of only the tiles that contain foreground pixels.
    // This is the work queue the GPU pipeline consumes in Phase 1.
    //
    // total_storage_elems: cumulative pixel count across all non-empty tiles.
    // Used to allocate label_storage exactly, with no wasted space for empty tiles
    // (which are the majority in sparse pathology images).
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

            // Scan this tile's pixels, exit as soon as any foreground pixel is found.
            bool is_empty = true;
            for (int row = origin_y; row < origin_y + tile_height && is_empty; row++) {
                const uint8_t* ptr = binary_mask.ptr(row) + origin_x;
                for (int col = 0; col < tile_width; col++)
                    if (ptr[col]) { is_empty = false; break; }
            }

            if (is_empty) {
                // Empty tile: no foreground pixels, so no components and all edge labels
                // are 0 (background). Fully resolved here, GPU never sees this tile.
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

    double scan_ms = phase_timer.elapsed_ms();

    // label_storage: a single flat int32 buffer holding per-pixel label results for all
    // non-empty tiles packed end-to-end. Each tile's block starts at its storage_offset.
    // Phase 1 writes into it; Phase 3 reads via local_label_mat (a cv::Mat view, no copies).
    //
    // Pre-faulting: OSes allocate memory lazily. new[] only reserves virtual address space;
    // physical RAM pages are mapped on first write, triggering a page fault per 4 KB page.
    // Without this loop, those faults would fire during Phase 1's copyTo calls, stalling
    // the CPU side of the pipeline mid-flight. Touching one element per page now, in parallel,
    // pays the fault cost upfront in bulk before the timed pipeline starts.
    //
    // The "? 1" guard avoids allocating 0 bytes (undefined behavior in some allocators)
    // when the image has no foreground pixels at all.
    std::unique_ptr<int32_t[]> label_storage(new int32_t[total_storage_elems ? total_storage_elems : 1]);
    {
        int32_t* p = label_storage.get();
        int64_t n_pages = ((int64_t)total_storage_elems * sizeof(int32_t) + 4095) / 4096;
        #pragma omp parallel for schedule(static)
        for (int64_t pg = 0; pg < n_pages; pg++)
            p[pg * (4096 / sizeof(int32_t))] = 0;
    }

    double phase0_ms = phase_timer.elapsed_ms();
    double prefault_ms = phase0_ms - scan_ms;

    // --- Phase 1: Double-buffered GPU CCL ---
    //
    // Processes non-empty tiles using two buffer slots (0 and 1) and two CUDA streams,
    // so GPU work on tile t+1 overlaps with CPU result-collection for tile t.
    //
    // Buffer slots alternate: slot = t % 2. Each slot has its own pinned host buffer
    // and device memory, so one slot can be in use by the GPU while the other is being
    // read by the CPU.
    //
    // Each iteration t:
    //   1. Launch GPU work for tile t+1 on streams[1-slot] (async, returns immediately)
    //   2. Wait for tile t to finish on streams[slot] (cudaStreamSynchronize blocks until done)
    //   3. Copy tile t results from pinned buffer into label_storage, fill in tile metadata
    //
    // Steps 1 and 3 overlap in time: while the CPU copies tile t, the GPU is already
    // computing tile t+1 on the other stream.
    //
    // running_global_offset: cumulative foreground label count across all collected tiles.
    // Each tile's local labels (1..N) are shifted by this to become globally unique.
    //
    // ne_pending_offset[slot]: snapshot of running_global_offset taken when the tile on
    // that slot was launched. Needed because by the time we collect tile t, running_global_offset
    // has already advanced for tile t+1.
    std::cout << "--- Phase 1: Per-tile GPU CCL (tile size: " << tile_size << ") ---\n";
    phase_timer.start();

    int running_global_offset = 0;
    int total_local_labels = 0;
    int total_non_empty = (int)non_empty_list.size();

    // ne_pending_offset[slot]: snapshot of running_global_offset taken when the tile on
    // that slot was launched. By the time we collect a tile, running_global_offset has
    // already advanced for the next tile, so we need this saved value.
    // Initialized to {0,0} - slot 0 is correct for tile 0 (no tiles before it).
    int ne_pending_offset[2] = {0, 0};

    // Seed: launch tile 0 before the loop so the GPU is already working on iteration 0.
    if (total_non_empty > 0) {
        const TileID& id = non_empty_list[0];
        LabelTileGPU(binary_mask.ptr(id.origin_y) + id.origin_x, binary_mask.step[0], id.tile_height, id.tile_width, gpu_bufs[0], streams[0]);
    }

    // Each iteration collects tile t. Before waiting, it launches tile t+1 so the
    // GPU stays busy while the CPU reads results - that is the overlap.
    for (int t = 0; t < total_non_empty; t++) {
        int slot = t % 2; // which buffer slot tile t is on (alternates 0/1)

        if (t + 1 < total_non_empty) {
            const TileID& next = non_empty_list[t + 1];
            // launch tile t+1 on the other slot before waiting for t, so GPU stays busy while CPU reads t
            LabelTileGPU(binary_mask.ptr(next.origin_y) + next.origin_x, binary_mask.step[0], next.tile_height, next.tile_width, gpu_bufs[1 - slot], streams[1 - slot]);
        }

        cudaStreamSynchronize(streams[slot]); // wait for tile t's GPU work to finish

        const TileID& prev_id = non_empty_list[t];
        TileInfo& cur_tile = all_tiles[prev_id.tile_row * num_tile_cols + prev_id.tile_col]; // metadata entry for tile t
        cur_tile.global_label_offset = ne_pending_offset[slot]; // cumulative fg label count of all tiles before this one (saved at launch time, since running_global_offset has since advanced)
        int num_fg = *gpu_bufs[slot].h_counter; // number of foreground components found by the GPU in this tile
        // pitch (row stride) = max_cols * sizeof(int32_t): the GPU buffer is allocated for
        // the worst-case tile width, so rows are spaced max_cols apart regardless of actual width.
        cv::Mat src_mat(prev_id.tile_height, prev_id.tile_width, CV_32S, gpu_bufs[slot].host_labels, gpu_bufs[slot].max_cols * sizeof(int32_t));
        int32_t* dst_ptr = label_storage.get() + prev_id.storage_offset; // destination in flat label_storage at this tile's pre-assigned slot
        // label_storage is tightly packed (no padding between rows), so pitch = actual width.
        cv::Mat dst_mat(prev_id.tile_height, prev_id.tile_width, CV_32S, dst_ptr, prev_id.tile_width * sizeof(int32_t));
        src_mat.copyTo(dst_mat); // copy from pitched pinned buffer into tightly packed label_storage
        cur_tile.local_label_mat = dst_mat; // point tile's label mat at its label_storage region, no extra alloc
        cur_tile.num_local_labels = num_fg + 1; // +1 for background label 0
        extract_tile_edges(cur_tile, ne_pending_offset[slot]); // convert edge pixel labels to globally unique range
        running_global_offset += num_fg; // advance global offset by fg component count of this tile
        total_local_labels += num_fg;

        // Record the offset for the next tile now that this tile's fg count is added in.
        if (t + 1 < total_non_empty)
            ne_pending_offset[1 - slot] = running_global_offset;
    }

    // Release GPU resources now that all tiles are collected.
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    FreeGPUTileBuffers(gpu_bufs[0]);
    FreeGPUTileBuffers(gpu_bufs[1]);

    double phase1_ms = phase_timer.elapsed_ms();
    int max_global_label = running_global_offset;

    std::cout << "\nTotal local labels before merge: " << total_local_labels << "\n";
    std::cout << "Global label range: [1, " << max_global_label << "]\n\n";

    std::cout << "--- Phase 2: Seam Merge (CPU Union-Find) ---\n";
    UnionFind union_find(max_global_label + 1); // size +1 because labels are 1-indexed; index 0 is unused
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

    std::cout << "Time - Phase 0 (scan + alloc): " << phase0_ms << " ms"
              << "  [scan: " << scan_ms << " ms, pre-fault: " << prefault_ms << " ms]\n";
    std::cout << "Time - Phase 1 (GPU tile CCL): " << phase1_ms << " ms\n";
    std::cout << "Time - Phase 2 (seam merge): " << phase2_ms << " ms\n";
    std::cout << "Time - Phase 3 (relabel): " << phase3_ms << " ms\n";
    std::cout << "Time - Total: " << total_ms << " ms\n";
    std::cout << "Throughput: " << throughput_mpix_s << " MPixels/s\n\n";

    return label_map;
}
