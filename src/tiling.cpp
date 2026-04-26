#include "tiling.hpp"
#include <iostream>
#include <cstring>
#include <opencv2/core.hpp>


void extract_tile_edges(TileInfo& tile, int running_global_offset) {
    // Shift each edge pixel's local label into the globally unique range.
    // Background (local label 0) stays 0 — it has no identity across tiles.
    tile.left_edge_labels.resize(tile.height);
    tile.right_edge_labels.resize(tile.height);
    for (int row = 0; row < tile.height; row++) {
        int local_left = tile.local_label_mat.at<int>(row, 0);
        int local_right = tile.local_label_mat.at<int>(row, tile.width - 1);

        if (local_left == 0) tile.left_edge_labels[row] = 0;
        else tile.left_edge_labels[row] = local_left + running_global_offset;

        if (local_right == 0) tile.right_edge_labels[row] = 0;
        else tile.right_edge_labels[row] = local_right + running_global_offset;
    }

    tile.top_edge_labels.resize(tile.width);
    tile.bottom_edge_labels.resize(tile.width);
    for (int col = 0; col < tile.width; col++) {
        int local_top = tile.local_label_mat.at<int>(0, col);
        int local_bottom = tile.local_label_mat.at<int>(tile.height - 1, col);

        if (local_top == 0) tile.top_edge_labels[col] = 0;
        else tile.top_edge_labels[col] = local_top + running_global_offset;

        if (local_bottom == 0) tile.bottom_edge_labels[col] = 0;
        else tile.bottom_edge_labels[col] = local_bottom + running_global_offset;
    }
}


std::vector<TileInfo> extract_tiles(const cv::Mat& binary_mask, int tile_size, int& out_max_global_label, int& out_total_local_labels) {
    int image_width = binary_mask.cols;
    int image_height = binary_mask.rows;
    int num_tile_cols = (image_width + tile_size - 1) / tile_size;
    int num_tile_rows = (image_height + tile_size - 1) / tile_size;

    std::cout << "Tile grid: " << num_tile_cols << " x " << num_tile_rows << " = " << num_tile_cols * num_tile_rows << " tiles\n";

    std::vector<TileInfo> all_tiles;
    all_tiles.reserve(num_tile_cols * num_tile_rows);

    int running_global_offset = 0;
    int total_local_labels = 0;

    for (int tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        for (int tile_col = 0; tile_col < num_tile_cols; tile_col++) {
            int origin_x = tile_col * tile_size;
            int origin_y = tile_row * tile_size;
            int tile_width = std::min(tile_size, image_width  - origin_x);
            int tile_height = std::min(tile_size, image_height - origin_y);

            // Zero-copy view into the binary mask for this tile's region.
            // image_width is the row stride: tile rows are not contiguous in
            // memory, so OpenCV needs to know how many bytes to skip per row.
            uchar* tile_start = const_cast<uchar*>(binary_mask.ptr(origin_y)) + origin_x;
            cv::Mat tile_mat(tile_height, tile_width, CV_8UC1, tile_start, static_cast<size_t>(image_width));

            TileInfo tile;
            tile.tile_col = tile_col; tile.tile_row = tile_row;
            tile.origin_x = origin_x; tile.origin_y = origin_y;
            tile.width = tile_width; tile.height = tile_height;

            // Skipping empty tiles as whole slide images mostly sparse
            if (cv::countNonZero(tile_mat) == 0) {
                tile.num_local_labels = 1;  // just background
                tile.global_label_offset = running_global_offset;
                // edge label vectors left empty — all zeros means no foreground to merge
                tile.left_edge_labels.assign(tile_height, 0);
                tile.right_edge_labels.assign(tile_height, 0);
                tile.top_edge_labels.assign(tile_width, 0);
                tile.bottom_edge_labels.assign(tile_width, 0);
                all_tiles.push_back(std::move(tile));
                continue;
            } 

            cv::Mat tile_local_label_mat;
            int num_labels_with_bg = cv::connectedComponents(tile_mat, tile_local_label_mat, 8, CV_32S);
            int num_fg_components = num_labels_with_bg - 1;

            tile.num_local_labels = num_labels_with_bg;
            tile.global_label_offset = running_global_offset;
            tile.local_label_mat = std::move(tile_local_label_mat);

            extract_tile_edges(tile, running_global_offset);
            all_tiles.push_back(std::move(tile));

            running_global_offset += num_fg_components;
            total_local_labels += num_fg_components;
        }
    }

    out_max_global_label = running_global_offset;
    out_total_local_labels = total_local_labels;

    std::cout << "\nTotal local labels before merge: " << total_local_labels << "\n";
    std::cout << "Global label range: [1, " << running_global_offset << "]\n\n";

    return all_tiles;
}


int merge_seams(std::vector<TileInfo>& tiles, UnionFind& union_find, int num_tile_cols, int num_tile_rows) {
    int total_merges = 0;

    for (int tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        for (int tile_col = 0; tile_col < num_tile_cols - 1; tile_col++) {
            TileInfo& left_tile = tiles[tile_row * num_tile_cols + tile_col];
            TileInfo& right_tile = tiles[tile_row * num_tile_cols + tile_col + 1];
            int seam_length = std::min((int)left_tile.right_edge_labels.size(), (int)right_tile.left_edge_labels.size());
            for (int pixel = 0; pixel < seam_length; pixel++) {
                int l = left_tile.right_edge_labels[pixel];
                int r = right_tile.left_edge_labels[pixel];
                if (l != 0 && r != 0 && union_find.Find(l) != union_find.Find(r)) {
                    union_find.Union(l, r);
                    total_merges++;
                }
            }
        }
    }

    for (int tile_row = 0; tile_row < num_tile_rows - 1; tile_row++) {
        for (int tile_col = 0; tile_col < num_tile_cols; tile_col++) {
            TileInfo& top_tile = tiles[tile_row * num_tile_cols + tile_col];
            TileInfo& bottom_tile = tiles[(tile_row + 1) * num_tile_cols + tile_col];
            int seam_length = std::min((int)top_tile.bottom_edge_labels.size(), (int)bottom_tile.top_edge_labels.size());
            for (int pixel = 0; pixel < seam_length; pixel++) {
                int t = top_tile.bottom_edge_labels[pixel];
                int b = bottom_tile.top_edge_labels[pixel];
                if (t != 0 && b != 0 && union_find.Find(t) != union_find.Find(b)) {
                    union_find.Union(t, b);
                    total_merges++;
                }
            }
        }
    }

    return total_merges;
}


cv::Mat build_label_map(const std::vector<TileInfo>& tiles, UnionFind& union_find, int image_height, int image_width, int max_global_label, int& out_num_final_components) {
    // At this point every foreground pixel in the image has a globally unique label
    // (an integer in [1, max_global_label]) assigned during Phase 1. Phase 2 ran
    // union-find over the seams to record which of those global labels belong to the
    // same connected component across tile boundaries.
    //
    // This function has two jobs:
    //   1. Resolve the union-find to produce a compact final component ID for every
    //      global label (IDs are 1..num_final_components, no gaps).
    //   2. Write those final IDs into a full-image int32 label map.

    // --- Step 1: Build a lookup table from global label → final component ID ---
    //
    // We do this in a serial loop rather than calling Find() inside the parallel pixel
    // loop below. Find() performs path compression (it writes to the union-find parent
    // array to flatten the tree), so calling it from multiple threads simultaneously
    // would cause data races. Doing it once here, serially, is safe and fast.
    //
    // root_to_final_id : maps a root label (canonical representative of a component)
    //                    to its assigned final ID. Populated on first encounter.
    // label_to_final_id: maps every global label directly to its final ID, so the
    //                    parallel pixel loop below only needs a single array lookup.
    std::vector<int> root_to_final_id(max_global_label + 1, 0);
    std::vector<int> label_to_final_id(max_global_label + 1, 0);
    int num_final_components = 0;
    for (int g = 1; g <= max_global_label; g++) {
        int root = union_find.Find(g); // which component does label g belong to?
        if (root_to_final_id[root] == 0) // first time we see this root → assign it an ID
            root_to_final_id[root] = ++num_final_components;
        label_to_final_id[g] = root_to_final_id[root]; // record final ID for fast lookup
    }

    // --- Step 2: Write the full-image label map in parallel ---
    //
    // label_map is the output: same spatial dimensions as the input image, int32 per pixel.
    // Allocated uninitialized here because every pixel is written by the loop below —
    // no pixel is left unwritten, so no garbage values escape.
    cv::Mat label_map(image_height, image_width, CV_32S);

    // Each tile owns a distinct, non-overlapping rectangle of label_map, so parallel
    // writes across tiles cannot race. label_to_final_id is read-only in this loop.
    // schedule(dynamic) because tiles vary in size and empty vs non-empty tiles differ
    // greatly in work — dynamic scheduling balances load across threads.
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < (int)tiles.size(); t++) {
        const TileInfo& tile = tiles[t];

        if (tile.local_label_mat.empty()) {
            // Empty tile (all background): local_label_mat was never written,
            // so there are no foreground labels to look up. Write zeros directly.
            for (int row = 0; row < tile.height; row++)
                memset(label_map.ptr<int>(tile.origin_y + row) + tile.origin_x, 0, tile.width * sizeof(int));
            continue;
        }

        for (int row = 0; row < tile.height; row++) {
            const int* src = tile.local_label_mat.ptr<int>(row);  // local label per pixel (0 = background, 1..N = foreground)
            int* dst = label_map.ptr<int>(tile.origin_y + row) + tile.origin_x;
            for (int col = 0; col < tile.width; col++) {
                int local = src[col];
                if (local == 0) { dst[col] = 0; continue; }  // background pixel → 0
                // local + global_label_offset converts the tile-local label into the
                // globally unique label assigned during Phase 1. label_to_final_id then
                // maps that to the final merged component ID from Phase 2.
                dst[col] = label_to_final_id[local + tile.global_label_offset];
            }
        }
    }

    out_num_final_components = num_final_components;
    return label_map;
}
