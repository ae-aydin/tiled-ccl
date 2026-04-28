#include "tiling.hpp"
#include <iostream>
#include <cstring>
#include <opencv2/core.hpp>


// Converts edge pixel labels from tile-local to globally unique by adding running_global_offset; background (0) is left as 0.
void extract_tile_edges(TileInfo& tile, int running_global_offset) {
    // Shift each edge pixel's local label into the globally unique range.
    // Background (local label 0) stays 0 - it has no identity across tiles.
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


// Splits binary_mask into tiles, runs per-tile CCL, assigns globally unique label offsets, and extracts edge labels for seam merging.
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


// Walks horizontal and vertical seams between adjacent tiles and unions global labels that touch across the boundary.
int merge_seams(std::vector<TileInfo>& tiles, UnionFind& union_find, int num_tile_cols, int num_tile_rows) {
    int total_merges = 0;

    for (int tile_row = 0; tile_row < num_tile_rows; tile_row++) {
        for (int tile_col = 0; tile_col < num_tile_cols - 1; tile_col++) {
            TileInfo& left_tile = tiles[tile_row * num_tile_cols + tile_col];
            TileInfo& right_tile = tiles[tile_row * num_tile_cols + tile_col + 1];
            int seam_length = std::min((int)left_tile.right_edge_labels.size(), (int)right_tile.left_edge_labels.size());
            for (int pixel = 0; pixel < seam_length; pixel++) {
                int left_label = left_tile.right_edge_labels[pixel];
                int right_label = right_tile.left_edge_labels[pixel];
                if (left_label && right_label && union_find.Find(left_label) != union_find.Find(right_label)) {
                    union_find.Union(left_label, right_label);
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
                int top_label = top_tile.bottom_edge_labels[pixel];
                int bottom_label = bottom_tile.top_edge_labels[pixel];
                if (top_label && bottom_label && union_find.Find(top_label) != union_find.Find(bottom_label)) {
                    union_find.Union(top_label, bottom_label);
                    total_merges++;
                }
            }
        }
    }

    return total_merges;
}


// Phase 3: resolve global labels to compact final IDs, then write the output image.
// Label hierarchy: local (per-tile, resets each tile) -> global (cumulative count of local labels across tiles) -> root (UF representative) -> final (sequential 1..N)
// tiles: all tiles with their local label mats and edge label vectors
// union_find: UF structure populated by merge_seams; maps global labels to component roots
// image_height: full image height for the output label map
// image_width: full image width for the output label map
// max_global_label: highest global label assigned across all tiles (upper bound for table sizes)
// out_num_final_components: number of distinct foreground components found in the whole image
// returns: int32 label map of the full image; 0 = background, 1..N = component IDs
cv::Mat build_label_map(const std::vector<TileInfo>& tiles, UnionFind& union_find, int image_height, int image_width, int max_global_label, int& out_num_final_components) {
    // Build lookup tables serially. Find() does path compression (shortens the tree by writing
    // UnionFind.parent[x] = root for every node visited). Two threads calling Find() on labels
    // that share an ancestor would race on those parent[] writes, which is UB even if both write the same value.
    // root_to_final_id: assigns a sequential ID the first time each UF root is encountered.
    // label_to_final_id: pre-resolved flat lookup used safely in the parallel pixel loop below.
    std::vector<int> root_to_final_id(max_global_label + 1, 0);
    std::vector<int> label_to_final_id(max_global_label + 1, 0);
    int num_final_components = 0;
    for (int g = 1; g <= max_global_label; g++) {
        int root = union_find.Find(g);
        if (root_to_final_id[root] == 0)
            root_to_final_id[root] = ++num_final_components;
        label_to_final_id[g] = root_to_final_id[root];
    }

    // Allocated uninitialized (no cv::Mat::zeros) because tiles are non-overlapping and together
    // cover the full image, so every pixel gets written exactly once in the loop below.
    cv::Mat label_map(image_height, image_width, CV_32S);

    // schedule(dynamic): threads pick tiles one at a time from a queue rather than getting
    // a fixed pre-assigned chunk. Needed because empty tiles (just a memset) are much faster
    // than non-empty ones, so a static split would leave some threads idle while others finish late.
    #pragma omp parallel for schedule(dynamic)
    for (int t = 0; t < (int)tiles.size(); t++) {
        const TileInfo& tile = tiles[t];

        if (tile.local_label_mat.empty()) {
            for (int row = 0; row < tile.height; row++)
                memset(label_map.ptr<int>(tile.origin_y + row) + tile.origin_x, 0, tile.width * sizeof(int));
            continue;
        }

        for (int row = 0; row < tile.height; row++) {
            const int* src = tile.local_label_mat.ptr<int>(row);
            int* dst = label_map.ptr<int>(tile.origin_y + row) + tile.origin_x;
            for (int col = 0; col < tile.width; col++) {
                int local = src[col];
                if (local == 0) { dst[col] = 0; continue; }
                // local -> global (add offset) -> final ID (table lookup)
                dst[col] = label_to_final_id[local + tile.global_label_offset];
            }
        }
    }

    out_num_final_components = num_final_components;
    return label_map;
}
