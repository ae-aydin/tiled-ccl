#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "union_find.hpp"

struct TileInfo {
    int tile_col, tile_row;
    int origin_x, origin_y; // top-left corner of this tile in the full image
    int width, height; // tile size
    int global_label_offset; // add to a local label (> 0) to get its globally unique label
    int num_local_labels; // includes background (label 0)
    cv::Mat local_label_mat; // CV_32S, per-pixel local labels — kept for the relabel pass

    // One global label per pixel along each edge.
    // Used during seam merging to find which components touch across tile boundaries.
    std::vector<int> left_edge_labels, right_edge_labels;
    std::vector<int> top_edge_labels, bottom_edge_labels;
};

// Fills the four edge label vectors of a tile from its local_label_mat.
void extract_tile_edges(TileInfo& tile, int running_global_offset);

// Phase 1: slice the image into tiles and run CCL on each.
// Returns all tile metadata. Outputs the highest global label assigned and the total count of local labels across all tiles.
std::vector<TileInfo> extract_tiles(const cv::Mat& binary_mask, int tile_size, int& out_max_global_label, int& out_total_local_labels);

// Phase 2: walk all horizontal and vertical seams, union labels that touch.
// Returns the number of merges performed.
int merge_seams(std::vector<TileInfo>& tiles, UnionFind& union_find, int num_tile_cols, int num_tile_rows);

// Phase 3: apply union-find results to produce a full-image label map (CV_32S).
// Outputs the number of final unique components.
cv::Mat build_label_map(const std::vector<TileInfo>& tiles, UnionFind& union_find, int image_height, int image_width, int max_global_label, int& out_num_final_components);
