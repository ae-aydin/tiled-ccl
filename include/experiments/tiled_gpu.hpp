#pragma once
#include <cstdint>
#include <cstddef>
#include <opencv2/opencv.hpp>

// cudaStream_t for plain C++ translation units that don't include cuda_runtime.h.
// .cu files include cuda_runtime.h first, which defines the real type identically.
#ifndef __CUDA_RUNTIME_H__
struct CUstream_st;
typedef CUstream_st* cudaStream_t;
#endif

// Pre-allocated device buffers + pinned host output for a tile up to max_rows × max_cols.
struct GPUTileBuffers {
    uint8_t* device_mask;
    int* device_labels;
    int* device_compact; // scratch: root flat_idx → compact ID (1..N)
    int* d_counter; // device-side atomic counter for compact ID assignment
    int32_t* host_labels; // pinned; row pitch = max_cols elements; compact 1..N after LabelTileGPU
    int* h_counter; // pinned; receives num_fg after sync
    int max_rows;
    int max_cols;
    size_t img_pitch_bytes;
    size_t labels_pitch_bytes;
};

GPUTileBuffers AllocGPUTileBuffers(int max_rows, int max_cols);
void FreeGPUTileBuffers(GPUTileBuffers& bufs);

void LabelTileGPU(const uint8_t* h_tile, size_t h_tile_stride, int rows, int cols, const GPUTileBuffers& bufs, cudaStream_t stream);

cv::Mat run_tiled_gpu_ccl(const cv::Mat& binary_mask, int tile_size = 1024);
