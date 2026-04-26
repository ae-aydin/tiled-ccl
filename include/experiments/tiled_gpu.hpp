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

// Pre-allocated buffers for processing one tile on the GPU.
// Allocated once at startup (AllocGPUTileBuffers) and reused for every tile —
// avoids per-tile cudaMalloc/Free which is extremely slow.
// Sized for the largest possible tile (max_rows × max_cols).
struct GPUTileBuffers {
    // --- Device buffers (live in GPU VRAM) ---

    // Destination for the H2D transfer of the tile's binary mask (uint8, 0 or 255).
    // Pitched allocation: rows are padded to a GPU-friendly stride so 2D memory
    // accesses are aligned to transaction boundaries — faster coalesced reads in kernels.
    uint8_t* device_mask;

    // Working buffer for the union-find label array during the CCL kernel.
    // Each pixel starts as its own root; the kernel merges neighbours until
    // all pixels in the same component share the same root. Also pitched.
    int* device_labels;

    // Scratch buffer for the compaction step.
    // After CCL, active roots are scattered across device_labels (e.g. roots at
    // positions 1, 7, 42...). Compaction remaps them to a dense range 1..N so
    // the output labels are consecutive integers with no gaps.
    // Indexed by flat pixel position → stores the compact ID assigned to that root.
    int* device_compact;

    // Atomic counter on the device used during compaction.
    // Each root claims a unique compact ID via atomicAdd on this counter.
    // Reset to 0 before each tile; its final value equals the number of
    // foreground components found in the tile (num_fg).
    int* d_counter;

    // --- Pinned host buffers (live in CPU RAM, page-locked) ---
    // Pinned memory bypasses the OS virtual memory system, allowing the GPU's DMA
    // engine to transfer directly to/from it without CPU involvement. This is what
    // makes async D2H transfers (cudaMemcpyAsync) possible — the GPU writes here
    // concurrently while the CPU is doing other work on a different stream.

    // Receives the compact per-pixel label result after D2H transfer.
    // Row pitch = max_cols elements (not pitched by CUDA — flat host layout).
    // Values are in range 1..num_fg (foreground) or 0 (background).
    // Phase 1 reads this to copy results into label_storage.
    int32_t* host_labels;

    // Receives the foreground component count (final value of d_counter) after D2H.
    // Read after cudaStreamSynchronize to know how many labels this tile produced.
    int* h_counter;

    // --- Dimensions ---
    int max_rows;        // maximum tile height this buffer was allocated for
    int max_cols;        // maximum tile width this buffer was allocated for
    size_t img_pitch_bytes;     // actual row stride of device_mask in bytes (≥ max_cols)
    size_t labels_pitch_bytes;  // actual row stride of device_labels in bytes (≥ max_cols * sizeof(int))
};

GPUTileBuffers AllocGPUTileBuffers(int max_rows, int max_cols);
void FreeGPUTileBuffers(GPUTileBuffers& bufs);

void LabelTileGPU(const uint8_t* h_tile, size_t h_tile_stride, int rows, int cols, const GPUTileBuffers& bufs, cudaStream_t stream);

cv::Mat run_tiled_gpu_ccl(const cv::Mat& binary_mask, int tile_size = 1024);
