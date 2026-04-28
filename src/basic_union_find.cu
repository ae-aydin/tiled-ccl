// Copyright (c) 2020, the YACCLAB contributors.
// BSD 3-Clause License — see licenses/YACCLAB_LICENSE for full text.
//
// Adapted from YACCLAB's UF_naive for standalone use:
//   - Removed YACCLAB framework (labeling_algorithms.h, register.h, GpuLabeling2D)
//   - Replaced OpenCV CUDA bindings (PtrStepSzb/Zi) with plain pitched pointers
//   - Added LabelTileGPU() host wrapper
//   - Kernel bodies are unchanged from the original

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "experiments/tiled_gpu.hpp"

// Thread block dimensions for all kernels. Each block has BLOCK_X * BLOCK_Y = 256 threads.
// BLOCK_X = 32 matches the warp size (the GPU's smallest unit of parallel execution),
// which ensures the first row of every block is warp-aligned for efficient memory access.
#define BLOCK_X 32
#define BLOCK_Y 8

// Anonymous namespace: everything here is file-local.
namespace {
    // Returns the root index of the union-find tree by walking up parent pointers.
    // Labels are stored as (flat_index + 1), with 0 reserved for background.
    // No path compression here: during Merge, thousands of threads traverse the tree
    // concurrently, so writing parent pointers back inline would race with other threads
    // doing the same traversal. Path compression is done safely in a separate
    // PathCompression kernel after all Merge work is finished.
    // __device__ means this function runs on the GPU and can only be called from GPU code.
    __device__ unsigned Find(const int* labels, unsigned n) {
        unsigned label = labels[n];
        assert(label > 0);
        while (label - 1 != n) {
            n = label - 1;
            label = labels[n];
            assert(label > 0);
        }
        return n;
    }

    // Merges the two components containing pixels a and b.
    // Uses atomicMin to point the larger root at the smaller one: atomicMin(labels + b, a + 1)
    // writes a+1 into labels[b] only if the current value is greater, and returns the old value.
    // If old == b+1 (b was still its own root), the merge succeeded. Otherwise another thread
    // already changed b's root, so we retry with the updated roots. The loop converges because
    // roots always move to smaller indices.
    __device__ void Union(int* labels, unsigned a, unsigned b) {
        bool done;
        do {
            a = Find(labels, a);
            b = Find(labels, b);
            if (a < b) {
                int old = atomicMin(labels + b, a + 1);
                done = (old == b + 1); // true if b was still a root when we wrote
                b = old - 1;
            } else if (b < a) {
                int old = atomicMin(labels + a, b + 1);
                done = (old == a + 1);
                a = old - 1;
            } else {
                done = true; // already in the same component
            }
        } while (!done);
    }

    // Each foreground pixel initializes its label to its own flat index + 1.
    // Background pixels get 0.
    __global__ void Initialization(const uint8_t* img, int img_pitch, int* labels, int labels_pitch, int cols, int rows) {
        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        if (x < (unsigned)cols && y < (unsigned)rows) {
            unsigned img_index = y * img_pitch + x;
            unsigned labels_index = y * labels_pitch + x;
            if (img[img_index]) {
                labels[labels_index] = labels_index + 1;
            } else {
                labels[labels_index] = 0;
            }
        }
    }

    // Each foreground pixel unions with its 8-connected foreground neighbors
    // that appear above or to the left (avoids double-processing).
    __global__ void Merge(int* labels, int labels_pitch, int cols, int rows) {
        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        if (x < (unsigned)cols && y < (unsigned)rows) {
            unsigned idx = y * labels_pitch + x;
            if (labels[idx]) {
                if (y > 0) {
                    if (x > 0 && labels[idx - labels_pitch - 1])
                        Union(labels, idx, idx - labels_pitch - 1);
                    if (labels[idx - labels_pitch])
                        Union(labels, idx, idx - labels_pitch);
                    if (x + 1 < (unsigned)cols && labels[idx - labels_pitch + 1])
                        Union(labels, idx, idx - labels_pitch + 1);
                }
                if (x > 0 && labels[idx - 1])
                    Union(labels, idx, idx - 1);
            }
        }
    }

    // Each foreground pixel writes its root index + 1 as its final label.
    __global__ void PathCompression(int* labels, int labels_pitch, int cols, int rows) {
        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        if (x < (unsigned)cols && y < (unsigned)rows) {
            unsigned idx = y * labels_pitch + x;
            if (labels[idx]) {
                labels[idx] = Find(labels, idx) + 1;
            }
        }
    }

    // Root pixels (labels[idx] == idx + 1) atomically claim a compact ID 1..N.
    // compact[idx] maps root flat_idx to compact ID; non-roots are left zero.
    // atomicAdd(counter, 1) returns the old value, so +1 makes IDs start at 1.
    __global__ void AssignCompactIDs(const int* labels, int* compact, int* counter, int labels_pitch, int cols, int rows) {
        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        if (x < (unsigned)cols && y < (unsigned)rows) {
            unsigned idx = y * labels_pitch + x;
            if (labels[idx] != 0 && labels[idx] == (int)(idx + 1))
                compact[idx] = atomicAdd(counter, 1) + 1;
        }
    }

    // Each foreground pixel looks up its root in compact[] and writes the compact ID.
    // After this, labels[idx] is 0 (background) or in 1..num_fg (foreground), with no gaps.
    __global__ void ApplyCompactIDs(int* labels, const int* compact, int labels_pitch, int cols, int rows) {
        unsigned x = blockIdx.x * BLOCK_X + threadIdx.x;
        unsigned y = blockIdx.y * BLOCK_Y + threadIdx.y;
        if (x < (unsigned)cols && y < (unsigned)rows) {
            unsigned idx = y * labels_pitch + x;
            if (labels[idx] != 0)
                labels[idx] = compact[labels[idx] - 1]; // labels[idx]-1 is root flat_idx
        }
    }

}

GPUTileBuffers AllocGPUTileBuffers(int max_rows, int max_cols) {
    GPUTileBuffers bufs;
    bufs.max_rows = max_rows;
    bufs.max_cols = max_cols;
    // cudaMallocPitch allocates GPU memory with automatic row padding so that each row
    // starts at an address aligned to the GPU's memory transaction size. This enables
    // coalesced (grouped) memory reads in kernels, which is critical for GPU throughput.
    // The actual row stride in bytes is written into img_pitch_bytes / labels_pitch_bytes.
    cudaMallocPitch(&bufs.device_mask, &bufs.img_pitch_bytes, max_cols * sizeof(uint8_t), max_rows);
    cudaMallocPitch(&bufs.device_labels, &bufs.labels_pitch_bytes, max_cols * sizeof(int), max_rows);
    cudaMalloc(&bufs.device_compact, bufs.labels_pitch_bytes * max_rows); // same shape as device_labels
    cudaMalloc(&bufs.d_counter, sizeof(int)); // single int on device for atomic counting
    cudaMallocHost(&bufs.host_labels, (size_t)max_rows * max_cols * sizeof(int32_t)); // pinned CPU memory for fast D2H
    cudaMallocHost(&bufs.h_counter, sizeof(int)); // pinned CPU memory to receive d_counter after D2H
    return bufs;
}

void FreeGPUTileBuffers(GPUTileBuffers& bufs) {
    cudaFree(bufs.device_mask);
    cudaFree(bufs.device_labels);
    cudaFree(bufs.device_compact);
    cudaFree(bufs.d_counter);
    cudaFreeHost(bufs.host_labels);
    cudaFreeHost(bufs.h_counter);
    bufs = {}; // zero-initialize the struct so dangling pointers don't linger
}

void LabelTileGPU(const uint8_t* h_tile, size_t h_tile_stride, int rows, int cols, const GPUTileBuffers& bufs, cudaStream_t stream) {
    // Convert byte-based pitches to element-based pitches for kernel indexing.
    int img_pitch = (int)(bufs.img_pitch_bytes / sizeof(uint8_t));
    int labels_pitch = (int)(bufs.labels_pitch_bytes / sizeof(int));

    // Async host-to-device copy of the tile's binary mask into pitched GPU memory.
    // cudaMemcpy2DAsync handles the stride mismatch: h_tile rows are h_tile_stride bytes apart
    // (full image row width), while device_mask rows are img_pitch_bytes apart (padded).
    cudaMemcpy2DAsync(bufs.device_mask, bufs.img_pitch_bytes, h_tile, h_tile_stride, cols * sizeof(uint8_t), rows, cudaMemcpyHostToDevice, stream);

    // dim3 is CUDA's 3D integer vector for specifying grid and block dimensions.
    // block_size: each block has BLOCK_X * BLOCK_Y threads arranged in a 2D tile.
    // grid_size: ceiling division ensures every pixel gets a thread even if dims are not multiples.
    dim3 block_size(BLOCK_X, BLOCK_Y);
    dim3 grid_size((cols + BLOCK_X - 1) / BLOCK_X, (rows + BLOCK_Y - 1) / BLOCK_Y);

    // CCL pipeline. <<<grid, block, 0, stream>>> is CUDA kernel launch syntax.
    // 0 = no dynamic shared memory. All launches go to the same stream so they run in order.
    Initialization<<<grid_size, block_size, 0, stream>>>(bufs.device_mask, img_pitch, bufs.device_labels, labels_pitch, cols, rows);
    Merge<<<grid_size, block_size, 0, stream>>>(bufs.device_labels, labels_pitch, cols, rows);
    PathCompression<<<grid_size, block_size, 0, stream>>>(bufs.device_labels, labels_pitch, cols, rows);

    // Compaction: remap scattered root indices to a dense 1..N range.
    cudaMemsetAsync(bufs.device_compact, 0, bufs.labels_pitch_bytes * bufs.max_rows, stream); // clear compact buffer
    cudaMemsetAsync(bufs.d_counter, 0, sizeof(int), stream);                                  // reset atomic counter
    AssignCompactIDs<<<grid_size, block_size, 0, stream>>>(bufs.device_labels, bufs.device_compact, bufs.d_counter, labels_pitch, cols, rows);
    ApplyCompactIDs<<<grid_size, block_size, 0, stream>>>(bufs.device_labels, bufs.device_compact, labels_pitch, cols, rows);

    // Async device-to-host copies. All operations above are queued on the same stream,
    // so these copies start only after all kernels finish. Caller must cudaStreamSynchronize
    // before reading host_labels or h_counter.
    cudaMemcpy2DAsync(bufs.host_labels, bufs.max_cols * sizeof(int32_t), bufs.device_labels, bufs.labels_pitch_bytes, cols * sizeof(int32_t), rows, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(bufs.h_counter, bufs.d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream);
}
