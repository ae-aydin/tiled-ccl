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

#define BLOCK_X 32
#define BLOCK_Y 8

namespace {
    // Returns the root index of the union-find tree.
    // Labels are stored as (index + 1), with 0 reserved for background.
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

    // Merges the union-find trees rooted at a and b via atomic-min on roots.
    __device__ void Union(int* labels, unsigned a, unsigned b) {
        bool done;
        do {
            a = Find(labels, a);
            b = Find(labels, b);
            if (a < b) {
                int old = atomicMin(labels + b, a + 1);
                done = (old == b + 1);
                b = old - 1;
            } else if (b < a) {
                int old = atomicMin(labels + a, b + 1);
                done = (old == a + 1);
                a = old - 1;
            } else {
                done = true;
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
    // compact[idx] maps root flat_idx → compact ID; non-roots are left zero.
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
    // After this, labels[idx] ∈ {0} ∪ {1..num_fg} with no gaps.
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
    cudaMallocPitch(&bufs.device_mask, &bufs.img_pitch_bytes, max_cols * sizeof(uint8_t), max_rows);
    cudaMallocPitch(&bufs.device_labels, &bufs.labels_pitch_bytes, max_cols * sizeof(int), max_rows);
    cudaMalloc(&bufs.device_compact, bufs.labels_pitch_bytes * max_rows);
    cudaMalloc(&bufs.d_counter, sizeof(int));
    cudaMallocHost(&bufs.host_labels, (size_t)max_rows * max_cols * sizeof(int32_t));
    cudaMallocHost(&bufs.h_counter, sizeof(int));
    return bufs;
}

void FreeGPUTileBuffers(GPUTileBuffers& bufs) {
    cudaFree(bufs.device_mask);
    cudaFree(bufs.device_labels);
    cudaFree(bufs.device_compact);
    cudaFree(bufs.d_counter);
    cudaFreeHost(bufs.host_labels);
    cudaFreeHost(bufs.h_counter);
    bufs = {};
}

void LabelTileGPU(const uint8_t* h_tile, size_t h_tile_stride, int rows, int cols, const GPUTileBuffers& bufs, cudaStream_t stream) {
    int img_pitch = (int)(bufs.img_pitch_bytes    / sizeof(uint8_t));
    int labels_pitch = (int)(bufs.labels_pitch_bytes / sizeof(int));

    cudaMemcpy2DAsync(bufs.device_mask, bufs.img_pitch_bytes, h_tile, h_tile_stride, cols * sizeof(uint8_t), rows, cudaMemcpyHostToDevice, stream);

    dim3 block_size(BLOCK_X, BLOCK_Y);
    dim3 grid_size((cols + BLOCK_X - 1) / BLOCK_X, (rows + BLOCK_Y - 1) / BLOCK_Y);

    Initialization<<<grid_size, block_size, 0, stream>>>(bufs.device_mask, img_pitch, bufs.device_labels, labels_pitch, cols, rows);
    Merge<<<grid_size, block_size, 0, stream>>>(bufs.device_labels, labels_pitch, cols, rows);
    PathCompression<<<grid_size, block_size, 0, stream>>>(bufs.device_labels, labels_pitch, cols, rows);

    cudaMemsetAsync(bufs.device_compact, 0, bufs.labels_pitch_bytes * bufs.max_rows, stream);
    cudaMemsetAsync(bufs.d_counter, 0, sizeof(int), stream);
    AssignCompactIDs<<<grid_size, block_size, 0, stream>>>(bufs.device_labels, bufs.device_compact, bufs.d_counter, labels_pitch, cols, rows);
    ApplyCompactIDs<<<grid_size, block_size, 0, stream>>>(bufs.device_labels, bufs.device_compact, labels_pitch, cols, rows);

    // Async D2H — caller must cudaStreamSynchronize before reading host_labels / h_counter.
    cudaMemcpy2DAsync(bufs.host_labels, bufs.max_cols * sizeof(int32_t), bufs.device_labels, bufs.labels_pitch_bytes, cols * sizeof(int32_t), rows, cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(bufs.h_counter, bufs.d_counter, sizeof(int), cudaMemcpyDeviceToHost, stream);
}
