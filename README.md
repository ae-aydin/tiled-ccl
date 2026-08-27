# Tiled Gigapixel CCL

Out-of-core tile-based connected components labeling (CCL) pipeline for binary images that exceed GPU memory (e.g., gigapixel pathology slides). Partitions on host, labels tiles on GPU with adapted union-find, merges cross-tile equivalences host-side.

## Results
The pipeline correctly handles gigapixel inputs that exceed consumer GPU VRAM (verified on five immunohistochemistry slides against OpenCV and CPU-tiled baselines). At tested tile sizes it does not yet improve runtime; ablation isolates overhead to the upfront scan and buffer allocation. Batch-mode amortization and alternatives like GPUDirect Storage are discussed as paths to close the gap.

## Build & Execute

```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

Run (set OpenCV pixel limit before process start):
```shell
OPENCV_IO_MAX_IMAGE_PIXELS=10000000000 ./ccl [image_path] [experiment] [tile_size] [-o]
```

Experiments: `baseline`, `tiled`, `tiled_gpu` (default), `verify`, `verify_gpu`. Tile size defaults to 1024.

## Environment
- Linux (Fedora 43)
- CUDA 13.2
- CMake 3.31.11
- OpenCV 4.11
- GCC 15.2.1
- C++17

## License
- BSD-3-Clause for `src/basic_union_find.cu`
- MIT for rest
