# Tiled Gigapixel CCL

## CMP674 Parallel Computing with GPUs Project

### Environment

- Linux (Fedora 43)
- CUDA 13.2
- CMake 3.31.11
- OpenCV 4.11
- GCC 15.2.1
- C++17

### Build & Execute

**For Linux:**

```shell
# Build
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 
make -j$(nproc) 

# Run
# We set the pixel limit here so OpenCV doesn't crash on gigapixel files
# The env var must be set before the process starts — OpenCV caches the pixel limit at library load time
OPENCV_IO_MAX_IMAGE_PIXELS=10000000000 ./ccl [options]
```
