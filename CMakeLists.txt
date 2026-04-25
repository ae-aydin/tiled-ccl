cmake_minimum_required(VERSION 3.18)
project(GigapixelCCL LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(ccl
    main.cpp
    src/tiling.cpp
    src/visualization.cpp
    src/verify.cpp
    src/basic_union_find.cu
    src/experiments/baseline_cpu.cpp
    src/experiments/tiled_cpu.cpp
    src/experiments/tiled_gpu.cu
)

set_target_properties(ccl PROPERTIES CUDA_ARCHITECTURES "89")
target_link_libraries(ccl ${OpenCV_LIBS} OpenMP::OpenMP_CXX CUDA::cudart)
target_include_directories(ccl SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
