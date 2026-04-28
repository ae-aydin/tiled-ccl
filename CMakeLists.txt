cmake_minimum_required(VERSION 3.18)
project(GigapixelCCL LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

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

target_compile_options(ccl PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-march=native -ffast-math>
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --extra-device-vectorization>
)
