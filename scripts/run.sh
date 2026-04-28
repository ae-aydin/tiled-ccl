#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
CCL=$SCRIPT_DIR/../build/ccl

if [ $# -eq 0 ]; then
    echo "Usage: $0 <image> [experiment] [tile_size]"
    echo "       experiment: baseline | tiled | tiled_cpu | tiled_gpu (default: tiled_gpu)"
    echo "       tile_size:  default 1024"
    exit 1
fi

OPENCV_IO_MAX_IMAGE_PIXELS=10000000000 "$CCL" "$@"
