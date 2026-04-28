#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$SCRIPT_DIR/..
CCL=$ROOT/build/ccl
IMAGES_DIR=$ROOT/gpx_masks
OUTPUTS_DIR=$ROOT/outputs

EXPERIMENT=${1:-tiled_gpu}
TILE_SIZE=${2:-1024}

# accept tiled_cpu as alias for tiled
[ "$EXPERIMENT" = "tiled_cpu" ] && EXPERIMENT="tiled"

if [[ "$EXPERIMENT" != "baseline" && "$EXPERIMENT" != "tiled" && "$EXPERIMENT" != "tiled_gpu" ]]; then
    echo "Usage: $0 [baseline|tiled|tiled_cpu|tiled_gpu] [tile_size]"
    exit 1
fi

mkdir -p "$OUTPUTS_DIR"
OUTPUT="$OUTPUTS_DIR/${EXPERIMENT}_tile${TILE_SIZE}_$(date +%Y%m%d_%H%M%S).txt"

{
    echo "Experiment : $EXPERIMENT"
    echo "Tile size  : $TILE_SIZE"
    echo "Date       : $(date)"
    echo "----------------------------------------"
} | tee "$OUTPUT"

for img in "$IMAGES_DIR"/*.png; do
    name=$(basename "$img")
    echo ">>> $name" | tee -a "$OUTPUT"
    OPENCV_IO_MAX_IMAGE_PIXELS=10000000000 "$CCL" "$img" "$EXPERIMENT" "$TILE_SIZE" 2>&1 | tee -a "$OUTPUT"
    echo "----------------------------------------" | tee -a "$OUTPUT"
done

echo "Results saved to: $OUTPUT"
