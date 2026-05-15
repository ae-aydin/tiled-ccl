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

awk '
/^Time - / {
    label = $0
    sub(/^Time - /, "", label)
    sub(/ \(.*/, "", label)
    sub(/:.*/, "", label)
    label = label " (ms)"
    split($0, parts, /: /)
    value = parts[2] + 0
    sum[label] += value; sum2[label] += value * value; count[label]++
    if (!(label in seen)) { seen[label] = 1; keys[++nkeys] = label }
    if ($0 ~ /^Time - Phase 0/) {
        split($0, s, /scan: /);      sv = s[2] + 0
        split($0, p, /pre-fault: /); pv = p[2] + 0
        for (si = 0; si < 2; si++) {
            sl = (si == 0) ? "  scan (ms)" : "  pre-fault (ms)"
            sv2 = (si == 0) ? sv : pv
            sum[sl] += sv2; sum2[sl] += sv2 * sv2; count[sl]++
            if (!(sl in seen)) { seen[sl] = 1; keys[++nkeys] = sl }
        }
    }
}
/^Throughput: / {
    label = "Throughput (MPix/s)"
    value = $2 + 0
    sum[label] += value; sum2[label] += value * value; count[label]++
    if (!(label in seen)) { seen[label] = 1; keys[++nkeys] = label }
}
END {
    n = count[keys[1]]
    printf "\n=== Summary (%d images) ===\n", n
    printf "%-24s  %10s  %10s\n", "Metric", "Mean", "StdDev"
    printf "%-24s  %10s  %10s\n", "------------------------", "----------", "----------"
    for (i = 1; i <= nkeys; i++) {
        k = keys[i]
        ni = count[k]
        mean = sum[k] / ni
        variance = (ni > 1) ? (sum2[k] - ni * mean * mean) / (ni - 1) : 0
        stddev = (variance > 0) ? sqrt(variance) : 0
        printf "%-24s  %10.1f  %10.1f\n", k, mean, stddev
    }
}
' "$OUTPUT" | tee -a "$OUTPUT"

echo "Results saved to: $OUTPUT"
