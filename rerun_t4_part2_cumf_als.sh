#!/usr/bin/env bash
# rerun_t4_part2_cumf_als.sh — re-run ONLY the justapr-vs-cuMF_ALS comparison
# (run_t4_everything.sh PART 2) on this machine.
#
# Why this exists: the 2026-07-21 T4/Colab run's cumf_als-master/main was a
# binary carried over from a different dev machine (built against a newer
# libstdc++). It was still `-x` (executable bit set) so both
# run_t4_everything.sh and run_justapr_vs_cumf.sh skipped rebuilding it and
# reused it as-is -> every one of the 25 cuMF legs failed at runtime with
# "GLIBCXX_x.y.zz not found", while justapr's own legs (which don't touch
# that binary) were fine. Both scripts now rebuild unconditionally / verify
# with `ldd` before trusting an existing binary, so this specific failure
# mode shouldn't recur — this script is the fast, PART-2-only path to get
# fresh results without re-running PARTs 1/3/4 (which already succeeded).
#
# Usage (from inside comparing_baseline/, e.g. on Colab after `cd` there):
#   DATA_DIR=/content bash rerun_t4_part2_cumf_als.sh
set -uo pipefail

DATA_DIR="${DATA_DIR:-/content}"
GPU="${GPU:-0}"                # Colab: single GPU, index 0
export CUDA_VISIBLE_DEVICES="$GPU"
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
ARCH="sm_${CAP:-75}"
SMS="${CAP:-75}"
echo "GPU=$GPU  ARCH=$ARCH  DATA_DIR=$DATA_DIR"

[[ -f justapr.cu && -f run_justapr_vs_cumf.sh ]] || {
    echo "ERROR: run this from inside comparing_baseline/ (justapr.cu / run_justapr_vs_cumf.sh not found here)"
    exit 1
}

NETFLIX="$DATA_DIR/netflix_ratings.bin"
ML20M="$DATA_DIR/ratings.bin"
ML10M="$DATA_DIR/ratings10.bin"
ML32M="$DATA_DIR/ratings32.bin"
ML100K="$DATA_DIR/ratings100.bin"
for f in "$NETFLIX" "$ML20M" "$ML10M" "$ML32M" "$ML100K"; do
    [[ -f "$f" ]] || echo "WARN: missing $f (that dataset's cuMF/justapr runs will be skipped)"
done

echo "=== Rebuilding cuMF_ALS for THIS machine (was the cause of the previous failure) ==="
make -C cumf_als-master clean
make -C cumf_als-master SMS="$SMS" main

echo "=== Sanity check: does the fresh binary actually run here? ==="
if ! ldd cumf_als-master/main >/dev/null 2>&1; then
    echo "ERROR: cumf_als-master/main still has unresolved shared-library deps after a clean rebuild."
    ldd cumf_als-master/main
    exit 1
fi
echo "OK — cuMF_ALS binary resolves cleanly on this machine."

echo "=== Running justapr (ship recipe) vs cuMF_ALS, all 5 datasets ==="
GPU="$GPU" ARCH="$ARCH" ./run_justapr_vs_cumf.sh "$ML10M" "$ML20M" "$ML32M" "$NETFLIX" "$ML100K"

echo "=== Done — see results/JUSTAPR_VS_CUMF_*.txt (newest timestamp) ==="
