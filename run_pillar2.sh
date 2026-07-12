#!/usr/bin/env bash
# run_pillar2.sh — Generate Table 3 (APR-BALS vs PyTorch ALS baseline)
#
# Runs the PyTorch GPU ALS baseline at K=16,32,64 on the same frozen split,
# then compiles APR-BALS main_experiment for the same K and runs both.
# Output: pillar2_results.txt — ready to paste into the paper.
#
# Prerequisites: pip install torch numpy scipy
# Usage:
#   ./run_pillar2.sh /path/to/netflix_ratings.bin
#   ARCH=sm_86 ./run_pillar2.sh /path/to/netflix_ratings.bin
set -uo pipefail

BIN="${1:-/home/pc/Desktop/2007080/netflix_ratings.bin}"
ARCH="${ARCH:-sm_86}"   # sm_86=RTX 3060, sm_75=T4
# The PyTorch baseline uses PLAIN lambda*I, so main_experiment is compiled
# with -DWEIGHTED_LAMBDA=0 below to keep this comparison like-for-like.
LAMBDA=0.1
K_VALUES=(16 32 64)
OUT="pillar2_results.txt"
SRC="main_experiment.cu"

[[ -f "$BIN" ]] || { echo "ERROR: $BIN not found"; exit 1; }
[[ -f "$SRC" ]] || { echo "ERROR: $SRC not found (run from the code directory)"; exit 1; }
command -v nvcc >/dev/null || { echo "ERROR: nvcc not in PATH"; exit 1; }
python3 -c "import torch" 2>/dev/null || { echo "ERROR: pip install torch"; exit 1; }

echo "=== APR-BALS Pillar 2 Benchmark (PyTorch ALS vs APR-BALS) ===" | tee "$OUT"
echo "Dataset : $BIN" | tee -a "$OUT"
echo "Arch    : $ARCH" | tee -a "$OUT"
echo "Lambda  : $LAMBDA" | tee -a "$OUT"
echo "Started : $(date)" | tee -a "$OUT"
echo "" | tee -a "$OUT"

# ─── Part 1: PyTorch ALS baseline ───────────────────────────────────────────
echo "━━━ Part 1: PyTorch GPU ALS baseline ━━━" | tee -a "$OUT"
echo "(Standard dense BLAS ops — no custom kernel, no mixed precision)" | tee -a "$OUT"
echo "" | tee -a "$OUT"

for K in "${K_VALUES[@]}"; do
    echo "── PyTorch ALS K=$K ──" | tee -a "$OUT"
    python3 baseline_als_gpu.py "$BIN" --K "$K" --lam $LAMBDA 2>&1 | tee -a "$OUT"
    echo "" | tee -a "$OUT"
done

# ─── Part 2: APR-BALS (same data, same K, same λ) ────────────────────────────
echo "━━━ Part 2: APR-BALS (custom sparse WMMA kernel) ━━━" | tee -a "$OUT"
echo "" | tee -a "$OUT"

for K in "${K_VALUES[@]}"; do
    echo "── APR-BALS K=$K ──" | tee -a "$OUT"
    BIN_OUT="/tmp/aprbals_k${K}"
    if nvcc -O3 -arch="$ARCH" -std=c++14 \
            -DK_DIM="$K" -DDENSE_NNZ_THRESH=4 -DGIANT_NNZ_THRESH=512 \
            -DWEIGHTED_LAMBDA=0 \
            "$SRC" -o "$BIN_OUT" 2>&1 | grep -v "warning"; then
        "$BIN_OUT" "$BIN" $LAMBDA 2>&1 | tee -a "$OUT"
    else
        echo "  [COMPILE ERROR for K=$K]" | tee -a "$OUT"
    fi
    echo "" | tee -a "$OUT"
done

# ─── Summary table ────────────────────────────────────────────────────────────
echo "━━━ Summary (extract from above) ━━━" | tee -a "$OUT"
echo "" | tee -a "$OUT"
echo "PyTorch ALS  wall times:" | tee -a "$OUT"
grep -E "^Wall time" "$OUT" | tee -a /dev/null || true
grep "Wall time" "$OUT" | head -10 | tee -a "$OUT" || true
echo "" | tee -a "$OUT"
echo "APR-BALS wall times:" | tee -a "$OUT"
grep "Training Complete in" "$OUT" | tee -a "$OUT" || true
echo "" | tee -a "$OUT"
echo "Test RMSE (PyTorch ALS):" | tee -a "$OUT"
grep "^Test  RMSE" "$OUT" | tee -a "$OUT" || true
echo "" | tee -a "$OUT"
echo "Test RMSE (APR-BALS):" | tee -a "$OUT"
grep "Test  RMSE  APR-BALS" "$OUT" | tee -a "$OUT" || true
echo "" | tee -a "$OUT"
echo "Finished: $(date)" | tee -a "$OUT"
echo "Full results: $OUT"
