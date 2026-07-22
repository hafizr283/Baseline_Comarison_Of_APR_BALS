#!/usr/bin/env bash
# run_t4_everything.sh — reproduce every result table/figure in the TPDS paper
# on a Tesla T4 (Colab), mirroring the exact protocols already validated on
# the RTX 3060 this session. Run from inside the uploaded `comparing_baseline/`
# directory. Each PART is independently toggleable and writes its own
# timestamped folder under results/, matching the project's existing
# convention (FIXLOG.md / RUNNING_SUMMARY.md), so nothing overwrites prior runs.
#
#   PART 1 -> Table VI + VII, Fig 41/42/43/44/45  (main_experiment.cu, own FP32 baseline)
#   PART 2 -> Table (cuMF_ALS per-K/F)             (justapr ship recipe vs cuMF_als)
#   PART 3 -> Tables (cuMF_CCD, cuMF_SGD) + full momentum grid
#   PART 4 -> Table (BALS-equivalent ladder)        (only_scalar_fp32.cu)
#
# Usage:
#   DATA_DIR=/content bash run_t4_everything.sh
set -uo pipefail

# ── Config — edit these ──────────────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-/content}"                # must contain the 5 .bin files below
GPU="${GPU:-0}"                                 # Colab: single GPU, index 0
RUN_PART1=${RUN_PART1:-1}
RUN_PART2=${RUN_PART2:-1}
RUN_PART3=${RUN_PART3:-1}
RUN_PART4=${RUN_PART4:-1}

NETFLIX="$DATA_DIR/netflix_ratings.bin"
ML20M="$DATA_DIR/ratings.bin"
ML10M="$DATA_DIR/ratings10.bin"
ML32M="$DATA_DIR/ratings32.bin"
ML100K="$DATA_DIR/ratings100.bin"

export CUDA_VISIBLE_DEVICES="$GPU"
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
ARCH="sm_${CAP:-75}"
echo "GPU=$GPU  ARCH=$ARCH  DATA_DIR=$DATA_DIR"
for f in "$NETFLIX" "$ML20M" "$ML10M" "$ML32M" "$ML100K"; do
    [[ -f "$f" ]] || echo "WARN: missing $f (that dataset's runs will be skipped)"
done

TS=$(date +%Y%m%d_%H%M%S)
BINDIR=$(mktemp -d /tmp/t4_bins_XXXXXX)
trap 'rm -rf "$BINDIR"' EXIT

# ═════════════════════════════════════════════════════════════════════════
# PART 1 — main_experiment.cu: own FP32 baseline vs APR-BALS, weighted-λ
#          (mirrors results/MAIN3060_WL_20260721_225757/ exactly)
#          -> Table VI, Table VII, Fig 41/42/43/44/45
# ═════════════════════════════════════════════════════════════════════════
if [[ "$RUN_PART1" == "1" ]]; then
  OUT1="results/T4_MAIN_WL_${TS}"
  mkdir -p "$OUT1"
  echo "=== PART 1: main_experiment.cu sweep -> $OUT1 ==="
  declare -A P1_DS=( [netflix]="$NETFLIX 1024" [ml20m]="$ML20M 512" [ml10m]="$ML10M 512" [ml100k]="$ML100K 512" )
  for name in netflix ml20m ml10m ml100k; do
      read -r BINFILE GT <<< "${P1_DS[$name]}"
      [[ -f "$BINFILE" ]] || continue
      for K in 16 32 48 64 96; do
          BIN="$BINDIR/me_${name}_k${K}"
          nvcc -O3 -arch="$ARCH" -std=c++14 -DK_DIM=$K -DCUMF_INIT=1 -DGIANT_NNZ_THRESH=$GT \
              main_experiment.cu -o "$BIN" 2> "$OUT1/${name}_k${K}.build.log"
          if [[ ! -x "$BIN" ]]; then echo "  $name K=$K BUILD FAILED (see log)"; continue; fi
          "$BIN" "$BINFILE" 0.048 > "$OUT1/${name}_k${K}.txt" 2>&1
          W=$(grep -oP 'Wall-time speedup:\s+\K[\d.]+' "$OUT1/${name}_k${K}.txt")
          echo "  $name K=$K  wall=${W:-?}x"
      done
  done
  echo "PART 1 done: $OUT1"
fi

# ═════════════════════════════════════════════════════════════════════════
# PART 2 — justapr (ship recipe) vs cuMF_als, per-K/F
#          -> Table (cuMF_ALS speed/accuracy, full grid)
# ═════════════════════════════════════════════════════════════════════════
if [[ "$RUN_PART2" == "1" ]]; then
  echo "=== PART 2: justapr vs cuMF_als ==="
  # Always rebuild: a cumf_als-master/main carried over from a different
  # machine (e.g. checked in from the dev box) is "executable" but can be
  # linked against a libstdc++ newer than this machine's -> fails at
  # runtime with "GLIBCXX_x.y.zz not found", not at the [[ -x ]] check
  # (measured 2026-07-21 on a T4/Colab run: 25/25 cuMF legs failed this way
  # while justapr's own legs were fine). Matches PART 3's unconditional
  # rebuild of cuMF_CCD/cuMF_SGD below.
  make -C cumf_als-master clean >/dev/null 2>&1
  make -C cumf_als-master SMS=75 main
  GPU="$GPU" ARCH="$ARCH" ./run_justapr_vs_cumf.sh "$ML10M" "$ML20M" "$ML32M" "$NETFLIX" "$ML100K"
  echo "PART 2 done: see results/JUSTAPR_VS_CUMF_*.txt (newest)"
fi

# ═════════════════════════════════════════════════════════════════════════
# PART 3 — justapr vs cuMF_CCD + cuMF_SGD (also exercises momentum at every
#          K/dataset as a byproduct of the ship-recipe build)
#          -> Tables (cuMF_CCD, cuMF_SGD), full momentum grid
# ═════════════════════════════════════════════════════════════════════════
if [[ "$RUN_PART3" == "1" ]]; then
  echo "=== PART 3: justapr vs cuMF_CCD/cuMF_SGD ==="
  sed -i 's/sm_86/sm_75/g; s/compute_86/compute_75/g' cumf_ccd-master/Makefile
  sed -i 's/sm_86/sm_75/g; s/compute_86/compute_75/g' cumf_sgd-master/singleGPU/Makefile
  make -C cumf_ccd-master clean >/dev/null 2>&1; make -C cumf_ccd-master
  make -C cumf_sgd-master/singleGPU clean >/dev/null 2>&1; make -C cumf_sgd-master/singleGPU
  make -C cumf_sgd-master/test clean >/dev/null 2>&1; make -C cumf_sgd-master/test

  # run_ccd_sgd_comparison.py resolves justapr.cu, the CCD/SGD binaries, AND its
  # own results/ output dir relative to ITS OWN file location (HERE = dirname(__file__)),
  # so it must be edited and run in place — copying it elsewhere (e.g. /tmp) breaks
  # every one of those paths silently (this is what happened last run: it looked
  # for /tmp/justapr.cu and wrote output to /tmp/results/).
  cp run_ccd_sgd_comparison.py run_ccd_sgd_comparison.py.orig
  sed -i "s#/home/pc/Desktop/2007080#$DATA_DIR#g" run_ccd_sgd_comparison.py

  export WORKDIR=/content/scratch/conv
  mkdir -p "$WORKDIR"
  GPU="$GPU" ARCH="$ARCH" python3 run_ccd_sgd_comparison.py
  mv run_ccd_sgd_comparison.py.orig run_ccd_sgd_comparison.py   # restore original paths
  echo "PART 3 done: see results/JUSTAPR_VS_CCD_SGD_*.txt (newest) + results/logs_ccd_sgd_*/"
fi

# ═════════════════════════════════════════════════════════════════════════
# PART 4 — BALS-equivalent ladder (only_scalar_fp32.cu, K=32, λ=0.048)
#          (mirrors results/BALS_EQUIV_VALIDATION_20260721_205101/)
#          -> Table (BALS-equivalent ladder), Fig 47
# ═════════════════════════════════════════════════════════════════════════
if [[ "$RUN_PART4" == "1" ]]; then
  OUT4="results/T4_BALS_EQUIV_${TS}"
  mkdir -p "$OUT4"
  echo "=== PART 4: BALS-equivalent ladder -> $OUT4 ==="
  nvcc -O3 -arch="$ARCH" -std=c++14 -DK_DIM=32 -DBALS_REORDER=1 \
      only_scalar_fp32.cu -o "$BINDIR/bals_base_k32"
  nvcc -O3 -arch="$ARCH" -std=c++14 -DK_DIM=32 -DBALS_REORDER=1 -DBALS_SYMTILE=1 \
      only_scalar_fp32.cu -o "$BINDIR/bals_symtile_k32"
  declare -A P4_DS=( [ml10m]="$ML10M" [ml20m]="$ML20M" [ml32m]="$ML32M" [netflix]="$NETFLIX" )
  for name in ml10m ml20m ml32m netflix; do
      BINFILE="${P4_DS[$name]}"
      [[ -f "$BINFILE" ]] || continue
      "$BINDIR/bals_base_k32"    "$BINFILE" 0.048 > "$OUT4/${name}_bals_base_k32.txt"    2>&1
      "$BINDIR/bals_symtile_k32" "$BINFILE" 0.048 > "$OUT4/${name}_bals_symtile_k32.txt" 2>&1
      B=$(grep -oP 'Wall time:\s+\K[\d.]+' "$OUT4/${name}_bals_base_k32.txt" | tail -1)
      S=$(grep -oP 'Wall time:\s+\K[\d.]+' "$OUT4/${name}_bals_symtile_k32.txt" | tail -1)
      echo "  $name  base=${B:-?}ms  symtile=${S:-?}ms"
  done
  echo "PART 4 done: $OUT4"
fi

echo "=== ALL REQUESTED PARTS DONE ==="
echo "Zipping results/ for download..."
zip -rq "results_t4_${TS}.zip" results/
echo "-> results_t4_${TS}.zip  (download this and hand it back)"
