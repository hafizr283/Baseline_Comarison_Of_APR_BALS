#!/usr/bin/env bash
# run_scalar_sweep.sh — debugging K sweep for the scalar-FP32-only pipeline
# (only_scalar_fp32.cu). One compile + one run per K, with the ptxas -v
# register/spill report saved per K (the classic scalar-baseline failure mode
# is register spill at K>=48 — see FIXLOG 07-04).
#
# Usage:
#   ./run_scalar_sweep.sh netflix_ratings.bin [ratings.bin ...]
#
# Knobs (env vars):
#   ARCH=sm_75        Default: auto-detected from nvidia-smi compute_cap
#                     (T4 -> sm_75, RTX 30xx -> sm_86). Override only if
#                     nvidia-smi is unavailable or you cross-compile.
#   K_VALUES="16 32 48 64 96"
#   LAMBDA=0.05       argv[2] of the binary. Weighted-λ (default build) wants
#                     ~0.02–0.06; plain-λ (WEIGHTED=0) was tuned at 0.1.
#   WEIGHTED=1        1 = cumf_als-style weighted λ (ALS-WR, comparable to
#                     cumf_als RMSE); 0 = plain λ (main_experiment.cu parity).
#   MAXIT=50          -DMAX_ITERS cap (use 10 for quick sanity sweeps).
#   EXTRA_FLAGS=""    extra nvcc flags, e.g. "-DITER_TIMING" or "-DDEBUG_SYNC"
#   VERBOSE=0         1 = echo full program output to terminal too.
#
# Output:
#   scalar_results/<name>_k<K>.txt          full log (compile + ptxas + run)
#   scalar_results/SCALAR_SUMMARY_<ts>.txt  grand table
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

[[ $# -lt 1 ]] && { echo "Usage: $0 <file1.bin> [file2.bin ...]"; exit 1; }

# Auto-detect GPU arch (compute_cap "7.5" -> sm_75). An arch mismatch does
# not fail at compile time — it fails at the first launch with "no kernel
# image is available for execution on the device".
if [[ -z "${ARCH:-}" ]]; then
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
    if [[ "$CAP" =~ ^[0-9]+$ ]]; then
        ARCH="sm_${CAP}"
        echo "Auto-detected GPU arch: $ARCH"
    else
        ARCH="sm_86"
        echo "WARN: could not detect compute_cap via nvidia-smi — defaulting to $ARCH (override with ARCH=sm_75 etc.)"
    fi
fi
SRC="only_scalar_fp32.cu"
K_VALUES=(${K_VALUES:-16 32 48 64 96})
LAMBDA="${LAMBDA:-0.05}"
WEIGHTED="${WEIGHTED:-1}"
MAXIT="${MAXIT:-50}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"
VERBOSE="${VERBOSE:-0}"

OUTDIR="scalar_results"
mkdir -p "$OUTDIR"
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="$OUTDIR/SCALAR_SUMMARY_${TS}.txt"
BINDIR=$(mktemp -d /tmp/scalarfp32_XXXXXX)
trap "rm -rf '$BINDIR'" EXIT

show() { echo "$@" | tee -a "$SUMMARY"; }
hr()   { show "════════════════════════════════════════════════════════════════"; }

# Lines worth seeing live even with VERBOSE=0
KEEP='Regularization|Weighted-lambda|users,.*items|GPU memory after setup|WARN|FATAL|error|Error|Iter |Converged|LHS\+RHS|Cholesky:|Total FLOPs|Training Complete|Train RMSE|Test  RMSE|Params:'

command -v nvcc >/dev/null || { echo "ERROR: nvcc not in PATH"; exit 1; }
[[ -f "$SRC" ]] || { echo "ERROR: $SRC not found in $(pwd)"; exit 1; }

name_of() {
    case "$(basename "$1")" in
        ratings100.bin)      echo "ml100k"  ;;
        ratings10.bin)       echo "ml10m"   ;;
        ratings.bin)         echo "ml20m"   ;;
        netflix_ratings.bin) echo "netflix" ;;
        *) basename "$1" .bin ;;
    esac
}

declare -A R_WALL R_CMP R_UGF R_IGF R_TR R_TE R_IT R_SPILL
DATASETS=()

hr
show "  SCALAR-FP32-ONLY K SWEEP  —  $#  dataset(s)"
show "  Arch: $ARCH | K: ${K_VALUES[*]} | lambda=$LAMBDA | WEIGHTED_LAMBDA=$WEIGHTED | MAX_ITERS=$MAXIT"
show "  Extra flags: ${EXTRA_FLAGS:-<none>}"
show "  Started: $(date)"
hr; show ""

for BIN_PATH in "$@"; do
    [[ -f "$BIN_PATH" ]] || { show "✗ SKIP — file not found: $BIN_PATH"; continue; }
    NAME=$(name_of "$BIN_PATH")
    DATASETS+=("$NAME")
    show "▆▆▆  DATASET: $NAME   ($BIN_PATH)"

    for K in "${K_VALUES[@]}"; do
        LOG="$OUTDIR/${NAME}_k${K}.txt"
        : > "$LOG"
        BINF="$BINDIR/scalar_${NAME}_k${K}"

        echo "── compile: K=$K WEIGHTED_LAMBDA=$WEIGHTED MAX_ITERS=$MAXIT $EXTRA_FLAGS ──" >> "$LOG"
        if ! nvcc -O3 -arch="$ARCH" -std=c++14 -Xptxas -v \
                 -DK_DIM="$K" -DWEIGHTED_LAMBDA="$WEIGHTED" -DMAX_ITERS="$MAXIT" \
                 $EXTRA_FLAGS "$SRC" -o "$BINF" >> "$LOG" 2>&1; then
            show "    K=$K  [COMPILE ERROR — see $LOG]"
            R_WALL[$NAME,$K]="ERR"; continue
        fi
        # Register-spill sanity for the template that actually RUNS at this K.
        # All RPT variants {1,2,4,8,16,32} are instantiated and the unused big
        # ones spill heavily — that is expected and harmless. The running one
        # is RPT = XB/(DZ*ROW_SPLIT): K16->2, K32->4, K48->2, K64->4, K96->4.
        # Large spill ON THIS ONE = the FIXLOG 07-04 collapse pattern.
        case "$K" in
            16|48) RPT=2 ;;
            *)     RPT=4 ;;
        esac
        SPILL=$(awk -v pat="BALS_blockILi${RPT}ELi" \
                    'index($0, pat){f=1} f && /spill stores/{print; f=0}' "$LOG" \
                | grep -oP '\d+(?= bytes spill stores)' | head -1)
        R_SPILL[$NAME,$K]="${SPILL:-0}"

        echo "── run: $BIN_PATH lambda=$LAMBDA ──" >> "$LOG"
        OUT="$("$BINF" "$BIN_PATH" "$LAMBDA" 2>&1)"; RC=$?
        echo "$OUT" >> "$LOG"
        if [[ "$VERBOSE" == "1" ]]; then echo "$OUT"; else echo "$OUT" | grep -E "$KEEP" || true; fi
        if [[ $RC -ne 0 ]]; then
            show "    K=$K  [RUNTIME ERROR rc=$RC — see $LOG]"
            R_WALL[$NAME,$K]="ERR"; continue
        fi

        R_WALL[$NAME,$K]=$(grep -oP 'Training Complete in \K[\d.]+' <<< "$OUT" | tail -1)
        R_CMP[$NAME,$K]=$(grep  -oP 'Compute time:\s+\K[\d.]+'      <<< "$OUT" | tail -1)
        R_UGF[$NAME,$K]=$(grep  -oP 'User LHS\+RHS:.*\|\s+\K[\d.]+(?= GFlops)' <<< "$OUT" | tail -1)
        R_IGF[$NAME,$K]=$(grep  -oP 'Item LHS\+RHS:.*\|\s+\K[\d.]+(?= GFlops)' <<< "$OUT" | tail -1)
        R_TR[$NAME,$K]=$(grep   -oP 'Train RMSE:\s+\K[\d.]+'        <<< "$OUT" | tail -1)
        R_TE[$NAME,$K]=$(grep   -oP 'Test  RMSE:\s+\K[\d.]+'        <<< "$OUT" | tail -1)
        R_IT[$NAME,$K]=$(grep   -oP '\(\K\d+(?= iters\))'           <<< "$OUT" | tail -1)
        show "    K=$K ✓  wall ${R_WALL[$NAME,$K]:-?}s | iters ${R_IT[$NAME,$K]:-?} | test RMSE ${R_TE[$NAME,$K]:-?} | spill ${R_SPILL[$NAME,$K]:-?}B"
    done
    show ""
done

hr
show "GRAND SUMMARY — scalar FP32 only (arch $ARCH, lambda=$LAMBDA, weighted=$WEIGHTED)"
hr
printf "| %-8s | %-3s | %-5s | %-8s | %-11s | %-7s | %-7s | %-9s | %-9s | %-7s |\n" \
    "dataset" "K" "iters" "wall s" "compute ms" "u GF/s" "i GF/s" "trainRMSE" "testRMSE" "spill B" | tee -a "$SUMMARY"
printf "|----------|-----|-------|----------|-------------|---------|---------|-----------|-----------|---------|\n" | tee -a "$SUMMARY"
for NAME in "${DATASETS[@]}"; do
    for K in "${K_VALUES[@]}"; do
        printf "| %-8s | %-3s | %-5s | %-8s | %-11s | %-7s | %-7s | %-9s | %-9s | %-7s |\n" \
            "$NAME" "$K" "${R_IT[$NAME,$K]:-?}" "${R_WALL[$NAME,$K]:-?}" \
            "${R_CMP[$NAME,$K]:-?}" "${R_UGF[$NAME,$K]:-?}" "${R_IGF[$NAME,$K]:-?}" \
            "${R_TR[$NAME,$K]:-?}" "${R_TE[$NAME,$K]:-?}" "${R_SPILL[$NAME,$K]:-?}" | tee -a "$SUMMARY"
    done
done
show ""
show "Finished : $(date)"
show "Summary  : $SUMMARY"
show "Full logs: $OUTDIR/<name>_k<K>.txt (compile + ptxas -v + run output)"
hr
