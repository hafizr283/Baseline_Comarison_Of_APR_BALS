#!/usr/bin/env bash
# run_rmse_comparison.sh — RMSE + speed comparison on the SAME frozen split:
#
#   1. main_experiment.cu   (FP32 baseline + APR-BALS, weighted-λ ALS-WR)
#   2. only_scalar_fp32.cu  (standalone scalar FP32 baseline, weighted-λ)
#   3. cumf_als-master/     (cuMF_als HPDC16, CUDA-12 port — the external
#                            reference; it ALWAYS uses weighted-λ)
#
# All three read the identical .bin (identical train/test split), use the
# same λ, the same convergence protocol (train-RMSE every 5 iters, delta <
# 0.001, max 50), so train/test RMSE are directly comparable. cuMF only
# supports rank F as a multiple of 10, so each K is paired with the nearest
# F: 16→20, 32→30, 48→50, 64→60, 96→100 (position-wise zip of the two lists).
#
# Usage:
#   ./run_rmse_comparison.sh netflix_ratings.bin
#
# Knobs (env vars):
#   GPU=1               physical GPU index (becomes CUDA_VISIBLE_DEVICES)
#   ARCH=sm_86          default: auto-detected from nvidia-smi
#   LAMBDA=0.048        weighted-λ for ALL codes (cumf_als Netflix tune)
#   K_VALUES="16 32 48 64 96"
#   CUMF_F="20 30 50 60 100"   set CUMF_F="" to skip cuMF
#   THETA_BATCH=3       cuMF THETA_BATCH (3 = the Netflix reference config)
#   RUN_MAIN=1          run main_experiment (baseline + APR legs)
#   RUN_SCALAR=1        run the standalone scalar FP32 binary too
#   INIT=1              1 = compile our binaries with -DCUMF_INIT=1 (factors
#                       ~U(0,0.2) like cuMF; ~1/3 fewer ALS iters + slightly
#                       better test RMSE — the fair setting for THIS script).
#                       0 = legacy U(0.1,1.1) init.
#   MAXIT=50            iteration cap for our binaries
#   VERBOSE=0           1 = echo full program output
#
# Full netflix sweep is ~20-25 min (the FP32 baseline legs dominate).
# Output: results/RMSE_COMPARISON_<ts>.txt + full logs in results/logs_<ts>/
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

[[ $# -lt 1 ]] && { echo "Usage: $0 <dataset.bin>"; exit 1; }
BIN_PATH="$1"
[[ -f "$BIN_PATH" ]] || { echo "ERROR: file not found: $BIN_PATH"; exit 1; }

GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
if [[ -z "${ARCH:-}" ]]; then
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
    [[ "$CAP" =~ ^[0-9]+$ ]] && ARCH="sm_${CAP}" || ARCH="sm_86"
fi
LAMBDA="${LAMBDA:-0.048}"
K_VALUES=(${K_VALUES:-16 32 48 64 96})
CUMF_F=(${CUMF_F:-20 30 50 60 100})
THETA_BATCH="${THETA_BATCH:-3}"
RUN_MAIN="${RUN_MAIN:-1}"
RUN_SCALAR="${RUN_SCALAR:-1}"
INIT="${INIT:-1}"
MAXIT="${MAXIT:-50}"
VERBOSE="${VERBOSE:-0}"

command -v nvcc >/dev/null || { echo "ERROR: nvcc not in PATH"; exit 1; }
[[ -f main_experiment.cu && -f only_scalar_fp32.cu ]] || { echo "ERROR: run from comparing_baseline/"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="results"; LOGDIR="$OUTDIR/logs_${TS}"
mkdir -p "$LOGDIR"
SUMMARY="$OUTDIR/RMSE_COMPARISON_${TS}.txt"
BINDIR=$(mktemp -d /tmp/rmsecmp_XXXXXX)
trap "rm -rf '$BINDIR'" EXIT

show() { echo "$@" | tee -a "$SUMMARY"; }
runlog() {  # runlog <logfile> <cmd...>
    local LOG="$1"; shift
    if [[ "$VERBOSE" == "1" ]]; then "$@" 2>&1 | tee "$LOG"; else "$@" > "$LOG" 2>&1; fi
}

show "════════════════════════════════════════════════════════════════════════"
show " RMSE COMPARISON on identical frozen split: $BIN_PATH"
show " GPU=$GPU arch=$ARCH | lambda=$LAMBDA (weighted ALS-WR everywhere) | CUMF_INIT=$INIT"
show " K: ${K_VALUES[*]} | cuMF F: ${CUMF_F[*]:-<skipped>} | protocol: RMSE %5, tol 0.001, max $MAXIT"
show " Started: $(date)"
show "════════════════════════════════════════════════════════════════════════"

# row storage:  ROWS  "<order>|<rank>|<code>|<iters>|<wall>|<msiter>|<train>|<test>"
ROWS=()
msiter() { awk -v w="$1" -v i="$2" 'BEGIN{ if (i>0) printf "%.0f", w*1000.0/i; else printf "?" }'; }

# ── 1+2. our binaries per K ─────────────────────────────────────────────────
for idx in "${!K_VALUES[@]}"; do
    K="${K_VALUES[$idx]}"

    if [[ "$RUN_MAIN" == "1" ]]; then
        echo ">> [main_experiment K=$K] compile..."
        if nvcc -O3 -arch="$ARCH" -std=c++14 -DK_DIM="$K" -DCUMF_INIT="$INIT" main_experiment.cu -o "$BINDIR/exp_k$K" 2> "$LOGDIR/compile_exp_k$K.txt"; then
            echo ">> [main_experiment K=$K] run (FP32 baseline + APR)..."
            LOG="$LOGDIR/main_k${K}.txt"
            runlog "$LOG" "$BINDIR/exp_k$K" "$BIN_PATH" "$LAMBDA"
            # leg 1 = baseline, leg 2 = APR (order of run_training calls)
            WALLS=($(grep -oP 'Training Complete in \K[\d.]+' "$LOG"))
            ITERS=($(grep -oP '=== Profiling: .*\(\K\d+(?= iters\))' "$LOG"))
            BTR=$(grep -oP 'Train RMSE  baseline:\s+\K[\d.]+' "$LOG")
            BTE=$(grep -oP 'Test  RMSE  baseline:\s+\K[\d.]+' "$LOG")
            ATR=$(grep -oP 'Train RMSE  APR-BALS:\s+\K[\d.]+' "$LOG")
            ATE=$(grep -oP 'Test  RMSE  APR-BALS:\s+\K[\d.]+' "$LOG")
            if [[ -n "$BTE" && -n "$ATE" ]]; then
                ROWS+=("$((idx*10))|K=$K|APR-BALS mixed (weighted-l)|${ITERS[1]:-?}|${WALLS[1]:-?}|$(msiter "${WALLS[1]:-0}" "${ITERS[1]:-0}")|$ATR|$ATE")
                ROWS+=("$((idx*10+1))|K=$K|FP32 baseline (main_exp)|${ITERS[0]:-?}|${WALLS[0]:-?}|$(msiter "${WALLS[0]:-0}" "${ITERS[0]:-0}")|$BTR|$BTE")
                show "  K=$K main_experiment ✓  APR test ${ATE} (${ITERS[1]:-?} it, ${WALLS[1]:-?}s) | base test ${BTE} (${ITERS[0]:-?} it, ${WALLS[0]:-?}s)"
            else
                show "  K=$K main_experiment ✗ RUN FAILED — see $LOG"
            fi
        else
            show "  K=$K main_experiment ✗ COMPILE FAILED — see $LOGDIR/compile_exp_k$K.txt"
        fi
    fi

    if [[ "$RUN_SCALAR" == "1" ]]; then
        echo ">> [only_scalar_fp32 K=$K] compile..."
        if nvcc -O3 -arch="$ARCH" -std=c++14 -DK_DIM="$K" -DCUMF_INIT="$INIT" -DMAX_ITERS="$MAXIT" only_scalar_fp32.cu -o "$BINDIR/scalar_k$K" 2> "$LOGDIR/compile_scalar_k$K.txt"; then
            echo ">> [only_scalar_fp32 K=$K] run..."
            LOG="$LOGDIR/scalar_k${K}.txt"
            runlog "$LOG" "$BINDIR/scalar_k$K" "$BIN_PATH" "$LAMBDA"
            SW=$(grep -oP 'Training Complete in \K[\d.]+' "$LOG" | tail -1)
            SI=$(grep -oP '=== Profiling: .*\(\K\d+(?= iters\))' "$LOG" | tail -1)
            STR=$(grep -oP 'Train RMSE:\s+\K[\d.]+' "$LOG" | tail -1)
            STE=$(grep -oP 'Test  RMSE:\s+\K[\d.]+' "$LOG" | tail -1)
            if [[ -n "$STE" ]]; then
                ROWS+=("$((idx*10+2))|K=$K|FP32 scalar (standalone)|${SI:-?}|${SW:-?}|$(msiter "${SW:-0}" "${SI:-0}")|$STR|$STE")
                show "  K=$K only_scalar_fp32 ✓  test ${STE} (${SI:-?} it, ${SW:-?}s)"
            else
                show "  K=$K only_scalar_fp32 ✗ RUN FAILED — see $LOG"
            fi
        else
            show "  K=$K only_scalar_fp32 ✗ COMPILE FAILED — see $LOGDIR/compile_scalar_k$K.txt"
        fi
    fi
done

# ── 3. cuMF per F ───────────────────────────────────────────────────────────
if [[ ${#CUMF_F[@]} -gt 0 ]]; then
    if [[ ! -f cumf_als-master/main ]]; then
        echo ">> [cuMF] building (make main)..."
        ( cd cumf_als-master && make main > "../$LOGDIR/compile_cumf.txt" 2>&1 ) \
            || show "  cuMF ✗ BUILD FAILED — see $LOGDIR/compile_cumf.txt"
    fi
    # a Windows->Linux sync can drop the exec bit on a prebuilt binary
    [[ -f cumf_als-master/main ]] && chmod +x cumf_als-master/main 2>/dev/null
    if [[ ! -x cumf_als-master/main ]]; then
        show "  cuMF ✗ SKIPPED — cumf_als-master/main not built/executable"
    else
        ABS_BIN=$(readlink -f "$BIN_PATH")
        for idx in "${!CUMF_F[@]}"; do
            F="${CUMF_F[$idx]}"
            echo ">> [cuMF F=$F] run..."
            LOG="$LOGDIR/cumf_f${F}.txt"
            # device 0 = the one GPU visible through CUDA_VISIBLE_DEVICES
            ( cd cumf_als-master && runlog "../$LOG" ./main "$ABS_BIN" "$LAMBDA" "$F" 1 "$THETA_BATCH" 0 "$MAXIT" 0.001 )
            rm -f cumf_als-master/cumf_XT_f*.bin cumf_als-master/cumf_thetaT_f*.bin
            CW=$(grep -oP 'wall=\K[\d.]+(?= ms)' "$LOG" | tail -1)
            CI=$(grep -oP '=== Profiling: .*\(\K\d+(?= iters\))' "$LOG" | tail -1)
            CTR=$(grep -oP 'Train RMSE: \K(nan|[\d.]+)(?= \| Test)' "$LOG" | tail -1)
            CTE=$(grep -oP 'Test RMSE: \K(nan|[\d.]+)' "$LOG" | tail -1)
            if [[ -n "$CTE" ]]; then
                CWS=$(awk -v w="$CW" 'BEGIN{printf "%.3f", w/1000.0}')
                ROWS+=("$((idx*10+3))|F=$F|cuMF-ALS (HPDC16, FP32+CG)|${CI:-?}|${CWS:-?}|$(msiter "${CWS:-0}" "${CI:-0}")|$CTR|$CTE")
                show "  F=$F cuMF ✓  test ${CTE} (${CI:-?} it, ${CWS:-?}s)"
                [[ "$CTE" == "nan" ]] && show "    ⚠ cuMF test RMSE is nan: this dataset has entities with 0 train" \
                                      && show "      ratings and cuMF has no empty-entity guard (netflix is safe)."
            else
                show "  F=$F cuMF ✗ RUN FAILED — see $LOG"
            fi
        done
    fi
fi

# ── grand table (grouped by rank: K and its nearest F together) ─────────────
show ""
show "════════════════════════════════════════════════════════════════════════"
show " GRAND TABLE — identical split, identical protocol, lambda=$LAMBDA weighted"
show "════════════════════════════════════════════════════════════════════════"
{
    printf "| %-5s | %-27s | %-5s | %-8s | %-8s | %-10s | %-9s |\n" \
        "rank" "code" "iters" "wall s" "ms/iter" "train RMSE" "test RMSE"
    printf "|-------|-----------------------------|-------|----------|----------|------------|-----------|\n"
    printf '%s\n' "${ROWS[@]}" | sort -t'|' -k1,1n | while IFS='|' read -r _ RANK CODE IT WALL MSI TR TE; do
        printf "| %-5s | %-27s | %-5s | %-8s | %-8s | %-10s | %-9s |\n" \
            "$RANK" "$CODE" "$IT" "$WALL" "$MSI" "$TR" "$TE"
    done
} | tee -a "$SUMMARY"
show ""
show "Reading guide:"
show " * test-RMSE is now apples-to-apples: ALL rows use weighted-lambda ALS-WR."
show " * within a rank group: APR-BALS vs FP32 rows = the precision claim;"
show "   APR-BALS vs cuMF ms/iter = the external-baseline speed claim."
show " * cuMF rank F is the nearest multiple of 10 to K, so tiny test-RMSE"
show "   differences across the K/F pair are partly the rank difference."
show ""
show "Finished : $(date)"
show "Summary  : $SUMMARY"
show "Full logs: $LOGDIR/"
