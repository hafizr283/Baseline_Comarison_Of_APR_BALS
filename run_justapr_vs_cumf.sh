#!/usr/bin/env bash
# run_justapr_vs_cumf.sh — justapr.cu (APR-BALS, SHIP recipe) vs cuMF-ALS
# (HPDC16), across every frozen dataset .bin in this repo. Skips the FP32
# baseline legs (main_experiment.cu / only_scalar_fp32.cu) on purpose — see
# FIXLOG SYNC-2026-07-20b/d: those are validated separately and are slow
# (K=96 FP32 alone is ~150s/dataset on Netflix); this script is for the fast
# "our shipped mixed-precision APR vs the external reference" comparison.
#
# Ship recipe (FIXLOG SYNC-2026-07-20b), applied per K:
#   K=96      -> -DCHOL_MP=2 (FP16 tiles, no refine)  — measured: solve -23%,
#                test RMSE exact vs FP32, 0 zeroed solves, weighted-lambda ONLY.
#   K<=64     -> no CHOL_MP (FP32 tiled/batched solver of record)
#   ALWAYS    -> -DWEIGHTED_LAMBDA=1 -DCUMF_INIT=1 (ALS-WR + cuMF-scale init;
#                required for CHOL_MP — plain lambda diverges, 07-20b/c)
#
# Usage:
#   ./run_justapr_vs_cumf.sh                       # all known datasets
#   ./run_justapr_vs_cumf.sh ml100k ml10m           # named subset
#   ./run_justapr_vs_cumf.sh /path/to/custom.bin    # explicit .bin path(s)
#
# Knobs (env vars): GPU, ARCH, LAMBDA, K_VALUES, CUMF_F, THETA_BATCH, MAXIT,
#                    VERBOSE — same meaning as run_rmse_comparison.sh.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

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
MAXIT="${MAXIT:-50}"
VERBOSE="${VERBOSE:-0}"

command -v nvcc >/dev/null || { echo "ERROR: nvcc not in PATH"; exit 1; }
[[ -f justapr.cu ]] || { echo "ERROR: run from comparing_baseline/"; exit 1; }

# ── known dataset registry (name -> .bin path) ─────────────────────────────
declare -A KNOWN=(
    [ml100k]="/home/pc/Desktop/2007080/ratings100.bin"
    [ml10m]="/home/pc/Desktop/2007080/ratings10.bin"
    [ml20m]="/home/pc/Desktop/2007080/ratings.bin"
    [ml32m]="/home/pc/Desktop/2007080/ratings32.bin"
    [netflix]="/home/pc/Desktop/2007080/netflix_ratings.bin"
)
DATASET_NAMES=(); DATASET_PATHS=()
if [[ $# -eq 0 ]]; then
    for n in ml100k ml10m ml20m ml32m netflix; do
        [[ -f "${KNOWN[$n]}" ]] && DATASET_NAMES+=("$n") && DATASET_PATHS+=("${KNOWN[$n]}")
    done
else
    for arg in "$@"; do
        if [[ -n "${KNOWN[$arg]:-}" ]]; then
            DATASET_NAMES+=("$arg"); DATASET_PATHS+=("${KNOWN[$arg]}")
        elif [[ -f "$arg" ]]; then
            DATASET_NAMES+=("$(basename "$arg" .bin)"); DATASET_PATHS+=("$arg")
        else
            echo "ERROR: unknown dataset name or missing file: $arg"; exit 1
        fi
    done
fi
[[ ${#DATASET_NAMES[@]} -gt 0 ]] || { echo "ERROR: no datasets found"; exit 1; }

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="results"; LOGDIR="$OUTDIR/logs_justapr_vs_cumf_${TS}"
mkdir -p "$LOGDIR"
SUMMARY="$OUTDIR/JUSTAPR_VS_CUMF_${TS}.txt"
BINDIR=$(mktemp -d /tmp/japrcmp_XXXXXX)
trap "rm -rf '$BINDIR'" EXIT

show() { echo "$@" | tee -a "$SUMMARY"; }
runlog() { local LOG="$1"; shift
    if [[ "$VERBOSE" == "1" ]]; then "$@" 2>&1 | tee "$LOG"; else "$@" > "$LOG" 2>&1; fi
}
msiter() { awk -v w="$1" -v i="$2" 'BEGIN{ if (i>0) printf "%.0f", w*1000.0/i; else printf "?" }'; }

show "════════════════════════════════════════════════════════════════════════"
show " justapr (APR-BALS, ship recipe) vs cuMF-ALS — GPU=$GPU arch=$ARCH lambda=$LAMBDA"
show " datasets: ${DATASET_NAMES[*]}"
show " K/F pairs: ${K_VALUES[*]} / ${CUMF_F[*]} | protocol: RMSE %5, tol 0.001, max $MAXIT"
show " Started: $(date)"
show "════════════════════════════════════════════════════════════════════════"

# ── build justapr once per K (dataset-independent) ──────────────────────────
declare -A JUSTAPR_OK
for K in "${K_VALUES[@]}"; do
    # Heavy-ball momentum beta=0.3 (07-21b) at EVERY K: converges in ~15 iters
    # vs 20-25, test RMSE equal-or-better, ~1.30x wall on Netflix across K16-96.
    # MBETA env overrides beta. Precision-/rank-independent (works with the FP32
    # solver at K<=64 too, no overflow risk there). Re-check fails==0 on new sets.
    FLAGS="-DK_DIM=$K -DWEIGHTED_LAMBDA=1 -DCUMF_INIT=1 -DMOMENTUM=1"
    # K=96 additionally uses FP16 solver tiles (07-20b: solve -23%).
    [[ "$K" == "96" ]] && FLAGS="$FLAGS -DCHOL_MP=2"
    echo ">> [justapr K=$K] compile ($FLAGS)..."
    if nvcc -O3 -arch="$ARCH" -std=c++14 $FLAGS justapr.cu -o "$BINDIR/apr_k$K" \
        2> "$LOGDIR/compile_apr_k$K.txt"; then
        JUSTAPR_OK[$K]=1
    else
        JUSTAPR_OK[$K]=0
        show "  K=$K justapr ✗ COMPILE FAILED — see $LOGDIR/compile_apr_k$K.txt"
    fi
done

# ── build cuMF once (dataset-independent) ───────────────────────────────────
# Don't trust a pre-existing binary just because it's executable: one carried
# over from a different machine (different libstdc++/glibc) is still -x but
# fails at runtime ("GLIBCXX_x.y.zz not found") — measured 2026-07-21, a
# stale 3060-built main silently reused on a T4 box, 25/25 cuMF legs failed.
# `ldd` catches an unresolvable shared-library dependency without having to
# actually run the binary.
NEED_CUMF_BUILD=1
if [[ -x cumf_als-master/main ]] && ldd cumf_als-master/main >/dev/null 2>&1; then
    NEED_CUMF_BUILD=0
fi
if [[ "$NEED_CUMF_BUILD" == "1" ]]; then
    echo ">> [cuMF] building (make main)..."
    ( cd cumf_als-master && make clean >/dev/null 2>&1 && make main > "../$LOGDIR/compile_cumf.txt" 2>&1 )
fi
chmod +x cumf_als-master/main 2>/dev/null
CUMF_OK=0; [[ -x cumf_als-master/main ]] && ldd cumf_als-master/main >/dev/null 2>&1 && CUMF_OK=1

for di in "${!DATASET_NAMES[@]}"; do
    DNAME="${DATASET_NAMES[$di]}"; BIN_PATH="${DATASET_PATHS[$di]}"
    show ""
    show "── dataset: $DNAME  ($BIN_PATH) ──"
    ROWS=()

    for idx in "${!K_VALUES[@]}"; do
        K="${K_VALUES[$idx]}"
        if [[ "${JUSTAPR_OK[$K]}" == "1" ]]; then
            echo ">> [$DNAME justapr K=$K] run..."
            LOG="$LOGDIR/${DNAME}_apr_k${K}.txt"
            runlog "$LOG" "$BINDIR/apr_k$K" "$BIN_PATH" "$LAMBDA"
            AW=$(grep -oP 'Training Complete in \K[\d.]+' "$LOG" | tail -1)
            AI=$(grep -oP 'Converged at iteration \K\d+' "$LOG" | tail -1)
            [[ -z "$AI" ]] && AI=$(grep -oP '=== Profiling: .*\(\K\d+(?= iters\))' "$LOG" | tail -1)
            ATR=$(grep -oP 'Train RMSE: \K(nan|inf|[\d.]+)(?= \| Test)' "$LOG" | tail -1)
            ATE=$(grep -oP 'Test RMSE: \K(nan|inf|[\d.]+)' "$LOG" | tail -1)
            FAILN=$(grep -oP 'non-finite solves zeroed = \K\d+' "$LOG" | tail -1)
            if [[ -n "$ATE" ]]; then
                FLAG=""
                [[ -n "$FAILN" && "$FAILN" != "0" ]] && FLAG=" [!] $FAILN zeroed solves"
                ROWS+=("$((idx*10))|K=$K|justapr APR-BALS (ship)|${AI:-?}|${AW:-?}|$(msiter "${AW:-0}" "${AI:-0}")|$ATR|$ATE|$FLAG")
                show "  K=$K justapr ✓  test ${ATE} (${AI:-?} it, ${AW:-?}s)${FLAG}"
            else
                show "  K=$K justapr ✗ RUN FAILED — see $LOG"
            fi
        fi
    done

    if [[ "$CUMF_OK" == "1" ]]; then
        ABS_BIN=$(readlink -f "$BIN_PATH")
        for idx in "${!CUMF_F[@]}"; do
            F="${CUMF_F[$idx]}"
            echo ">> [$DNAME cuMF F=$F] run..."
            LOG="$LOGDIR/${DNAME}_cumf_f${F}.txt"
            ( cd cumf_als-master && runlog "../$LOG" ./main "$ABS_BIN" "$LAMBDA" "$F" 1 "$THETA_BATCH" 0 "$MAXIT" 0.001 )
            rm -f cumf_als-master/cumf_XT_f*.bin cumf_als-master/cumf_thetaT_f*.bin
            CW=$(grep -oP 'wall=\K[\d.]+(?= ms)' "$LOG" | tail -1)
            CI=$(grep -oP '=== Profiling: .*\(\K\d+(?= iters\))' "$LOG" | tail -1)
            CTR=$(grep -oP 'Train RMSE: \K(nan|[\d.]+)(?= \| Test)' "$LOG" | tail -1)
            CTE=$(grep -oP 'Test RMSE: \K(nan|[\d.]+)' "$LOG" | tail -1)
            if [[ -n "$CTE" ]]; then
                CWS=$(awk -v w="$CW" 'BEGIN{printf "%.3f", w/1000.0}')
                ROWS+=("$((idx*10+5))|F=$F|cuMF-ALS (HPDC16, FP32+CG)|${CI:-?}|${CWS:-?}|$(msiter "${CWS:-0}" "${CI:-0}")|$CTR|$CTE|")
                show "  F=$F cuMF ✓  test ${CTE} (${CI:-?} it, ${CWS:-?}s)"
                [[ "$CTE" == "nan" ]] && show "    ⚠ cuMF test RMSE is nan (dataset has 0-train entities; no empty-entity guard)"
            else
                show "  F=$F cuMF ✗ RUN FAILED — see $LOG"
            fi
        done
    fi

    show ""
    show "  table ($DNAME):"
    {
        printf "  | %-5s | %-27s | %-5s | %-8s | %-8s | %-10s | %-9s |\n" \
            "rank" "code" "iters" "wall s" "ms/iter" "train RMSE" "test RMSE"
        printf "  |-------|-----------------------------|-------|----------|----------|------------|-----------|\n"
        printf '%s\n' "${ROWS[@]}" | sort -t'|' -k1,1n | while IFS='|' read -r _ RANK CODE IT WALL MSI TR TE FLAG; do
            printf "  | %-5s | %-27s | %-5s | %-8s | %-8s | %-10s | %-9s |%s\n" \
                "$RANK" "$CODE" "$IT" "$WALL" "$MSI" "$TR" "$TE" "$FLAG"
        done
    } | tee -a "$SUMMARY"
done

show ""
show "Finished : $(date)"
show "Summary  : $SUMMARY"
show "Full logs: $LOGDIR/"
