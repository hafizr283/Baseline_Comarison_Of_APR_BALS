#!/usr/bin/env bash
# run_final.sh — APR-BALS ONE-SHOT pipeline for ALL datasets at once.
#
# Pass any number of preprocessed .bin files; each is run through the full
# GPU pipeline (GT tuning sweep + K sweep) AND the CPU baseline sweep.
#
# Usage:
#   ./run_final.sh ratings100.bin ratings10.bin ratings.bin netflix_ratings.bin
#   ./run_final.sh *.bin
#
# Knobs (env vars):
#   ARCH=sm_86        GPU arch (sm_75=T4/Colab, sm_86=RTX 30xx/Ampere). Default sm_86.
#   LAMBDA=0.048      Regularization passed to the binary. main_experiment.cu
#                     now defaults to WEIGHTED lambda (ALS-WR, cumf_als-style),
#                     whose good range is ~0.02-0.06 — NOT the old plain-λ 0.1.
#                     For the old plain-λ behavior: WEIGHTED=0 LAMBDA=0.1.
#   WEIGHTED=1        1 = weighted λ (default), 0 = plain λ·I (pre-07-12 runs).
#   VERBOSE=1         Echo the FULL program output to the terminal too
#                     (default 0 = terminal shows only the curated summary;
#                      the full log is ALWAYS saved to the per-dataset file).
#   SKIP_GT_SWEEP=1   Skip Phase 1 and use a known-good GT per dataset (fast rerun).
#
# Output (one folder, one file per dataset + one grand summary):
#   final results/<name>_gpu.txt
#   final results/ALL_SUMMARY_<timestamp>.txt
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

[[ $# -lt 1 ]] && { echo "Usage: $0 <file1.bin> [file2.bin ...]"; exit 1; }

# ── Tunable config ──────────────────────────────────────────────────────
ARCH="${ARCH:-sm_86}"
SRC="main_experiment.cu"          # canonical experiment source (reads .bin)
DENSE_THRESH=4                    # DENSE_NNZ_THRESH (tile density cutoff for WMMA)
K_TUNE=16                         # K used for GT tuning (fastest compile+run)
GT_SWEEP=(128 256 512 1024 2048)  # GIANT_NNZ_THRESH values to sweep
K_VALUES=(16 32 48 64 96)         # final experiment K values (full supported set)
GT_MAX_DELTA_PCT=1.0              # reject GT candidates whose Test Δ% exceeds this
VERBOSE="${VERBOSE:-0}"
SKIP_GT_SWEEP="${SKIP_GT_SWEEP:-0}"
WEIGHTED="${WEIGHTED:-1}"         # 1 = weighted λ (ALS-WR), 0 = plain λ·I
LAMBDA="${LAMBDA:-$([[ "$WEIGHTED" == "1" ]] && echo 0.048 || echo 0.1)}"

OUTDIR="final results"
mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="$OUTDIR/ALL_SUMMARY_${TIMESTAMP}.txt"
BINDIR=$(mktemp -d /tmp/aprbals_XXXXXX)
trap "rm -rf '$BINDIR'" EXIT

# Known-good GT per dataset name (used only when SKIP_GT_SWEEP=1).
declare -A KNOWN_GT=( [ml100k]=128 [ml10m]=1024 [ml20m]=1024 [netflix]=1024 )

# ── Helpers ─────────────────────────────────────────────────────────────
# show() -> terminal + summary file.   flog() -> per-dataset file only.
show() { echo "$@" | tee -a "$SUMMARY"; }
flog() { echo "$@" >> "$GPU_FILE"; }
hr()   { show "════════════════════════════════════════════════════════════════"; }

# Curated lines worth seeing live; everything else goes to the file only.
KEEP='Wall-time speedup|Compute speedup|Test delta|Train delta|RMSE  (baseline|APR)|Giant split|Precision Tiers|Total FLOPs|Throughput:|Training Complete|users,.*items'

# Print full output to a dataset file; mirror only curated lines to terminal
# (or everything if VERBOSE=1).
capture() {            # capture <datafile> <command...>
    local datafile="$1"; shift
    local out
    out="$("$@" 2>&1)"; local rc=$?
    echo "$out" >> "$datafile"
    if [[ "$VERBOSE" == "1" ]]; then
        echo "$out"
    else
        echo "$out" | grep -E "$KEEP" || true
    fi
    return $rc
}

compile() {            # compile <K> <GT> <bin>
    nvcc -O3 -arch="$ARCH" -std=c++14 \
         -DK_DIM="$1" -DDENSE_NNZ_THRESH="$DENSE_THRESH" -DGIANT_NNZ_THRESH="$2" \
         -DWEIGHTED_LAMBDA="$WEIGHTED" \
         "$SRC" -o "$3"
}

extract_wall()    { grep -oP 'Wall-time speedup:\s+\K[\d.]+'             <<< "$1" | tail -1 || true; }
extract_compute() { grep -oP 'Compute speedup:\s+\K[\d.]+'              <<< "$1" | tail -1 || true; }
extract_tdabs()   { grep -oP 'Test delta=\K[\d.]+(?= \()'              <<< "$1" | tail -1 || true; }
extract_tdpct()   { grep -oP 'Test delta=[\d.]+ \(\K[\d.]+(?=%)'       <<< "$1" | tail -1 || true; }
extract_giants()  { grep -oP 'Giant split: User giants=\K[^,]+'        <<< "$1" | tail -1 || true; }

flt_gt() { echo "${1:-0} > ${2:-0}" | bc -l; }
flt_lt() { echo "${1:-0} < ${2:-0}" | bc -l; }

# Map a .bin filename to a short dataset name.
name_of() {
    case "$(basename "$1")" in
        ratings100.bin)     echo "ml100k" ;;
        ratings10.bin)      echo "ml10m"  ;;
        ratings.bin)        echo "ml20m"  ;;
        netflix_ratings.bin) echo "netflix" ;;
        *) basename "$1" .bin ;;
    esac
}

# ── Preflight ────────────────────────────────────────────────────────────
command -v nvcc >/dev/null || { echo "ERROR: nvcc not found in PATH."; exit 1; }
[[ -f "$SRC" ]] || { echo "ERROR: $SRC not found in $(pwd)."; exit 1; }

# ═════════════════════════════════════════════════════════════════════════
# MASTER LOOP — one full pipeline per .bin file given on the command line
# ═════════════════════════════════════════════════════════════════════════
declare -A SUM_WALL SUM_COMP SUM_TDPCT SUM_BESTGT
DATASETS=()

hr
show "  APR-BALS FINAL PIPELINE — $#  dataset(s)"
show "  Arch: $ARCH | DENSE_THRESH=$DENSE_THRESH | GT sweep: ${GT_SWEEP[*]} | K: ${K_VALUES[*]}"
show "  Lambda: $LAMBDA ($([[ "$WEIGHTED" == "1" ]] && echo 'WEIGHTED ALS-WR, cumf_als-style' || echo 'plain lambda*I'))"
show "  Started: $(date)"
hr; show ""

for BIN_PATH in "$@"; do
    if [[ ! -f "$BIN_PATH" ]]; then
        show "✗ SKIP — file not found: $BIN_PATH"; continue
    fi
    NAME=$(name_of "$BIN_PATH")
    DATASETS+=("$NAME")
    GPU_FILE="$OUTDIR/${NAME}_gpu.txt"
    : > "$GPU_FILE"          # truncate fresh

    hr
    show "▆▆▆  DATASET: $NAME   ($BIN_PATH)"
    hr

    # ── PHASE 1 — GIANT_NNZ_THRESH tuning (K=K_TUNE) ─────────────────────
    declare -A GT_WALL GT_COMP GT_TDPCT GT_TDABS GT_GIANTS
    GT_WALL=(); GT_COMP=(); GT_TDPCT=(); GT_TDABS=(); GT_GIANTS=()
    best_gt=""; best_wall="0"

    if [[ "$SKIP_GT_SWEEP" == "1" ]]; then
        best_gt="${KNOWN_GT[$NAME]:-1024}"
        show "  [Phase 1 skipped] Using known GT=$best_gt for $NAME"
    else
        show "  PHASE 1 ▸ GT tuning (K=$K_TUNE)  reject if Test Δ% > ${GT_MAX_DELTA_PCT}%"
        for GT in "${GT_SWEEP[@]}"; do
            BINF="$BINDIR/${NAME}_tune_gt${GT}"
            echo "── GT=$GT (K=$K_TUNE) ──" >> "$GPU_FILE"
            if ! compile "$K_TUNE" "$GT" "$BINF" >> "$GPU_FILE" 2>&1; then
                show "    GT=$GT  [COMPILE ERROR — skipped]"; continue
            fi
            OUT="$("$BINF" "$BIN_PATH" "$LAMBDA" 2>&1)"; echo "$OUT" >> "$GPU_FILE"
            GT_WALL[$GT]=$(extract_wall "$OUT");  GT_COMP[$GT]=$(extract_compute "$OUT")
            GT_TDABS[$GT]=$(extract_tdabs "$OUT"); GT_TDPCT[$GT]=$(extract_tdpct "$OUT")
            GT_GIANTS[$GT]=$(extract_giants "$OUT")
            flog "    GT=$GT  Wall ${GT_WALL[$GT]:-?}x | Compute ${GT_COMP[$GT]:-?}x | Test Δ% ${GT_TDPCT[$GT]:-?}% | giants ${GT_GIANTS[$GT]:-?}"
            echo "    GT=$GT  ✓"
        done
        for GT in "${GT_SWEEP[@]}"; do
            [[ -z "${GT_WALL[$GT]:-}" ]] && continue
            if (( $(flt_lt "${GT_TDPCT[$GT]:-9}" "$GT_MAX_DELTA_PCT") )) && \
               (( $(flt_gt "${GT_WALL[$GT]:-0}" "$best_wall") )); then
                best_wall="${GT_WALL[$GT]}"; best_gt="$GT"
            fi
        done
        [[ -z "$best_gt" ]] && { best_gt="${KNOWN_GT[$NAME]:-1024}"; show "  (no GT passed delta gate — falling back to GT=$best_gt)"; }
        show "  ★ BEST GT = $best_gt"
    fi
    SUM_BESTGT[$NAME]="$best_gt"
    show ""

    # ── PHASE 2 — full K sweep at best GT ────────────────────────────────
    show "  PHASE 2 ▸ K sweep (GT=$best_gt)"
    for K in "${K_VALUES[@]}"; do
        BINF="$BINDIR/${NAME}_k${K}"
        echo "── K=$K (GT=$best_gt) ──" >> "$GPU_FILE"
        if ! compile "$K" "$best_gt" "$BINF" >> "$GPU_FILE" 2>&1; then
            show "    K=$K  [COMPILE ERROR — skipped]"
            SUM_WALL[$NAME,$K]="ERR"; SUM_COMP[$NAME,$K]="ERR"; SUM_TDPCT[$NAME,$K]="ERR"; continue
        fi
        OUT="$(capture "$GPU_FILE" "$BINF" "$BIN_PATH" "$LAMBDA")"
        SUM_WALL[$NAME,$K]=$(extract_wall "$OUT")
        SUM_COMP[$NAME,$K]=$(extract_compute "$OUT")
        SUM_TDPCT[$NAME,$K]=$(extract_tdpct "$OUT")
        flog "    K=$K → Wall ${SUM_WALL[$NAME,$K]:-?}x | Compute ${SUM_COMP[$NAME,$K]:-?}x | Test Δ% ${SUM_TDPCT[$NAME,$K]:-?}%"
        echo "    K=$K  ✓"
    done
    show ""
    show "  saved: $GPU_FILE"
    show ""
done

# ═════════════════════════════════════════════════════════════════════════
# GRAND SUMMARY — one table across every dataset × K
# ═════════════════════════════════════════════════════════════════════════
hr
show "GRAND SUMMARY — all datasets (arch $ARCH)"
hr
printf "| %-8s | %-3s | %-4s | %-12s | %-12s | %-9s |\n" \
    "dataset" "K" "GT" "Wall x" "Compute x" "TestΔ%" | tee -a "$SUMMARY"
printf "|----------|-----|------|--------------|--------------|-----------|\n" | tee -a "$SUMMARY"
for NAME in "${DATASETS[@]}"; do
    for K in "${K_VALUES[@]}"; do
        printf "| %-8s | %-3s | %-4s | %-12s | %-12s | %-9s |\n" \
            "$NAME" "$K" "${SUM_BESTGT[$NAME]:-?}" \
            "${SUM_WALL[$NAME,$K]:-?}x" "${SUM_COMP[$NAME,$K]:-?}x" \
            "${SUM_TDPCT[$NAME,$K]:-?}%" | tee -a "$SUMMARY"
    done
done
show ""
hr
show "Finished : $(date)"
show "Summary  : $SUMMARY"
show "Per-dataset full logs in: $OUTDIR/<name>_gpu.txt"
hr
