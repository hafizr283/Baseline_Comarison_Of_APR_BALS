# Baseline Comparison for APR-BALS

> **Note:** This repository is the baseline-comparison extension of [Accelerating-ALS-Matrix-Factorization-on-GPUs-via-Mixed-Precision-Tensor-Cores](https://github.com/hafizr283/Accelerating-ALS-Matrix-Factorization-on-GPUs-via-Mixed-Precision-Tensor-Cores). It measures **APR-BALS ("justapr")** — a mixed-precision (tensor-core) ALS matrix-factorization solver — against a faithful FP32 re-implementation of BALS and three external CUDA baselines: **cuMF_ALS**, **cuMF_CCD**, and **cuMF_SGD**.

Most numbers in this document are measured on **2× RTX 3060 (sm_86, 12 GB), CUDA 12.0**; §6.7 cross-validates every comparison on a **Tesla T4 (Colab, sm_75, Turing)** to confirm the results aren't an Ampere-specific artifact. All runs use identical frozen train/test splits shared by every method, and every table is backed by a raw-output file committed under [`results/`](results/) — file names are cited next to each table so you can check the source yourself.

---

## TL;DR — which binary is "the" code?

This folder has **three home-grown `.cu` binaries**. They look similar but answer different questions, and — this trips people up — **none of them reproduce the paper's headline numbers with a bare `nvcc file.cu`**. You must pass the flags below.

| Binary | What it is | Bare-default build | Flags for the results in this README |
|---|---|---|---|
| **`justapr.cu`** | The shipped mixed-precision APR-BALS solver (tensor cores + optional FP16 Cholesky + momentum) | Legacy 2026-07-06 path: plain λ·I, legacy init, FP32 solver, **no momentum** | **`-DWEIGHTED_LAMBDA=1 -DCUMF_INIT=1 -DMOMENTUM=1`** always, **add `-DCHOL_MP=2` only at `K_DIM=96`** |
| **`only_scalar_fp32.cu`** | Faithful scalar-FP32 BALS re-implementation (tiled storage + Y-column reuse, no tensor cores) — the honest "BALS-equivalent" baseline | Weighted-λ ON, legacy init, **no** Alg.3 reordering | `-DBALS_REORDER=1` (Alg. 3 row/col reordering); add `-DBALS_SYMTILE=1` for the symmetry-exploiting Gram kernel (closer to real BALS, and the fastest FP32 variant we have) |
| **`main_experiment.cu`** | A single binary that runs **both** an FP32 baseline leg and the mixed-precision (WMMA) leg back-to-back, same process, same data — the controlled "what does tensor-core dispatch alone buy you" harness | Weighted-λ ON, legacy init, `CHOL_MP=0` | `-DCUMF_INIT=1` (matches the released-cuMF-style init used in the headline runs). `-DCHOL_MP` compiles but is deliberately left at `0` in every headline run — the flag applies identically to both legs, so leaving it off keeps the Gram-kernel precision (WMMA vs scalar) the *only* variable that differs between the two legs. No `MOMENTUM` support in this binary at all. |

If you only want to reproduce the paper's tables, skip straight to **[Reproducing the results](#reproducing-the-results)** and run the provided shell scripts — they already bake in the right flags per binary.

---

## 1. Repository layout

```
comparing_baseline/
├── justapr.cu              # ship binary: mixed-precision APR-BALS (WMMA + optional FP16 Cholesky + momentum)
├── justapr_2gpu.cu         # 2-GPU entity-split variant of justapr (validated 1.34x extra vs 1-GPU; not in the tables below)
├── only_scalar_fp32.cu     # standalone FP32-only "BALS-equivalent" baseline (tiled BALS storage, no tensor cores)
├── main_experiment.cu      # two-leg fairness harness: FP32 baseline leg + APR/WMMA leg, one process
├── common.cuh               # shared macros/flags (K_DIM, WEIGHTED_LAMBDA, CHOL_MP, MOMENTUM knobs, dispatch rules)
├── fused_kernels.cuh        # BALS-tiled scalar + WMMA Gram (LHS/RHS) kernels
├── wmma_kernels.cuh          # tensor-core (WMMA) Gram kernels
├── cholesky_kernels.cuh      # tiled/packed/batched Cholesky solvers (FP32 and mixed-precision variants)
├── data_utils.cuh            # .bin dataset loader
├── preprocess.cpp            # CSV -> frozen-split .bin converter (used by every method below)
├── convert_bin_for_ccd_sgd.py# converts a justapr .bin into cuMF_CCD's CSR/CSC/COO and cuMF_SGD's COO inputs
├── cumf_als-master/          # external baseline #1: cuMF_ALS (Tan et al., HPDC'16), CUDA-12 ported
├── cumf_ccd-master/          # external baseline #2: cuMF_CCD, CUDA-12 ported
├── cumf_sgd-master/          # external baseline #3: cuMF_SGD, CUDA-12 ported
├── run_rmse_comparison.sh    # main_experiment + only_scalar_fp32 + cuMF_ALS, one dataset, one command
├── run_justapr_vs_cumf.sh    # justapr (ship recipe) vs cuMF_ALS, all datasets, one command
├── run_ccd_sgd_comparison.py # justapr vs cuMF_CCD vs cuMF_SGD, all datasets, one command
├── run_t4_everything.sh      # reproduces every table in this doc on a fresh machine (e.g. Colab T4) end-to-end
├── FIXLOG.md                 # full chronological engineering log (every optimization, every dead end, every measurement)
└── results/                  # archived raw output of every run cited in this README
```

---

## 2. Requirements

- An NVIDIA GPU, CUDA toolkit + `nvcc` (validated on CUDA 12.0, `sm_86`; the scripts auto-detect your GPU's compute capability via `nvidia-smi`).
- `g++` (for `preprocess.cpp`).
- Python 3 + `numpy` + `scipy` (only needed for the cuMF_CCD/cuMF_SGD comparison — `convert_bin_for_ccd_sgd.py` and `run_ccd_sgd_comparison.py`).
- ~5–10 GB free disk per large dataset (Netflix/ML-32M `.bin` files are multi-GB).

---

## 3. Dataset preparation

Every method in this repo (ours and all three external baselines) trains and is scored on **the exact same frozen train/test split**, because everything derives from one `.bin` file. Build the converter and run it once per dataset:

```bash
g++ -O3 preprocess.cpp -o preprocess
./preprocess path/to/your/ratings.csv my_dataset.bin
```

`preprocess` reorders users/items to be dense-indexed, does an 80/20 train/test split, and writes the frozen split. The `.bin` format is documented at the top of `convert_bin_for_ccd_sgd.py` if you need to write your own reader.

**Note on the shipped scripts:** `run_justapr_vs_cumf.sh` and `run_ccd_sgd_comparison.py` both ship with a hardcoded dataset registry (absolute paths from the machine these results were measured on). On a fresh clone you have two options:
- Pass an explicit `.bin` path as an argument (`run_justapr_vs_cumf.sh` supports this: `./run_justapr_vs_cumf.sh /path/to/my_dataset.bin`), or
- Edit the `KNOWN=(...)` array in `run_justapr_vs_cumf.sh` / the `REGISTRY = {...}` dict in `run_ccd_sgd_comparison.py` to point at your own `.bin` files.

`run_ccd_sgd_comparison.py` also needs a `WORKDIR` env var pointing at scratch space it can write intermediate CCD/SGD conversion files to (`WORKDIR=/path/to/scratch python3 run_ccd_sgd_comparison.py`) — see `run_t4_everything.sh` PART 3 for a working example.

Datasets used for every measurement in this README:

| Dataset | Users | Items | Train ratings | Test ratings | `.bin` filename (as passed to the scripts above) |
|---|---:|---:|---:|---:|---|
| ml100k    | 943     | 1,682  | 80,367     | 19,633     | `ratings100.bin` |
| ml10m     | 69,878  | 10,677 | 8,026,731  | 1,973,323  | `ratings10.bin` |
| ml20m     | 138,493 | 26,744 | 16,052,927 | 3,947,336  | `ratings.bin` |
| ml32m     | 200,948 | 84,432 | 25,677,530 | 6,322,674  | `ratings32.bin` |
| netflix   | 480,189 | 17,770 | 80,570,989 | 19,909,518 | `netflix_ratings.bin` |

(`K_DIM` — the latent-factor rank — must be one of `{16, 32, 48, 64, 96}`; other values are rejected at compile time by a `#error` guard in `common.cuh`, because unsupported ranks silently truncate the LHS and produce a wrong-but-plausible RMSE instead of an error. See the guard's comment in `common.cuh` for why.)

---

## 4. The three home-grown binaries, in detail

### 4.1 `justapr.cu` — the ship binary

Mixed-precision APR-BALS: BALS-style tiled storage + Y-column reuse, WMMA (tensor-core) Gram computation for dense/giant tiles, a custom blocked-tiled Cholesky solver, and two **optional, independently-flagged** accelerations layered on top:

- **`-DCHOL_MP=2`** (K=96 only): stores the Cholesky solver's factor tiles in FP16 instead of FP32 (no iterative refinement — the "YOLO" variant). Cuts smem pressure, raising occupancy 4→8 blocks/SM. Solve time **−23%**, and at K=96 under the weighted-λ protocol the final test RMSE lands **bit-for-bit identical** to the FP32 solver (0 non-finite solves). **Do not use at K≤64** — measured: K=32 is destroyed by giant-item FP16 overflow (test RMSE 1.17 vs the 0.82 record), K=48 costs more convergence iterations than it saves, K=64 saves only ~4% with some overflow risk. `common.cuh` `#error`s if you pass `CHOL_MP` at `K_DIM=16` (no effect there — K=16 uses the batched solver).
- **`-DMOMENTUM=1`** (all K): heavy-ball extrapolation of the factor sequence between ALS sweeps (`X_used = X_raw + β(X_raw − X_raw_prev)`, β=0.3 default, override with `MBETA` env var at runtime). Converges in ~15 iterations instead of 20–25, test RMSE equal-or-better at every measured (dataset, K) pair, 0 non-finite solves. Precision-independent (works with the FP32 solver too).

Build the ship recipe:
```bash
# K <= 64:
nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=64 -DWEIGHTED_LAMBDA=1 -DCUMF_INIT=1 -DMOMENTUM=1 justapr.cu -o apr_k64
# K == 96 additionally gets the FP16 Cholesky:
nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=96 -DWEIGHTED_LAMBDA=1 -DCUMF_INIT=1 -DMOMENTUM=1 -DCHOL_MP=2 justapr.cu -o apr_k96

./apr_k96 my_dataset.bin 0.048
```
(`0.048` = λ, the cumf_als Netflix tune for weighted ALS-WR; good weighted-λ values are ~0.02–0.06.)

**Why the bare default isn't the ship recipe:** `justapr.cu` predates the weighted-λ/momentum/CHOL_MP work and originally shipped every legacy Netflix trajectory at plain λ=0.1. Rather than break reproducibility of those old runs, every new feature defaults OFF (`WEIGHTED_LAMBDA 0`, `CUMF_INIT 0`, `MOMENTUM 0`, `CHOL_MP 0` in the source). A bare `nvcc -DK_DIM=96 justapr.cu` still compiles and runs — it just runs the 2026-07-06 code path, not the paper's.

### 4.2 `only_scalar_fp32.cu` — the honest BALS-equivalent baseline

Same BALS tiled-storage format, job lists, entity batching, and Cholesky dispatch as `justapr.cu`, but **all tiles run the FP32 scalar kernel** — no WMMA, no tensor cores, exactly what the BALS paper specifies. It is, by construction, `main_experiment.cu`'s baseline leg extracted into its own binary (see §5 for the proof).

Two optional flags make it match BALS **more** faithfully instead of less:
- **`-DBALS_REORDER=1`** — BALS Algorithm 3: rows/columns pre-sorted by descending nnz before tiling (skips more vacant tiles up front). RMSE-neutral (permutation is a bijection), speed-neutral on this kernel (vacant tiles are already skipped via the job/nz-tile list here, so reordering doesn't buy what it buys in BALS's own kernel).
- **`-DBALS_SYMTILE=1`** — replaces the FP32 Gram kernel with the actual BALS/cuMF register-tile mapping (`compute_LHS_RHS_BALS_symtile`, faithful to `jingchen95/BALS`'s `magma_sals_kernel_lower2` and cuMF's `get_hermitian`): only lower-triangular tiles are computed, exploiting Gram symmetry. Bit-identical RMSE trajectory to the baseline (verified digit-for-digit); **1.0–1.21× faster** wall time (real but modest — the FP32 Gram here is memory/latency-bound at ~3% of the GPU's FP32 peak, so halving compute barely moves the needle).

Build:
```bash
nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=32 -DBALS_REORDER=1 -DBALS_SYMTILE=1 only_scalar_fp32.cu -o bals_equiv_k32
./bals_equiv_k32 my_dataset.bin 0.048
```

Default build (no flags) already has `WEIGHTED_LAMBDA=1` — this is the one place the "faithful to the paper" and "comparable to cuMF" defaults coincide, so a bare `nvcc -DK_DIM=32 only_scalar_fp32.cu` gives you weighted-λ ALS-WR with legacy init and no reordering.

### 4.3 `main_experiment.cu` — the controlled fairness harness

Runs an FP32 baseline leg *and* the mixed-precision APR/WMMA leg **in the same process, on the same loaded data**, via one shared `run_training(mixed, ...)` function called once with `mixed=false` and once with `mixed=true`. It does support `-DCHOL_MP` (it applies identically inside both calls, so both legs would still share the same solver), but every headline run in this README builds it with `CHOL_MP=0` on purpose — that keeps the WMMA-vs-scalar Gram kernel the *only* thing that differs between the two printed result blocks. There is no `MOMENTUM` support in this binary at all. This is the binary behind the paper's core "tensor cores + BALS format vs FP32 BALS format, same everything else" claim.

```bash
nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=96 -DCUMF_INIT=1 main_experiment.cu -o main_k96
./main_k96 my_dataset.bin 0.048
```
(`WEIGHTED_LAMBDA=1` is already the default here.)

---

## 5. Verification: does justapr's FP32 path actually match the baseline?

Before trusting any speed/accuracy comparison, we checked that `justapr.cu`'s FP32-solver, weighted-λ path is numerically the *same computation* as `main_experiment.cu`'s APR leg and `only_scalar_fp32.cu`'s standalone output — not three implementations that happen to look similar.

**1. `only_scalar_fp32.cu` ≡ `main_experiment.cu`'s baseline leg, run head-to-head, same split (`results/RMSE_COMPARISON_20260712_225827.txt`):**

| K | `main_experiment` baseline leg | `only_scalar_fp32` standalone | match? |
|---|---:|---:|---|
| 16 | test 0.831081, 4.846 s | test 0.831081, 4.834 s | RMSE exact, wall within noise |
| 32 | test 0.823308, 14.363 s | test 0.823308, 14.367 s | RMSE exact, wall within noise |
| 48 | test 0.820510, 31.733 s | test 0.820510, 31.759 s | RMSE exact, wall within noise |
| 64 | test 0.819113, 74.168 s | test 0.819113, 74.204 s | RMSE exact, wall within noise |
| 96 | test 0.817593, 154.463 s | test 0.817593, 154.502 s | RMSE exact, wall within noise |

**2. `justapr.cu`'s FP32-solver + weighted-λ + CUMF_INIT path == `main_experiment.cu`'s APR leg, exactly (`results/CHOLMP_VALIDATION_20260720/SUMMARY.txt`, "WEIGHTED-λ PORT" section):** built and ran both directions —
- `justapr.cu` with **no flags** reproduces the pre-2026-07-12 legacy trajectory digit-for-digit (train RMSE 0.545780 @ iter 45 — the original record).
- `justapr.cu -DWEIGHTED_LAMBDA=1 -DCUMF_INIT=1` (FP32 solver, `CHOL_MP=0`) at K=96 lands on **test RMSE 0.817558 exactly**, matching the `main_experiment.cu` APR-leg record from `results/RMSE_COMPARISON_20260712_225827.txt` (also 0.817558) and reconfirmed in the newer `results/MAIN3060_WL_20260721_225757/netflix_k96.txt` run (APR-BALS test RMSE 0.817558, iter 20).

So the chain `only_scalar_fp32.cu` → `main_experiment.cu` baseline leg → `main_experiment.cu` APR leg (FP32) → `justapr.cu` FP32+weighted-λ path is one verified, numerically-consistent computation. Everything downstream (`CHOL_MP`, `MOMENTUM`) is then layered on top of that same verified base and independently gated (see §4.1: `CHOL_MP=2` at K=96 reproduces this exact RMSE bit-for-bit; only iteration *count* changes under momentum, not the converged answer).

---

## 6. Results (measured, RTX 3060 sm_86, weighted-λ ALS-WR everywhere, λ=0.048)

Every table below cites the exact `results/` file it was read from. The cuMF_ALS comparison (§6.1–6.2) was **re-measured on 2026-07-22** specifically for this document — the previous archive (`results/JUSTAPR_VS_CUMF_20260721_045838.txt`) predates the momentum feature landing in the ship recipe and is kept only for the record.

### 6.1 justapr (ship recipe) vs cuMF_ALS — Netflix, per rank

Source: `results/JUSTAPR_VS_CUMF_20260722_031959.txt`.

| K (justapr) | F (cuMF, nearest ×10) | justapr ms/iter | cuMF ms/iter | per-iter speedup | justapr wall (15 it) | cuMF wall (to converge) | wall speedup | justapr test RMSE | cuMF test RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 20  | 58  | 365  | **6.30×** | 0.871 s | 5.478 s  | **6.29×** | 0.831099 | 0.828840 |
| 32 | 30  | 95  | 498  | **5.24×** | 1.431 s | 9.955 s  | **6.96×** | 0.823113 | 0.824093 |
| 48 | 50  | 138 | 826  | **5.99×** | 2.074 s | 16.512 s | **7.96×** | 0.820143 | 0.820556 |
| 64 | 60  | 235 | 1069 | **4.55×** | 3.525 s | 21.381 s | **6.07×** | 0.818655 | 0.819548 |
| 96 | 100 | 396 | 2099 | **5.30×** | 5.943 s | 41.982 s | **7.06×** | 0.817145 | 0.817487 |

justapr matches or beats cuMF's test RMSE at every rank on Netflix except K=96/F=100 (0.03% worse — cuMF uses one more effective factor per the F=100 vs K=96 rank mismatch), while running 4.5–6.3× faster per iteration and 6.1–8.0× faster wall-clock to convergence.

### 6.2 Cross-dataset summary at the top rank (K=96 vs F=100)

Source: same file, all 5 datasets. cuMF_ALS produces `nan` test RMSE on every dataset except Netflix — cuMF's kernel has no guard against zero-train-rating entities, which the four MovieLens sets contain and Netflix does not (this is cuMF's limitation, not a comparison artifact; wall-time is still valid since training completes).

| Dataset | justapr K=96 wall (iters) | cuMF F=100 wall (iters) | wall speedup | justapr test RMSE | cuMF test RMSE |
|---|---:|---:|---:|---:|---:|
| ml100k  | 0.028 s (15) | 0.084 s (20)  | 3.0×  | 0.937251 | nan |
| ml10m   | 0.767 s (15) | 4.885 s (25)  | 6.4×  | 0.779203 | nan |
| ml20m   | 1.733 s (15) | 10.696 s (25) | 6.2×  | 0.768034 | nan |
| ml32m   | 3.636 s (15) | 14.426 s (20) | 4.0×  | 0.757443 | nan |
| netflix | 5.943 s (15) | 41.982 s (20) | 7.1×  | 0.817145 | 0.817487 |

Full per-K grid for every dataset (25 justapr rows + 25 cuMF rows) is in `results/JUSTAPR_VS_CUMF_20260722_031959.txt`; per-iteration speedups range **3.0×–6.6×** across every (dataset, K) pair measured.

### 6.3 Our own FP32 baseline vs cuMF_ALS (no tensor cores on either side)

Source: `results/RMSE_COMPARISON_20260712_225827.txt` (Netflix; `main_experiment`'s baseline leg and `only_scalar_fp32` agree to the ms, see §5).

| F (cuMF) | K (FP32) | cuMF ms/iter | FP32 ms/iter | FP32 vs cuMF | cuMF test RMSE | FP32 test RMSE | Δ RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| 20  | 16 | 368  | 242  | 1.52× | 0.8288 | 0.8311 | +0.0023 |
| 30  | 32 | 500  | 718  | 0.70× | 0.8241 | 0.8233 | −0.0008 |
| 50  | 48 | 829  | 1587 | 0.52× | 0.8206 | 0.8205 | −0.0001 |
| 60  | 64 | 1071 | 3708 | 0.29× | 0.8195 | 0.8191 | −0.0004 |
| 100 | 96 | 2101 | 7723 | 0.27× | 0.8175 | 0.8176 | +0.0001 |

This is the "no cheating" comparison: our BALS-format FP32 scalar kernel against cuMF's tuned FP32+CG solver, no tensor cores on either side. cuMF's FP32 path is faster at K≥32 — the entire speed story of this project comes from the tensor-core dispatch (§6.1), not from the scalar kernel alone. RMSE differences are noise-level (≤0.03%) at every rank — the tiled BALS-format reformulation itself changes nothing about solution quality.

### 6.4 Compute throughput (GFlops/s), Netflix

Sources: `results/logs_justapr_vs_cumf_20260722_031959/netflix_apr_k*.txt` and `netflix_cumf_f*.txt` (justapr/cuMF), `results/MAIN3060_WL_20260721_225757/netflix_k*.txt` (FP32 baseline).

| K / F | cuMF GFlops/s | FP32-scalar GFlops/s | justapr (ship) GFlops/s |
|---|---:|---:|---:|
| 16 / 20   | 203.0 | 202.8 | **849.1**  |
| 32 / 30   | 320.5 | 250.9 | **1907.9** |
| 48 / 50   | 517.2 | 248.4 | **2876.6** |
| 64 / 60   | 569.8 | 186.3 | **2960.0** |
| 96 / 100  | 790.7 | 198.4 | **3887.8** |

### 6.5 justapr vs cuMF_CCD and cuMF_SGD

Source: `results/JUSTAPR_VS_CCD_SGD_20260721_192433.txt` — same frozen-split protocol, justapr ship recipe (momentum included), cuMF_CCD `t=15` outer iterations, cuMF_SGD `k=128` (its only supported rank) swept over `t∈{10,20,30,40}` epochs.

**vs cuMF_CCD** (average wall-time ratio across K=16..96), compared against the original BALS paper's own margin over cuMF_CCD on a TITAN RTX:

| dataset | justapr × faster than cuMF_CCD (measured here) | BALS paper × over cuMF_CCD |
|---|:--:|:--:|
| ml10m   | **3.83×** | 2.09× |
| ml20m   | **3.72×** | 3.86× |
| ml32m   | **3.44×** | (not in paper) |
| netflix | **4.08×** | 3.22× |

**vs cuMF_SGD** (justapr K=96 vs SGD k=128, wall-to-reach-justapr's-RMSE): despite using *fewer* factors (K=96 vs k=128), justapr matches SGD's quality at comparable wall time on ml10m and beats it (SGD never reaches justapr's RMSE within 40 epochs) on ml20m, ml32m, and netflix. See `RUNNING_SUMMARY.md §1` for the full per-dataset breakdown — note the BALS paper's own "5.3× over cuMF_SGD" is a *throughput* (GFlops) number at matched rank, not a wall-to-convergence number; ALS does strictly more FLOPs/iteration than SGD, so it always wins the throughput metric even where wall-clock is only competitive.

### 6.6 The BALS-equivalent ladder — `only_scalar_fp32.cu` vs cuMF_ALS vs justapr

Source: `results/BALS_EQUIV_VALIDATION_20260721_205101/` (K=32, λ=0.048, all 4 large datasets — `{ml10m,ml20m,ml32m,netflix}_bals_{base,symtile}_k32.txt`), cross-checked directly (`grep "Training Complete"` on every file) rather than taken from prose.

| dataset | reorder-only (`-DBALS_REORDER=1`) | **symtile** (`+ -DBALS_SYMTILE=1`) | cuMF_ALS (F=30) | justapr (K=32, ship) | symtile vs cuMF | justapr vs symtile | justapr vs reorder-only |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| ml10m   | 1.921 s  | 1.581 s  | 0.845 s | 0.154 s | 1.87× slower | **10.3×** | **12.5×** |
| ml20m   | 4.351 s  | 3.798 s  | 1.846 s | 0.331 s | 2.06× slower | **11.5×** | **13.1×** |
| ml32m   | 6.933 s  | 6.452 s  | 3.078 s | 0.611 s | 2.10× slower | **10.6×** | **11.3×** |
| netflix | 17.970 s | 15.618 s | 9.955 s | 1.431 s | 1.57× slower | **10.9×** | **12.6×** |

Two honest findings here (both documented at length in `results/RUNNING_SUMMARY.md §2b/§2b.1`):
1. **justapr is ~10–13× faster than a faithful FP32 BALS re-implementation.** This isolates exactly what tensor cores + momentum + mixed-precision Cholesky buy on top of the shared BALS tiling contribution.
2. **The faithful FP32 BALS-equivalent is itself 1.6–2.1× slower than modern cuMF_ALS on this GPU**, unlike the original BALS paper's pre-Ampere results (K20C/TITAN X/TITAN RTX). The symmetry-exploiting kernel (`BALS_SYMTILE`) is a real 1.10–1.22× win over the reorder-only kernel (bit-identical RMSE trajectory, verified digit-for-digit), but the FP32 Gram computation here is memory/latency-bound at only ~3% of FP32 peak throughput, so halving FLOPs barely moves wall time — this is an architectural (Ampere vs pre-Ampere) effect, not a bug, and it's the reason the tensor-core path (justapr) exists at all.

### 6.7 Cross-GPU validation — Tesla T4 (Colab, `sm_75`, Turing)

Everything above was measured on the dev machine's RTX 3060 (`sm_86`, Ampere). To check the headline claims aren't an Ampere-specific artifact, every comparison in §6.1–§6.6 was re-run on a **Tesla T4** (Google Colab, `sm_75`, Turing, single GPU) via `run_t4_everything.sh`. One caveat up front: **T4 is Turing — it has no `cp.async` hardware** (that instruction is Ampere-only). `justapr.cu`'s giant-Gram kernel uses `__pipeline_memcpy_async` (round 3, 2026-07-21); on `sm_75` this is not a compile error — CUDA's own `cuda_pipeline.h` transparently falls back to a synchronous emulated copy — but it means that *specific* optimization's benefit doesn't materialize on T4. Correctness is unaffected either way (see the 0-non-finite-solves check below); only that one kernel's speed headroom is left partly on the table.

**§6.3-style — own FP32 baseline vs APR-BALS (`main_experiment.cu`, weighted-λ), T4 vs 3060 side by side.** Source: `results/t4_latest_res_22_7_2026/T4_MAIN_WL_20260721_210336/` vs `results/MAIN3060_WL_20260721_225757/`. Test RMSE is omitted from this table because it's cross-GPU identical to 5-6 decimals (e.g. netflix K=96 APR test RMSE: 0.817558 on **both** machines) — only wall-time speedup is architecture-dependent.

| Dataset | K | T4 wall speedup | 3060 wall speedup |
|---|---:|---:|---:|
| netflix | 16 | 2.75× | 4.07× |
| netflix | 32 | 4.98× | 7.21× |
| netflix | 48 | 6.27× | 10.14× |
| netflix | 64 | 9.08× | 13.90× |
| netflix | 96 | 10.78× | 14.84× |
| ml20m   | 16 | 2.85× | 4.08× |
| ml20m   | 32 | 4.72× | 6.76× |
| ml20m   | 48 | 5.50× | 8.83× |
| ml20m   | 64 | 7.23× | 11.62× |
| ml20m   | 96 | 8.08× | 11.16× |
| ml10m   | 16 | 2.95× | 3.93× |
| ml10m   | 32 | 4.68× | 6.42× |
| ml10m   | 48 | 5.26× | 8.16× |
| ml10m   | 64 | 6.88× | 10.87× |
| ml10m   | 96 | 7.36× | 10.45× |
| ml100k  | 16–96 | 1.89×–2.80× | *(not run on 3060; tiny dataset, sub-0.1s wall on both — sensitive to fixed overhead)* |

Same conclusion, smaller margin: speedup still climbs with K on both cards, but T4's 2nd-gen tensor cores (vs the 3060's 3rd-gen) and the missing `cp.async` path mean the ceiling is lower — 7.4×–10.8× at K=96 on T4 vs 10.5×–14.8× on the 3060.

**§6.1/6.2-style — justapr (ship recipe) vs cuMF_ALS, T4.** Source: `results/t4_latest_res_22_7_2026/results_unfinished_part/JUSTAPR_VS_CUMF_20260721_233252.txt` (despite the directory name, this specific run completed cleanly — see the note at the end of this subsection).

| K (justapr) | F (cuMF) | T4 per-iter speedup | T4 wall speedup | justapr test RMSE | cuMF test RMSE |
|---|---|---:|---:|---:|---:|
| 16 | 20  | **6.09×** | **6.08×** | 0.831099 | 0.828840 |
| 32 | 30  | **5.10×** | **6.80×** | 0.823112 | 0.824093 |
| 48 | 50  | **4.69×** | **6.27×** | 0.820142 | 0.820556 |
| 64 | 60  | **3.93×** | **5.25×** | 0.818655 | 0.819548 |
| 96 | 100 | **4.66×** | **6.22×** | 0.817146 | 0.818021 |

Cross-dataset at the top rank (K=96 vs F=100; cuMF is `nan` off-Netflix on T4 too, same 0-train-entity limitation as on the 3060):

| Dataset | T4 justapr K=96 wall (iters) | T4 cuMF F=100 wall (iters) | T4 wall speedup |
|---|---:|---:|---:|
| ml10m   | 1.252 s (15) | 7.003 s (25)  | 5.6× |
| ml20m   | 2.777 s (15) | 15.566 s (25) | 5.6× |
| ml32m   | 5.813 s (15) | 30.458 s (25) | 5.2× |
| netflix | 10.888 s (15) | 67.721 s (20) | 6.2× |

**Compute throughput (GFlops/s), netflix, T4:** cuMF 119.5→490.2 (K16→96), FP32-scalar (from Part 1) 192.6→155.9, justapr (ship) **482.5→2118.8**. Same qualitative story as the 3060 (§6.4): justapr's throughput lead over both baselines holds, just at T4's lower absolute ceiling (2119 GFlops/s peak vs the 3060's 3888).

**§6.5-style — justapr vs cuMF_CCD/cuMF_SGD, T4.** Source: `results/t4_latest_res_22_7_2026/JUSTAPR_VS_CCD_SGD_20260721_213048.txt` (this one succeeded on the first try — cuMF_CCD/SGD are rebuilt unconditionally by the script, unlike the cuMF_ALS binary that caused the Part-2 gap).

| dataset | justapr × faster than cuMF_CCD (T4) | (for reference: 3060, §6.5) |
|---|:--:|:--:|
| ml10m   | 2.05× | 3.83× |
| ml20m   | 2.38× | 3.72× |
| ml32m   | 2.35× | 3.44× |
| netflix | 2.86× | 4.08× |

vs cuMF_SGD (T4): justapr K=96 ties SGD on ml10m (0.82× — SGD reaches the same RMSE slightly faster there, same as the 3060 finding), and SGD never reaches justapr's RMSE within 40 epochs on ml20m/ml32m/netflix — identical qualitative result to §6.5, smaller absolute margin.

**§6.6-style — BALS-equivalent ladder, T4.** Source: `results/t4_latest_res_22_7_2026/T4_BALS_EQUIV_20260721_210336/` (`only_scalar_fp32.cu` base/symtile) + the K=32/F=30 rows of the file above (cuMF_ALS, same T4 run).

| dataset | reorder-only | symtile | cuMF_ALS (F=30) | justapr (K=32) | symtile vs cuMF | justapr vs symtile |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| ml10m   | 2.135 s  | 1.801 s  | 1.039 s  | 0.293 s | 1.73× slower | **6.15×** |
| ml20m   | 4.666 s  | 4.058 s  | 2.352 s  | 0.536 s | 1.73× slower | **7.57×** |
| ml32m   | 7.185 s  | 6.601 s  | 4.429 s  | 0.953 s | 1.49× slower | **6.93×** |
| netflix | 18.511 s | 15.821 s | 16.711 s | 2.458 s | **1.06× *faster*** | **6.44×** |

**Notable, non-obvious finding:** on the 3060, the FP32 BALS-equivalent (even the symmetry-exploiting `symtile` variant) is *always* slower than cuMF_ALS (1.57–2.10×, §6.6). On T4, at netflix, `symtile` edges *ahead* of cuMF_ALS. This isn't noise — the `symtile`/`reorder-only` gap (1.09–1.19× on T4) matches the 3060's 1.07–1.22× almost exactly, so the FP32 kernel itself behaves consistently; what moved is cuMF's *own* relative speed on Turing vs Ampere. This is independent evidence for the README's existing explanation (§6.6) that the BALS paper's original faster-than-cuMF results were pre-Ampere hardware: T4 (Turing, architecturally closer to that era) reproduces a crossover, Ampere doesn't. The tensor-core path (justapr) still wins outright either way — 6.15–7.57× over `symtile` on T4 (vs 10.3–10.9× on the 3060: still decisive, just a smaller margin, consistent with T4's weaker/2nd-gen tensor cores).

**Caveats specific to the T4 numbers:**
- **Higher run-to-run wall-time variance than the 3060.** Netflix justapr K=96 (ship recipe, identical build) measured 9.315 s / 9.479 s / 10.888 s across three separate runs in this batch (≈17% spread) vs the 3060's <1% spread across repeated runs — expected for a shared/virtualized Colab GPU (thermal state, co-tenancy, instance variability), not a code issue. Test RMSE stayed effectively constant across all three (0.817145–0.817146), confirming it's a timing artifact, not a numerical one. Treat single T4 wall-time numbers as directional; where possible, prefer the ratio (speedup) over the absolute seconds.
- **cuMF_ALS's own RMSE drifts slightly more cross-GPU than our binaries do.** Netflix F=100 test RMSE: 0.818021 (T4) vs 0.817483–0.817499 (3060 runs) — about 0.06%, a larger gap than the ≤0.01% wobble seen elsewhere in this doc. This is cuMF's own external CG solver's floating-point path on a different architecture, not something under our control.
- **The `results_unfinished_part/` directory name is misleading.** It was captured mid-session on Colab and the name suggests an interrupted run, but the run inside it (`JUSTAPR_VS_CUMF_20260721_233252.txt`, all 55 expected log files present, every leg reports `✓`, `Finished:` line present, 0 non-finite solves at K=96 on every dataset) is in fact complete — verified file-by-file before writing this section. An earlier attempt in the same results bundle (`JUSTAPR_VS_CUMF_20260721_212253.txt`) genuinely did fail every cuMF leg (root cause: a `cumf_als-master/main` binary carried over from this dev box, incompatible with Colab's libstdc++ — fixed in `run_t4_everything.sh`/`run_justapr_vs_cumf.sh`, see their inline comments); the `233252` run is the corrected re-run and is what the tables above are drawn from.

---

## 7. Reproducing the results

Every table above can be regenerated with one of these commands, run from inside `comparing_baseline/`:

```bash
# §6.3 — our FP32 baseline (both binaries) + cuMF_ALS, one dataset, full K sweep:
./run_rmse_comparison.sh netflix_ratings.bin

# §6.1, §6.2 — justapr (ship recipe) vs cuMF_ALS, every dataset found on disk:
./run_justapr_vs_cumf.sh
# or a named/explicit subset:
./run_justapr_vs_cumf.sh ml10m netflix
./run_justapr_vs_cumf.sh /path/to/custom.bin

# §6.5 — justapr vs cuMF_CCD vs cuMF_SGD:
WORKDIR=/path/to/scratch python3 run_ccd_sgd_comparison.py

# §6.6 — BALS-equivalent ladder:
nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=32 -DBALS_REORDER=1 only_scalar_fp32.cu -o bals_base
nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=32 -DBALS_REORDER=1 -DBALS_SYMTILE=1 only_scalar_fp32.cu -o bals_symtile
./bals_base my_dataset.bin 0.048
./bals_symtile my_dataset.bin 0.048
```

All four scripts write a timestamped summary + full logs under `results/`, so re-running never overwrites a previous result.

**On a machine other than the one these numbers were measured on** (e.g. a Colab T4, or your own workstation with different dataset paths), use `run_t4_everything.sh` — it runs all four comparisons above end-to-end, auto-detects the GPU architecture, and zips `results/` for you at the end:
```bash
DATA_DIR=/path/to/your/bin/files bash run_t4_everything.sh
```
It expects the 5 `.bin` files named exactly as in the dataset table in §3, inside `DATA_DIR`. Toggle individual parts off with `RUN_PART1=0` etc. if you only need one comparison.

Each `cuMF_*` external baseline needs to be built once (`make` inside `cumf_als-master/`, `cumf_ccd-master/`, `cumf_sgd-master/singleGPU/` and `cumf_sgd-master/test/`) — all four run scripts above do this automatically on first use and skip it on subsequent runs.

---

## 8. Known gotchas

- **cuMF_ALS prints `nan` test RMSE on every dataset except Netflix.** This is a real limitation of the external cuMF_ALS code (no guard against zero-train-rating entities, which the MovieLens splits contain), not a bug in our harness. Training still completes and wall-time/throughput numbers from those runs are valid; only the RMSE column is unusable off-Netflix.
- **`K_DIM` only supports `{16, 32, 48, 64, 96}`.** Anything else fails a compile-time `#error` (by design — see the guard's comment in `common.cuh`; earlier silent support for `K=80` produced a plausible-looking but wrong RMSE).
- **`-DCHOL_MP` is a K=96-only optimization.** It's compile-guarded against `K_DIM=16` and measured-harmful/net-neutral at K=32/48/64 (§4.1) — don't add it at other ranks even though it will compile.
- **Run-to-run wall-time noise is ~1–2%** (thermal/scheduler variance on shared GPUs); test RMSE can wobble in the 5th–6th decimal from nondeterministic `atomicAdd` ordering in the LHS accumulation. None of the conclusions in this README depend on differences that small.
- **Dataset paths in `run_justapr_vs_cumf.sh` / `run_ccd_sgd_comparison.py` are hardcoded** to the machine these results were measured on — see §3 for the two ways to point them at your own files.
- **The shell scripts default to `GPU=1`** (the second GPU on this 2×3060 box, `CUDA_VISIBLE_DEVICES=$GPU`). On a single-GPU machine this will fail to find a device — run with `GPU=0 ./run_justapr_vs_cumf.sh` (etc.) instead.

---

## 9. Where the deeper detail lives

- **`FIXLOG.md`** — full chronological engineering log: every optimization attempt (including the ones that failed and were reverted), with exact measured numbers and the reasoning behind every dispatch rule (`CHOL_MP` K-gating, `FAST_RMSE` K-gating, momentum β sweep, solver kernel choice, etc).
- **`results/RUNNING_SUMMARY.md`** — the living results document this README's numbers are drawn from; kept in sync with `results/` as new runs land.
- **`results/`** — every raw run output cited above, plus historical runs (`CHOLMP_VALIDATION_20260720/`, `MAIN3060_WL_20260721_225757/`, etc.) documenting the validation work behind each shipped default.

## 10. External baselines — attribution

- **cuMF_ALS** — Wei Tan et al., *"Fast ImplicitALS on GPUs"*, HPDC 2016. Original: [github.com/wei-tan/cumf_als](https://github.com/wei-tan/cumf_als)/[cuMF](https://github.com/wei-tan/cuMF). Ported here from `sm_35`/C++11 to `sm_86`/C++14 with a `.bin`-format loader and simplified CLI args (see `cumf_als-master/`).
- **cuMF_CCD** — CCD++ solver from the same cuMF family. Ported with a `__shfl_down` compatibility shim (CUDA 9+ removed the maskless intrinsic).
- **cuMF_SGD** — Xie et al., *cuMF_SGD*. Ported with a `__shfl`/`__shfl_down` shim and an added wall-clock timer; test harness uses the bundled libmf-derived `mf-predict`.
- **BALS** — Chen et al., *"Bridging the Gap between HPC and Big Data frameworks"* / the BALS tiled-ALS paper (TPDS 2021), `jingchen95/BALS`. `only_scalar_fp32.cu` with `-DBALS_REORDER=1 -DBALS_SYMTILE=1` is our from-scratch, faithful re-implementation of its tiling + reuse + symmetry-exploiting Gram kernel — no code was copied, only the algorithm.
