# Accelerating ALS Matrix Factorization on GPUs via Mixed-Precision Tensor Cores

> Adaptive-Precision Blocked Alternating Least Squares — a GPU implementation of
> explicit-feedback matrix factorization that runs the dense parts of each ALS
> update on FP16 **tensor cores (WMMA)** while keeping the numerically sensitive
> parts in FP32. It reaches **up to ~15× wall-clock (~18× compute) speedup** over
> an FP32 GPU baseline **with ≤ 0.1 % change in test RMSE**.

This repository contains the code, preprocessing, and benchmarking harness for the
undergraduate thesis *"Accelerating ALS Matrix Factorization on GPUs via
Mixed-Precision Tensor Cores."*

> **This is the fresh, finalized version of the project.** The original development
> repo — [hafizr283/GPU_Programming_Essentials](https://github.com/hafizr283/GPU_Programming_Essentials)
> — was a draft/practice workspace with many experimental and one-off files, so it
> was rebuilt clean here. That older repo also includes a **step-by-step tutorial on
> how ALS works**, which is worth reading first if you're new to the algorithm.

---

## 1. What this does

Given a sparse user–item rating matrix `R`, ALS factorizes it into `P` (users × K)
and `Q` (items × K) such that `R ≈ P Qᵀ`, by alternately solving the closed-form
ridge-regression update for each row:

```
x_u = (Y_{I_u}ᵀ Y_{I_u} + λI)⁻¹ Y_{I_u}ᵀ r_u
```

The contribution (APR-BALS) is a **precision-dispatch** on top of the BALS tiled
format: each entity is routed at runtime to one of three code paths depending on
how many ratings it has, so the expensive `YᵀY` outer-product accumulation is done
on FP16 tensor cores for the dense rows and in plain FP32 for the sparse rows —
recovering the tensor-core speedup **without** the accuracy loss that a blanket
FP16 conversion would cause.

**Two-pillar evaluation** (see `BENCHMARKING.md`):
- **Pillar 1 (main claim):** APR-BALS vs an FP32 baseline **in the same binary, on
  the same GPU, same data** — only the precision-dispatch changes. This is the
  speedup ratio the thesis reports.
- **Pillar 2 (external baseline):** a standard PyTorch/cuBLAS GPU ALS
  (`baseline_als_gpu.py`) and optionally cuMF, scored on the **identical frozen
  split** via `bench_eval_rmse.py`.

---

## 2. Headline results

APR-BALS vs the FP32 baseline in the **same binary, same GPU, same data** (only the
precision-dispatch differs). Ratios are measured on one GPU (the machine cancels);
test RMSE is machine-independent. Numbers below are the final run on an RTX 3060
(sm_86); the ratios reproduce on a Tesla T4 (sm_75).

**Netflix** (speedup grows with K):

| K  | Wall speedup | Compute speedup | Test ΔRMSE |
|----|--------------|-----------------|------------|
| 16 | 4.61×        | 5.47×           | 0.009 %    |
| 32 | 7.75×        | 9.80×           | 0.027 %    |
| 48 | 10.98×       | 14.18×          | 0.051 %    |
| 64 | 14.22×       | 17.24×          | 0.083 %    |
| 96 | **15.36×**   | 17.90×          | 0.124 %    |

**Other datasets (K = 64):** ML-20M → 12.10× wall / 12.53× compute (ΔRMSE 0.018 %);
ML-10M → 11.37× wall / 11.79× compute (ΔRMSE 0.007 %).

So: **up to ~15× wall-clock / ~18× compute on Netflix**, with test-RMSE change
≤ 0.13 %. K = 64 is the best "large speedup at < 0.1 % RMSE" operating point; K = 96
edges the wall-clock higher but crosses 0.1 %. Full tables in
[`results/ALL_SUMMARY.txt`](results/ALL_SUMMARY.txt).

Accuracy note: best **generalization** is at **K = 16 / K = 32** (lowest test RMSE);
higher K trades accuracy for speed, and the FP32 baseline overfits identically — so
the growing Δ at high K is a property of the data, not a weakness of the
mixed-precision path.

Supported latent dimensions: **K ∈ {16, 32, 48, 64, 96}** (enforced at compile
time; see the guard note in `common.cuh`).

---

## 3. Repository layout

```
├── README.md                 ← this file
├── BENCHMARKING.md           evaluation methodology (how the claims are made unattackable)
├── DEBUG_NOTES.txt           ALL debugging/verification details in one file (read this
│                             before touching kernels — gotchas, debug switches, records)
│
├── preprocess.cpp            CSV → frozen .bin (train/test split + CSR) — build this first
│
├── common.cuh                constants, K guard, tile sizes, dispatch thresholds
├── data_utils.cuh            builds the BALS tiled/segmented sparse format
├── fused_kernels.cuh         scalar FP32 path (sparse entities) + fused helpers
├── wmma_kernels.cuh          FP16 tensor-core (WMMA) path for dense/giant entities
├── cholesky_kernels.cuh      batched / packed / tiled Cholesky solvers
├── main_experiment.cu        MAIN: runs FP32 baseline + APR in one binary → speedup ratio
│                             (weighted-λ ALS-WR by default; -DWEIGHTED_LAMBDA=0 for plain λ)
├── only_scalar_fp32.cu       standalone scalar-FP32 baseline (main_experiment minus all
│                             WMMA/FP16 launches; same weighted-λ switch)
├── justapr.cu                APR-only variant (no baseline pass; plain λ)
│
├── baseline_als_gpu.py       Pillar-2 external baseline (PyTorch/cuBLAS GPU ALS, plain λ)
├── bench_eval_rmse.py        scores ANY P,Q on the frozen .bin split (fair comparison)
│
├── run_rmse_comparison.sh    ONE-SHOT: APR + FP32 baselines + cuMF on the same split → table
├── run_scalar_sweep.sh       K sweep of only_scalar_fp32 (with register-spill check)
├── run_final.sh              one-shot pipeline: GT tuning sweep + full K sweep, all datasets
├── run_pillar2.sh            PyTorch-baseline comparison (pinned to plain λ)
├── cumf_als-master/          cuMF_als (HPDC 2016) CUDA-12 port — external reference baseline
└── results/                  official APR sweep summary + comparison-run outputs
```

---

## 4. Requirements

- **NVIDIA GPU with tensor cores** — compute capability ≥ 7.0 (Volta/Turing/Ampere).
  Tested on Tesla T4 (`sm_75`, Colab) and RTX 3060 (`sm_86`).
- **CUDA Toolkit 11+** (`nvcc` on PATH).
- A C++14 host compiler (for `preprocess.cpp`).
- For the Pillar-2 baseline: **Python 3.8+**, `pip install torch numpy scipy`
  (CUDA-enabled PyTorch).

Check your toolchain:
```bash
nvcc --version && nvidia-smi
```

---

## 5. Data & preprocessing

### 5.1 Input format
A CSV in **MovieLens `ratings.csv`** layout — a header line, then one rating per
line:
```
userId,movieId,rating,timestamp
1,296,5.0,1147880044
1,306,3.5,1147868817
...
```
IDs are 1-based in the file. Netflix data must be converted to this same 4-column
CSV first.

### 5.2 Build the preprocessor
```bash
g++ -O3 -std=c++14 preprocess.cpp -o preprocess
```

### 5.3 Create the frozen `.bin`
```bash
./preprocess  path/to/ratings.csv  ratings.bin
```
The preprocessor does everything the training code depends on, and freezes it:
1. **Frequency reordering** — users and items are relabelled most-popular-first, so
   the non-zeros cluster in the top-left tiles (this is what makes BALS's dense-tile
   path pay off).
2. **80/20 train/test split, stratified per user, seed = 42** — deterministic and
   reproducible.
3. **CSR build** for both user-major and item-major orders.
4. Serializes header + train COO + test COO + both CSR arrays into one
   little-endian `.bin`.

Because the split lives **inside the `.bin`**, every consumer (APR-BALS, the FP32
baseline, the PyTorch baseline, cuMF) reads the *identical* arrays — so RMSE is
directly comparable across all of them. Inspect a `.bin`:
```bash
python bench_eval_rmse.py ratings.bin
```

### 5.4 Expected filenames (used by `run_final.sh`)
| Dataset        | `.bin` name           | short name |
|----------------|-----------------------|------------|
| MovieLens 100K | `ratings100.bin`      | `ml100k`   |
| MovieLens 10M  | `ratings10.bin`       | `ml10m`    |
| MovieLens 20M  | `ratings.bin`         | `ml20m`    |
| Netflix        | `netflix_ratings.bin` | `netflix`  |

### 5.5 Where to get the data
The raw datasets are **not** shipped here (hundreds of MB, separate licenses):
- MovieLens — https://grouplens.org/datasets/movielens/
- Netflix Prize — https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
  (convert to the `userId,movieId,rating,timestamp` CSV layout above, then run
  `preprocess`).

---

## 6. Build & run the factorization

The latent dimension `K` is a **compile-time constant** (`-DK_DIM`) so the WMMA
kernels can be specialized. Pick your GPU arch with `-arch`.

```bash
# K=64 on Ampere (RTX 30xx); use -arch=sm_75 for T4/Colab
nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=64 main_experiment.cu -o apr_k64

# run on a preprocessed split (arg 2 = lambda; weighted-λ wants ~0.02–0.06)
./apr_k64 ratings.bin 0.048
```

**Regularization (since 2026-07-12):** both `main_experiment.cu` and
`only_scalar_fp32.cu` default to **weighted-λ regularization (ALS-WR,
cuMF-style: `diag += nnz_entity · λ`)** so their RMSE is directly comparable
with cuMF. Good weighted λ is **~0.02–0.06** (0.048 = cuMF's Netflix tune).
Compile with `-DWEIGHTED_LAMBDA=0` to get the original plain `λ·I` behavior
(tuned at λ = 0.1); that path is bit-preserved. Details in `DEBUG_NOTES.txt`.

Optional compile knobs (all have sensible defaults in `common.cuh`):

| Macro                     | Meaning                                              | Default |
|---------------------------|------------------------------------------------------|---------|
| `K_DIM`                   | latent factors — one of {16,32,48,64,96}             | 16      |
| `WEIGHTED_LAMBDA`         | 1 = weighted-λ ALS-WR (cuMF-style), 0 = plain λ·I    | 1       |
| `CUMF_INIT`               | 1 = U(0,0.2) init like cuMF (~⅓ fewer iters, better test RMSE); 0 = legacy U(0.1,1.1) | 0 |
| `DENSE_NNZ_THRESH`        | tile-density cutoff for routing a tile to WMMA       | 4       |
| `GIANT_NNZ_THRESH`        | nnz cutoff for the giant-entity WMMA reduction path  | 1024    |
| `BATCHED_CHOLESKY_MAX_K`  | K at/below which the thread-per-system solver is used | 16     |

### What it prints
`main_experiment.cu` runs **two full trainings** — an FP32 baseline pass and the
APR pass — then reports:
- **Compute speedup** and **Wall-time speedup** (APR vs FP32 baseline),
- Train/Test RMSE for both, with the absolute and % delta,
- FLOP count, throughput (GFlops/s), and per-phase timing.

### Getting the factor matrices `P` and `Q`
After training, the user factors `P` (users × K) and item factors `Q` (items × K)
are copied back to host into `h_X` and `h_Y` at the end of `main()`. The default
build reports metrics rather than dumping the matrices; to persist the
factorization, add a `write_vec`-style dump of `h_X`/`h_Y` there (the reader in
`bench_eval_rmse.py` shows the exact binary layout to match). `justapr.cu` runs a
single APR training without the baseline pass if you only want the factors/profile.

---

## 7. Reproduce the full sweep (one command)

`run_final.sh` compiles and runs the whole pipeline for every `.bin` you pass:
a `GIANT_NNZ_THRESH` tuning sweep at K=16, then the full K sweep {16,32,48,64,96}
at the best threshold, writing per-dataset logs and a grand summary table.

```bash
# sm_86 by default; export ARCH=sm_75 for T4/Colab
ARCH=sm_86 ./run_final.sh ratings100.bin ratings10.bin ratings.bin netflix_ratings.bin
```
Output lands in `final results/<name>_gpu.txt` and
`final results/ALL_SUMMARY_<timestamp>.txt`. `SKIP_GT_SWEEP=1` reuses known-good
thresholds for a fast rerun; `VERBOSE=1` echoes full program output.
Regularization knobs: `WEIGHTED=1 LAMBDA=0.048` by default; use
`WEIGHTED=0 LAMBDA=0.1` to reproduce the pre-2026-07-12 plain-λ sweeps.

---

## 8. External baseline (Pillar 2)

**One-shot RMSE + speed comparison** — APR-BALS, the FP32 baselines, and
cuMF_als on the identical frozen split, one grand table (~20–25 min for the
full Netflix sweep):
```bash
GPU=1 ./run_rmse_comparison.sh /path/to/netflix_ratings.bin
```
Individual baselines:
```bash
# PyTorch / cuBLAS GPU ALS — no custom kernels, no mixed precision (plain λ)
python baseline_als_gpu.py ratings.bin --K 32 --lam 0.1 --iters 50

# cuMF_als (HPDC 2016) CUDA-12 port — rank F must be a multiple of 10
cd cumf_als-master && make main
./main /path/to/netflix_ratings.bin 0.048 60 1 3 0 50 0.001
#      <bin>  [lambda] [F] [X_BATCH] [THETA_BATCH] [DEVICE] [MAX_ITERS] [TOL]

# score any other model's P,Q on the identical split
python bench_eval_rmse.py ratings.bin
```
`run_pillar2.sh` automates the PyTorch comparison (pinned to plain λ so it is
like-for-like). See `BENCHMARKING.md` §3–§4 for the protocol rules (do **not**
claim to beat the Netflix Prize — the split differs), and `DEBUG_NOTES.txt`
§4 for the cuMF port notes and its recorded Netflix results.

---

## 9. Citation

If you use this code, please cite the thesis:

```bibtex
@mastersthesis{aprbals,
  title  = {Accelerating ALS Matrix Factorization on GPUs via Mixed-Precision Tensor Cores},
  author = {Hafizur Rahman},
  year   = {2026},
  note   = {APR-BALS}
}
```

Built on the BALS tiled sparse format (Blocked Alternating Least Squares for
Parallel Sparse Matrix Factorization on GPUs).
"# Baseline_Comarison_Of_APR_BALS" 
