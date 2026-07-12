# Modified cuMF for Baseline Comparison

> **Note:** This repository is the extension repository of [Accelerating-ALS-Matrix-Factorization-on-GPUs-via-Mixed-Precision-Tensor-Cores](https://github.com/hafizr283/Accelerating-ALS-Matrix-Factorization-on-GPUs-via-Mixed-Precision-Tensor-Cores) for comparing baseline.

This folder contains a modified version of [cuMF](https://github.com/wei-tan/cuMF) (CUDA-Accelerated ALS), adapted specifically to integrate with our baseline comparison framework.

The original cuMF codebase has been modified to:
- Accept preprocessed binary (`.bin`) datasets instead of text-based CSR/CSC/COO matrices.
- Take simplified hyperparameter arguments (Lambda, F, max iterations, etc.) from the command line, enabling direct comparison against APR and BALS baselines.

## 1. Build Instructions

To compile the `main` executable, navigate to this directory and run:
```bash
make clean
make main
```

## 2. Dataset Preparation

Before running cuMF, you must convert your raw dataset (e.g., CSV) into the binary format required by the framework. Use the `preprocess.cpp` tool located in the parent directory:

```bash
# In the parent directory (comparing_baseline):
g++ -O3 preprocess.cpp -o preprocess
./preprocess path/to/your/ratings.csv my_dataset.bin
```

## 3. Running the RMSE Comparison

The easiest way to run this code and compare its performance and RMSE against the other baseline models is to use the `run_rmse_comparison.sh` script located in the parent directory.

```bash
# In the parent directory:
./run_rmse_comparison.sh my_dataset.bin
```

This script will automatically compile cuMF (if needed), execute it alongside the custom scalar FP32/CUDA baselines on the same `.bin` dataset, and output a comparison table of execution speeds and Root Mean Square Error (RMSE).

## 4. Running cuMF Manually

If you need to run this modified cuMF independently, navigate to this directory and use the following syntax:

```bash
./main <dataset.bin> <lambda> <F> <X_BATCH> <THETA_BATCH> <device_id> <max_iterations> <tolerance>
```

**Example:**
```bash
./main /absolute/path/to/my_dataset.bin 0.048 100 1 3 0 20 0.001
```

*Note: Rank value `F` has to be a multiple of 10, e.g., 10, 50, 100.*

## 5. Performance and RMSE Results

Here are the benchmarking results on identical frozen splits on the Netflix dataset.

### 5.1 APR-BALS vs cuMF
Comparison of our accelerated mixed-precision baseline (APR-BALS) against cuMF.

| F (cuMF) | K (APR) | cuMF (ms/iter) | APR (ms/iter) | APR Speedup | cuMF GFlops/s | APR GFlops/s | cuMF RMSE | APR RMSE | Δ RMSE |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 16 | 368 | 53 | **6.94x** | 201.6 | 918.0 | 0.8288 | 0.8311 | +0.0023 |
| 30 | 32 | 500 | 97 | **5.15x** | 318.8 | 1866.2 | 0.8241 | 0.8233 | -0.0008 |
| 50 | 48 | 829 | 152 | **5.45x** | 515.2 | 2592.2 | 0.8206 | 0.8205 | -0.0001 |
| 60 | 64 | 1071 | 273 | **3.92x** | 568.8 | 2532.8 | 0.8195 | 0.8191 | -0.0004 |
| 100| 96 | 2101 | 529 | **3.97x** | 789.9 | 2893.5 | 0.8175 | 0.8176 | +0.0001 |

### 5.2 FP32 Baseline vs cuMF
Comparison of the standard scalar FP32 baseline against cuMF.

| F (cuMF) | K (FP32) | cuMF (ms/iter) | FP32 (ms/iter) | FP32 Speedup | cuMF GFlops/s | FP32 GFlops/s | cuMF RMSE | FP32 RMSE | Δ RMSE |
|---|---|---|---|---|---|---|---|---|---|
| 20 | 16 | 368 | 242 | **1.52x** | 201.6 | 202.2 | 0.8288 | 0.8311 | +0.0023 |
| 30 | 32 | 500 | 718 | **0.70x** | 318.8 | 251.3 | 0.8241 | 0.8233 | -0.0008 |
| 50 | 48 | 829 | 1587 | **0.52x** | 515.2 | 248.6 | 0.8206 | 0.8205 | -0.0001 |
| 60 | 64 | 1071 | 3708 | **0.29x** | 568.8 | 186.3 | 0.8195 | 0.8191 | -0.0004 |
| 100| 96 | 2101 | 7723 | **0.27x** | 789.9 | 198.3 | 0.8175 | 0.8176 | +0.0001 |

*Reading guide:*
* *test-RMSE is apples-to-apples: ALL rows use weighted-lambda ALS-WR.*
* *cuMF rank F is the nearest multiple of 10 to K, so tiny test-RMSE differences across the K/F pair are partly due to the rank difference.*

