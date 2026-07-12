# Modified cuMF for Baseline Comparison

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

Here are the benchmarking results on identical frozen splits comparing **APR-BALS** (our baseline) against the standard FP32 baseline and **cuMF** on the Netflix dataset. 

As shown, the APR-BALS implementation achieves up to **14.6x speedup** over the standard FP32 baseline and up to **6.9x speedup** over cuMF, with virtually identical test RMSE (differences < 0.001).

| rank | code | ms/iter | Speedup (vs FP32) | Speedup (vs cuMF) | train RMSE | test RMSE |
|---|---|---|---|---|---|---|
| **K=16** | **APR-BALS mixed** | **53** | **4.56x** | **6.94x** | 0.782323 | 0.831057 |
| | FP32 baseline | 242 | 1.00x | - | 0.782356 | 0.831081 |
| *(F=20)* | cuMF-ALS | 368 | - | 1.00x | 0.771220 | 0.828840 |
| **K=32** | **APR-BALS mixed** | **97** | **7.40x** | **5.15x** | 0.752649 | 0.823282 |
| | FP32 baseline | 718 | 1.00x | - | 0.752687 | 0.823308 |
| *(F=30)* | cuMF-ALS | 500 | - | 1.00x | 0.753987 | 0.824093 |
| **K=48** | **APR-BALS mixed** | **152** | **10.44x** | **5.45x** | 0.736522 | 0.820479 |
| | FP32 baseline | 1587 | 1.00x | - | 0.736567 | 0.820510 |
| *(F=50)* | cuMF-ALS | 829 | - | 1.00x | 0.734113 | 0.820556 |
| **K=64** | **APR-BALS mixed** | **273** | **13.58x** | **3.92x** | 0.726029 | 0.819080 |
| | FP32 baseline | 3708 | 1.00x | - | 0.726076 | 0.819113 |
| *(F=60)* | cuMF-ALS | 1071 | - | 1.00x | 0.727628 | 0.819547 |
| **K=96** | **APR-BALS mixed** | **529** | **14.60x** | **3.97x** | 0.712923 | 0.817558 |
| | FP32 baseline | 7723 | 1.00x | - | 0.712971 | 0.817593 |
| *(F=100)*| cuMF-ALS | 2101 | - | 1.00x | 0.711658 | 0.817499 |

*Reading guide:*
* *test-RMSE is apples-to-apples: ALL rows use weighted-lambda ALS-WR.*
* *cuMF rank F is the nearest multiple of 10 to K, so tiny test-RMSE differences across the K/F pair are partly the rank difference.*
* *Speedup metrics are calculated using the `ms/iter` timing from the respective baseline.*

---
**Original Authors:**
- [Wei Tan](https://github.com/wei-tan)
- [Shiyu Chang](https://github.com/code-terminator)
- [Liangliang Cao](https://github.com/llcao)
