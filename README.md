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

---
**Original Authors:**
- [Wei Tan](https://github.com/wei-tan)
- [Shiyu Chang](https://github.com/code-terminator)
- [Liangliang Cao](https://github.com/llcao)
