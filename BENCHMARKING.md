# APR-BALS Benchmarking & Paper Plan

Goal: make the "we are faster, with no accuracy loss" claim **unattackable by a reviewer**.
The whole strategy rests on one idea:

> **Never compare wall-clock across machines. Compare a RATIO on ONE machine, and
> compare ACCURACY as an absolute number (it is machine-independent).**

---

## 1. Two axes — separate them, and the "machine-dependent" critique dies

**Axis A — Accuracy (test RMSE): 100% machine-independent.**
RMSE at a given K on a given split is a pure number; a 3060, a T4, and a CPU all
produce the same value. So RMSE can sit in a table next to *anyone's* published
number **without running their code** — provided the split protocol matches (see §4).

**Axis B — Speed: report the RATIO on fixed hardware, never raw seconds as the claim.**
"X× faster than baseline B on the same GPU" holds everything constant except the
technique. The machine cancels. Reproduce the ratio on ≥2 GPUs (T4 + 3060 — you
already do) and the hardware objection is gone.

---

## 2. Pillar 1 — self-relative (this is the CONTRIBUTION; already built)

APR-BALS vs FP32 BALS in the **same binary, same machine, same data, same
everything except the precision-dispatch**. Only one variable changes → cleanest
possible controlled experiment.

- Claim: **"up to ~30× compute / ~27× wall speedup at ≤0.1% test-RMSE change."**
- Evidence you already have: Netflix K=64 on T4 → 30.47× compute, 27.69× wall,
  test ΔRMSE +0.0834%. ML-20M + Netflix threshold sweeps flat in accuracy.
- Reproduced on T4 (sm_75) and 3060 (sm_86) → machine-independent ratio.

This alone is publishable. Pillar 2 only adds "…and faster than *other people's* code."

---

## 3. Pillar 2 — external baseline (this is what earns "better than others")

**Do NOT reimplement a paper yourself** — a reviewer will say you crippled it.
Run someone's *released, optimized* code on *your* GPU, same `.bin`, same K.

### 3a. PRIMARY: PyTorch GPU ALS baseline (`baseline_als_gpu.py`)
- Implements the identical closed-form ALS update as APR-BALS.
- Uses PyTorch standard GPU ops (cuBLAS matmul, cuSolver batched solve).
- No custom kernels, no mixed precision, no sparse-structure exploitation.
- Reads our exact frozen `.bin` split → RMSE is directly comparable.
- K can be anything (16, 32, 64) — no alignment constraint.
- Run: `python3 baseline_als_gpu.py netflix_ratings.bin --K 32 --lam 0.1`
- Paper description: *"GPU ALS using standard dense BLAS operations (PyTorch/
  cuBLAS), without exploiting the tiled sparse structure of the rating matrix
  or mixed-precision tensor cores."*

### 3b. OPTIONAL: cuMF_als (Tan et al., HPDC 2016)
- NVIDIA open-source explicit GPU ALS — the most prestigious external baseline.
- **BLOCKER: cuMF requires rank F to be a multiple of 10.** K=16,32,64 are all
  invalid. Nearest valid comparisons: K=20 (vs our 16), K=30 (vs our 32),
  K=40 or K=60 (vs our 64). RMSE at nearby K is similar but not identical.
- Build: `./setup_cumf.sh netflix_ratings.bin 20`
- If cuMF compiles (CUDA 11+ sometimes breaks it), use K=20/30/40 results as a
  third column in the comparison table and note the K offset.
- If cuMF does NOT compile: `baseline_als_gpu.py` is the comparison. State this
  honestly in the paper ("we compare against a GPU ALS reference implementation
  using standard library operations").

### 3c. `implicit` — do NOT use for RMSE comparison
- Optimizes the confidence-weighted implicit-feedback objective, not explicit
  RMSE. Raw dot-product predictions differ in scale → unfair win. Use only for
  throughput reference if at all.

---

## 4. The protocol trap (this is what gets papers rejected)

- Your 80/20 random split is **NOT** the canonical Netflix probe/qualifying set.
  → You may **not** claim you beat the Netflix Prize (0.8567). Different split.
- **Good news: your split is already frozen** — it is serialized inside the `.bin`
  (train triplets + test triplets). Every run (baseline, APR, cuMF via the loader)
  reads the *identical* arrays. So "freeze the split" is already satisfied; just
  make the external baseline read the same `.bin` (see `bench_eval_rmse.py`).
- Rule: when you quote any external RMSE, either use their exact split OR re-run
  their code on your split. Never mix protocols in one table.

---

## 5. Metrics to report (so speed isn't hand-wavy)

1. **Time-to-target-RMSE** (primary, fairest) — fix a test RMSE, measure seconds
   to reach it. Folds in per-iter cost AND convergence rate.
2. **Per-iteration time × iterations-to-converge.**
3. **Throughput (GFlops/s) and % of theoretical peak** — your best *hardware-
   normalized* number. "APR hits X% of FP16 tensor peak vs baseline's Y% of FP32
   peak." Blunts any residual machine-dependence worry.
4. **Test RMSE at convergence** (Axis A, machine-independent).

---

## 6. Accuracy story (from the λ sweep — already run)

- APR matches the FP32 baseline to **<0.1%** at every K (K64: baseline 1.0737 vs
  APR 1.0746). The claim is *"identical to FP32 reference,"* not *"best RMSE ever."*
- **λ does not rescue high K**: even λ=1.0, K48/K64 generalize worse than K32.
  Best generalization is **K=16/K=32** (Netflix test ~0.86–0.88). This is a
  property of the *data*, and the baseline overfits identically — so it is NOT a
  weakness of APR-BALS. Frame it that way.
- Recommended cited points: **K=16, K=32** for accuracy; **K=64** for peak speedup
  (note baseline+APR converge to the same overfit RMSE, delta negligible).

---

## 7. Roofline table template (fill from your own logs)

| K | Path | GFlops/s (APR) | GFlops/s (FP32) | % FP16 peak | % FP32 peak | Speedup |
|---|------|----------------|-----------------|-------------|-------------|---------|
| 16 | dense/scalar | … | … | … | … | … |
| 32 | dense/scalar | … | … | … | … | … |
| 48 | dense/scalar | … | … | … | … | … |
| 64 | dense/scalar | … | … | … | … | … |
| 96 | dense/scalar | … | … | … | … | … |

Peak refs (fill for your card): 3060 FP16 tensor ≈ 51 TFLOPs, FP32 ≈ 12.7 TFLOPs;
T4 FP16 tensor ≈ 65 TFLOPs, FP32 ≈ 8.1 TFLOPs. % of peak shows you are extracting
real hardware efficiency, not just winning on a slow baseline.

---

## 8. Paper tables to produce

- **T1 Accuracy:** dataset × K → {baseline test RMSE, APR test RMSE, Δ%}. Shows ≤0.1%.
- **T2 Speed:** dataset × K → {baseline s, APR s, compute×, wall×} on 3060 AND T4.
- **T3 vs cuMF:** dataset × K → {cuMF time-to-RMSE, APR time-to-RMSE, RMSE both}.
- **T4 Roofline:** §7.
- **T5 Ablation:** effect of the precision-dispatch threshold (you have the sweep).

Deliver T1+T2+T4+T5 from data you already have. T3 is the one new run (cuMF).
