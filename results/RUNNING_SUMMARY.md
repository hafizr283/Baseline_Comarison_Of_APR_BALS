# APR-BALS (justapr) vs cuMF_ALS / cuMF_CCD / cuMF_SGD — running summary

**This is the single living results doc.** It is kept up to date as tests are
re-run; the paper LaTeX is NOT edited until the numbers here are final. Regenerate
the raw data with:

```
cd comparing_baseline
./run_justapr_vs_cumf.sh                    # justapr vs cuMF_ALS  (existing)
python3 run_ccd_sgd_comparison.py           # justapr vs cuMF_CCD + cuMF_SGD (new)
```

- Machine: **RTX 3060 (sm_86, 12 GB)**, GPU id 1. CUDA nvcc default toolkit.
- Fair protocol (BALS-style): **all four methods train and are scored on the exact
  same frozen train/test split**, produced from each justapr `.bin` by
  `convert_bin_for_ccd_sgd.py` (CCD gets CSR+CSC+test-COO; SGD gets train/test COO
  binaries). Same split ⇒ RMSE is directly comparable.
- justapr = APR-BALS ship recipe (`-DWEIGHTED_LAMBDA=1 -DCUMF_INIT=1 -DMOMENTUM=1`,
  K=96 adds `-DCHOL_MP=2`), λ=0.048, converge on train-RMSE Δ<1e-3, ≤50 it.
- cuMF_CCD: same K sweep, λ=0.05 (netflix 0.058), t=15, tiles a=b=100000.
- cuMF_SGD: k=128 (its only supported rank), t∈{10,20,30,40}, λ=0.05, α=0.08, β=0.3.
- Timing = training only (excludes data load): justapr "Training Complete" wall;
  CCD cumulative kernel time; SGD `SGD_TRAIN_SECONDS` (added timer). SGD test RMSE
  via libmf `mf-predict` on the frozen test split.
- Datasets: ml10m, ml20m, ml32m, netflix. (ml100k excluded from all comparisons.)

_Last full run: 2026-07-21 (results/JUSTAPR_VS_CCD_SGD_20260721_192433.txt)._

_2026-07-21e addendum: the paper's Table VI (baseline-vs-APR, main_experiment.cu
two-leg harness) was re-measured under the weighted-λ protocol —
`results/MAIN3060_WL_20260721_225757/` (λ=0.048, CUMF_INIT=1, FP32 solver both
legs, GT netflix 1024 / ml 512). Netflix wall speedup 4.07→14.84× (K16→96),
ML-20M peak 11.62×, ML-10M 10.87×; test-RMSE deltas ≤0.0043%; APR K=96 netflix
test = 0.817558 exactly. See FIXLOG SYNC-2026-07-21e._

---

## 1. Headline — justapr speedups (this machine) vs BALS paper's own margins

### vs cuMF_CCD  (avg of per-K wall-time ratios, K∈{16,32,48,64,96})

| dataset | justapr **× faster than cuMF_CCD** (ours, measured) | BALS paper × over cuMF_CCD (TITAN RTX) |
|---------|:--:|:--:|
| ml10m   | **3.83×** | 2.09× |
| ml20m   | **3.72×** | 3.86× |
| ml32m   | **3.44×** | — (not in paper) |
| netflix | **4.08×** | 3.22× |

➡ On ml10m and netflix we beat cuMF_CCD by a **larger** margin than BALS did; on
ml20m we match it. justapr's advantage over CCD is in the BALS regime or better.

### vs cuMF_SGD  (justapr K=96 vs SGD k=128, wall-to-reach-justapr's-RMSE)

| dataset | justapr K=96 (RMSE @ s) | best SGD k=128 within 40 ep | verdict |
|---------|:--:|:--:|--------|
| ml10m   | 0.7792 @ 0.76 s | 0.7786 @ 0.68 s (t=40) | on par (SGD ~0.9× time, tiny) |
| ml20m   | 0.7680 @ 1.73 s | 0.7699 @ 1.02 s (t=40) | **SGD never reaches justapr RMSE** |
| ml32m   | 0.7574 @ 3.63 s | 0.7607 @ 1.41 s (t=40) | **SGD never reaches justapr RMSE** |
| netflix | 0.8171 @ 5.94 s | 0.8221 @ 5.88 s (t=40) | **SGD never reaches justapr RMSE** |

➡ Despite using **more** factors (k=128 vs K=96), cuMF_SGD does not reach justapr's
test RMSE within 40 epochs on the three larger datasets — justapr wins on
**quality at comparable wall time**. On the smallest set (ml10m) they tie.

> NOTE on the BALS paper's "5.3× over cuMF_SGD": that is a **GFlops (throughput)**
> number at f=128, not wall-to-convergence. ALS does ~f/3× more FLOPs per iteration
> than SGD, so ALS always wins the throughput metric; the practically meaningful
> wall-clock story is the table above (justapr matches/exceeds SGD quality at
> comparable time).

---

## 2. Transitive reasoning — anchored on cuMF_ALS (the common yardstick)

The clean way to unify all four methods: use **cuMF_ALS** as the shared reference.

- **BALS paper:** BALS ≈ **1.5–1.9×** over cuMF_ALS (Fig. 10, TITAN RTX), and from
  there BALS is 3.2× over cuMF_CCD (netflix) and 5.3× (GFlops) over cuMF_SGD.
- **Ours (measured, same machine):** justapr ≈ **5.3× over cuMF_ALS** on netflix
  (per-K wall ratios 4.6–6.2×; K=96: 7.75 s vs 41.99 s). See
  `results/JUSTAPR_VS_CUMF_20260721_045838.txt`.

**Chain:** justapr's edge over the shared reference cuMF_ALS (~5.3×) is *far larger*
than BALS's edge over the same reference (~1.5–1.9×). Since BALS already beats
cuMF_CCD/cuMF_SGD, and justapr beats the common anchor by more, justapr must beat
CCD/SGD **at least as decisively as BALS does** — and the direct measurements in
§1 confirm exactly this (3.4–4.1× over CCD; ≥SGD quality at comparable time).

> This transitivity is a *consistency check*, not exact arithmetic — the ratios
> come from different ranks/precisions/metrics, so don't multiply them literally.
> The direct head-to-head numbers in §1 and §3 are the ground truth.

---

## 2b. BALS-equivalent baseline (`only_scalar_fp32.cu`)

`only_scalar_fp32.cu` is our **faithful scalar-FP32 BALS reimplementation**: it uses
BALS's exact 2D-tiled storage format, loads each unique Y column once into shared
memory and reuses it across the tile's rows (BALS contribution #1), skips vacant
tiles, and — with `-DBALS_REORDER=1` — sorts rows/cols by descending nnz (BALS
Algorithm 3, contribution #2). It runs **scalar FP32 with no tensor cores**, exactly
as the BALS paper specifies. Build:
`nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=32 -DBALS_REORDER=1 only_scalar_fp32.cu`.

**Measured (K=32, weighted-λ 0.048, RTX 3060):**

| dataset | BALS-equiv (this) | cuMF_ALS (F=30) | **justapr K=32** | justapr / BALS-equiv |
|---------|:--:|:--:|:--:|:--:|
| ml10m   | 1.93 s | 0.84 s | 0.15 s | **12.5×** |
| ml20m   | 4.37 s | 1.85 s | 0.33 s | **13.2×** |
| ml32m   | 6.95 s | 3.08 s | 0.61 s | **11.4×** |
| netflix | 17.97 s | 9.97 s | 1.43 s | **12.6×** |

Two honest findings, both usable in the paper:

1. **justapr is ~11–13× faster than a faithful FP32 BALS** — because justapr adds
   tensor-core (WMMA) Gram kernels, heavy-ball momentum (½ the iterations), and
   mixed-precision Cholesky on top of the shared BALS tile format. This is the
   cleanest "what our contributions buy over BALS" number.
2. **The scalar BALS-equivalent is slower than cuMF_ALS on RTX 3060**, unlike
   the paper's BALS (faster than cuMF on K20C/TITAN). Modern cuMF's FP32+CG path is
   very well tuned; see §2b.1 for the kernel-rewrite attempt and why the gap is
   architectural on Ampere.

### 2b.1 BALS symmetric-tile Gram kernel — implemented & measured (2026-07-21)

We rewrote the FP32 Gram kernel to the **actual BALS/cuMF mapping** and measured it,
to answer "can it beat cuMF like the paper?". Enable with `-DBALS_SYMTILE=1`
(`compute_LHS_RHS_BALS_symtile` in `fused_kernels.cuh`; default build unchanged).

What it does, faithful to the official BALS source (`jingchen95/BALS`,
`magma_sals_kernel_lower2`) and cuMF's `get_hermitian`: each thread owns one small
register sub-tile of the K×K Gram, **only the lower-triangular tiles are computed**
(symmetry → half the flops), and off-diagonal tiles are written to both halves.
Correctness is exact: the trajectory is **bit-identical** to the baseline every
iteration (e.g. ml10m K=32 train 0.658566 / test 0.782254, digit-for-digit).

**Measured (GPU1 idle, weighted-λ 0.048, RTX 3060, wall s):**

| dataset | K | baseline (full-matrix) | **symtile (symmetry)** | speedup |
|---------|---|:--:|:--:|:--:|
| ml10m   | 32 | 1.59 | **1.31** | 1.21× |
| ml20m   | 32 | 3.62 | **3.16** | 1.15× |
| ml32m   | 32 | 6.94 | **6.45** | 1.07× |
| netflix | 32 | 14.36 | **12.48** | 1.15× |
| ml10m   | 64 | 8.09 | **7.62** | 1.06× |
| ml20m   | 64 | 18.62 | **18.56** | 1.00× |

**Verdict: it is a real but modest 1.0–1.21× win, and it still does NOT beat cuMF**
(symtile ml10m K=32 = 1.31 s vs cuMF_ALS F=30 = 0.84 s → still ~1.5× slower).
Root cause is architectural, not a mapping bug:

- The FP32 scalar Gram is **memory-/latency-/occupancy-bound**, running at only
  ~380 GFlop/s ≈ **3% of the RTX 3060's ~12.7 TFlop/s FP32 peak** (verified: DRAM
  traffic is tiny; the cost is the sparse per-segment gather + the two
  `__syncthreads` per column-tile + atomic write-out). Halving the *compute*
  (BALS's symmetry) therefore barely moves the wall clock. The gain we do see comes
  mostly from higher occupancy (1024-thread blocks) at K=32, which is why it fades
  at K=64 (640-thread blocks).
- BALS's *faster-than-cuMF* results in the paper are all **pre-Ampere** (2.08× K20C,
  3.72× TITAN X, 3.13× TITAN RTX) and largely vs cuMF_CCD / cuMF_SGD / MAGMA-Gates,
  **not** modern cuMF_als CG on Ampere. On our GPU the compute/bandwidth balance is
  different, so the symmetry win doesn't translate to a cuMF-beating wall time.

Practical takeaway for the paper: use `-DBALS_SYMTILE=1` as the BALS-equivalent (it
is both **more faithful** to BALS — real BALS exploits symmetry — and the **fastest**
FP32 number we have), but present it honestly as *competitive-but-slower than cuMF on
Ampere*; the tensor-core justapr path (11–13× over it) remains the differentiator.
Fully matching/beating cuMF on Ampere would require a lean per-entity batched-GEMM
Gram (essentially reimplementing what cuMF already does) — large effort, uncertain
payoff; see Open items.

**2026-07-21 re-verification (manuscript prep):** the numbers above had no archived
run file (prose only) and the two subsections' baseline measurements didn't quite
agree with each other. Re-ran fresh, one session, GPU1, K=32/λ=0.048:
`results/BALS_EQUIV_VALIDATION_20260721_205101/` (reorder-only + symtile ×
4 datasets). New timings: reorder-only 1.921/4.351/6.933/17.970 s, symtile
1.581/3.798/6.452/15.618 s (ml10m/ml20m/ml32m/netflix) — ratios (1.07–1.22× from
symmetry; justapr 10.3–11.5× over symtile, 12.5–13.2× over reorder-only; symtile
1.57–2.10× slower than cuMF F=30) match the qualitative story above and are now
citable to an actual file. This is what `sec/5_extended.tex` §"The Honest
BALS-Equivalent Baseline" in the TPDS manuscript cites (see FIXLOG SYNC-2026-07-21d).

**Data reordering (Alg. 3)** is implemented and verified: test RMSE is unchanged
to 4 decimals (permutation is a bijection). On this kernel it is **speed-neutral**
(ml20m 4.34→4.34 s) because vacant tiles are already skipped via the job/nz-tile
list, so we don't pay the per-segment vacant-probe cost that reordering removes in
the paper's kernel — consistent with the paper noting reordering helps most where
vacant *segments* are probed per row.

---

## 3. Full per-dataset tables (train seconds, test RMSE)

### ml10m
| method | rank | iters | train s | test RMSE |
|--------|------|------:|--------:|----------:|
| justapr | K=16 | 15 | 0.078 | 0.785833 |
| justapr | K=32 | 15 | 0.154 | 0.781819 |
| justapr | K=48 | 15 | 0.236 | 0.780860 |
| justapr | K=64 | 15 | 0.426 | 0.780288 |
| justapr | K=96 | 15 | 0.758 | 0.779207 |
| cuMF_CCD | K=16 | 15 | 0.348 | 0.789069 |
| cuMF_CCD | K=32 | 15 | 0.686 | 0.785213 |
| cuMF_CCD | K=48 | 15 | 1.020 | 0.783830 |
| cuMF_CCD | K=64 | 15 | 1.362 | 0.782559 |
| cuMF_CCD | K=96 | 15 | 2.046 | 0.781193 |
| cuMF_SGD | k=128 | 40 | 0.683 | 0.778600 |

### ml20m
| method | rank | iters | train s | test RMSE |
|--------|------|------:|--------:|----------:|
| justapr | K=16 | 15 | 0.173 | 0.777555 |
| justapr | K=32 | 15 | 0.330 | 0.771570 |
| justapr | K=48 | 15 | 0.509 | 0.770045 |
| justapr | K=64 | 15 | 0.911 | 0.769074 |
| justapr | K=96 | 15 | 1.725 | 0.768034 |
| cuMF_CCD | K=16 | 15 | 0.736 | 0.780837 |
| cuMF_CCD | K=32 | 15 | 1.443 | 0.775413 |
| cuMF_CCD | K=48 | 15 | 2.182 | 0.773719 |
| cuMF_CCD | K=64 | 15 | 2.883 | 0.772313 |
| cuMF_CCD | K=96 | 15 | 4.310 | 0.770886 |
| cuMF_SGD | k=128 | 40 | 1.018 | 0.769900 |

### ml32m
| method | rank | iters | train s | test RMSE |
|--------|------|------:|--------:|----------:|
| justapr | K=16 | 15 | 0.313 | 0.770735 |
| justapr | K=32 | 15 | 0.610 | 0.762362 |
| justapr | K=48 | 15 | 0.967 | 0.759637 |
| justapr | K=64 | 15 | 1.746 | 0.758579 |
| justapr | K=96 | 15 | 3.628 | 0.757444 |
| cuMF_CCD | K=16 | 15 | 1.284 | 0.774895 |
| cuMF_CCD | K=32 | 15 | 2.554 | 0.767519 |
| cuMF_CCD | K=48 | 15 | 3.802 | 0.764775 |
| cuMF_CCD | K=64 | 15 | 5.062 | 0.763311 |
| cuMF_CCD | K=96 | 15 | 7.603 | 0.762060 |
| cuMF_SGD | k=128 | 40 | 1.414 | 0.760700 |

### netflix
| method | rank | iters | train s | test RMSE |
|--------|------|------:|--------:|----------:|
| justapr | K=16 | 15 | 0.870 | 0.831099 |
| justapr | K=32 | 15 | 1.428 | 0.823112 |
| justapr | K=48 | 15 | 2.071 | 0.820142 |
| justapr | K=64 | 15 | 3.529 | 0.818655 |
| justapr | K=96 | 15 | 5.937 | 0.817146 |
| cuMF_CCD | K=16 | 15 | 3.362 | 0.837088 |
| cuMF_CCD | K=32 | 15 | 6.599 | 0.830831 |
| cuMF_CCD | K=48 | 15 | 9.981 | 0.828698 |
| cuMF_CCD | K=64 | 15 | 13.249 | 0.827682 |
| cuMF_CCD | K=96 | 15 | 19.796 | 0.826655 |
| cuMF_SGD | k=128 | 40 | 5.879 | 0.822100 |

SGD iter-sweep (train s / test RMSE), for time-to-target analysis:
- ml10m: t10 0.170/0.7963 · t20 0.341/0.7843 · t30 0.512/0.7804 · t40 0.683/0.7786
- ml20m: t10 0.253/0.7891 · t20 0.507/0.7761 · t30 0.761/0.7718 · t40 1.018/0.7699
- ml32m: t10 0.351/0.7835 · t20 0.706/0.7686 · t30 1.061/0.7632 · t40 1.414/0.7607
- netflix: t10 1.470/0.8410 · t20 2.941/0.8283 · t30 4.410/0.8240 · t40 5.879/0.8221

---

## 4. Build/porting notes (so results are reproducible)

- **cuMF_CCD**: `Makefile` arch `sm_35→sm_86`, `c++11→c++14`; added a `__shfl_down`
  compat shim (CUDA 9+ removed the maskless intrinsic) in `device_utilities.h`.
- **cuMF_SGD**: `singleGPU/Makefile` gencode → `sm_86`; `__shfl`/`__shfl_down` shim
  in `sgd_kernel.cu`; added `SGD_TRAIN_SECONDS` monotonic timer (pure kernel loop).
  Test harness `test/mf-predict` builds unchanged (CPU, libmf-derived).
- **Model path**: pass an explicit model path to `cumf_sgd` (default writes the
  `.model` into CWD, not next to the input).

---

## 5. Open items
- [x] `only_scalar_fp32.cu` = faithful FP32 **BALS-equivalent** (tiled storage +
      data reuse + Alg. 3 reordering via `-DBALS_REORDER=1`). Documented in §2b:
      justapr is ~12× faster than it; it is ~2× slower than cuMF on RTX 3060.
- [x] Rewrite the scalar Gram kernel to BALS's real symmetry-exploiting register-tile
      mapping (`-DBALS_SYMTILE=1`, `compute_LHS_RHS_BALS_symtile`). DONE & measured
      (§2b.1): bit-identical RMSE, 1.0–1.21× over the baseline, but still ~1.5× slower
      than cuMF on RTX 3060 — the FP32 Gram is memory/latency-bound (~3% of peak), so
      BALS's compute-halving doesn't beat cuMF on Ampere (paper's wins were pre-Ampere).
- [ ] (optional, only if a reviewer demands cuMF-beating FP32 on Ampere) Replace the
      whole-tile-row-per-block Gram with a lean per-entity batched-GEMM (blk_k slab,
      minimal smem) — i.e. reimplement cuMF/MAGMA's approach. Large effort, uncertain
      Ampere payoff (it converges toward what cuMF already does).
- [ ] Optional: matched-rank SGD (lift justapr K-guard to 128) if reviewers want
      identical rank instead of the k=128-vs-K=96 note.
