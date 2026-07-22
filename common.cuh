#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <random>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#ifdef USE_CUSOLVER
#include <cusolverDn.h>
#endif

using namespace std;
using namespace nvcuda;

#define TILE_SIZE 32
#ifndef XB
#define XB 32          // BALS tile height (rows). Larger XB => more Y-column reuse
#endif                 // per tile (BALS's core win) but more accumulator registers.
#ifndef YB
#define YB 256         // BALS tile width (cols) => shared-mem column budget.
#endif
#define TILES_PER_CHUNK 8
#define TAU1 0.001f

#ifndef K_DIM
#define K_DIM 16
#endif

// ── Supported-K guard ────────────────────────────────────────────────────
// APR-BALS supports EXACTLY K in {16,32,48,64,96}. Two independent reasons:
//   (1) WMMA kernels are dispatched by KT=K/16; only KT in {1,2,3,4,6} are
//       instantiated. A missing KT (e.g. K=80 -> KT=5, K=128 -> KT=8) makes
//       n_dense/n_giant fall to 0 -> every entity routes to the slow scalar
//       path (no error, just ~30x slower and wrong tiering).
//   (2) The scalar kernel's register tiling needs TRp*RR_C >= K with TRp<=32.
//       RR_C only reaches the needed value at the five supported K; anything
//       else (K=80) silently computes a truncated LHS -> wrong RMSE.
// K=80 wasted a full debugging session for exactly this reason. Fail loudly
// at compile time instead. To ADD a K: instantiate the wmma<KT>/wmma_giant<KT>
// templates AND verify RR_C coverage, THEN extend this list. See FIXLOG.md.
#if (K_DIM != 16) && (K_DIM != 32) && (K_DIM != 48) && (K_DIM != 64) && (K_DIM != 96)
#error "Unsupported K_DIM. APR-BALS supports only K in {16,32,48,64,96}. Other values (e.g. 80,128) silently produce a wrong LHS and route all work to the scalar path. See the supported-K guard note in common.cuh / FIXLOG.md."
#endif
#ifndef DENSE_NNZ_THRESH
#define DENSE_NNZ_THRESH 4
#endif
#ifndef GIANT_NNZ_THRESH
#define GIANT_NNZ_THRESH 1024
#endif
#ifndef WARPS_PER_BLOCK
#define WARPS_PER_BLOCK 4
#endif
#ifndef GIANT_WARPS
#define GIANT_WARPS 4
#endif
// Solver split point (measured on Netflix, RTX 3060, 2026-07-04):
//   K=16: thread-per-system batched wins (16.6 vs 21.5 ms/iter user solve —
//         1 KB/thread local matrix suits L1; packed wastes warp lanes).
//   K=32: warp-packed wins 2.1x (106.6 -> 49.7 ms/iter user solve; batched
//         streams 4 KB/thread of locals at ~42 GFlops/s). Lifted the K=32
//         Netflix headline from 5.25x/4.69x to 7.74x/6.47x, ΔRMSE unchanged.
// Override per-build with -DBATCHED_CHOLESKY_MAX_K=<K> (32 = old behavior).
#ifndef BATCHED_CHOLESKY_MAX_K
#define BATCHED_CHOLESKY_MAX_K 16
#endif
#ifndef ENTITY_BATCH_SIZE
#define ENTITY_BATCH_SIZE 60000
#endif

// FAST_RMSE (07-21): convergence-check RMSE runs a warp-per-entity CSR
// kernel with fp16 factor gathers (compute_RMSE_csr_half) instead of the
// random-gather fp32 COO kernel, which measured ~3.8 ns/rating (378 ms/call
// = 16% of the Netflix K=96 iteration). Predictions use the same fp16-rounded
// factors the WMMA Gram path trains with, so the check value differs from
// fp32-exact only in the 4th decimal; the FINAL reported train/test RMSE is
// re-computed with the fp32 COO kernel after convergence. -DFAST_RMSE=0
// restores the legacy per-check path (reproduces pre-07-21 trajectory
// prints digit-for-digit).
// Default ON for K>=32 only: measured Netflix crossover (07-21) — the fast
// path saves 0.04/0.25/0.45/2.46 s at K=32/48/64/96 but LOSES 0.14 s at
// K=16, where the fp16 factor rows are only 32 B (little gather locality to
// recover) and the two extra fp16-convert kernels + warp-per-entity overhead
// dominate. K_DIM arrives as a -D compile macro, so this is a valid #if.
#ifndef FAST_RMSE
#define FAST_RMSE (K_DIM >= 32 ? 1 : 0)
#endif

// ── Mixed-precision tiled Cholesky experiment (SYNC-2026-07-20) ──────────
// GPU-VALIDATED 2026-07-20 (FIXLOG SYNC-2026-07-20b). Measured verdicts:
//   * REQUIRES the weighted-λ ALS-WR protocol (WEIGHTED_LAMBDA=1 + CUMF_INIT,
//     λ≈0.048). Under plain λ=0.1, κ(A) ~ 1e5+ on heavy entities and EVERY
//     16-bit variant diverges catastrophically (measured, all four).
//   * Ship recipe: K=96 → -DCHOL_MP=2 (fp16, no IR): solve −23%, Netflix APR
//     wall 10.59→9.44 s, test RMSE = fp32 record EXACTLY, 0 zeroed solves.
//   * K≤64: keep FP32 (this flag OFF). K=64 gains only −4% (61 giant-item
//     overflows with fp16; bf16 clean but +7e-5 test); K=48 costs +5
//     convergence iters (net SLOWER); K=32 is DESTROYED by giant-item
//     overflow (test 1.17 vs 0.82) at only 0.018% zeroed solves.
//   * Gate lesson: zeroed solves concentrate on the highest-nnz entities —
//     treat ANY nonzero fail count as a red flag, not a percentage.
//   * CHOL_REFINE converges in the weighted regime but eats the speed win
//     (mp+IR is slower than fp32 at K=64, −9% at K=96 vs −24% noref): with
//     conditioning fixed by weighted-λ, refinement is unnecessary.
// CHOL_MP selects the SMEM STORAGE type of the tiled solver's factor tiles.
// The RHS (sb), diagonal inverses (invD) and all register math stay FP32;
// only the K*K tile array — the smem-occupancy limiter — shrinks:
//   0 = FP32 (default). Production kernel cholesky_solve_tiled, UNTOUCHED;
//       the mp kernel below is not even compiled.
//   1 = BF16 tiles (__nv_bfloat16). FP32-range exponent -> no overflow, but
//       only 8 mantissa bits -> CHOL_REFINE defaults ON.
//   2 = FP16 tiles (__half). 11 mantissa bits (more accurate than BF16) but
//       max 65504 -> giant items with diag = sum(f^2) over ~200k ratings CAN
//       overflow. Non-finite solves are zeroed and counted (d_chol_fail,
//       printed per run) — the "YOLO" test: CHOL_REFINE defaults OFF.
// Why: at K=96 the tiled solver is smem-capacity bound (23.6 KB/block -> 4
// blocks/SM at MaxShared) and latency-skeleton bound at that 33% occupancy
// (07-06). 16-bit tiles cut smem to ~12.3 KB -> 8 blocks/SM. NOTE this only
// moves K=96: K=64 is already at 8 blocks/SM (11.1 KB), K=48/K=32 are
// register/launch bound. Occupancy math in FIXLOG SYNC-2026-07-20.
// CHOL_REFINE = 1 adds ONE FP32 iterative-refinement step inside the solve
// kernel: r = b - (A+λI)x re-reading the fp32 LHS from gmem (upper triangle,
// symmetric expansion), then two tile-parallel trisolves with the 16-bit L.
// Costs roughly one extra load phase + ~2 bwd-solve phases per solve.
#ifndef CHOL_MP
#define CHOL_MP 0
#endif
#ifndef CHOL_REFINE
#define CHOL_REFINE (CHOL_MP == 1 ? 1 : 0)
#endif
#if CHOL_MP != 0 && K_DIM == 16
#error "CHOL_MP has no effect at K=16 (K=16 uses the batched FP32 solver). Build K=16 without CHOL_MP."
#endif
#if CHOL_MP == 1
#include <cuda_bf16.h>
typedef __nv_bfloat16 chol_store_t;
#elif CHOL_MP == 2
typedef __half chol_store_t;
#endif

// ── Stale-L cache experiment (SYNC-2026-07-20, wired in justapr.cu ONLY) ──
// GPU-VALIDATED 2026-07-20: MEASURED-DEAD as implemented. At K=64 under the
// weighted-λ protocol every cadence (REFRESH 3/5/8) collapses the model to
// zeros within 2-5 stale iterations (train→3.7643, tens of thousands of
// zeroed solves): the single stale-L IR step diverges per-entity on light/
// fast-drifting entities and the zeroed factors cascade. The per-entity
// residual-gated rescue (implemented behind this flag, τ = CHOL_STALE_TAU)
// was ALSO measured dead 07-20c with a τ-bracket: τ=0.5 → 56% gate-fail AND
// divergence; τ=0.1 → 76% gate-fail, oscillation, wall 15.2 s vs 5.8 s
// fresh. No τ gives both stability and savings — κ(50-200)·per-iter-drift
// ≥ 1 for most entities. Kept compilable for the record only; do not re-run.
// See FIXLOG SYNC-2026-07-20b/c.
// ALS re-factorizes every A_e from scratch each iteration, but after warmup
// Y (and therefore A_u = Y^T Y + λI) changes slowly. CHOL_STALE=1 keeps each
// entity's tiled L (BF16, lower tiles + the L11^-T upper halves = exactly
// the smem tile image, (K/16)(K/16+1)/2*256 elems) in global memory. Fresh
// iterations (iter < CHOL_STALE_WARMUP, or iter % CHOL_STALE_REFRESH == 0)
// run the full mp solve and overwrite the cache; stale iterations skip the
// factorization entirely: x0 = trisolves with the cached L, then ONE FP32
// refinement step against the FRESH A (which the LHS kernels still build
// every iteration — only the O(K^3) factorization is skipped).
// Requires CHOL_MP=1 (the cache format IS the bf16 tile image).
// VRAM: users*elems*2B — Netflix K=64 ≈ 2.46 GB (fits), K=96 ≈ 5.16 GB
// (needs -DENTITY_BATCH_SIZE=30000 to shrink the two LHS buffers). ml20m
// users=138k: trivial. Validate at K=64 first.
#ifndef CHOL_STALE
#define CHOL_STALE 0
#endif
#if CHOL_STALE
#if CHOL_MP != 1
#error "CHOL_STALE requires CHOL_MP=1 (the L cache stores the BF16 tile image)."
#endif
#ifndef CHOL_STALE_WARMUP
#define CHOL_STALE_WARMUP 5
#endif
#ifndef CHOL_STALE_REFRESH
#define CHOL_STALE_REFRESH 5
#endif
// Residual gate for stale iterations (2026-07-20b rescue attempt): a stale
// solve whose x0 residual has ||r|| > TAU*||b|| (or is non-finite) is
// discarded and the entity gets a full mp re-solve + cache refresh in a
// second, list-driven launch. TAU must sit above the bf16 factor's own
// residual noise (~u_bf16*kappa) but below "diverging"; sweep if in doubt.
#ifndef CHOL_STALE_TAU
#define CHOL_STALE_TAU 0.5f
#endif
#endif

// Human-readable solver-precision banner string (printed at startup).
#if CHOL_MP == 0
#define CHOL_MP_STR "FP32 tiled (production)"
#elif CHOL_MP == 1 && CHOL_REFINE
#define CHOL_MP_STR "BF16 tiles + 1-step FP32 refinement"
#elif CHOL_MP == 1
#define CHOL_MP_STR "BF16 tiles, NO refinement"
#elif CHOL_MP == 2 && CHOL_REFINE
#define CHOL_MP_STR "FP16 tiles + 1-step FP32 refinement"
#else
#define CHOL_MP_STR "FP16 tiles, NO refinement (YOLO)"
#endif
