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
#define XB 32
#define YB 256
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
