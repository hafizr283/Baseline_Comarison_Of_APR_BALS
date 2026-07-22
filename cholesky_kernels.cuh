#pragma once
#include "common.cuh"

#ifdef USE_CUSOLVER
__global__ void add_lambda_diag(float* __restrict__ d_LHS_all, int K, int num_entities, float lambda) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)num_entities * K;
    if (idx >= total) return;
    int ent = (int)(idx / K);
    int d   = (int)(idx % K);
    d_LHS_all[(long long)ent * K * K + (long long)d * K + d] += lambda;
}

__global__ void build_ptr_array(float** __restrict__ ptrs, float* __restrict__ base, long long stride, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ptrs[i] = base + (long long)i * stride;
}
#endif

__global__ void cholesky_solve_cooperative(float* __restrict__ d_LHS_all, float* __restrict__ d_X, int K, int num_entities, float lambda,
                                           const int* __restrict__ d_nnz_w = nullptr) {
    int ent = blockIdx.x;
    if (ent >= num_entities) return;
    if (d_nnz_w) { int n = d_nnz_w[ent]; lambda = lambda * (n > 0 ? n : 1); }   // fused weighted-λ (07-21)

    extern __shared__ float sMem[];
    float* sA = sMem;
    float* sb = sMem + K * K;
    int tid = threadIdx.x;

    long long lhs_base = (long long)ent * K * K;
    int rhs_base = ent * K;

    for (int idx = tid; idx < K * K; idx += 32) {
        int r = idx / K, c = idx % K;
        if (r <= c) sA[idx] = d_LHS_all[lhs_base + idx];
    }
    for (int i = tid; i < K; i += 32)     sb[i] = d_X[rhs_base + i];
    __syncwarp();

    for (int i = tid; i < K; i += 32) sA[i * K + i] += lambda;
    __syncwarp();

    for (int j = 0; j < K; j++) {
        float dp = 0.0f;
        for (int k = tid; k < j; k += 32) { float x = sA[k * K + j]; dp += x * x; }
        for (int off = 16; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off);
        if (tid == 0) {
            float s = sA[j * K + j] - dp;
            sA[j * K + j] = (s > 1e-10f) ? sqrtf(s) : 1e-5f;
        }
        __syncwarp();

        float inv = 1.0f / sA[j * K + j];
        int i = j + 1 + tid;
        if (i < K) {
            float v = sA[j * K + i];
            for (int k = 0; k < j; k++) v -= sA[k * K + i] * sA[k * K + j];
            sA[j * K + i] = v * inv;
        }
        if (K > 32) {
            i += 32;
            if (i < K) {
                float v = sA[j * K + i];
                for (int k = 0; k < j; k++) v -= sA[k * K + i] * sA[k * K + j];
                sA[j * K + i] = v * inv;
            }
        }
        if (K > 64) {
            i += 32;
            if (i < K) {
                float v = sA[j * K + i];
                for (int k = 0; k < j; k++) v -= sA[k * K + i] * sA[k * K + j];
                sA[j * K + i] = v * inv;
            }
        }
        __syncwarp();
    }

    for (int i = 0; i < K; i++) {
        float dp = 0.0f;
        for (int j = tid; j < i; j += 32) dp += sA[j * K + i] * sb[j];
        for (int off = 16; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off);
        if (tid == 0) sb[i] = (sb[i] - dp) / sA[i * K + i];
        __syncwarp();
    }
    for (int i = K - 1; i >= 0; i--) {
        float dp = 0.0f;
        for (int j = i + 1 + tid; j < K; j += 32) dp += sA[i * K + j] * sb[j];
        for (int off = 16; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off);
        if (tid == 0) sb[i] = (sb[i] - dp) / sA[i * K + i];
        __syncwarp();
    }

    for (int i = tid; i < K; i += 32) d_X[rhs_base + i] = sb[i];
}

// Packed lower-triangular cooperative Cholesky.
// Stores only K*(K+1)/2 floats in shared memory (vs K*K for cooperative),
// nearly doubling occupancy at K=64 (6 warps/SM → ~11 warps/SM).
// Also eliminates 32-way bank conflicts in the forward/backward solve.
template <int K>
__global__ void cholesky_solve_packed(float* __restrict__ d_LHS_all, float* __restrict__ d_X, int num_entities, float lambda,
                                      const int* __restrict__ d_nnz_w = nullptr) {
    int ent = blockIdx.x;
    if (ent >= num_entities) return;
    if (d_nnz_w) { int n = d_nnz_w[ent]; lambda = lambda * (n > 0 ? n : 1); }   // fused weighted-λ (07-21)

    constexpr int NT = K * (K + 1) / 2;
    extern __shared__ float sMem[];
    float* sA = sMem;       // NT floats — lower triangle, row-major packed
    float* sb = sMem + NT;  // K  floats — RHS

    int tid = threadIdx.x;
    long long lhs_base = (long long)ent * K * K;

    // Load the UPPER triangle (the WMMA kernels no longer write the lower mirror).
    // Enumerate elements in upper-triangle row-major MEMORY order — row j holds
    // columns c = j..K-1, contiguous at [j*K + j] — so consecutive threads read
    // consecutive addresses (coalesced). Scatter into the packed *lower* layout
    // sA[c*(c+1)/2 + j] in shared memory, where scattered writes are free.
    // Elements before row j: U(j) = j*(2K - j + 1)/2.
    for (int idx = tid; idx < NT; idx += 32) {
        int j = (int)(((2 * K + 1) - sqrtf((float)((2 * K + 1) * (2 * K + 1) - 8 * idx))) * 0.5f);
        while (j * (2 * K - j + 1) / 2 > idx) j--;
        while ((j + 1) * (2 * K - j) / 2 <= idx) j++;
        int c = j + (idx - j * (2 * K - j + 1) / 2);
        sA[c * (c + 1) / 2 + j] = d_LHS_all[lhs_base + (long long)j * K + c];
    }
    for (int i = tid; i < K; i += 32) sb[i] = d_X[ent * K + i];
    __syncwarp();

    // Lambda on diagonal: element (i,i) is at packed index i*(i+1)/2 + i = i*(i+3)/2
    for (int i = tid; i < K; i += 32) sA[i*(i+3)/2] += lambda;
    __syncwarp();

    // Cholesky factorization (lower triangular, Cholesky-Banachiewicz)
    for (int j = 0; j < K; j++) {
        int bj = j*(j+1)/2;          // base index for row j in packed storage
        float dp = 0.0f;
        for (int k = tid; k < j; k += 32) { float x = sA[bj + k]; dp += x * x; }
        for (int off = 16; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off);
        if (tid == 0) { float s = sA[bj + j] - dp; sA[bj + j] = (s > 1e-10f) ? sqrtf(s) : 1e-5f; }
        __syncwarp();

        float inv = 1.0f / sA[bj + j];
        // Update sub-diagonal entries of column j: rows i = j+1..K-1
        int i = j + 1 + tid;
        if (i < K) {
            int bi = i*(i+1)/2;
            float v = sA[bi + j];
            for (int k = 0; k < j; k++) v -= sA[bi + k] * sA[bj + k];
            sA[bi + j] = v * inv;
        }
        if (K > 32) {
            i += 32;
            if (i < K) {
                int bi = i*(i+1)/2;
                float v = sA[bi + j];
                for (int k = 0; k < j; k++) v -= sA[bi + k] * sA[bj + k];
                sA[bi + j] = v * inv;
            }
        }
        if (K > 64) {
            i += 32;
            if (i < K) {
                int bi = i*(i+1)/2;
                float v = sA[bi + j];
                for (int k = 0; k < j; k++) v -= sA[bi + k] * sA[bj + k];
                sA[bi + j] = v * inv;
            }
        }
        __syncwarp();
    }

    // Forward solve: L y = b
    for (int i = 0; i < K; i++) {
        int bi = i*(i+1)/2;
        float dp = 0.0f;
        for (int j = tid; j < i; j += 32) dp += sA[bi + j] * sb[j];
        for (int off = 16; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off);
        if (tid == 0) sb[i] = (sb[i] - dp) / sA[bi + i];
        __syncwarp();
    }

    // Backward solve: L^T x = y  (L^T[i][j] = L[j][i], j > i, packed at j*(j+1)/2+i)
    for (int i = K - 1; i >= 0; i--) {
        float dp = 0.0f;
        for (int j = i + 1 + tid; j < K; j += 32) dp += sA[j*(j+1)/2 + i] * sb[j];
        for (int off = 16; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off);
        if (tid == 0) sb[i] = (sb[i] - dp) / sA[i*(i+3)/2];
        __syncwarp();
    }

    for (int i = tid; i < K; i += 32) d_X[ent * K + i] = sb[i];
}

// Multi-warp packed solver: TPS threads per system (TPS multiple of 32).
// Same math and packed layout as cholesky_solve_packed, but the dominant
// factorization row-updates are spread over TPS threads (~1 row each at
// TPS>=K) instead of 3 serial chunks per lane of one warp. The diagonal dp
// and the forward/backward triangular solves stay in warp 0 (shuffle-only
// reductions); other warps wait at the barriers. Bit-identical to packed
// (same per-element summation order). Measured on 60k systems (sm_86):
// K=96: 92 -> 74 ms, K=64: 25.5 -> 23.5 ms, K=48: no gain (keep packed<48>).
// The kernel is smem-capacity bound (K=96: 19KB/system -> 5 systems/SM), so
// bigger TPS only shortens the per-system critical path; TPS=96 is the
// measured optimum for K=64/96.
template <int K, int TPS>
__global__ void cholesky_solve_packed_mw(float* __restrict__ d_LHS_all, float* __restrict__ d_X, int num_entities, float lambda) {
    int ent = blockIdx.x;
    if (ent >= num_entities) return;

    constexpr int NT = K * (K + 1) / 2;
    extern __shared__ float sMem[];
    float* sA = sMem;       // NT floats — lower triangle, row-major packed
    float* sb = sMem + NT;  // K  floats — RHS

    int tid = threadIdx.x;
    long long lhs_base = (long long)ent * K * K;

    // Coalesced upper-triangle read -> packed lower scatter (see packed kernel)
    for (int idx = tid; idx < NT; idx += TPS) {
        int j = (int)(((2 * K + 1) - sqrtf((float)((2 * K + 1) * (2 * K + 1) - 8 * idx))) * 0.5f);
        while (j * (2 * K - j + 1) / 2 > idx) j--;
        while ((j + 1) * (2 * K - j) / 2 <= idx) j++;
        int c = j + (idx - j * (2 * K - j + 1) / 2);
        sA[c * (c + 1) / 2 + j] = d_LHS_all[lhs_base + (long long)j * K + c];
    }
    for (int i = tid; i < K; i += TPS) sb[i] = d_X[ent * K + i];
    __syncthreads();   // loader and lambda-add touch the same diagonal elements from different warps
    for (int i = tid; i < K; i += TPS) sA[i*(i+3)/2] += lambda;
    __syncthreads();

    for (int j = 0; j < K; j++) {
        int bj = j*(j+1)/2;
        if (tid < 32) {   // warp 0: diagonal element
            float dp = 0.0f;
            for (int k = tid; k < j; k += 32) { float x = sA[bj + k]; dp += x * x; }
            for (int off = 16; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off);
            if (tid == 0) { float s = sA[bj + j] - dp; sA[bj + j] = (s > 1e-10f) ? sqrtf(s) : 1e-5f; }
        }
        __syncthreads();
        float inv = 1.0f / sA[bj + j];
        for (int i = j + 1 + tid; i < K; i += TPS) {
            int bi = i*(i+1)/2;
            float v = sA[bi + j];
            for (int k = 0; k < j; k++) v -= sA[bi + k] * sA[bj + k];
            sA[bi + j] = v * inv;
        }
        __syncthreads();
    }

    if (tid < 32) {   // warp 0: forward + backward triangular solves
        for (int i = 0; i < K; i++) {
            int bi = i*(i+1)/2;
            float dp = 0.0f;
            for (int j = tid; j < i; j += 32) dp += sA[bi + j] * sb[j];
            for (int off = 16; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off);
            if (tid == 0) sb[i] = (sb[i] - dp) / sA[bi + i];
            __syncwarp();
        }
        for (int i = K - 1; i >= 0; i--) {
            float dp = 0.0f;
            for (int j = i + 1 + tid; j < K; j += 32) dp += sA[j*(j+1)/2 + i] * sb[j];
            for (int off = 16; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off);
            if (tid == 0) sb[i] = (sb[i] - dp) / sA[i*(i+3)/2];
            __syncwarp();
        }
    }
    __syncthreads();
    for (int i = tid; i < K; i += TPS) d_X[ent * K + i] = sb[i];
}

// ── Tiled blocked right-looking Cholesky + solve (SYNC-2026-07-05) ──────────
// Replaces packed/packed_mw for K>=32 (measured on 60k systems, RTX 3060:
// K32 6.6->3.6 ms, K48 11.5->6.0 ms, K64 22.8->10.9 ms, K96 69.4->30.8 ms =
// 1.86x/1.93x/2.08x/2.26x). The packed kernels are latency/issue bound: 2*K
// block barriers + K serial dot chains per system, plus 2*K warp-serial
// triangular-solve steps, plus an isqrt scatter decode recomputed per element
// per system. This kernel restructures the factorization around 16x16 tiles:
//   * Per 16-column panel: POTRF(16x16, warp 0) | TRSM rows (one thread/row,
//     fully parallel) | SYRK trailing update (2x2 register-blocked dots).
//     Block-wide barriers drop from 2*K to ~3*(K/16).
//   * POTRF(p+1) runs on warp 0 CONCURRENTLY with the bulk SYRK (tile
//     (p+1,p+1) is updated first in a small SYRK-a phase), hiding the only
//     remaining serial phase for every panel but the first.
//   * The forward solve is FOLDED into the panels (bordered-matrix trick:
//     b rides as one extra TRSM row + one SYRK dot per remaining row).
//   * Each panel's TRSM also solves 16 identity rows = rows of L11^-T (upper
//     triangular), scattered into the never-read upper half of the diagonal
//     tile; the backward solve per tile-column is then a parallel matvec
//     instead of 16 warp-serial substitution steps.
//   * d_map holds the host-precomputed load scatter (upper-tri memory index ->
//     tiled smem offset): ldg+ldg+sts instead of ~20 decode instructions.
// Row stride 17 inside tiles kills bank conflicts on column walks. Summation
// order differs from packed (blocked vs column-at-a-time), so results match
// to FP32 reorder noise, NOT bit-identical; baseline and APR use the same
// solver so the comparison stays apples-to-apples.
// Dispatch (measured optimum threads/system): K32 -> 64, K48 -> 96,
// K64/96 -> 128. K=16 keeps cholesky_solve_batched (1 tile = pure overhead).
#define CHOL_TS 16
#define CHOL_TSTR 17
#define CHOL_TFL (CHOL_TS * CHOL_TSTR)

__device__ __forceinline__ int chol_tofs(int R, int C) { return (R * (R + 1) / 2 + C) * CHOL_TFL; }

__device__ __forceinline__ void chol_potrf16_w0(float* sD, float* iD, int lane) {
    for (int j = 0; j < CHOL_TS; j++) {
        float x = (lane < j) ? sD[j * CHOL_TSTR + lane] : 0.0f;
        float dp = x * x;
        for (int off = 8; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off, 16);
        if (lane == 0) {
            float s = sD[j * (CHOL_TSTR + 1)] - dp;
            s = (s > 1e-10f) ? sqrtf(s) : 1e-5f;
            sD[j * (CHOL_TSTR + 1)] = s;
            iD[j] = 1.0f / s;
        }
        __syncwarp();
        float inv = iD[j];
        int i = j + 1 + lane;
        if (i < CHOL_TS) {
            float v = sD[i * CHOL_TSTR + j];
            for (int k = 0; k < j; k++) v -= sD[i * CHOL_TSTR + k] * sD[j * CHOL_TSTR + k];
            sD[i * CHOL_TSTR + j] = v * inv;
        }
        __syncwarp();
    }
}

template <int K>
constexpr int cholesky_tiled_smem() {
    return ((K / CHOL_TS) * (K / CHOL_TS + 1) / 2 * CHOL_TFL + K + (K / CHOL_TS) * CHOL_TS) * (int)sizeof(float);
}

// Host: build the load scatter map, float4-vectorized (SYNC-2026-07-06).
// Upper-triangle row j holds cols c=j..K-1 contiguously at j*K+j. A 4-aligned
// chunk [c0,c0+3] never crosses a 16-tile boundary (4 | 16), so its four smem
// slots are base, +17, +34, +51 — one offset per chunk. Layout: nvec vector
// entries {.x = gmem float4 index, .y = smem base}, then scalar head/tail
// entries {.x = gmem float offset, .y = smem offset}. Alignment holds for all
// supported K (K % 4 == 0 -> row starts j*K are 4-element aligned).
// Same floats land in the same smem slots -> solver output is bit-identical
// to the scalar-map version (verified by memcmp on 60k systems, all K).
inline void build_cholesky_tile_map(int K, std::vector<int2>& map, int& nvec) {
    std::vector<int2> vec, sc;
    for (int j = 0; j < K; j++)
        for (int c = j; c < K; ) {
            int R = c >> 4, C = j >> 4;
            int smem = (R * (R + 1) / 2 + C) * CHOL_TFL + (c & 15) * CHOL_TSTR + (j & 15);
            if ((c & 3) == 0 && c + 3 < K) {
                vec.push_back(make_int2((j * K + c) >> 2, smem));
                c += 4;
            } else {
                sc.push_back(make_int2(j * K + c, smem));
                c += 1;
            }
        }
    nvec = (int)vec.size();
    map = vec;
    map.insert(map.end(), sc.begin(), sc.end());
}

template <int K, int NTHREADS>
__global__ void cholesky_solve_tiled(const float* __restrict__ d_LHS_all, float* __restrict__ d_X,
                                     const int2* __restrict__ d_map, int nvec, int ntot,
                                     int num_entities, float lambda,
                                     const int* __restrict__ d_nnz_w = nullptr) {
    constexpr int NT  = K / CHOL_TS;
    constexpr int NTT = NT * (NT + 1) / 2;

    int ent = blockIdx.x;
    if (ent >= num_entities) return;
    // Fused weighted-λ (07-21): with d_nnz_w set, the per-entity λ·nnz diag
    // add happens here instead of a separate RMW kernel over the 2.2 GB LHS
    // buffer (was 27 ms/iter at K=96). Same fp32 add on the same fp32 value
    // → bit-identical to the two-kernel sequence.
    float lam = lambda;
    if (d_nnz_w) { int n = d_nnz_w[ent]; lam = lambda * (n > 0 ? n : 1); }

    extern __shared__ float sMem[];
    float* sA   = sMem;                    // NTT tiles, stride-17 rows
    float* sb   = sMem + NTT * CHOL_TFL;   // K   — RHS / solution
    float* invD = sb + K;                  // NT*16 — all diagonal inverses

    const int tid = threadIdx.x;
    const long long lhs_base = (long long)ent * K * K;
    const float4* src4 = (const float4*)(d_LHS_all + lhs_base);

    for (int idx = tid; idx < ntot; idx += NTHREADS) {
        int2 m = d_map[idx];
        if (idx < nvec) {   // float4 chunk: 4 row-consecutive elements, smem stride 17
            float4 v = src4[m.x];
            sA[m.y]                 = v.x;
            sA[m.y +     CHOL_TSTR] = v.y;
            sA[m.y + 2 * CHOL_TSTR] = v.z;
            sA[m.y + 3 * CHOL_TSTR] = v.w;
        } else {            // scalar head/tail element
            sA[m.y] = d_LHS_all[lhs_base + m.x];
        }
    }
    for (int i = tid; i < K; i += NTHREADS) sb[i] = d_X[ent * K + i];
    __syncthreads();
    for (int i = tid; i < K; i += NTHREADS)
        sA[chol_tofs(i >> 4, i >> 4) + (i & 15) * (CHOL_TSTR + 1)] += lam;
    __syncthreads();

    if (tid < 32) chol_potrf16_w0(sA, invD, tid);   // prologue: POTRF(0)
    __syncthreads();

    for (int p = 0; p < NT; p++) {
        float* sD = sA + chol_tofs(p, p);
        float* iD = invD + p * CHOL_TS;

        // TRSM(p): panel rows + b-row (t==nrows) + 16 identity rows (t>nrows,
        // solutions = rows of L11^-T -> upper half of the diagonal tile).
        int nrows = (NT - 1 - p) * CHOL_TS;
        for (int t = tid; t < nrows + 1 + CHOL_TS; t += NTHREADS) {
            float a[CHOL_TS];
            float* row = nullptr;
            int irow = t - nrows - 1;
            if (t < nrows) {
                row = sA + chol_tofs(p + 1 + (t >> 4), p) + (t & 15) * CHOL_TSTR;
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++) a[c] = row[c];
            } else if (t == nrows) {
                row = sb + p * CHOL_TS;
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++) a[c] = row[c];
            } else {
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++) a[c] = (c == irow) ? 1.0f : 0.0f;
            }
            #pragma unroll
            for (int c = 0; c < CHOL_TS; c++) {
                float v = a[c];
                #pragma unroll
                for (int k = 0; k < c; k++) v -= a[k] * sD[c * CHOL_TSTR + k];
                a[c] = v * iD[c];
            }
            if (row) {
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++) row[c] = a[c];
            } else {
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++)
                    if (c > irow) sD[irow * CHOL_TSTR + c] = a[c];
            }
        }
        __syncthreads();

        if (p + 1 < NT) {
            // SYRK-a: only what POTRF(p+1) needs — tile(p+1,p+1) + b-segment.
            const float* Pn = sA + chol_tofs(p + 1, p);
            float* Dn = sA + chol_tofs(p + 1, p + 1);
            for (int e = tid; e < 64; e += NTHREADS) {
                int r0 = (e >> 3) * 2, c0 = (e & 7) * 2;
                const float *a0 = Pn + r0 * CHOL_TSTR, *a1 = a0 + CHOL_TSTR;
                const float *b0 = Pn + c0 * CHOL_TSTR, *b1 = b0 + CHOL_TSTR;
                float s00 = 0, s01 = 0, s10 = 0, s11 = 0;
                #pragma unroll
                for (int k = 0; k < CHOL_TS; k++) {
                    float x0 = a0[k], x1 = a1[k], y0 = b0[k], y1 = b1[k];
                    s00 += x0 * y0; s01 += x0 * y1; s10 += x1 * y0; s11 += x1 * y1;
                }
                Dn[r0 * CHOL_TSTR + c0] -= s00;       Dn[r0 * CHOL_TSTR + c0 + 1] -= s01;
                Dn[(r0 + 1) * CHOL_TSTR + c0] -= s10; Dn[(r0 + 1) * CHOL_TSTR + c0 + 1] -= s11;
            }
            for (int i = (p + 1) * CHOL_TS + tid; i < (p + 2) * CHOL_TS; i += NTHREADS) {
                const float* Lr = Pn + (i & 15) * CHOL_TSTR;
                float dp = 0.0f;
                #pragma unroll
                for (int k = 0; k < CHOL_TS; k++) dp += Lr[k] * sb[p * CHOL_TS + k];
                sb[i] -= dp;
            }
            __syncthreads();

            // warp 0: POTRF(p+1)  ∥  tid>=32: SYRK-b (remaining trailing tiles,
            // 2x2 register-blocked) + b-tail rows.
            if (tid < 32) {
                chol_potrf16_w0(sA + chol_tofs(p + 1, p + 1), invD + (p + 1) * CHOL_TS, tid);
            } else {
                int m = NT - 1 - p;
                int wtid = tid - 32, wn = NTHREADS - 32;
                for (int e = wtid; e < m * (m + 1) / 2 * 64 - 64; e += wn) {
                    int t = (e >> 6) + 1;   // skip tile 0 = (p+1,p+1), done in SYRK-a
                    int r0 = ((e >> 3) & 7) * 2, c0 = (e & 7) * 2;
                    int Rp = 0;
                    while ((Rp + 1) * (Rp + 2) / 2 <= t) Rp++;
                    int Cp = t - Rp * (Rp + 1) / 2;
                    const float* P1 = sA + chol_tofs(p + 1 + Rp, p);
                    const float* P2 = sA + chol_tofs(p + 1 + Cp, p);
                    const float *a0 = P1 + r0 * CHOL_TSTR, *a1 = a0 + CHOL_TSTR;
                    const float *b0 = P2 + c0 * CHOL_TSTR, *b1 = b0 + CHOL_TSTR;
                    float s00 = 0, s01 = 0, s10 = 0, s11 = 0;
                    #pragma unroll
                    for (int k = 0; k < CHOL_TS; k++) {
                        float x0 = a0[k], x1 = a1[k], y0 = b0[k], y1 = b1[k];
                        s00 += x0 * y0; s01 += x0 * y1; s10 += x1 * y0; s11 += x1 * y1;
                    }
                    float* D = sA + chol_tofs(p + 1 + Rp, p + 1 + Cp);
                    D[r0 * CHOL_TSTR + c0] -= s00;       D[r0 * CHOL_TSTR + c0 + 1] -= s01;
                    D[(r0 + 1) * CHOL_TSTR + c0] -= s10; D[(r0 + 1) * CHOL_TSTR + c0 + 1] -= s11;
                }
                for (int i = (p + 2) * CHOL_TS + wtid; i < K; i += wn) {
                    const float* Lr = sA + chol_tofs(i >> 4, p) + (i & 15) * CHOL_TSTR;
                    float dp = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < CHOL_TS; k++) dp += Lr[k] * sb[p * CHOL_TS + k];
                    sb[i] -= dp;
                }
            }
            __syncthreads();
        }
    }

    // Backward solve: per tile-column, x_C = L(C,C)^-T y_C as one parallel
    // matvec (diag from invD + upper-half rows), then rows above update.
    for (int C = NT - 1; C >= 0; C--) {
        if (tid < 32) {
            const float* sD = sA + chol_tofs(C, C);
            float xi = 0.0f;
            if (tid < CHOL_TS) {
                xi = invD[C * CHOL_TS + tid] * sb[C * CHOL_TS + tid];
                for (int c = tid + 1; c < CHOL_TS; c++) xi += sD[tid * CHOL_TSTR + c] * sb[C * CHOL_TS + c];
            }
            __syncwarp();
            if (tid < CHOL_TS) sb[C * CHOL_TS + tid] = xi;
        }
        __syncthreads();
        for (int i = tid; i < C * CHOL_TS; i += NTHREADS) {
            const float* Lc = sA + chol_tofs(C, i >> 4);
            float dp = 0.0f;
            #pragma unroll
            for (int k = 0; k < CHOL_TS; k++) dp += Lc[k * CHOL_TSTR + (i & 15)] * sb[C * CHOL_TS + k];
            sb[i] -= dp;
        }
        __syncthreads();
    }

    for (int i = tid; i < K; i += NTHREADS) d_X[ent * K + i] = sb[i];
}

// ── Mixed-precision tiled Cholesky (SYNC-2026-07-20, EXPERIMENT) ────────────
// cholesky_solve_tiled_mp<K,NTH,ST>: same five-idea tiled structure as
// cholesky_solve_tiled above, but the factor tiles (sA) live in shared memory
// as ST ∈ {__nv_bfloat16, __half} while ALL register math, the RHS (sb) and
// the diagonal inverses (invD) stay FP32. Rationale + occupancy math: see
// CHOL_MP in common.cuh and FIXLOG SYNC-2026-07-20. Compiled only when
// CHOL_MP != 0 — the FP32 production kernel above is untouched.
//
// Precision notes (for whoever validates this):
//  * This is NOT textbook "Cholesky in half precision": the trailing matrix
//    is stored 16-bit too, so every panel's SYRK update ROUNDS the trailing
//    A to ST. Effective factor error ~ u_ST * (K/16 panels), worse than a
//    plain 16-bit factor. That is the price of halving smem — the whole
//    point of the experiment.
//  * FP16 (u=2^-11) is MORE accurate than BF16 (u=2^-8); BF16's win is
//    range (no overflow). κ(A)·u < 1 needed for refinement to converge:
//    κ≈500 estimated -> FP16 comfortably converges, BF16 is borderline.
//    Hence the test order in FIXLOG: FP16-noref first.
//  * Non-finite solves (FP16 overflow / breakdown) are zeroed (the entity
//    keeps a zero factor vector for one iteration, self-heals next iter,
//    RMSE stays finite) and counted once per entity-solve in *d_fail.
#if CHOL_MP != 0

__device__ __forceinline__ float chol_ld(const float* p)         { return *p; }
__device__ __forceinline__ float chol_ld(const __half* p)        { return __half2float(*p); }
__device__ __forceinline__ void  chol_st(float* p, float v)      { *p = v; }
__device__ __forceinline__ void  chol_st(__half* p, float v)     { *p = __float2half(v); }
#if CHOL_MP == 1
__device__ __forceinline__ float chol_ld(const __nv_bfloat16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ void  chol_st(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
#endif

// Bytes of dynamic smem: 16-bit tiles + fp32 sb/invD (+ fp32 residual buffer
// when refining). Tile block is NTT*272*sizeof(ST); 272*2 = 544 = 4*136, so
// the fp32 arrays that follow stay 4-byte aligned for every NTT.
template <int K, typename ST>
constexpr int cholesky_tiled_smem_mp() {
    return (K / CHOL_TS) * (K / CHOL_TS + 1) / 2 * CHOL_TFL * (int)sizeof(ST)
         + (K + (K / CHOL_TS) * CHOL_TS + (CHOL_REFINE ? K : 0)) * (int)sizeof(float);
}

// potrf16 on warp 0, ST tile storage / FP32 math. invD gets the inverse of
// the ROUNDED stored diagonal so trisolves and factor stay consistent.
template <typename ST>
__device__ __forceinline__ void chol_potrf16_w0_mp(ST* sD, float* iD, int lane) {
    for (int j = 0; j < CHOL_TS; j++) {
        float x = (lane < j) ? chol_ld(sD + j * CHOL_TSTR + lane) : 0.0f;
        float dp = x * x;
        for (int off = 8; off > 0; off >>= 1) dp += __shfl_down_sync(0xffffffff, dp, off, 16);
        if (lane == 0) {
            float s = chol_ld(sD + j * (CHOL_TSTR + 1)) - dp;
            s = (s > 1e-10f) ? sqrtf(s) : 1e-5f;
            chol_st(sD + j * (CHOL_TSTR + 1), s);
            iD[j] = 1.0f / chol_ld(sD + j * (CHOL_TSTR + 1));
        }
        __syncwarp();
        float inv = iD[j];
        int i = j + 1 + lane;
        if (i < CHOL_TS) {
            float v = chol_ld(sD + i * CHOL_TSTR + j);
            for (int k = 0; k < j; k++) v -= chol_ld(sD + i * CHOL_TSTR + k) * chol_ld(sD + j * CHOL_TSTR + k);
            chol_st(sD + i * CHOL_TSTR + j, v * inv);
        }
        __syncwarp();
    }
}

// Forward solve L·v' = v on the factored tile image. Uses the L11^-T rows
// stashed in the diagonal tiles' upper halves: L^-1[i][r] = L^-T[r][i] =
// sD[r*17+i] for r<i, diag = invD — so each tile-column is one parallel
// 16-wide matvec + a fully parallel rows-below update (no serial
// substitution). Ends fully synchronized.
template <int K, int NTHREADS, typename ST>
__device__ __forceinline__ void chol_tiled_fwd(const ST* sA, const float* invD, float* v, int tid) {
    constexpr int NT = K / CHOL_TS;
    for (int C = 0; C < NT; C++) {
        if (tid < 32) {
            const ST* sD = sA + chol_tofs(C, C);
            float yi = 0.0f;
            if (tid < CHOL_TS) {
                yi = invD[C * CHOL_TS + tid] * v[C * CHOL_TS + tid];
                for (int r = 0; r < tid; r++)
                    yi += chol_ld(sD + r * CHOL_TSTR + tid) * v[C * CHOL_TS + r];
            }
            __syncwarp();
            if (tid < CHOL_TS) v[C * CHOL_TS + tid] = yi;
        }
        __syncthreads();
        for (int i = (C + 1) * CHOL_TS + tid; i < K; i += NTHREADS) {
            const ST* Lr = sA + chol_tofs(i >> 4, C) + (i & 15) * CHOL_TSTR;
            float dp = 0.0f;
            #pragma unroll
            for (int k = 0; k < CHOL_TS; k++) dp += chol_ld(Lr + k) * v[C * CHOL_TS + k];
            v[i] -= dp;
        }
        __syncthreads();
    }
}

// Backward solve L^T·v' = v — identical structure to the FP32 kernel's
// backward phase (per tile-column matvec with invD + upper-half rows, then
// rows-above update). Ends fully synchronized.
template <int K, int NTHREADS, typename ST>
__device__ __forceinline__ void chol_tiled_bwd(const ST* sA, const float* invD, float* v, int tid) {
    constexpr int NT = K / CHOL_TS;
    for (int C = NT - 1; C >= 0; C--) {
        if (tid < 32) {
            const ST* sD = sA + chol_tofs(C, C);
            float xi = 0.0f;
            if (tid < CHOL_TS) {
                xi = invD[C * CHOL_TS + tid] * v[C * CHOL_TS + tid];
                for (int c = tid + 1; c < CHOL_TS; c++)
                    xi += chol_ld(sD + tid * CHOL_TSTR + c) * v[C * CHOL_TS + c];
            }
            __syncwarp();
            if (tid < CHOL_TS) v[C * CHOL_TS + tid] = xi;
        }
        __syncthreads();
        for (int i = tid; i < C * CHOL_TS; i += NTHREADS) {
            const ST* Lc = sA + chol_tofs(C, i >> 4);
            float dp = 0.0f;
            #pragma unroll
            for (int k = 0; k < CHOL_TS; k++) dp += chol_ld(Lc + k * CHOL_TSTR + (i & 15)) * v[C * CHOL_TS + k];
            v[i] -= dp;
        }
        __syncthreads();
    }
}

// FP32 residual r = b − (A+λI)·x against the entity's gmem LHS. ONLY the
// upper triangle of the gmem LHS is valid (mirror store removed 07-03):
// row i takes j>=i from A[i][j], j<i from A[j][i] by symmetry. In
// WEIGHTED_LAMBDA builds the λ·nnz term is already baked into the gmem diag
// and lambda arrives as 0 — consistent in both regularization modes.
// dA/db are the ENTITY-base pointers. Caller must __syncthreads() after.
template <int K, int NTHREADS>
__device__ __forceinline__ void chol_residual_fp32(const float* __restrict__ dA,
                                                   const float* __restrict__ db,
                                                   const float* x, float lambda,
                                                   float* r, int tid) {
    for (int i = tid; i < K; i += NTHREADS) {
        const float* Arow = dA + i * K;
        float acc = db[i] - lambda * x[i];
        for (int j = i; j < K; j++) acc -= Arow[j] * x[j];
        for (int j = 0; j < i; j++) acc -= dA[j * K + i] * x[j];
        r[i] = acc;
    }
}

#if CHOL_STALE
// Elements per entity in the L cache: the full tile image, unpadded —
// lower/diag tiles of L plus the L11^-T upper halves land in exactly
// NTT*16*16 slots (nothing is wasted: off-diagonal tiles of L are full).
template <int K>
__host__ __device__ constexpr int chol_cache_elems() { return (K / CHOL_TS) * (K / CHOL_TS + 1) / 2 * CHOL_TS * CHOL_TS; }
#endif

// d_ent_list/d_ent_cnt (optional, default off): list-driven re-solve mode for
// the residual-gated stale path — block b solves entity d_ent_list[b], grid
// is launched at worst-case size and blocks past *d_ent_cnt exit immediately
// (count is device-side, so the launch stays stream-ordered with no host
// sync; same pattern cost as the 07-04d dead-launch note).
template <int K, int NTHREADS, typename ST>
__global__ void cholesky_solve_tiled_mp(const float* __restrict__ d_LHS_all, float* __restrict__ d_X,
                                        const int2* __restrict__ d_map, int nvec, int ntot,
                                        int num_entities, float lambda,
                                        int* __restrict__ d_fail, ST* __restrict__ d_Lcache,
                                        const int* __restrict__ d_ent_list = nullptr,
                                        const int* __restrict__ d_ent_cnt = nullptr,
                                        const int* __restrict__ d_nnz_w = nullptr) {
    constexpr int NT  = K / CHOL_TS;
    constexpr int NTT = NT * (NT + 1) / 2;

    int ent = blockIdx.x;
    if (d_ent_list) {
        if (ent >= *d_ent_cnt) return;
        ent = d_ent_list[ent];
    } else if (ent >= num_entities) return;
    // Fused weighted-λ (07-21). The 16-bit tiles round on the smem store, so
    // the diag add must NOT go through a fp16 round-trip: with d_nnz_w set,
    // the diag pass below re-reads the fp32 gmem value, adds λ·nnz in fp32,
    // and converts once — fl16(fl32(A+λn)), bit-identical to the retired
    // pre-add kernel. lam also feeds the refine residual (gmem no longer
    // carries the λ term).
    float lam = lambda;
    if (d_nnz_w) { int n = d_nnz_w[ent]; lam = lambda * (n > 0 ? n : 1); }

    extern __shared__ unsigned char sRawMP[];
    ST*    sA   = (ST*)sRawMP;                                     // NTT tiles, stride-17 rows, 16-bit
    float* sb   = (float*)(sRawMP + NTT * CHOL_TFL * sizeof(ST));  // K — RHS/solution, FP32
    float* invD = sb + K;                                          // NT*16 — diag inverses, FP32
#if CHOL_REFINE
    float* sr   = invD + NT * CHOL_TS;                             // K — residual/correction, FP32
#endif
    __shared__ int sBad;

    const int tid = threadIdx.x;
    if (tid == 0) sBad = 0;
    const long long lhs_base = (long long)ent * K * K;
    const float4* src4 = (const float4*)(d_LHS_all + lhs_base);

    // Same float4 load map as the FP32 kernel; values are rounded to ST on
    // the smem store (conversion is register-side, gmem traffic unchanged).
    for (int idx = tid; idx < ntot; idx += NTHREADS) {
        int2 m = d_map[idx];
        if (idx < nvec) {
            float4 v = src4[m.x];
            chol_st(sA + m.y,                 v.x);
            chol_st(sA + m.y +     CHOL_TSTR, v.y);
            chol_st(sA + m.y + 2 * CHOL_TSTR, v.z);
            chol_st(sA + m.y + 3 * CHOL_TSTR, v.w);
        } else {
            chol_st(sA + m.y, d_LHS_all[lhs_base + m.x]);
        }
    }
    for (int i = tid; i < K; i += NTHREADS) sb[i] = d_X[ent * K + i];
    __syncthreads();
    for (int i = tid; i < K; i += NTHREADS) {
        ST* dg = sA + chol_tofs(i >> 4, i >> 4) + (i & 15) * (CHOL_TSTR + 1);
        if (d_nnz_w) chol_st(dg, d_LHS_all[lhs_base + (long long)i * K + i] + lam);
        else         chol_st(dg, chol_ld(dg) + lambda);
        // Overflow gate on the STORED diag. An FP16 inf diag does not go NaN:
        // iD becomes 1/inf = 0 and every column silently zeroes — the final
        // non-finite check never fires and the fail counter lies. A is Gram
        // (|a_ij| <= sqrt(a_ii*a_jj)), so any entry overflowing 16-bit range
        // implies a diag overflow: checking the K diag slots catches it all.
        float dv = chol_ld(dg);
        if (dv != dv || dv > 1e30f || dv < -1e30f) sBad = 1;
    }
    __syncthreads();
    if (sBad) {
        if (tid == 0 && d_fail) atomicAdd(d_fail, 1);
        for (int i = tid; i < K; i += NTHREADS) d_X[ent * K + i] = 0.0f;
        return;
    }

    if (tid < 32) chol_potrf16_w0_mp(sA, invD, tid);   // prologue: POTRF(0)
    __syncthreads();

    for (int p = 0; p < NT; p++) {
        ST*    sD = sA + chol_tofs(p, p);
        float* iD = invD + p * CHOL_TS;

        // TRSM(p): panel rows + b-row (fp32, lives in sb) + 16 identity rows
        // (solutions = rows of L11^-T -> upper half of the diagonal tile).
        int nrows = (NT - 1 - p) * CHOL_TS;
        for (int t = tid; t < nrows + 1 + CHOL_TS; t += NTHREADS) {
            float a[CHOL_TS];
            ST* rowS = nullptr;
            int irow = t - nrows - 1;
            if (t < nrows) {
                rowS = sA + chol_tofs(p + 1 + (t >> 4), p) + (t & 15) * CHOL_TSTR;
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++) a[c] = chol_ld(rowS + c);
            } else if (t == nrows) {
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++) a[c] = sb[p * CHOL_TS + c];
            } else {
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++) a[c] = (c == irow) ? 1.0f : 0.0f;
            }
            #pragma unroll
            for (int c = 0; c < CHOL_TS; c++) {
                float v = a[c];
                #pragma unroll
                for (int k = 0; k < c; k++) v -= a[k] * chol_ld(sD + c * CHOL_TSTR + k);
                a[c] = v * iD[c];
            }
            if (rowS) {
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++) chol_st(rowS + c, a[c]);
            } else if (t == nrows) {
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++) sb[p * CHOL_TS + c] = a[c];
            } else {
                #pragma unroll
                for (int c = 0; c < CHOL_TS; c++)
                    if (c > irow) chol_st(sD + irow * CHOL_TSTR + c, a[c]);
            }
        }
        __syncthreads();

        if (p + 1 < NT) {
            // SYRK-a: tile(p+1,p+1) + b-segment (what POTRF(p+1) needs).
            const ST* Pn = sA + chol_tofs(p + 1, p);
            ST*       Dn = sA + chol_tofs(p + 1, p + 1);
            for (int e = tid; e < 64; e += NTHREADS) {
                int r0 = (e >> 3) * 2, c0 = (e & 7) * 2;
                const ST *a0 = Pn + r0 * CHOL_TSTR, *a1 = a0 + CHOL_TSTR;
                const ST *b0 = Pn + c0 * CHOL_TSTR, *b1 = b0 + CHOL_TSTR;
                float s00 = 0, s01 = 0, s10 = 0, s11 = 0;
                #pragma unroll
                for (int k = 0; k < CHOL_TS; k++) {
                    float x0 = chol_ld(a0 + k), x1 = chol_ld(a1 + k);
                    float y0 = chol_ld(b0 + k), y1 = chol_ld(b1 + k);
                    s00 += x0 * y0; s01 += x0 * y1; s10 += x1 * y0; s11 += x1 * y1;
                }
                chol_st(Dn + r0 * CHOL_TSTR + c0,           chol_ld(Dn + r0 * CHOL_TSTR + c0)           - s00);
                chol_st(Dn + r0 * CHOL_TSTR + c0 + 1,       chol_ld(Dn + r0 * CHOL_TSTR + c0 + 1)       - s01);
                chol_st(Dn + (r0 + 1) * CHOL_TSTR + c0,     chol_ld(Dn + (r0 + 1) * CHOL_TSTR + c0)     - s10);
                chol_st(Dn + (r0 + 1) * CHOL_TSTR + c0 + 1, chol_ld(Dn + (r0 + 1) * CHOL_TSTR + c0 + 1) - s11);
            }
            for (int i = (p + 1) * CHOL_TS + tid; i < (p + 2) * CHOL_TS; i += NTHREADS) {
                const ST* Lr = Pn + (i & 15) * CHOL_TSTR;
                float dp = 0.0f;
                #pragma unroll
                for (int k = 0; k < CHOL_TS; k++) dp += chol_ld(Lr + k) * sb[p * CHOL_TS + k];
                sb[i] -= dp;
            }
            __syncthreads();

            // warp 0: POTRF(p+1)  ∥  tid>=32: SYRK-b + b-tail rows.
            if (tid < 32) {
                chol_potrf16_w0_mp(sA + chol_tofs(p + 1, p + 1), invD + (p + 1) * CHOL_TS, tid);
            } else {
                int m = NT - 1 - p;
                int wtid = tid - 32, wn = NTHREADS - 32;
                for (int e = wtid; e < m * (m + 1) / 2 * 64 - 64; e += wn) {
                    int t = (e >> 6) + 1;   // skip tile 0 = (p+1,p+1), done in SYRK-a
                    int r0 = ((e >> 3) & 7) * 2, c0 = (e & 7) * 2;
                    int Rp = 0;
                    while ((Rp + 1) * (Rp + 2) / 2 <= t) Rp++;
                    int Cp = t - Rp * (Rp + 1) / 2;
                    const ST* P1 = sA + chol_tofs(p + 1 + Rp, p);
                    const ST* P2 = sA + chol_tofs(p + 1 + Cp, p);
                    const ST *a0 = P1 + r0 * CHOL_TSTR, *a1 = a0 + CHOL_TSTR;
                    const ST *b0 = P2 + c0 * CHOL_TSTR, *b1 = b0 + CHOL_TSTR;
                    float s00 = 0, s01 = 0, s10 = 0, s11 = 0;
                    #pragma unroll
                    for (int k = 0; k < CHOL_TS; k++) {
                        float x0 = chol_ld(a0 + k), x1 = chol_ld(a1 + k);
                        float y0 = chol_ld(b0 + k), y1 = chol_ld(b1 + k);
                        s00 += x0 * y0; s01 += x0 * y1; s10 += x1 * y0; s11 += x1 * y1;
                    }
                    ST* D = sA + chol_tofs(p + 1 + Rp, p + 1 + Cp);
                    chol_st(D + r0 * CHOL_TSTR + c0,           chol_ld(D + r0 * CHOL_TSTR + c0)           - s00);
                    chol_st(D + r0 * CHOL_TSTR + c0 + 1,       chol_ld(D + r0 * CHOL_TSTR + c0 + 1)       - s01);
                    chol_st(D + (r0 + 1) * CHOL_TSTR + c0,     chol_ld(D + (r0 + 1) * CHOL_TSTR + c0)     - s10);
                    chol_st(D + (r0 + 1) * CHOL_TSTR + c0 + 1, chol_ld(D + (r0 + 1) * CHOL_TSTR + c0 + 1) - s11);
                }
                for (int i = (p + 2) * CHOL_TS + wtid; i < K; i += wn) {
                    const ST* Lr = sA + chol_tofs(i >> 4, p) + (i & 15) * CHOL_TSTR;
                    float dp = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < CHOL_TS; k++) dp += chol_ld(Lr + k) * sb[p * CHOL_TS + k];
                    sb[i] -= dp;
                }
            }
            __syncthreads();
        }
    }

    // Backward solve (identical tile-column matvec structure to FP32 kernel).
    chol_tiled_bwd<K, NTHREADS, ST>(sA, invD, sb, tid);

    // Breakdown / FP16-overflow guard: zero the entity's solution (keeps the
    // run finite; the entity self-heals on the next ALS iteration) and count.
    // (v != v catches NaN; the magnitude test catches ±inf. Deliberately not
    // isfinite(): ambiguous in device code under `using namespace std`.)
    for (int i = tid; i < K; i += NTHREADS) {
        float v = sb[i];
        if (v != v || v > 1e30f || v < -1e30f) sBad = 1;
    }
    __syncthreads();
    if (sBad) {
        if (tid == 0 && d_fail) atomicAdd(d_fail, 1);
        for (int i = tid; i < K; i += NTHREADS) d_X[ent * K + i] = 0.0f;
        return;
    }

#if CHOL_REFINE
    // One FP32 iterative-refinement step: d_X still holds b (final store is
    // below), the gmem LHS still holds fp32 A — residual is true FP32.
    chol_residual_fp32<K, NTHREADS>(d_LHS_all + lhs_base, d_X + ent * K, sb, lam, sr, tid);
    __syncthreads();
    chol_tiled_fwd<K, NTHREADS, ST>(sA, invD, sr, tid);
    chol_tiled_bwd<K, NTHREADS, ST>(sA, invD, sr, tid);
    for (int i = tid; i < K; i += NTHREADS) sb[i] += sr[i];
#endif

#if CHOL_STALE
    // Export the tile image (L lower/diag + L11^-T upper halves) to the
    // per-entity cache, unpadded, coalesced on the gmem side. d_Lcache is
    // the BATCH-base pointer (host pre-offsets by bstart).
    if (d_Lcache) {
        ST* dst = d_Lcache + (long long)ent * chol_cache_elems<K>();
        for (int idx = tid; idx < chol_cache_elems<K>(); idx += NTHREADS) {
            int t = idx >> 8, off = idx & 255;
            dst[idx] = sA[t * CHOL_TFL + (off >> 4) * CHOL_TSTR + (off & 15)];
        }
    }
#endif

    for (int i = tid; i < K; i += NTHREADS) d_X[ent * K + i] = sb[i];
}

#if CHOL_STALE
// Bytes of dynamic smem for the stale-refine kernel: tiles + sb/invD/sr.
template <int K, typename ST>
constexpr int cholesky_stale_smem() {
    return (K / CHOL_TS) * (K / CHOL_TS + 1) / 2 * CHOL_TFL * (int)sizeof(ST)
         + 3 * K * (int)sizeof(float);
}

// Stale-L solve: NO factorization. x0 from trisolves with the CACHED L
// (previous fresh iteration), then ONE FP32 refinement step against the
// FRESH A that the LHS kernels built this iteration. FLOPs ~ 4 trisolves +
// 1 matvec ≈ K^3/8 vs K^3/3 + panel skeleton for the full factorization;
// gmem ~ cache (2B/elem) + b + one sweep of the fp32 LHS for the residual.
// If Y drifted too far since the last refresh this degrades to a bad
// preconditioner — watch train RMSE between refreshes (see FIXLOG).
// Residual gate (d_bad_list non-null): after x0, the FP32 residual r=b−Ax0 is
// already computed for the IR step — if ||r||² > τ²·||b||² (or non-finite)
// the stale L is a bad preconditioner for THIS entity this iteration: skip
// the IR, leave b untouched in d_X, and append the entity to d_bad_list; the
// caller then runs the full mp solve on just that list (which also refreshes
// the entity's cache slot). This is the per-entity adaptive refresh — the
// fixed-cadence version measured-dead 07-20b (divergence cascade).
template <int K, int NTHREADS, typename ST>
__global__ void cholesky_stale_refine(const float* __restrict__ d_LHS_all, float* __restrict__ d_X,
                                      const ST* __restrict__ d_Lcache,
                                      int num_entities, float lambda, int* __restrict__ d_fail,
                                      int* __restrict__ d_bad_cnt = nullptr,
                                      int* __restrict__ d_bad_list = nullptr,
                                      int* __restrict__ d_bad_total = nullptr,
                                      float tau = 0.5f,
                                      const int* __restrict__ d_nnz_w = nullptr) {
    constexpr int NT  = K / CHOL_TS;
    constexpr int NTT = NT * (NT + 1) / 2;

    int ent = blockIdx.x;
    if (ent >= num_entities) return;
    // Fused weighted-λ (07-21): residual needs the per-entity λ·nnz that the
    // gmem LHS no longer carries.
    float lam = lambda;
    if (d_nnz_w) { int n = d_nnz_w[ent]; lam = lambda * (n > 0 ? n : 1); }

    extern __shared__ unsigned char sRawSt[];
    ST*    sA   = (ST*)sRawSt;
    float* sb   = (float*)(sRawSt + NTT * CHOL_TFL * sizeof(ST));
    float* invD = sb + K;
    float* sr   = invD + K;
    __shared__ int sBad;

    const int tid = threadIdx.x;
    if (tid == 0) sBad = 0;

    const ST* src = d_Lcache + (long long)ent * chol_cache_elems<K>();
    for (int idx = tid; idx < chol_cache_elems<K>(); idx += NTHREADS) {
        int t = idx >> 8, off = idx & 255;
        sA[t * CHOL_TFL + (off >> 4) * CHOL_TSTR + (off & 15)] = src[idx];
    }
    for (int i = tid; i < K; i += NTHREADS) sb[i] = d_X[ent * K + i];
    __syncthreads();
    // invD recomputed from the stored (rounded) diagonal — identical to what
    // the exporting solve used, so factor and trisolves stay consistent.
    for (int i = tid; i < K; i += NTHREADS)
        invD[i] = 1.0f / chol_ld(sA + chol_tofs(i >> 4, i >> 4) + (i & 15) * (CHOL_TSTR + 1));
    __syncthreads();

    // x0 = (L_old L_old^T)^-1 b
    chol_tiled_fwd<K, NTHREADS, ST>(sA, invD, sb, tid);
    chol_tiled_bwd<K, NTHREADS, ST>(sA, invD, sb, tid);

    // One refinement step against the FRESH A (d_X still holds b here).
    chol_residual_fp32<K, NTHREADS>(d_LHS_all + (long long)ent * K * K, d_X + ent * K, sb, lam, sr, tid);
    __syncthreads();

    if (d_bad_list) {
        __shared__ float sR2, sB2;
        if (tid == 0) { sR2 = 0.0f; sB2 = 0.0f; }
        __syncthreads();
        float r2 = 0.0f, b2 = 0.0f;
        for (int i = tid; i < K; i += NTHREADS) {
            r2 += sr[i] * sr[i];
            float bv = d_X[ent * K + i];
            b2 += bv * bv;
        }
        atomicAdd(&sR2, r2); atomicAdd(&sB2, b2);
        __syncthreads();
        // negated form so a NaN residual also fails the gate
        if (!(sR2 <= tau * tau * sB2)) {
            if (tid == 0) {
                int slot = atomicAdd(d_bad_cnt, 1);
                d_bad_list[slot] = ent;
                if (d_bad_total) atomicAdd(d_bad_total, 1);
            }
            return;   // d_X keeps b — the list re-solve reads it as the RHS
        }
    }

    chol_tiled_fwd<K, NTHREADS, ST>(sA, invD, sr, tid);
    chol_tiled_bwd<K, NTHREADS, ST>(sA, invD, sr, tid);
    for (int i = tid; i < K; i += NTHREADS) sb[i] += sr[i];

    for (int i = tid; i < K; i += NTHREADS) {
        float v = sb[i];
        if (v != v || v > 1e30f || v < -1e30f) sBad = 1;
    }
    __syncthreads();
    if (sBad) {
        if (tid == 0 && d_fail) atomicAdd(d_fail, 1);
        for (int i = tid; i < K; i += NTHREADS) d_X[ent * K + i] = 0.0f;
        return;
    }
    for (int i = tid; i < K; i += NTHREADS) d_X[ent * K + i] = sb[i];
}
#endif // CHOL_STALE
#endif // CHOL_MP

// Name of the tiled solver actually dispatched at this build's precision —
// used by the .cu hosts for cudaFuncSetAttribute (carveout) calls.
#if CHOL_MP == 0
#define CHOL_TILED_SOLVER(KK, NTH) cholesky_solve_tiled<KK, NTH>
#else
#define CHOL_TILED_SOLVER(KK, NTH) cholesky_solve_tiled_mp<KK, NTH, chol_store_t>
#endif

template <int K>
__global__ void cholesky_solve_batched(float* __restrict__ d_LHS_all, float* __restrict__ d_X, int num_entities, float lambda,
                                       const int* __restrict__ d_nnz_w = nullptr) {
    int ent = blockIdx.x * blockDim.x + threadIdx.x;
    if (ent >= num_entities) return;
    // Fused weighted-λ (07-21): same fp32 add as the retired pre-add kernel.
    float lam = lambda;
    if (d_nnz_w) { int n = d_nnz_w[ent]; lam = lambda * (n > 0 ? n : 1); }

    float A[K * K];
    float b[K];
    long long lhs_base = (long long)ent * K * K;
    int rhs_base = ent * K;

    // Full sequential K*K load ON PURPOSE: the compiler vectorizes it (float4)
    // and the warp streams contiguous blocks. The lower-triangle entries are
    // garbage (mirror store removed) but the math below only ever reads the
    // upper triangle, so they are loaded and ignored. An upper-only double
    // loop measured ~40 ms/iter SLOWER at K=32 (broken vectorization).
    for (int i = 0; i < K * K; i++) A[i] = d_LHS_all[lhs_base + i];
    for (int i = 0; i < K; i++)     b[i] = d_X[rhs_base + i];
    for (int i = 0; i < K; i++)     A[i * K + i] += lam;

    for (int j = 0; j < K; j++) {
        float s = A[j * K + j];
        for (int k = 0; k < j; k++) s -= A[k * K + j] * A[k * K + j];
        A[j * K + j] = (s > 1e-10f) ? sqrtf(s) : 1e-5f;
        float inv = 1.0f / A[j * K + j];
        for (int i = j + 1; i < K; i++) {
            float v = A[j * K + i];
            for (int k = 0; k < j; k++) v -= A[k * K + i] * A[k * K + j];
            A[j * K + i] = v * inv;
        }
    }
    for (int i = 0; i < K; i++) {
        float s = b[i];
        for (int j = 0; j < i; j++) s -= A[j * K + i] * b[j];
        b[i] = s / A[i * K + i];
    }
    for (int i = K - 1; i >= 0; i--) {
        float s = b[i];
        for (int j = i + 1; j < K; j++) s -= A[i * K + j] * b[j];
        b[i] = s / A[i * K + i];
    }
    for (int i = 0; i < K; i++) d_X[rhs_base + i] = b[i];
}
