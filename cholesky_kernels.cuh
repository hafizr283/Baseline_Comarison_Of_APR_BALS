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

__global__ void cholesky_solve_cooperative(float* __restrict__ d_LHS_all, float* __restrict__ d_X, int K, int num_entities, float lambda) {
    int ent = blockIdx.x;
    if (ent >= num_entities) return;

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
__global__ void cholesky_solve_packed(float* __restrict__ d_LHS_all, float* __restrict__ d_X, int num_entities, float lambda) {
    int ent = blockIdx.x;
    if (ent >= num_entities) return;

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
                                     int num_entities, float lambda) {
    constexpr int NT  = K / CHOL_TS;
    constexpr int NTT = NT * (NT + 1) / 2;

    int ent = blockIdx.x;
    if (ent >= num_entities) return;

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
        sA[chol_tofs(i >> 4, i >> 4) + (i & 15) * (CHOL_TSTR + 1)] += lambda;
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

template <int K>
__global__ void cholesky_solve_batched(float* __restrict__ d_LHS_all, float* __restrict__ d_X, int num_entities, float lambda) {
    int ent = blockIdx.x * blockDim.x + threadIdx.x;
    if (ent >= num_entities) return;

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
    for (int i = 0; i < K; i++)     A[i * K + i] += lambda;

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
