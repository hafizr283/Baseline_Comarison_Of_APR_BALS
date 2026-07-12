#pragma once
#include "common.cuh"

// One-time FP32 -> FP16 feature conversion, run once per phase (before the
// user/item entity-batch loop). The WMMA kernels then gather half-precision
// rows directly: half the global read bandwidth of gathering FP32 and
// converting per element. Values are identical to __float2half at load,
// so the LHS/RHS results are bit-identical to the previous scheme.
__global__ void convert_fp32_to_fp16(const float* __restrict__ src,
                                     half* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

template <int KT>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * 32, 1)
compute_LHS_RHS_wmma(int n_dense, int entity_offset,
                     const int* __restrict__ offsets,
                     const int* __restrict__ col_indices,
                     const float* __restrict__ values,
                     const half* __restrict__ d_Feat,
                     float* __restrict__ d_LHS_all, float* __restrict__ d_RHS_all,
                     int batch_start) {
    const int K  = KT * 16;
    const int NT = KT * (KT + 1) / 2;
    const int RHS_PER_LANE = (K + 31) / 32;

    int w    = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int u    = entity_offset + blockIdx.x * WARPS_PER_BLOCK + w;
    if (u >= n_dense) return;

    int start = offsets[u];
    int nnz_u = offsets[u + 1] - start;

    __shared__ half  sS[WARPS_PER_BLOCK][16 * (KT * 16)];
    __shared__ float sR[WARPS_PER_BLOCK][16];
    half*  S = sS[w];
    float* R = sR[w];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) wmma::fill_fragment(c_frag[t], 0.0f);

    float rhs_acc[RHS_PER_LANE];
    #pragma unroll
    for (int j = 0; j < RHS_PER_LANE; j++) rhs_acc[j] = 0.0f;

    // 8 halves per 16-byte int4 load (features are already FP16)
    const int HVECS_PER_ROW = K / 8;
    const int TOTAL_HVECS = 16 * HVECS_PER_ROW;   // multiple of 32 for all supported K
    // Register prefetch double-buffer (K<=48): the per-tile gather chain
    // (col_indices load -> feature int4 load -> smem store -> syncwarp) used
    // to serialize with the MMAs, so short-loop entities (user side: ~10
    // tiles/warp) ran ~4x slower per tile than long-loop ones (item side).
    // Issue tile t+1's global loads into registers BEFORE consuming tile t's
    // MMAs, so the gather latency hides under tensor-core work. Same bytes,
    // same tile order, same accumulation order -> bit-identical results.
    // Single smem buffer (commit happens after the end-of-tile syncwarp), so
    // occupancy is unchanged. Disabled for K>=64: PFV int4s cost 4*PFV regs
    // and K>=96 already runs near the 240-reg ceiling; K>=64 WMMA is not the
    // bottleneck there (the Cholesky solve is, see FIXLOG 07-04c).
    const int  PFV    = (TOTAL_HVECS + 31) / 32;   // int4 gathers per lane per tile
    const bool USE_PF = (KT <= 3);

    if (USE_PF) {
        int4  pf[PFV > 0 ? PFV : 1];
        float pr = 0.0f;
        #pragma unroll
        for (int j = 0; j < PFV; j++) {            // prefetch tile 0
            int e = lane + j * 32;
            int i = e / HVECS_PER_ROW, v = e % HVECS_PER_ROW;
            pf[j] = make_int4(0, 0, 0, 0);
            if (i < nnz_u) {
                int item = col_indices[start + i];
                pf[j] = ((const int4*)d_Feat)[(long long)item * HVECS_PER_ROW + v];
            }
        }
        if (lane < 16 && lane < nnz_u) pr = values[start + lane];

        for (int base = 0; base < nnz_u; base += 16) {
            #pragma unroll
            for (int j = 0; j < PFV; j++) {        // commit prefetched tile
                int e = lane + j * 32;
                int i = e / HVECS_PER_ROW, v = e % HVECS_PER_ROW;
                ((int4*)S)[i * HVECS_PER_ROW + v] = pf[j];
            }
            if (lane < 16) R[lane] = pr;
            __syncwarp();

            int nbase = base + 16;                 // issue next tile's gathers
            if (nbase < nnz_u) {
                #pragma unroll
                for (int j = 0; j < PFV; j++) {
                    int e = lane + j * 32;
                    int i = e / HVECS_PER_ROW, v = e % HVECS_PER_ROW;
                    int gi = nbase + i;
                    pf[j] = make_int4(0, 0, 0, 0);
                    if (gi < nnz_u) {
                        int item = col_indices[start + gi];
                        pf[j] = ((const int4*)d_Feat)[(long long)item * HVECS_PER_ROW + v];
                    }
                }
                int gi = nbase + lane;
                pr = (lane < 16 && gi < nnz_u) ? values[start + gi] : 0.0f;
            }

            int t = 0;
            #pragma unroll
            for (int m_kt = 0; m_kt < KT; m_kt++) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
                wmma::load_matrix_sync(a_frag, S + m_kt * 16, K);
                #pragma unroll
                for (int n_kt = m_kt; n_kt < KT; n_kt++) {
                    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                    wmma::load_matrix_sync(b_frag, S + n_kt * 16, K);
                    wmma::mma_sync(c_frag[t], a_frag, b_frag, c_frag[t]);
                    t++;
                }
            }

            #pragma unroll
            for (int j = 0; j < RHS_PER_LANE; j++) {
                int f = lane + j * 32;
                if (f < K) {
                    float acc = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 16; i++)
                        acc += __half2float(S[i * K + f]) * R[i];
                    rhs_acc[j] += acc;
                }
            }
            __syncwarp();
        }
    } else {
    for (int base = 0; base < nnz_u; base += 16) {
        for (int e = lane; e < TOTAL_HVECS; e += 32) {
            int i = e / HVECS_PER_ROW;
            int v = e % HVECS_PER_ROW;
            int gi = base + i;
            int4 h8 = make_int4(0, 0, 0, 0);   // 8 x half(0.0)
            if (gi < nnz_u) {
                int item = col_indices[start + gi];
                h8 = ((const int4*)d_Feat)[(long long)item * HVECS_PER_ROW + v];
            }
            ((int4*)S)[i * HVECS_PER_ROW + v] = h8;
        }
        if (lane < 16) {
            int gi = base + lane;
            R[lane] = (gi < nnz_u) ? values[start + gi] : 0.0f;
        }
        __syncwarp();

        int t = 0;
        #pragma unroll
        for (int m_kt = 0; m_kt < KT; m_kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
            wmma::load_matrix_sync(a_frag, S + m_kt * 16, K);
            #pragma unroll
            for (int n_kt = m_kt; n_kt < KT; n_kt++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, S + n_kt * 16, K);
                wmma::mma_sync(c_frag[t], a_frag, b_frag, c_frag[t]);
                t++;
            }
        }

        #pragma unroll
        for (int j = 0; j < RHS_PER_LANE; j++) {
            int f = lane + j * 32;
            if (f < K) {
                float acc = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    acc += __half2float(S[i * K + f]) * R[i];
                rhs_acc[j] += acc;
            }
        }
        __syncwarp();
    }
    }

    int t = 0;
    #pragma unroll
    for (int m_kt = 0; m_kt < KT; m_kt++) {
        #pragma unroll
        for (int n_kt = m_kt; n_kt < KT; n_kt++) {
            long long ubase = (long long)(u - batch_start) * K * K;
            wmma::store_matrix_sync(d_LHS_all + ubase + (long long)(m_kt * 16) * K + n_kt * 16,
                                    c_frag[t], K, wmma::mem_row_major);
            t++;
        }
    }
    #pragma unroll
    for (int j = 0; j < RHS_PER_LANE; j++) {
        int f = lane + j * 32;
        if (f < K) d_RHS_all[(long long)u * K + f] = rhs_acc[j];
    }
}

template <int KT, int NUM_WARPS>
__global__ void __launch_bounds__(NUM_WARPS * 32, 1)
compute_LHS_RHS_wmma_giant(int n_giant,
                           const int* __restrict__ offsets,
                           const int* __restrict__ col_indices,
                           const float* __restrict__ values,
                           const half* __restrict__ d_Feat,
                           float* __restrict__ d_LHS_all, float* __restrict__ d_RHS_all,
                           int entity_offset, int batch_start) {
    const int K  = KT * 16;
    const int NT = KT * (KT + 1) / 2;
    const int RHS_PER_LANE = (K + 31) / 32;

    int w    = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int global_u = entity_offset + blockIdx.x;
    // NOTE: n_giant here is the PER-BATCH giant count (ng_b = g1-g0), not an
    // absolute index bound. global_u is absolute, so comparing global_u >= n_giant
    // wrongly skipped every giant in batches after the first (entity_offset>0),
    // leaving their LHS as garbage. The grid has exactly ng_b blocks, so the
    // correct guard is on blockIdx.x.
    if (blockIdx.x >= n_giant) return;

    int start = offsets[global_u];
    int nnz_u = offsets[global_u + 1] - start;

    __shared__ half  sS[NUM_WARPS][16 * (KT * 16)];
    __shared__ float sR[NUM_WARPS][16];
    __shared__ float sRed[NUM_WARPS][16 * 16];
    __shared__ float sComb[16 * 16];
    __shared__ float sRedRHS[NUM_WARPS][KT * 16];
    half*  S = sS[w];
    float* R = sR[w];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[NT];
    #pragma unroll
    for (int t = 0; t < NT; t++) wmma::fill_fragment(c_frag[t], 0.0f);

    float rhs_acc[RHS_PER_LANE];
    #pragma unroll
    for (int j = 0; j < RHS_PER_LANE; j++) rhs_acc[j] = 0.0f;

    for (int base = w * 16; base < nnz_u; base += NUM_WARPS * 16) {
        // 8 halves per 16-byte int4 load (features are already FP16)
        const int HVECS_PER_ROW = K / 8;
        const int TOTAL_HVECS = 16 * HVECS_PER_ROW;
        for (int e = lane; e < TOTAL_HVECS; e += 32) {
            int i = e / HVECS_PER_ROW;
            int v = e % HVECS_PER_ROW;
            int gi = base + i;
            int4 h8 = make_int4(0, 0, 0, 0);   // 8 x half(0.0)
            if (gi < nnz_u) {
                int item = col_indices[start + gi];
                h8 = ((const int4*)d_Feat)[(long long)item * HVECS_PER_ROW + v];
            }
            ((int4*)S)[i * HVECS_PER_ROW + v] = h8;
        }
        if (lane < 16) {
            int gi = base + lane;
            R[lane] = (gi < nnz_u) ? values[start + gi] : 0.0f;
        }
        __syncwarp();

        int t = 0;
        #pragma unroll
        for (int m_kt = 0; m_kt < KT; m_kt++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
            wmma::load_matrix_sync(a_frag, S + m_kt * 16, K);
            #pragma unroll
            for (int n_kt = m_kt; n_kt < KT; n_kt++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, S + n_kt * 16, K);
                wmma::mma_sync(c_frag[t], a_frag, b_frag, c_frag[t]);
                t++;
            }
        }

        #pragma unroll
        for (int j = 0; j < RHS_PER_LANE; j++) {
            int f = lane + j * 32;
            if (f < K) {
                float acc = 0.0f;
                #pragma unroll
                for (int i = 0; i < 16; i++)
                    acc += __half2float(S[i * K + f]) * R[i];
                rhs_acc[j] += acc;
            }
        }
        __syncwarp();
    }

    #pragma unroll
    for (int j = 0; j < RHS_PER_LANE; j++) {
        int f = lane + j * 32;
        if (f < K) sRedRHS[w][f] = rhs_acc[j];
    }
    __syncthreads();
    for (int f = threadIdx.x; f < K; f += blockDim.x) {
        float s = 0.0f;
        #pragma unroll
        for (int wv = 0; wv < NUM_WARPS; wv++) s += sRedRHS[wv][f];
        d_RHS_all[(long long)global_u * K + f] = s;
    }

    long long ubase = (long long)(global_u - batch_start) * K * K;
    int t = 0;
    #pragma unroll
    for (int m_kt = 0; m_kt < KT; m_kt++) {
        #pragma unroll
        for (int n_kt = m_kt; n_kt < KT; n_kt++) {
            __syncthreads();
            wmma::store_matrix_sync(sRed[w], c_frag[t], 16, wmma::mem_row_major);
            __syncthreads();
            for (int e = threadIdx.x; e < 256; e += blockDim.x) {
                float s = 0.0f;
                #pragma unroll
                for (int wv = 0; wv < NUM_WARPS; wv++) s += sRed[wv][e];
                sComb[e] = s;
            }
            __syncthreads();
            for (int e = threadIdx.x; e < 256; e += blockDim.x) {
                int r = e >> 4, c = e & 15;
                d_LHS_all[ubase + (long long)(m_kt * 16 + r) * K + (n_kt * 16 + c)] = sComb[e];
            }
            t++;
        }
    }
}
