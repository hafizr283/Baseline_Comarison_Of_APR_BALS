#pragma once
#include "common.cuh"

template <int RPT, int RR, int RC>
__global__ void __launch_bounds__(1024, 1)
compute_LHS_RHS_BALS_block(int rows, int K, float lambda,
                           const int* __restrict__ tile_ptr,
                           const int* __restrict__ tile_colidx,
                           const int* __restrict__ seg_ptr,
                           const int* __restrict__ seg_colidx,
                           const float* __restrict__ seg_values,
                           const float* __restrict__ d_Feat,
                           float* __restrict__ d_LHS_all,
                           float* __restrict__ d_RHS_all,
                           const float* __restrict__ d_tile_density,
                           const int* __restrict__ nz_tile_list,
                           const int* __restrict__ nz_tile_ptr,
                           const int* __restrict__ job_tx,
                           const int* __restrict__ job_chunk,
                           const int* __restrict__ job_nchunks,
                           int batch_start, int batch_size) {
    int job     = blockIdx.x;
    int tx      = job_tx[job];
    // Early exit: this job's tile-row [tx*XB, (tx+1)*XB) lies entirely outside
    // the current entity batch — skip all compute, not just the writes.
    if (tx * XB >= batch_start + batch_size || tx * XB + XB <= batch_start) return;
    int chunk   = job_chunk[job];
    int nchunks = job_nchunks[job];
    int r_th = threadIdx.x;
    int c_th = threadIdx.y;
    int z_th = threadIdx.z;
    int TR   = blockDim.x;
    int TC   = blockDim.y;

    // Row-split: gridDim.y sibling blocks share this tile-row. Block y covers
    // rows [row_base, row_base + RPT*DZ) of the XB-row tile, so the persistent
    // accumulator block RPT*RR*RC stays register-resident (no local-mem spill).
    // Requires RPT * blockDim.z * gridDim.y == XB (host asserts).
    const int DZ = blockDim.z;
    const int row_base = blockIdx.y * (RPT * DZ);

    float lhs_vals[RPT][RR][RC];
    float rhs_vals[RPT][RR];
    #pragma unroll
    for (int r = 0; r < RPT; r++) {
        #pragma unroll
        for (int rr = 0; rr < RR; rr++) {
            rhs_vals[r][rr] = 0.0f;
            #pragma unroll
            for (int cc = 0; cc < RC; cc++) lhs_vals[r][rr][cc] = 0.0f;
        }
    }

    extern __shared__ float sY[];

    int nz_start = nz_tile_ptr[tx];
    int nz_end   = nz_tile_ptr[tx + 1];
    int tid = (z_th * TC + c_th) * TR + r_th;
    int total_threads = TR * TC * DZ;

    for (int nz_idx = nz_start + chunk; nz_idx < nz_end; nz_idx += nchunks) {
        int tile_id = nz_tile_list[nz_idx];
        int t_start = tile_ptr[tile_id];
        int t_end   = tile_ptr[tile_id + 1];
        int t_cnt   = t_end - t_start;
        float dt = d_tile_density[tile_id];

        if (dt >= TAU1) {
            half* sY_h = (half*)sY;
            int total_halfs = t_cnt * K;
            for (int i = tid; i < total_halfs; i += total_threads) {
                int col_idx_local = i / K;
                int f_idx = i % K;
                int global_col = tile_colidx[t_start + col_idx_local];
                sY_h[col_idx_local * K + f_idx] = __float2half(d_Feat[global_col * K + f_idx]);
            }
            __syncthreads();

            #pragma unroll
            for (int r = 0; r < RPT; r++) {
                int local_row = row_base + r * DZ + z_th;
                int global_row = tx * XB + local_row;
                if (global_row < rows) {
                    long long seg_base = (long long)tile_id * XB + local_row;
                    int s_start = seg_ptr[seg_base];
                    int s_end   = seg_ptr[seg_base + 1];
                    float l_v[RR][RC]; float r_v[RR];
                    #pragma unroll
                    for (int rr = 0; rr < RR; rr++) { r_v[rr] = 0.0f;
                        #pragma unroll
                        for (int cc = 0; cc < RC; cc++) l_v[rr][cc] = 0.0f; }
                    for (int i = s_start; i < s_end; i++) {
                        int local_c = seg_colidx[i];
                        float fr[RR], fc[RC];
                        #pragma unroll
                        for (int rr = 0; rr < RR; rr++) fr[rr] = __half2float(sY_h[local_c * K + r_th + rr * TR]);
                        #pragma unroll
                        for (int cc = 0; cc < RC; cc++) fc[cc] = __half2float(sY_h[local_c * K + c_th + cc * TC]);
                        #pragma unroll
                        for (int rr = 0; rr < RR; rr++)
                            #pragma unroll
                            for (int cc = 0; cc < RC; cc++) l_v[rr][cc] += fr[rr] * fc[cc];
                        if (c_th == 0) {
                            float rating = seg_values[i];
                            #pragma unroll
                            for (int rr = 0; rr < RR; rr++) r_v[rr] += fr[rr] * rating;
                        }
                    }
                    #pragma unroll
                    for (int rr = 0; rr < RR; rr++) {
                        #pragma unroll
                        for (int cc = 0; cc < RC; cc++) lhs_vals[r][rr][cc] += l_v[rr][cc];
                        if (c_th == 0) rhs_vals[r][rr] += r_v[rr];
                    }
                }
            }
            __syncthreads();
        } else {
            int total_floats = t_cnt * K;
            for (int i = tid; i < total_floats; i += total_threads) {
                int col_idx_local = i / K;
                int f_idx = i % K;
                int global_col = tile_colidx[t_start + col_idx_local];
                sY[col_idx_local * K + f_idx] = d_Feat[global_col * K + f_idx];
            }
            __syncthreads();

            #pragma unroll
            for (int r = 0; r < RPT; r++) {
                int local_row = row_base + r * DZ + z_th;
                int global_row = tx * XB + local_row;
                if (global_row < rows) {
                    long long seg_base = (long long)tile_id * XB + local_row;
                    int s_start = seg_ptr[seg_base];
                    int s_end   = seg_ptr[seg_base + 1];
                    float l_v[RR][RC]; float r_v[RR];
                    #pragma unroll
                    for (int rr = 0; rr < RR; rr++) { r_v[rr] = 0.0f;
                        #pragma unroll
                        for (int cc = 0; cc < RC; cc++) l_v[rr][cc] = 0.0f; }
                    for (int i = s_start; i < s_end; i++) {
                        int local_c = seg_colidx[i];
                        float fr[RR], fc[RC];
                        #pragma unroll
                        for (int rr = 0; rr < RR; rr++) fr[rr] = sY[local_c * K + r_th + rr * TR];
                        #pragma unroll
                        for (int cc = 0; cc < RC; cc++) fc[cc] = sY[local_c * K + c_th + cc * TC];
                        #pragma unroll
                        for (int rr = 0; rr < RR; rr++)
                            #pragma unroll
                            for (int cc = 0; cc < RC; cc++) l_v[rr][cc] += fr[rr] * fc[cc];
                        if (c_th == 0) {
                            float rating = seg_values[i];
                            #pragma unroll
                            for (int rr = 0; rr < RR; rr++) r_v[rr] += fr[rr] * rating;
                        }
                    }
                    #pragma unroll
                    for (int rr = 0; rr < RR; rr++) {
                        #pragma unroll
                        for (int cc = 0; cc < RC; cc++) lhs_vals[r][rr][cc] += l_v[rr][cc];
                        if (c_th == 0) rhs_vals[r][rr] += r_v[rr];
                    }
                }
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int r = 0; r < RPT; r++) {
        int local_row = row_base + r * DZ + z_th;
        int global_row = tx * XB + local_row;
        int buf_row = global_row - batch_start;
        if (global_row < rows && buf_row >= 0 && buf_row < batch_size) {
            #pragma unroll
            for (int rr = 0; rr < RR; rr++) {
                int gr = r_th + rr * TR;
                #pragma unroll
                for (int cc = 0; cc < RC; cc++) {
                    int gc = c_th + cc * TC;
                    atomicAdd(&d_LHS_all[(long long)buf_row * K * K + gc * K + gr], lhs_vals[r][rr][cc]);
                }
                if (c_th == 0) atomicAdd(&d_RHS_all[global_row * K + gr], rhs_vals[r][rr]);
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// BALS symmetric-tile Gram kernel (dx=dy=DX register tiling, lower triangle
// only) — the mapping the BALS paper (and cuMF's get_hermitian) use to beat
// cuMF_ALS, ported onto our BALS 2D tile format.
//
// Difference from compute_LHS_RHS_BALS_block (the faithful-but-slow baseline):
//   * Each ACTIVE thread owns ONE DX*DX output sub-tile (ti,tj) of the K*K
//     Gram matrix, with ti>=tj — i.e. only the NB*(NB+1)/2 lower-triangular
//     tiles (NB = K/DX). That HALVES the flops (the baseline computed the full
//     symmetric matrix) and RAISES arithmetic intensity to DX FMAs per smem
//     load (DX=4 -> 2x the baseline's 2x2=1-FMA/load; the baseline's tiny 2x2
//     register tile is what capped it at ~265 GFlop/s).
//   * Off-diagonal tiles are written to BOTH halves (transpose) so the solver
//     still receives the full symmetric K*K; diagonal tiles fill their block
//     once. RHS (X = Y^T r) is produced by the diagonal-tile owner of each
//     tile-row, so every output index is written exactly once.
// SAME as the baseline: BALS tile format, per-tile distinct-column smem load
// reused across the tile-row's rows, job/chunk list, entity batching, ROW_SPLIT
// via gridDim.y, atomic accumulation. FP32 only (only_scalar forces fp32, so
// the density/half branch is dropped here). RPT is kept templated but the host
// launches RPT=1 for every K (DZ*ROW_SPLIT = XB), which keeps the register
// accumulator at DX*DX+DX floats — no spill even under __launch_bounds__(1024).
// Layout: threadIdx.x = lower-triangular tile index (0..NT-1 active, padded to a
// warp so extra lanes still help the smem load), threadIdx.z = one of DZ rows
// processed CONCURRENTLY (this is what packs the block up to 1024 threads for
// latency hiding — measured ~15% faster than a no-idle-lane 576-thread flat
// layout, because this kernel is latency/occupancy-bound, not compute-bound).
// gridDim.y = ROW_SPLIT sibling blocks cover the remaining XB rows. Each thread
// keeps ONE DX*DX register accumulator (RPT=1), so register pressure is tiny.
template <int RPT, int DX>
__global__ void __launch_bounds__(1024, 1)
compute_LHS_RHS_BALS_symtile(int rows, int K, float lambda,
                             const int* __restrict__ tile_ptr,
                             const int* __restrict__ tile_colidx,
                             const int* __restrict__ seg_ptr,
                             const int* __restrict__ seg_colidx,
                             const float* __restrict__ seg_values,
                             const float* __restrict__ d_Feat,
                             float* __restrict__ d_LHS_all,
                             float* __restrict__ d_RHS_all,
                             const float* __restrict__ d_tile_density, // unused: fp32 forced
                             const int* __restrict__ nz_tile_list,
                             const int* __restrict__ nz_tile_ptr,
                             const int* __restrict__ job_tx,
                             const int* __restrict__ job_chunk,
                             const int* __restrict__ job_nchunks,
                             int batch_start, int batch_size) {
    constexpr int NB = K_DIM / DX;              // tiles per dimension
    constexpr int NT = NB * (NB + 1) / 2;       // lower-triangular tiles = active threads

    int job = blockIdx.x;
    int tx  = job_tx[job];
    if (tx * XB >= batch_start + batch_size || tx * XB + XB <= batch_start) return;
    int chunk   = job_chunk[job];
    int nchunks = job_nchunks[job];

    const int t   = threadIdx.x;                // tile index (0..NT-1 active, rest load-only)
    const int z   = threadIdx.z;                // row-plane within the block
    const int DZ  = blockDim.z;
    const int NTP = blockDim.x;                 // padded thread count (multiple of 32)
    const int row_base = blockIdx.y * (RPT * DZ);

    // Lower-triangular tile index t -> (ti, tj), ti>=tj, row-major.
    int ti = 0, tj = 0;
    {
        int base = 0;
        #pragma unroll
        for (int rr = 0; rr < NB; rr++) {
            if (t >= base && t < base + rr + 1) { ti = rr; tj = t - base; }
            base += rr + 1;
        }
    }

    float acc[RPT][DX][DX];
    float rhs[RPT][DX];
    #pragma unroll
    for (int p = 0; p < RPT; p++) {
        #pragma unroll
        for (int a = 0; a < DX; a++) { rhs[p][a] = 0.0f;
            #pragma unroll
            for (int b = 0; b < DX; b++) acc[p][a][b] = 0.0f; }
    }

    extern __shared__ float sY[];
    int nz_start = nz_tile_ptr[tx];
    int nz_end   = nz_tile_ptr[tx + 1];
    int tid = z * NTP + t;
    int total_threads = NTP * DZ;

    for (int nz_idx = nz_start + chunk; nz_idx < nz_end; nz_idx += nchunks) {
        int tile_id = nz_tile_list[nz_idx];
        int t_start = tile_ptr[tile_id];
        int t_cnt   = tile_ptr[tile_id + 1] - t_start;

        // Cooperative load of this column-tile's distinct Y columns into smem
        // (every thread helps, including the NT..NTP-1 compute-idle ones).
        int total_floats = t_cnt * K;
        for (int i = tid; i < total_floats; i += total_threads) {
            int c = i / K, f = i % K;
            sY[c * K + f] = d_Feat[(long long)tile_colidx[t_start + c] * K + f];
        }
        __syncthreads();

        if (t < NT) {
            #pragma unroll
            for (int p = 0; p < RPT; p++) {
                int local_row  = row_base + p * DZ + z;
                int global_row = tx * XB + local_row;
                if (global_row < rows) {
                    long long seg_base = (long long)tile_id * XB + local_row;
                    int s_start = seg_ptr[seg_base];
                    int s_end   = seg_ptr[seg_base + 1];
                    for (int i = s_start; i < s_end; i++) {
                        const float* col = &sY[seg_colidx[i] * K];
                        float fr[DX], fc[DX];
                        #pragma unroll
                        for (int a = 0; a < DX; a++) fr[a] = col[ti * DX + a];
                        #pragma unroll
                        for (int b = 0; b < DX; b++) fc[b] = col[tj * DX + b];
                        #pragma unroll
                        for (int a = 0; a < DX; a++)
                            #pragma unroll
                            for (int b = 0; b < DX; b++) acc[p][a][b] += fr[a] * fc[b];
                        if (ti == tj) {
                            float rating = seg_values[i];
                            #pragma unroll
                            for (int a = 0; a < DX; a++) rhs[p][a] += fr[a] * rating;
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (t < NT) {
        #pragma unroll
        for (int p = 0; p < RPT; p++) {
            int local_row  = row_base + p * DZ + z;
            int global_row = tx * XB + local_row;
            int buf_row    = global_row - batch_start;
            if (global_row < rows && buf_row >= 0 && buf_row < batch_size) {
                float* L = &d_LHS_all[(long long)buf_row * K * K];
                #pragma unroll
                for (int a = 0; a < DX; a++) {
                    int gr = ti * DX + a;
                    #pragma unroll
                    for (int b = 0; b < DX; b++) {
                        int gc = tj * DX + b;
                        atomicAdd(&L[(long long)gc * K + gr], acc[p][a][b]);   // lower (gr,gc)
                        if (ti != tj)
                            atomicAdd(&L[(long long)gr * K + gc], acc[p][a][b]); // transpose (gc,gr)
                    }
                    if (ti == tj)
                        atomicAdd(&d_RHS_all[(long long)global_row * K + gr], rhs[p][a]);
                }
            }
        }
    }
}

// FAST_RMSE convergence-check kernel (07-21). The COO kernel below runs at
// ~3.8 ns/rating on Netflix: both factor gathers are fully random, so every
// rating pays two cold ~384 B row reads (~62 GB/s effective). This variant
// walks the TRAIN CSR (or a test-side CSR built once on the host) one warp
// per entity: the X row is read once into registers, the Y gathers hit the
// fp16 factor table (Netflix K=96: 3.4 MB — mostly L2-resident), and each
// rating costs three coalesced 64 B row segments + a warp reduction.
// Numerics: predictions come from the SAME fp16-rounded factors the WMMA
// Gram path trains with (fp32 accumulate), so the RMSE it reports differs
// from the fp32-exact value only in the 4th decimal — fine for the delta<tol
// convergence decision. Reported finals stay fp32-exact: the host runs the
// COO kernel once after convergence.
template <int KD>
__global__ void compute_RMSE_csr_half(const int* __restrict__ offs, const int* __restrict__ cols,
                                      const float* __restrict__ vals, int num_ent,
                                      const half* __restrict__ Xh, const half* __restrict__ Yh,
                                      double* __restrict__ d_sq_err) {
    constexpr int RPL = (KD + 31) / 32;          // X-row elements per lane
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    __shared__ double s_err[8];                  // one slot per warp @ blockDim 256
    double acc = 0.0;
    if (warp < num_ent) {
        float x[RPL];
        #pragma unroll
        for (int r = 0; r < RPL; r++) {
            int k = lane + 32 * r;
            x[r] = (k < KD) ? __half2float(Xh[(long long)warp * KD + k]) : 0.0f;
        }
        const int j1 = offs[warp + 1];
        for (int j = offs[warp]; j < j1; j++) {
            const int it = cols[j];
            float p = 0.0f;
            #pragma unroll
            for (int r = 0; r < RPL; r++) {
                int k = lane + 32 * r;
                if (k < KD) p += x[r] * __half2float(Yh[(long long)it * KD + k]);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xffffffffu, p, o);
            if (lane == 0) { float e = vals[j] - p; acc += (double)(e * e); }
        }
    }
    const int wid = threadIdx.x >> 5;
    if (lane == 0) s_err[wid] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int w = 0; w < (int)(blockDim.x >> 5); w++) t += s_err[w];
        atomicAdd(d_sq_err, t);
    }
}

// const __restrict__ on all read-only pointers (same 07-04 fix as the other
// kernels — this one was missed): routes the user/item factor gathers through
// the read-only texture path. Per-thread math and order unchanged.
__global__ void compute_RMSE_kernel(const int* __restrict__ users, const int* __restrict__ items,
                                    const float* __restrict__ ratings, int nnz,
                                    const float* __restrict__ d_X, const float* __restrict__ d_Y,
                                    int K, double* __restrict__ d_sq_err) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    __shared__ double s_err[256];
    double thread_err = 0.0;
    if (idx < nnz) {
        int u = users[idx], it = items[idx];
        float pred = 0.0f;
        if ((K & 3) == 0) {
            // float4 loads; product order matches the scalar loop → bit-identical
            const float4* x4 = (const float4*)(d_X + (long long)u  * K);
            const float4* y4 = (const float4*)(d_Y + (long long)it * K);
            for (int k = 0; k < (K >> 2); k++) {
                float4 a = x4[k], b = y4[k];
                pred += a.x * b.x;
                pred += a.y * b.y;
                pred += a.z * b.z;
                pred += a.w * b.w;
            }
        } else {
            for (int k = 0; k < K; k++) pred += d_X[u * K + k] * d_Y[it * K + k];
        }
        float err = ratings[idx] - pred;
        thread_err = (double)(err * err);
    }
    s_err[tid] = thread_err;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_err[tid] += s_err[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(d_sq_err, s_err[0]);
}
