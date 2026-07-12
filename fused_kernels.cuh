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
