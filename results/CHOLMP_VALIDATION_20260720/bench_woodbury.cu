// bench_woodbury.cu — prototype + measurement for the Woodbury light-entity
// solve (external proposal, 2026-07-20 fact-check). For an entity with
// nnz < K, instead of Cholesky on A = Y^T Y + λI (K×K):
//   x = (1/λ)[ b − Y^T z ],  z = (G + λI)^{-1} (Y b),  G = Y Y^T (nnz×nnz)
// One block per entity; Y_Ω gathered as FP16 (same data source as the
// production WMMA-built LHS); all math FP32. Compared against the production
// tiled FP32 and CHOL_MP-fp16 solvers on the SAME systems (their K×K LHS is
// prebuilt from the same fp16-rounded Y, untimed) and a CPU fp64 reference.
// Build:
//   nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=96 -DCHOL_MP=2 -DCHOL_REFINE=0 \
//        -I<comparing_baseline> bench_woodbury.cu -o bench_woodbury
#include "common.cuh"
#include "cholesky_kernels.cuh"

#define CK(x) do { cudaError_t e = (x); if (e) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

__device__ __forceinline__ float whash01(unsigned s) {
    s ^= s >> 16; s *= 0x7feb352dU; s ^= s >> 15; s *= 0x846ca68bU; s ^= s >> 16;
    return (float)(s & 0xffffff) / (float)0x1000000;
}

// Item-feature pool, post-convergence-ish scale: U(-a, a), a = 2/sqrt(K).
__global__ void fill_pool(half* Y, int items, int K) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= items * K) return;
    float a = 2.0f / sqrtf((float)K);
    Y[t] = __float2half(a * (2.0f * whash01(1234567u + t) - 1.0f));
}

__global__ void fill_lists(int* idx, float* r, int n, int nnz, int items) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * nnz) return;
    idx[t] = (int)(whash01(777u + t * 2654435761u) * items) % items;
    r[t]   = 1.0f + 4.0f * whash01(99991u + t);
}

// Untimed: build the K×K LHS (upper triangle, row-major, garbage lower half —
// exactly the production layout) from the SAME fp16 Y rows + weighted λ diag,
// and the RHS b = Y^T r. One block per entity.
template <int K>
__global__ void build_A_b(const half* __restrict__ Y, const int* __restrict__ idx,
                          const float* __restrict__ r, float* __restrict__ A,
                          float* __restrict__ B, int n, int nnz, float lambda0) {
    int ent = blockIdx.x; if (ent >= n) return;
    const int* my = idx + (long long)ent * nnz;
    const float* mr = r + (long long)ent * nnz;
    float* myA = A + (long long)ent * K * K;
    float* myB = B + (long long)ent * K;
    for (int p = threadIdx.x; p < K * (K + 1) / 2 + K; p += blockDim.x) {
        if (p < K * (K + 1) / 2) {
            int i = (int)((sqrtf(8.0f * p + 1.0f) - 1.0f) * 0.5f);
            while (i * (i + 1) / 2 > p) i--;
            while ((i + 1) * (i + 2) / 2 <= p) i++;
            int j = p - i * (i + 1) / 2;          // j <= i ; store upper: A[j][i]
            float acc = 0.0f;
            for (int t = 0; t < nnz; t++) {
                const half* yr = Y + (long long)my[t] * K;
                acc += __half2float(yr[i]) * __half2float(yr[j]);
            }
            if (i == j) acc += lambda0 * nnz;
            myA[(long long)j * K + i] = acc;
        } else {
            int k = p - K * (K + 1) / 2;
            float acc = 0.0f;
            for (int t = 0; t < nnz; t++)
                acc += mr[t] * __half2float(Y[(long long)my[t] * K + k]);
            myB[k] = acc;
        }
    }
}

// ── The Woodbury kernel ────────────────────────────────────────────────────
// smem: sYh [MAXN*K] half | sG [MAXN*(MAXN+1)] float (stride MAXN+1, lower
// used) | sb [K] | sw [MAXN] | sz [MAXN] | sIdx [MAXN] int | sR [MAXN]
template <int K, int MAXN, int NTH>
__global__ void woodbury_solve(const half* __restrict__ Y, const int* __restrict__ idx,
                               const float* __restrict__ r, float* __restrict__ X,
                               int n, int nnz, float lambda0) {
    int ent = blockIdx.x; if (ent >= n) return;
    extern __shared__ unsigned char wraw[];
    half*  sYh  = (half*)wraw;
    float* sG   = (float*)(wraw + MAXN * K * sizeof(half));
    float* sb   = sG + MAXN * (MAXN + 1);
    float* sw   = sb + K;
    float* sz   = sw + MAXN;
    float* sR   = sz + MAXN;
    int*   sIdx = (int*)(sR + MAXN);
    const int tid = threadIdx.x;
    const int GS = MAXN + 1;
    const float lam = lambda0 * nnz;

    for (int i = tid; i < nnz; i += NTH) {
        sIdx[i] = idx[(long long)ent * nnz + i];
        sR[i]   = r[(long long)ent * nnz + i];
    }
    __syncthreads();
    // gather Y_Ω rows (coalesced along K)
    for (int t = tid; t < nnz * K; t += NTH)
        sYh[t] = Y[(long long)sIdx[t / K] * K + (t % K)];
    __syncthreads();
    // b = Y^T r  (K dots of length nnz)
    for (int k = tid; k < K; k += NTH) {
        float acc = 0.0f;
        for (int i = 0; i < nnz; i++) acc += sR[i] * __half2float(sYh[i * K + k]);
        sb[k] = acc;
    }
    // G = Y Y^T lower triangle + λ_e diag (nnz(nnz+1)/2 dots of length K)
    for (int p = tid; p < nnz * (nnz + 1) / 2; p += NTH) {
        int i = (int)((sqrtf(8.0f * p + 1.0f) - 1.0f) * 0.5f);
        while (i * (i + 1) / 2 > p) i--;
        while ((i + 1) * (i + 2) / 2 <= p) i++;
        int j = p - i * (i + 1) / 2;
        const half *ri = sYh + i * K, *rj = sYh + j * K;
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; k++) acc += __half2float(ri[k]) * __half2float(rj[k]);
        sG[i * GS + j] = (i == j) ? acc + lam : acc;
    }
    __syncthreads();
    // in-place right-looking Cholesky on sG (lower), column loop.
    // Trailing update is ROW-strided (no triangular index decode — the
    // per-element isqrt decode was O(n^3) decodes, the 07-06 packed-solver
    // mistake) and the scale step is folded into the same barrier window.
    for (int j = 0; j < nnz; j++) {
        if (tid == 0) sG[j * GS + j] = sqrtf(fmaxf(sG[j * GS + j], 1e-20f));
        __syncthreads();
        float inv = 1.0f / sG[j * GS + j];
        for (int i = j + 1 + tid; i < nnz; i += NTH) sG[i * GS + j] *= inv;
        __syncthreads();
        for (int i = j + 1 + tid; i < nnz; i += NTH) {
            float lij = sG[i * GS + j];
            for (int c = j + 1; c <= i; c++) sG[i * GS + c] -= lij * sG[c * GS + j];
        }
        __syncthreads();
    }
    // w = Y b (nnz dots of length K)
    for (int i = tid; i < nnz; i += NTH) {
        const half* ri = sYh + i * K;
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; k++) acc += __half2float(ri[k]) * sb[k];
        sw[i] = acc;
    }
    __syncthreads();
    // solve L z' = w, L^T z = z'  (warp 0, serial-ish; O(nnz^2) tiny)
    if (tid < 32) {
        for (int i = 0; i < nnz; i++) {
            float acc = 0.0f;
            for (int k = tid; k < i; k += 32) acc += sG[i * GS + k] * sz[k];
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
            if (tid == 0) sz[i] = (sw[i] - acc) / sG[i * GS + i];
            __syncwarp();
        }
        for (int i = nnz - 1; i >= 0; i--) {
            float acc = 0.0f;
            for (int k = i + 1 + tid; k < nnz; k += 32) acc += sG[k * GS + i] * sw[k];
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
            if (tid == 0) sw[i] = (sz[i] - acc) / sG[i * GS + i];   // sw now holds z
            __syncwarp();
        }
    }
    __syncthreads();
    // x = (b − Y^T z) / λ_e
    for (int k = tid; k < K; k += NTH) {
        float acc = 0.0f;
        for (int i = 0; i < nnz; i++) acc += __half2float(sYh[i * K + k]) * sw[i];
        X[(long long)ent * K + k] = (sb[k] - acc) / lam;
    }
}

// Variant B: one WARP per entity, WPB warps per block, __syncwarp only.
// Trades the block-barrier skeleton for per-warp independence; smem is
// partitioned per warp (same per-entity layout as the block variant).
template <int K, int MAXN, int WPB>
__global__ void woodbury_solve_warp(const half* __restrict__ Y, const int* __restrict__ idx,
                                    const float* __restrict__ r, float* __restrict__ X,
                                    int n, int nnz, float lambda0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int ent = blockIdx.x * WPB + warp;
    if (ent >= n) return;
    constexpr int GS = MAXN + 1;
    constexpr int WB = MAXN * K * (int)sizeof(half)
                     + (MAXN * GS + K + 3 * MAXN) * (int)sizeof(float)
                     + MAXN * (int)sizeof(int);
    extern __shared__ unsigned char wraw2[];
    unsigned char* my = wraw2 + (size_t)warp * WB;
    half*  sYh  = (half*)my;
    float* sG   = (float*)(my + MAXN * K * sizeof(half));
    float* sb   = sG + MAXN * GS;
    float* sw   = sb + K;
    float* sz   = sw + MAXN;
    float* sR   = sz + MAXN;
    int*   sIdx = (int*)(sR + MAXN);
    const float lam = lambda0 * nnz;

    for (int i = lane; i < nnz; i += 32) {
        sIdx[i] = idx[(long long)ent * nnz + i];
        sR[i]   = r[(long long)ent * nnz + i];
    }
    __syncwarp();
    for (int t = lane; t < nnz * K; t += 32)
        sYh[t] = Y[(long long)sIdx[t / K] * K + (t % K)];
    __syncwarp();
    for (int k = lane; k < K; k += 32) {
        float acc = 0.0f;
        for (int i = 0; i < nnz; i++) acc += sR[i] * __half2float(sYh[i * K + k]);
        sb[k] = acc;
    }
    for (int p = lane; p < nnz * (nnz + 1) / 2; p += 32) {
        int i = (int)((sqrtf(8.0f * p + 1.0f) - 1.0f) * 0.5f);
        while (i * (i + 1) / 2 > p) i--;
        while ((i + 1) * (i + 2) / 2 <= p) i++;
        int j = p - i * (i + 1) / 2;
        const half *ri = sYh + i * K, *rj = sYh + j * K;
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; k++) acc += __half2float(ri[k]) * __half2float(rj[k]);
        sG[i * GS + j] = (i == j) ? acc + lam : acc;
    }
    __syncwarp();
    for (int j = 0; j < nnz; j++) {
        if (lane == 0) sG[j * GS + j] = sqrtf(fmaxf(sG[j * GS + j], 1e-20f));
        __syncwarp();
        float inv = 1.0f / sG[j * GS + j];
        for (int i = j + 1 + lane; i < nnz; i += 32) sG[i * GS + j] *= inv;
        __syncwarp();
        for (int i = j + 1 + lane; i < nnz; i += 32) {
            float lij = sG[i * GS + j];
            for (int c = j + 1; c <= i; c++) sG[i * GS + c] -= lij * sG[c * GS + j];
        }
        __syncwarp();
    }
    for (int i = lane; i < nnz; i += 32) {
        const half* ri = sYh + i * K;
        float acc = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < K; k++) acc += __half2float(ri[k]) * sb[k];
        sw[i] = acc;
    }
    __syncwarp();
    for (int i = 0; i < nnz; i++) {
        float acc = 0.0f;
        for (int k = lane; k < i; k += 32) acc += sG[i * GS + k] * sz[k];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        acc = __shfl_sync(0xffffffff, acc, 0);
        if (lane == 0) sz[i] = (sw[i] - acc) / sG[i * GS + i];
        __syncwarp();
    }
    for (int i = nnz - 1; i >= 0; i--) {
        float acc = 0.0f;
        for (int k = i + 1 + lane; k < nnz; k += 32) acc += sG[k * GS + i] * sw[k];
        for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
        if (lane == 0) sw[i] = (sz[i] - acc) / sG[i * GS + i];
        __syncwarp();
    }
    __syncwarp();
    for (int k = lane; k < K; k += 32) {
        float acc = 0.0f;
        for (int i = 0; i < nnz; i++) acc += __half2float(sYh[i * K + k]) * sw[i];
        X[(long long)ent * K + k] = (sb[k] - acc) / lam;
    }
}

template <int K, int MAXN>
constexpr int woodbury_smem() {
    return MAXN * K * (int)sizeof(half)
         + (MAXN * (MAXN + 1) + K + 3 * MAXN) * (int)sizeof(float)
         + MAXN * (int)sizeof(int);
}

// CPU fp64 reference from the SAME fp16-rounded Y
static void cpu_ref(const std::vector<uint16_t>& Yh, const std::vector<int>& idx,
                    const std::vector<float>& r, std::vector<double>& x,
                    int K, int nnz, float lambda0, long long ent) {
    auto h2f = [](uint16_t h) { __half hh; memcpy(&hh, &h, 2); return (double)__half2float(hh); };
    std::vector<double> A((size_t)K * K, 0.0), b(K, 0.0);
    for (int t = 0; t < nnz; t++) {
        long long row = (long long)idx[ent * nnz + t] * K;
        for (int i = 0; i < K; i++) {
            double yi = h2f(Yh[row + i]);
            b[i] += (double)r[ent * nnz + t] * yi;
            for (int j = 0; j <= i; j++) A[i * K + j] += yi * h2f(Yh[row + j]);
        }
    }
    double lam = (double)lambda0 * nnz;
    for (int i = 0; i < K; i++) A[i * K + i] += lam;
    // fp64 Cholesky solve (lower stored)
    for (int j = 0; j < K; j++) {
        double s = A[j * K + j];
        for (int k = 0; k < j; k++) s -= A[j * K + k] * A[j * K + k];
        A[j * K + j] = sqrt(s);
        for (int i = j + 1; i < K; i++) {
            double v = A[i * K + j];
            for (int k = 0; k < j; k++) v -= A[i * K + k] * A[j * K + k];
            A[i * K + j] = v / A[j * K + j];
        }
    }
    std::vector<double> y(K);
    for (int i = 0; i < K; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= A[i * K + k] * y[k];
        y[i] = s / A[i * K + i];
    }
    x.assign(K, 0.0);
    for (int i = K - 1; i >= 0; i--) {
        double s = y[i];
        for (int k = i + 1; k < K; k++) s -= A[k * K + i] * x[k];
        x[i] = s / A[i * K + i];
    }
}

static void set_max_shared(const void* fn) {
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
}

template <typename Launch>
float time_solver(Launch launch, float* d_X, const float* d_B, size_t xbytes) {
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int w = 0; w < 2; w++) { CK(cudaMemcpy(d_X, d_B, xbytes, cudaMemcpyDeviceToDevice)); launch(); }
    CK(cudaDeviceSynchronize());
    float tot = 0;
    for (int rep = 0; rep < 10; rep++) {
        CK(cudaMemcpy(d_X, d_B, xbytes, cudaMemcpyDeviceToDevice));
        cudaEventRecord(t0); launch(); cudaEventRecord(t1);
        CK(cudaEventSynchronize(t1));
        float ms; cudaEventElapsedTime(&ms, t0, t1); tot += ms;
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return tot / 10.0f;
}

static void acc_check(const float* d_X, const std::vector<std::vector<double>>& refs,
                      int K, int ncheck, double* maxrel, double* meanrel) {
    std::vector<float> x((size_t)ncheck * K);
    CK(cudaMemcpy(x.data(), d_X, x.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double mx = 0, sum = 0; long long cnt = 0;
    for (int s = 0; s < ncheck; s++)
        for (int i = 0; i < K; i++) {
            double d = fabs((double)x[(size_t)s * K + i] - refs[s][i]) / (fabs(refs[s][i]) + 1e-6);
            if (d > mx) mx = d; sum += d; cnt++;
        }
    *maxrel = mx; *meanrel = sum / cnt;
}

int main(int argc, char** argv) {
    const int K = 96;
    int n = (argc > 1) ? atoi(argv[1]) : 60000;
    int items = 20000;
    float lambda0 = 0.048f;   // weighted per-rating λ (production tune)
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s | K=%d n=%d | weighted lambda0=%.3f | Y pool fp16\n", p.name, K, n, lambda0);

    half* d_Y; CK(cudaMalloc(&d_Y, (size_t)items * K * sizeof(half)));
    fill_pool<<<(items * K + 255) / 256, 256>>>(d_Y, items, K);

    std::vector<int2> hmap; int nvec = 0;
    build_cholesky_tile_map(K, hmap, nvec);
    int ntot = (int)hmap.size();
    int2* d_map; CK(cudaMalloc(&d_map, ntot * sizeof(int2)));
    CK(cudaMemcpy(d_map, hmap.data(), ntot * sizeof(int2), cudaMemcpyHostToDevice));

    float *d_A, *d_B, *d_X; int* d_fail;
    CK(cudaMalloc(&d_A, (size_t)n * K * K * sizeof(float)));
    CK(cudaMalloc(&d_B, (size_t)n * K * sizeof(float)));
    CK(cudaMalloc(&d_X, (size_t)n * K * sizeof(float)));
    CK(cudaMalloc(&d_fail, sizeof(int))); CK(cudaMemset(d_fail, 0, sizeof(int)));

    std::vector<uint16_t> Yh((size_t)items * K);
    CK(cudaMemcpy(Yh.data(), d_Y, Yh.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    const int ncheck = 256;
    int buckets[] = {8, 16, 24, 32, 48, 64, 76};
    printf("\n%6s | %12s | %12s | %26s\n", "nnz", "tiled fp32", "tiled fp16", "woodbury (maxrel/meanrel)");
    for (int bi = 0; bi < 7; bi++) {
        int nnz = buckets[bi];
        int *d_idx; float *d_r;
        CK(cudaMalloc(&d_idx, (size_t)n * nnz * sizeof(int)));
        CK(cudaMalloc(&d_r,   (size_t)n * nnz * sizeof(float)));
        fill_lists<<<(int)(((long long)n * nnz + 255) / 256), 256>>>(d_idx, d_r, n, nnz, items);
        build_A_b<K><<<n, 128>>>(d_Y, d_idx, d_r, d_A, d_B, n, nnz, lambda0);
        CK(cudaDeviceSynchronize());

        std::vector<int> hidx((size_t)ncheck * nnz);
        std::vector<float> hr((size_t)ncheck * nnz);
        CK(cudaMemcpy(hidx.data(), d_idx, hidx.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hr.data(), d_r, hr.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<std::vector<double>> refs(ncheck);
        for (int s = 0; s < ncheck; s++) cpu_ref(Yh, hidx, hr, refs[s], K, nnz, lambda0, s);

        size_t xb = (size_t)n * K * sizeof(float);
        set_max_shared((const void*)cholesky_solve_tiled<K, 128>);
        float ms32 = time_solver([&] {
            cholesky_solve_tiled<K, 128><<<n, 128, cholesky_tiled_smem<K>()>>>(d_A, d_X, d_map, nvec, ntot, n, 0.0f);
        }, d_X, d_B, xb);
        double mr32, er32; acc_check(d_X, refs, K, ncheck, &mr32, &er32);

        set_max_shared((const void*)cholesky_solve_tiled_mp<K, 128, half>);
        float ms16 = time_solver([&] {
            cholesky_solve_tiled_mp<K, 128, half><<<n, 128, cholesky_tiled_smem_mp<K, half>()>>>(d_A, d_X, d_map, nvec, ntot, n, 0.0f, d_fail, (half*)nullptr);
        }, d_X, d_B, xb);
        double mr16, er16; acc_check(d_X, refs, K, ncheck, &mr16, &er16);

        float msw; double mrw, erw;
        if (nnz <= 32) {
            set_max_shared((const void*)woodbury_solve<K, 32, 64>);
            msw = time_solver([&] {
                woodbury_solve<K, 32, 64><<<n, 64, woodbury_smem<K, 32>()>>>(d_Y, d_idx, d_r, d_X, n, nnz, lambda0);
            }, d_X, d_B, xb);
        } else {
            set_max_shared((const void*)woodbury_solve<K, 76, 128>);
            msw = time_solver([&] {
                woodbury_solve<K, 76, 128><<<n, 128, woodbury_smem<K, 76>()>>>(d_Y, d_idx, d_r, d_X, n, nnz, lambda0);
            }, d_X, d_B, xb);
        }
        CK(cudaGetLastError());
        acc_check(d_X, refs, K, ncheck, &mrw, &erw);

        // Variant B: warp-per-entity (WPB=4 -> 128 threads/block)
        float msb; double mrb, erb;
        if (nnz <= 32) {
            constexpr int SM32 = (32 * 96 * 2 + (32 * 33 + 96 + 3 * 32) * 4 + 32 * 4) * 4;
            cudaFuncSetAttribute((const void*)woodbury_solve_warp<K, 32, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM32);
            set_max_shared((const void*)woodbury_solve_warp<K, 32, 4>);
            msb = time_solver([&] {
                woodbury_solve_warp<K, 32, 4><<<(n + 3) / 4, 128, SM32>>>(d_Y, d_idx, d_r, d_X, n, nnz, lambda0);
            }, d_X, d_B, xb);
        } else {
            constexpr int SM76 = (76 * 96 * 2 + (76 * 77 + 96 + 3 * 76) * 4 + 76 * 4) * 2;
            cudaFuncSetAttribute((const void*)woodbury_solve_warp<K, 76, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM76);
            set_max_shared((const void*)woodbury_solve_warp<K, 76, 2>);
            msb = time_solver([&] {
                woodbury_solve_warp<K, 76, 2><<<(n + 1) / 2, 64, SM76>>>(d_Y, d_idx, d_r, d_X, n, nnz, lambda0);
            }, d_X, d_B, xb);
        }
        CK(cudaGetLastError());
        acc_check(d_X, refs, K, ncheck, &mrb, &erb);

        printf("%6d | %8.3f ms  | %8.3f ms  | %8.3f ms  %.1e/%.1e  (vs fp32 %.2fx, vs fp16 %.2fx)\n",
               nnz, ms32, ms16, msw, mrw, erw, ms32 / msw, ms16 / msw);
        printf("       |  acc %.0e/%.0e | acc %.0e/%.0e | warpB %8.3f ms  %.1e/%.1e (vs fp16 %.2fx)\n",
               mr32, er32, mr16, er16, msb, mrb, erb, ms16 / msb);
        CK(cudaFree(d_idx)); CK(cudaFree(d_r));
    }
    return 0;
}
