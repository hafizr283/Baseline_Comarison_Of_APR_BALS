// bench_mp.cu — FIXLOG SYNC-2026-07-20 checklist step 1: numeric sanity +
// ms/launch for cholesky_solve_tiled_mp (FP16/BF16 tiles, ±refine) and
// cholesky_stale_refine against the production FP32 tiled kernel and a CPU
// fp64 reference, on 60k synthetic SPD systems (same generator as
// bench_cholesky.cu). Build with the PRODUCTION header so exactly the shipped
// kernels are measured:
//   nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=96 -DCHOL_MP=1 -DCHOL_STALE=1 \
//        [-DCHOL_REFINE=0] -I<comparing_baseline> bench_mp.cu -o bench_mp
// (CHOL_MP=1 compiles BOTH half and bf16 instantiations; CHOL_REFINE is a
//  build flag -> build twice for the ±IR comparison.)
#include "common.cuh"
#include "cholesky_kernels.cuh"

#define CK(x) do { cudaError_t e = (x); if (e) { printf("CUDA err %s @%d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while (0)

// Same symmetric pseudo-random SPD generator as bench_cholesky.cu, plus a
// magnitude scale (scale>~680 at K=96 pushes the diag past FP16 max 65504 —
// the overflow-guard probe).
__device__ __forceinline__ float hash01(unsigned s) {
    s ^= s >> 16; s *= 0x7feb352dU; s ^= s >> 15; s *= 0x846ca68bU; s ^= s >> 16;
    return (float)(s & 0xffffff) / (float)0x1000000;
}
__global__ void fill_spd(float* A, float* X, int K, int n, float scale) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n * K * K;
    if (idx >= total) return;
    int ent = (int)(idx / (K * K));
    int e = (int)(idx % (K * K));
    int i = e / K, j = e % K;
    int lo = min(i, j), hi = max(i, j);
    float v = 2.0f * hash01(ent * 131071u + lo * 521u + hi * 31u + 7u) - 1.0f;
    if (i == j) v += (float)K;
    A[idx] = v * scale;
    if (e < K) X[(long long)ent * K + e] = 2.0f * hash01(ent * 92821u + e * 13u + 3u) - 1.0f;
}

static void cpu_chol_solve(const std::vector<float>& A, const std::vector<float>& b,
                           std::vector<double>& x, int K, double lambda) {
    std::vector<double> L(K * K, 0.0);
    for (int i = 0; i < K; i++)
        for (int j = i; j < K; j++) L[j * K + i] = A[i * K + j];   // upper -> lower
    for (int i = 0; i < K; i++) L[i * K + i] += lambda;
    for (int j = 0; j < K; j++) {
        double s = L[j * K + j];
        for (int k = 0; k < j; k++) s -= L[j * K + k] * L[j * K + k];
        L[j * K + j] = sqrt(s);
        for (int i = j + 1; i < K; i++) {
            double v = L[i * K + j];
            for (int k = 0; k < j; k++) v -= L[i * K + k] * L[j * K + k];
            L[i * K + j] = v / L[j * K + j];
        }
    }
    x.assign(K, 0.0);
    std::vector<double> y(K);
    for (int i = 0; i < K; i++) {
        double s = b[i];
        for (int k = 0; k < i; k++) s -= L[i * K + k] * y[k];
        y[i] = s / L[i * K + i];
    }
    for (int i = K - 1; i >= 0; i--) {
        double s = y[i];
        for (int k = i + 1; k < K; k++) s -= L[k * K + i] * x[k];
        x[i] = s / L[i * K + i];
    }
}

static void set_max_shared(const void* fn) {
    cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
}

struct Timed { float ms; double maxrel; double meanrel; int fails; };

// Accuracy vs the CPU fp64 subset + kernel-only timing (2 warmup + 10 reps).
template <typename Launch>
Timed run_case(Launch launch, float* d_X, float* d_X0, int K, int n,
               const std::vector<double>& xref_cpu, int ncheck, int* d_fail) {
    size_t xb = (size_t)n * K * sizeof(float);
    if (d_fail) CK(cudaMemset(d_fail, 0, sizeof(int)));
    CK(cudaMemcpy(d_X, d_X0, xb, cudaMemcpyDeviceToDevice));
    launch(d_X, n);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    int fails = 0;
    if (d_fail) CK(cudaMemcpy(&fails, d_fail, sizeof(int), cudaMemcpyDeviceToHost));
    double maxrel = 0.0, sumrel = 0.0; long long cnt = 0;
    if (!xref_cpu.empty()) {
        std::vector<float> x((size_t)ncheck * K);
        CK(cudaMemcpy(x.data(), d_X, (size_t)ncheck * K * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < x.size(); i++) {
            double d = fabs((double)x[i] - xref_cpu[i]) / (fabs(xref_cpu[i]) + 1e-6);
            if (d > maxrel) maxrel = d;
            sumrel += d; cnt++;
        }
    }
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    for (int r = 0; r < 2; r++) { CK(cudaMemcpy(d_X, d_X0, xb, cudaMemcpyDeviceToDevice)); launch(d_X, n); }
    CK(cudaDeviceSynchronize());
    float tot = 0;
    for (int r = 0; r < 10; r++) {
        CK(cudaMemcpy(d_X, d_X0, xb, cudaMemcpyDeviceToDevice));
        cudaEventRecord(t0); launch(d_X, n); cudaEventRecord(t1);
        CK(cudaEventSynchronize(t1));
        float ms; cudaEventElapsedTime(&ms, t0, t1); tot += ms;
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return { tot / 10.0f, maxrel, cnt ? sumrel / cnt : 0.0, fails };
}

template <int K, int NTH_PROD>
void bench_K(int n, float lambda) {
    printf("\n================ K = %d  (n = %d systems, lambda = %.3f, CHOL_REFINE=%d) ================\n",
           K, n, lambda, (int)CHOL_REFINE);
    float *d_A, *d_X, *d_X0; int* d_fail;
    CK(cudaMalloc(&d_A, (size_t)n * K * K * sizeof(float)));
    CK(cudaMalloc(&d_X, (size_t)n * K * sizeof(float)));
    CK(cudaMalloc(&d_X0, (size_t)n * K * sizeof(float)));
    CK(cudaMalloc(&d_fail, sizeof(int)));
    long long total = (long long)n * K * K;
    fill_spd<<<(int)((total + 255) / 256), 256>>>(d_A, d_X0, K, n, 1.0f);
    CK(cudaDeviceSynchronize());

    std::vector<int2> hmap; int nvec = 0;
    build_cholesky_tile_map(K, hmap, nvec);
    int ntot = (int)hmap.size();
    int2* d_map; CK(cudaMalloc(&d_map, ntot * sizeof(int2)));
    CK(cudaMemcpy(d_map, hmap.data(), ntot * sizeof(int2), cudaMemcpyHostToDevice));

    // CPU fp64 reference on the first ncheck systems
    const int ncheck = 512;
    std::vector<double> xref_cpu((size_t)ncheck * K);
    {
        std::vector<float> A((size_t)ncheck * K * K), b((size_t)ncheck * K);
        CK(cudaMemcpy(A.data(), d_A, A.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), d_X0, b.size() * sizeof(float), cudaMemcpyDeviceToHost));
        for (int s = 0; s < ncheck; s++) {
            std::vector<float> As(A.begin() + (size_t)s * K * K, A.begin() + (size_t)(s + 1) * K * K);
            std::vector<float> bs(b.begin() + (size_t)s * K, b.begin() + (size_t)(s + 1) * K);
            std::vector<double> xs;
            cpu_chol_solve(As, bs, xs, K, lambda);
            for (int i = 0; i < K; i++) xref_cpu[(size_t)s * K + i] = xs[i];
        }
    }

    set_max_shared((const void*)cholesky_solve_tiled<K, NTH_PROD>);
    Timed ref = run_case([&](float* X, int nn) {
        cholesky_solve_tiled<K, NTH_PROD><<<nn, NTH_PROD, cholesky_tiled_smem<K>()>>>(d_A, X, d_map, nvec, ntot, nn, lambda);
    }, d_X, d_X0, K, n, xref_cpu, ncheck, nullptr);
    printf("%-34s %8.3f ms   maxrel %.2e  meanrel %.2e\n", "fp32 tiled (production)", ref.ms, ref.maxrel, ref.meanrel);

#define MP_CASE(ST, STNAME, NTH) do {                                                                  \
        set_max_shared((const void*)cholesky_solve_tiled_mp<K, NTH, ST>);                              \
        Timed t = run_case([&](float* X, int nn) {                                                     \
            cholesky_solve_tiled_mp<K, NTH, ST><<<nn, NTH, cholesky_tiled_smem_mp<K, ST>()>>>(         \
                d_A, X, d_map, nvec, ntot, nn, lambda, d_fail, (ST*)nullptr);                          \
        }, d_X, d_X0, K, n, xref_cpu, ncheck, d_fail);                                                 \
        printf("%-34s %8.3f ms   maxrel %.2e  meanrel %.2e  fails %d\n",                               \
               STNAME " NTH=" #NTH, t.ms, t.maxrel, t.meanrel, t.fails);                               \
    } while (0)

    MP_CASE(__half, "fp16 tiles", 96);
    MP_CASE(__half, "fp16 tiles", 128);
    MP_CASE(__half, "fp16 tiles", 192);
    MP_CASE(__nv_bfloat16, "bf16 tiles", 96);
    MP_CASE(__nv_bfloat16, "bf16 tiles", 128);
    MP_CASE(__nv_bfloat16, "bf16 tiles", 192);
#undef MP_CASE

#if CHOL_STALE
    // Stale kernel: export the cache with a fresh mp solve, then time the
    // stale path against the SAME A (drift behavior is an ALS-level question;
    // this checks trisolve+IR machinery and the per-launch cost).
    {
        typedef __nv_bfloat16 ST;
        ST* d_Lc; CK(cudaMalloc(&d_Lc, (size_t)n * chol_cache_elems<K>() * sizeof(ST)));
        set_max_shared((const void*)cholesky_solve_tiled_mp<K, NTH_PROD, ST>);
        CK(cudaMemcpy(d_X, d_X0, (size_t)n * K * sizeof(float), cudaMemcpyDeviceToDevice));
        cholesky_solve_tiled_mp<K, NTH_PROD, ST><<<n, NTH_PROD, cholesky_tiled_smem_mp<K, ST>()>>>(
            d_A, d_X, d_map, nvec, ntot, n, lambda, d_fail, d_Lc);
        CK(cudaDeviceSynchronize());
        set_max_shared((const void*)cholesky_stale_refine<K, NTH_PROD, ST>);
        Timed t = run_case([&](float* X, int nn) {
            cholesky_stale_refine<K, NTH_PROD, ST><<<nn, NTH_PROD, cholesky_stale_smem<K, ST>()>>>(
                d_A, X, d_Lc, nn, lambda, d_fail);
        }, d_X, d_X0, K, n, xref_cpu, ncheck, d_fail);
        printf("%-34s %8.3f ms   maxrel %.2e  meanrel %.2e  fails %d\n",
               "stale-refine bf16 (exact-A)", t.ms, t.maxrel, t.meanrel, t.fails);
        CK(cudaFree(d_Lc));
    }
#endif

    // FP16 overflow probe: scale the systems so the diag passes 65504 —
    // every solve must be zeroed+counted, never NaN-poisoned silently.
    if (K == 96) {
        fill_spd<<<(int)((total + 255) / 256), 256>>>(d_A, d_X0, K, n, 800.0f);
        CK(cudaDeviceSynchronize());
        set_max_shared((const void*)cholesky_solve_tiled_mp<K, NTH_PROD, __half>);
        Timed t = run_case([&](float* X, int nn) {
            cholesky_solve_tiled_mp<K, NTH_PROD, __half><<<nn, NTH_PROD, cholesky_tiled_smem_mp<K, __half>()>>>(
                d_A, X, d_map, nvec, ntot, nn, lambda, d_fail, (__half*)nullptr);
        }, d_X, d_X0, K, n, {}, 0, d_fail);
        printf("%-34s %8.3f ms   fails %d / %d  (expect all: diag ~%.0f > 65504)\n",
               "fp16 overflow probe scale=800", t.ms, t.fails, n, 800.0 * K);
        // and bf16 must survive the same scale
        set_max_shared((const void*)cholesky_solve_tiled_mp<K, NTH_PROD, __nv_bfloat16>);
        std::vector<double> xr2((size_t)ncheck * K);
        {
            std::vector<float> A((size_t)ncheck * K * K), b((size_t)ncheck * K);
            CK(cudaMemcpy(A.data(), d_A, A.size() * sizeof(float), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), d_X0, b.size() * sizeof(float), cudaMemcpyDeviceToHost));
            for (int s = 0; s < ncheck; s++) {
                std::vector<float> As(A.begin() + (size_t)s * K * K, A.begin() + (size_t)(s + 1) * K * K);
                std::vector<float> bs(b.begin() + (size_t)s * K, b.begin() + (size_t)(s + 1) * K);
                std::vector<double> xs;
                cpu_chol_solve(As, bs, xs, K, lambda);
                for (int i = 0; i < K; i++) xr2[(size_t)s * K + i] = xs[i];
            }
        }
        Timed t2 = run_case([&](float* X, int nn) {
            cholesky_solve_tiled_mp<K, NTH_PROD, __nv_bfloat16><<<nn, NTH_PROD, cholesky_tiled_smem_mp<K, __nv_bfloat16>()>>>(
                d_A, X, d_map, nvec, ntot, nn, lambda, d_fail, (__nv_bfloat16*)nullptr);
        }, d_X, d_X0, K, n, xr2, ncheck, d_fail);
        printf("%-34s %8.3f ms   maxrel %.2e  meanrel %.2e  fails %d (expect 0)\n",
               "bf16 tiles scale=800", t2.ms, t2.maxrel, t2.meanrel, t2.fails);
    }

    CK(cudaFree(d_A)); CK(cudaFree(d_X)); CK(cudaFree(d_X0));
    CK(cudaFree(d_map)); CK(cudaFree(d_fail));
}

int main(int argc, char** argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 60000;
    float lambda = 0.1f;
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("GPU: %s | n=%d | CHOL_REFINE=%d\n", p.name, n, (int)CHOL_REFINE);
    bench_K<32, 64>(n, lambda);
    bench_K<48, 96>(n, lambda);
    bench_K<64, 128>(n, lambda);
    bench_K<96, 128>(n, lambda);
    return 0;
}
