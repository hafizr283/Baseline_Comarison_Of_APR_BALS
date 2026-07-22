#include "common.cuh"
#include "data_utils.cuh"
#include "fused_kernels.cuh"
#include "wmma_kernels.cuh"
#include "cholesky_kernels.cuh"

// Regularization mode — ported from main_experiment.cu (SYNC-2026-07-12) so
// the CHOL_MP/CHOL_STALE experiments can run in the production regime: plain
// λ·I at λ=0.1 gives κ(A) ~ ||A||/λ ~ 1e5+ on heavy Netflix entities, which
// breaks a 16-bit factorization outright (measured 2026-07-20: all four mp
// variants diverge in this binary with WEIGHTED_LAMBDA=0). Weighted-λ caps
// κ at O(K/λ) for every entity. Default stays 0: every legacy justapr record
// (plain λ trajectories) remains reproducible with no flags.
#ifndef WEIGHTED_LAMBDA
#define WEIGHTED_LAMBDA 0
#endif
#ifndef CUMF_INIT
#define CUMF_INIT 0
#endif
#if WEIGHTED_LAMBDA
// cumf_als parity (als.cu "weighted-lambda regularization"):
//   tt[diag] += (end - start) * lambda   with (end-start) = entity's train nnz.
// Runs on the compute stream after each batch's LHS accumulation; the solvers
// then get λ=0 so nothing is added twice. nnz=0 entities fall back to plain λ.
__global__ void add_weighted_lambda_diag(float* __restrict__ d_LHS_all,
                                         const int* __restrict__ d_nnz,
                                         int batch_start, int batch_n,
                                         int K, float lambda) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)batch_n * K) return;
    int e = (int)(idx / K);
    int d = (int)(idx % K);
    int n = d_nnz[batch_start + e];
    d_LHS_all[(long long)e * K * K + (long long)d * K + d] += lambda * (n > 0 ? n : 1);
}
#endif

// ── Heavy-ball momentum / DIIS-style extrapolation of the factor sequence
// (EXPERIMENT, SYNC-2026-07-21). ALS is a contractive fixed-point map near its
// solution; extrapolating along the update direction lowers the effective
// spectral radius and can reach the same fixed point in fewer sweeps. After a
// factor is solved this iteration (X_raw), we set
//     X_used = X_raw + beta*(X_raw - X_raw_prev)
// and store X_raw into the prev buffer. At the fixed point X_raw==X_raw_prev so
// the extrapolation term vanishes -> SAME converged factors (parity gate =
// final fp32 RMSE). beta==0 on iter 0 (no valid prev) just seeds the prev
// buffer. Guarded behind MOMENTUM: the default build is byte-identical.
#ifndef MOMENTUM
#define MOMENTUM 0
#endif
#if MOMENTUM
__global__ void momentum_extrapolate(float* __restrict__ x, float* __restrict__ xprev,
                                     long long n, float beta) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float raw = x[i];
    x[i]     = raw + beta * (raw - xprev[i]);
    xprev[i] = raw;
}
#endif

int main(int argc, char* argv[]) {
    printf("=== CODE VERSION: SYNC-2026-07-20 [CHOL-MP 16-bit tiled Cholesky + optional IR + stale-L cache — UNVALIDATED ON GPU; CHOL_MP=0 path = 07-06 code] ===\n");
    printf("Tiled Cholesky precision: %s%s\n", CHOL_MP_STR, CHOL_STALE ? " + stale-L cache" : "");
    printf("Regularization: %s | Init: %s\n",
           WEIGHTED_LAMBDA ? "weighted-lambda ALS-WR (diag += nnz*lambda)" : "plain lambda*I (legacy)",
           CUMF_INIT ? "cuMF-scale U(0,0.2)" : "legacy U(0.1,1.1)");
    const char* csv_path = (argc > 1) ? argv[1]
        : "/content/drive/MyDrive/gpu programming/gpu_last_time/IncludingDataset/netflix_ratings.csv";
    vector<int> raw_users, raw_items, test_users_h, test_items_h;
    vector<float> raw_ratings, test_ratings_h;
    vector<int> h_user_offsets, h_item_indices, h_item_offsets, h_user_indices;
    vector<float> h_user_ratings, h_item_ratings;
    int num_users = 0, num_items = 0;
    int nnz_train = 0, nnz_test = 0;

    string path_str = csv_path;
    if (path_str.length() >= 4 && path_str.substr(path_str.length() - 4) == ".bin") {
        cout << "Loading binary preprocessed dataset: " << csv_path << endl;
        FILE* f = fopen(csv_path, "rb");
        if (!f) { cerr << "Error opening binary file!" << endl; return 1; }
        int version;
        if(fread(&version, sizeof(int), 1, f) != 1) { cerr << "Error reading version" << endl; return 1; }
        if(fread(&num_users, sizeof(int), 1, f) != 1) { cerr << "Error reading num_users" << endl; return 1; }
        if(fread(&num_items, sizeof(int), 1, f) != 1) { cerr << "Error reading num_items" << endl; return 1; }
        if(fread(&nnz_train, sizeof(int), 1, f) != 1) { cerr << "Error reading nnz_train" << endl; return 1; }
        if(fread(&nnz_test, sizeof(int), 1, f) != 1) { cerr << "Error reading nnz_test" << endl; return 1; }

        auto read_vec_int = [&](vector<int>& v) {
            size_t sz; if(fread(&sz, sizeof(size_t), 1, f) != 1) return;
            v.resize(sz); if(sz > 0) { if(fread(v.data(), sizeof(int), sz, f) != sz) return; }
        };
        auto read_vec_float = [&](vector<float>& v) {
            size_t sz; if(fread(&sz, sizeof(size_t), 1, f) != 1) return;
            v.resize(sz); if(sz > 0) { if(fread(v.data(), sizeof(float), sz, f) != sz) return; }
        };

        read_vec_int(raw_users); read_vec_int(raw_items); read_vec_float(raw_ratings);
        read_vec_int(test_users_h); read_vec_int(test_items_h); read_vec_float(test_ratings_h);
        read_vec_int(h_user_offsets); read_vec_int(h_item_indices); read_vec_float(h_user_ratings);
        read_vec_int(h_item_offsets); read_vec_int(h_user_indices); read_vec_float(h_item_ratings);
        fclose(f);
        printf("%d users, %d items | train=%d, test=%d\n", num_users, num_items, nnz_train, nnz_test);
    } else {
        cerr << "ERROR: Please run 'preprocess' on the CSV first and pass the .bin file to this program!" << endl;
        return 1;
    }

    int K         = K_DIM;
    float lambda  = (argc > 2) ? atof(argv[2]) : 0.1f;
    // Weighted-λ is FUSED into the solvers (07-21): every solver takes an
    // optional per-entity nnz pointer and adds λ·nnz on its own diag pass —
    // bit-identical to the retired add_weighted_lambda_diag pre-pass, which
    // cost 27 ms/iter of pure RMW traffic at Netflix K=96. All solver call
    // sites now pass `lambda` + `d_nnzw` (nullptr in plain mode).
    int max_iters = 50;
    float tol     = 0.001f;
    int max_ent   = max(num_users, num_items);
    int yb_eff    = (K > 32) ? 128 : YB;

    srand(42);
    vector<float> h_X(num_users * K), h_Y(num_items * K);
#if CUMF_INIT
    // cuMF-scale init U(0,0.2) — same sequence as main_experiment.cu CUMF_INIT.
    for (int i = 0; i < num_users * K; i++) h_X[i] = 0.2f * ((float)rand() / (float)RAND_MAX);
    for (int i = 0; i < num_items * K; i++) h_Y[i] = 0.2f * ((float)rand() / (float)RAND_MAX);
#else
    for (int i = 0; i < num_users * K; i++) h_X[i] = 0.1f + (rand() % 100) / 100.0f;
    for (int i = 0; i < num_items * K; i++) h_Y[i] = 0.1f + (rand() % 100) / 100.0f;
#endif
    vector<float> h_X_init = h_X, h_Y_init = h_Y;

#if WEIGHTED_LAMBDA
    // Per-entity train nnz for weighted-λ (CSR offsets from the frozen .bin).
    int *d_u_nnz, *d_i_nnz;
    {
        vector<int> h_u_nnz(num_users), h_i_nnz(num_items);
        for (int u = 0; u < num_users; u++) h_u_nnz[u] = h_user_offsets[u + 1] - h_user_offsets[u];
        for (int i = 0; i < num_items; i++) h_i_nnz[i] = h_item_offsets[i + 1] - h_item_offsets[i];
        cudaMalloc(&d_u_nnz, sizeof(int) * num_users);
        cudaMalloc(&d_i_nnz, sizeof(int) * num_items);
        cudaMemcpy(d_u_nnz, h_u_nnz.data(), sizeof(int) * num_users, cudaMemcpyHostToDevice);
        cudaMemcpy(d_i_nnz, h_i_nnz.data(), sizeof(int) * num_items, cudaMemcpyHostToDevice);
    }
#endif

    float *d_X, *d_Y;
    cudaMalloc(&d_X,       sizeof(float) * num_users * K);
    cudaMalloc(&d_Y,       sizeof(float) * num_items * K);
    // Persistent FP16 feature copies for the WMMA gather path (converted once
    // per phase; halves the gather read bandwidth vs FP32 + per-element convert)
    half *d_X_half, *d_Y_half;
    cudaMalloc(&d_X_half,  sizeof(half) * num_users * K);
    cudaMalloc(&d_Y_half,  sizeof(half) * num_items * K);
    int lhs_batch = min(ENTITY_BATCH_SIZE, max_ent);
    // Double-buffered LHS + dual-stream pipeline (ported from
    // main_experiment.cu SYNC-2026-07-04d): the Cholesky solve of batch b
    // (stream sS) overlaps the LHS+RHS of batch b+1 (stream sC). Phase-level
    // ops stay on the legacy NULL stream, which joins both pipes and provides
    // all cross-phase ordering. Bit-identical math; scheduling only.
    float* d_LHS_buf[2];
    int    nbuf = 1;
    cudaMalloc(&d_LHS_buf[0], sizeof(float) * (long long)lhs_batch * K * K);
    if (cudaMalloc(&d_LHS_buf[1], sizeof(float) * (long long)lhs_batch * K * K) == cudaSuccess) {
        nbuf = 2;
    } else {
        cudaGetLastError();
        d_LHS_buf[1] = d_LHS_buf[0];
        printf("WARN: 2nd LHS buffer OOM -> solve/LHS+RHS overlap disabled (serial fallback)\n");
    }
    cudaStream_t sC, sS;
    {
        int prLo, prHi;
        cudaDeviceGetStreamPriorityRange(&prLo, &prHi);
        cudaStreamCreate(&sC);
        cudaStreamCreateWithPriority(&sS, cudaStreamDefault, prHi);
    }
    const int MAX_PIPE_EVT = 32;   // sync-only events (justapr has no per-batch solve timing)
    cudaEvent_t ev_lhs_u[MAX_PIPE_EVT], ev_lhs_i[MAX_PIPE_EVT], ev_sol_u[MAX_PIPE_EVT], ev_sol_i[MAX_PIPE_EVT];
    for (int i = 0; i < MAX_PIPE_EVT; i++) {
        cudaEventCreateWithFlags(&ev_lhs_u[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev_lhs_i[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev_sol_u[i], cudaEventDisableTiming);
        cudaEventCreateWithFlags(&ev_sol_i[i], cudaEventDisableTiming);
    }
    cudaMemcpy(d_X, h_X.data(), sizeof(float) * num_users * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y.data(), sizeof(float) * num_items * K, cudaMemcpyHostToDevice);

#if MOMENTUM
    float *d_Xprev = nullptr, *d_Yprev = nullptr;
    cudaMalloc(&d_Xprev, sizeof(float) * (long long)num_users * K);
    cudaMalloc(&d_Yprev, sizeof(float) * (long long)num_items * K);
    float mom_beta = 0.3f;   // validated optimum, Netflix K=96 (07-21): iter 20->15,
                             // 0 fails, test RMSE 0.81715 < baseline 0.81756. beta>=0.4
                             // overshoots (wobble delays convergence), 0.5 diverges.
    if (const char* e = getenv("MBETA")) mom_beta = atof(e);
    printf("[MOMENTUM] heavy-ball extrapolation ON, beta=%.3f\n", mom_beta);
#endif

    int   *d_train_users, *d_train_items, *d_test_users, *d_test_items;
    float *d_train_ratings, *d_test_ratings;
    double *d_sq_err;
    cudaMalloc(&d_train_users,   sizeof(int)   * nnz_train);
    cudaMalloc(&d_train_items,   sizeof(int)   * nnz_train);
    cudaMalloc(&d_train_ratings, sizeof(float) * nnz_train);
    cudaMalloc(&d_test_users,    sizeof(int)   * nnz_test);
    cudaMalloc(&d_test_items,    sizeof(int)   * nnz_test);
    cudaMalloc(&d_test_ratings,  sizeof(float) * nnz_test);
    cudaMalloc(&d_sq_err,        sizeof(double));
    cudaMemcpy(d_train_users,   raw_users.data(),       sizeof(int)   * nnz_train, cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_items,   raw_items.data(),       sizeof(int)   * nnz_train, cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_ratings, raw_ratings.data(),     sizeof(float) * nnz_train, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_users,    test_users_h.data(),    sizeof(int)   * nnz_test,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_items,    test_items_h.data(),    sizeof(int)   * nnz_test,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_ratings,  test_ratings_h.data(),  sizeof(float) * nnz_test,  cudaMemcpyHostToDevice);

#if FAST_RMSE
    // Test-side user CSR for the fast convergence-check RMSE (see common.cuh
    // FAST_RMSE). Built once on the host; only the fp64 summation order
    // changes relative to the COO enumeration.
    int *d_t_offsets, *d_t_colidx; float *d_t_vals;
    {
        vector<int> t_off(num_users + 1, 0), t_col(nnz_test);
        vector<float> t_val(nnz_test);
        for (int j = 0; j < nnz_test; j++) t_off[test_users_h[j] + 1]++;
        for (int u = 0; u < num_users; u++) t_off[u + 1] += t_off[u];
        vector<int> t_cur(t_off.begin(), t_off.end() - 1);
        for (int j = 0; j < nnz_test; j++) {
            int p = t_cur[test_users_h[j]]++;
            t_col[p] = test_items_h[j];
            t_val[p] = test_ratings_h[j];
        }
        cudaMalloc(&d_t_offsets, sizeof(int)   * (num_users + 1));
        cudaMalloc(&d_t_colidx,  sizeof(int)   * nnz_test);
        cudaMalloc(&d_t_vals,    sizeof(float) * nnz_test);
        cudaMemcpy(d_t_offsets, t_off.data(), sizeof(int)   * (num_users + 1), cudaMemcpyHostToDevice);
        cudaMemcpy(d_t_colidx,  t_col.data(), sizeof(int)   * nnz_test,        cudaMemcpyHostToDevice);
        cudaMemcpy(d_t_vals,    t_val.data(), sizeof(float) * nnz_test,        cudaMemcpyHostToDevice);
    }
#endif

    // Build BALS tile format
    vector<int>   user_tile_ptr, user_tile_colidx, user_seg_ptr, user_seg_colidx;
    vector<float> user_seg_values, user_tile_density;
    vector<int>   user_nz_tile_list, user_nz_tile_ptr;
    build_BALS_format(num_users, num_items, XB, yb_eff,
                      h_user_offsets, h_item_indices, h_user_ratings,
                      user_tile_ptr, user_tile_colidx,
                      user_seg_ptr, user_seg_colidx, user_seg_values,
                      user_tile_density, user_nz_tile_list, user_nz_tile_ptr);

    vector<int>   item_tile_ptr, item_tile_colidx, item_seg_ptr, item_seg_colidx;
    vector<float> item_seg_values, item_tile_density;
    vector<int>   item_nz_tile_list, item_nz_tile_ptr;
    build_BALS_format(num_items, num_users, XB, yb_eff,
                      h_item_offsets, h_user_indices, h_item_ratings,
                      item_tile_ptr, item_tile_colidx,
                      item_seg_ptr, item_seg_colidx, item_seg_values,
                      item_tile_density, item_nz_tile_list, item_nz_tile_ptr);

    printf("User: %d nonzero tiles out of %d (%.1f%% skipped)\n",
           (int)user_nz_tile_list.size(), (int)user_tile_density.size(),
           100.0f * (1.0f - (float)user_nz_tile_list.size() / user_tile_density.size()));
    printf("Item: %d nonzero tiles out of %d (%.1f%% skipped)\n",
           (int)item_nz_tile_list.size(), (int)item_tile_density.size(),
           100.0f * (1.0f - (float)item_nz_tile_list.size() / item_tile_density.size()));

    int u_fp16 = 0, u_fp32 = 0;
    for (float d : user_tile_density) { if (d >= TAU1) u_fp16++; else u_fp32++; }
    int i_fp16 = 0, i_fp32 = 0;
    for (float d : item_tile_density) { if (d >= TAU1) i_fp16++; else i_fp32++; }
    printf("APR-BALS Precision Tiers (User tiles): FP16 Tensor Core=%d (%.1f%%), FP32 Scalar=%d (%.1f%%)\n",
           u_fp16, 100.0f * u_fp16 / user_tile_density.size(),
           u_fp32, 100.0f * u_fp32 / user_tile_density.size());
    printf("APR-BALS Precision Tiers (Item tiles): FP16 Tensor Core=%d (%.1f%%), FP32 Scalar=%d (%.1f%%)\n",
           i_fp16, 100.0f * i_fp16 / item_tile_density.size(),
           i_fp32, 100.0f * i_fp32 / item_tile_density.size());

    auto print_density_stats = [](const vector<float>& d, const char* name) {
        float mn = 1e9f, mx = 0.0f, sum = 0.0f;
        int nonzero = 0;
        vector<float> sorted_d;
        for (float v : d) {
            if (v > 0) { sorted_d.push_back(v); nonzero++; }
            mn = min(mn, v); mx = max(mx, v); sum += v;
        }
        sort(sorted_d.begin(), sorted_d.end());
        printf("%s density stats: min=%.6f max=%.6f avg=%.6f nonzero=%d/%d\n",
               name, mn, mx, sum / d.size(), nonzero, (int)d.size());
        if (!sorted_d.empty()) {
            int n = (int)sorted_d.size();
            printf("  Nonzero percentiles: p25=%.6f p50=%.6f p75=%.6f p90=%.6f p99=%.6f\n",
                   sorted_d[n / 4], sorted_d[n / 2], sorted_d[3 * n / 4],
                   sorted_d[(int)(n * 0.9)], sorted_d[(int)(n * 0.99)]);
        }
    };
    print_density_stats(user_tile_density, "User");
    print_density_stats(item_tile_density, "Item");

    {
        long long u_nnz_fp16 = 0, u_nnz_fp32 = 0;
        for (int i = 0; i < (int)user_tile_density.size(); i++) {
            long long tnz = llroundf(user_tile_density[i] * XB * yb_eff);
            if (user_tile_density[i] >= TAU1) u_nnz_fp16 += tnz; else u_nnz_fp32 += tnz;
        }
        long long i_nnz_fp16 = 0, i_nnz_fp32 = 0;
        for (int i = 0; i < (int)item_tile_density.size(); i++) {
            long long tnz = llroundf(item_tile_density[i] * XB * yb_eff);
            if (item_tile_density[i] >= TAU1) i_nnz_fp16 += tnz; else i_nnz_fp32 += tnz;
        }
        printf("\n=== NNZ per Precision Tier ===\n");
        printf("User-side: FP16=%lld (%.1f%%)  FP32=%lld (%.1f%%)\n",
               u_nnz_fp16, 100.0*u_nnz_fp16/nnz_train, u_nnz_fp32, 100.0*u_nnz_fp32/nnz_train);
        printf("Item-side: FP16=%lld (%.1f%%)  FP32=%lld (%.1f%%)\n",
               i_nnz_fp16, 100.0*i_nnz_fp16/nnz_train, i_nnz_fp32, 100.0*i_nnz_fp32/nnz_train);
        printf("Thesis signal: dense tiles (FP16 path) hold %.1f%% of user-side train NNZ.\n",
               100.0*u_nnz_fp16/nnz_train);
        printf("If >40%%: precision dispatch covers most compute -> strong thesis.\n");
        printf("If <20%%: most compute in FP32 tiles -> thesis needs argument about dense-tile quality.\n");
    }

    // Upload tile data
    int *d_u_tile_ptr, *d_u_tile_colidx, *d_u_seg_ptr, *d_u_seg_colidx;
    float *d_u_seg_values;
    cudaMalloc(&d_u_tile_ptr,    sizeof(int)   * user_tile_ptr.size());
    cudaMalloc(&d_u_tile_colidx, sizeof(int)   * user_tile_colidx.size());
    cudaMalloc(&d_u_seg_ptr,     sizeof(int)   * user_seg_ptr.size());
    cudaMalloc(&d_u_seg_colidx,  sizeof(int)   * user_seg_colidx.size());
    cudaMalloc(&d_u_seg_values,  sizeof(float) * user_seg_values.size());
    cudaMemcpy(d_u_tile_ptr,    user_tile_ptr.data(),    sizeof(int)   * user_tile_ptr.size(),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_tile_colidx, user_tile_colidx.data(), sizeof(int)   * user_tile_colidx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_seg_ptr,     user_seg_ptr.data(),     sizeof(int)   * user_seg_ptr.size(),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_seg_colidx,  user_seg_colidx.data(),  sizeof(int)   * user_seg_colidx.size(),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_seg_values,  user_seg_values.data(),  sizeof(float) * user_seg_values.size(),  cudaMemcpyHostToDevice);

    float *d_u_tile_density;
    int   *d_u_nz_list, *d_u_nz_ptr;
    cudaMalloc(&d_u_tile_density, sizeof(float) * user_tile_density.size());
    cudaMemcpy(d_u_tile_density,  user_tile_density.data(), sizeof(float) * user_tile_density.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&d_u_nz_list, sizeof(int) * user_nz_tile_list.size());
    cudaMalloc(&d_u_nz_ptr,  sizeof(int) * user_nz_tile_ptr.size());
    cudaMemcpy(d_u_nz_list, user_nz_tile_list.data(), sizeof(int) * user_nz_tile_list.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_nz_ptr,  user_nz_tile_ptr.data(),  sizeof(int) * user_nz_tile_ptr.size(),  cudaMemcpyHostToDevice);

    int *d_i_tile_ptr, *d_i_tile_colidx, *d_i_seg_ptr, *d_i_seg_colidx;
    float *d_i_seg_values;
    cudaMalloc(&d_i_tile_ptr,    sizeof(int)   * item_tile_ptr.size());
    cudaMalloc(&d_i_tile_colidx, sizeof(int)   * item_tile_colidx.size());
    cudaMalloc(&d_i_seg_ptr,     sizeof(int)   * item_seg_ptr.size());
    cudaMalloc(&d_i_seg_colidx,  sizeof(int)   * item_seg_colidx.size());
    cudaMalloc(&d_i_seg_values,  sizeof(float) * item_seg_values.size());
    cudaMemcpy(d_i_tile_ptr,    item_tile_ptr.data(),    sizeof(int)   * item_tile_ptr.size(),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_tile_colidx, item_tile_colidx.data(), sizeof(int)   * item_tile_colidx.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_seg_ptr,     item_seg_ptr.data(),     sizeof(int)   * item_seg_ptr.size(),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_seg_colidx,  item_seg_colidx.data(),  sizeof(int)   * item_seg_colidx.size(),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_seg_values,  item_seg_values.data(),  sizeof(float) * item_seg_values.size(),  cudaMemcpyHostToDevice);

    float *d_i_tile_density;
    int   *d_i_nz_list, *d_i_nz_ptr;
    cudaMalloc(&d_i_tile_density, sizeof(float) * item_tile_density.size());
    cudaMemcpy(d_i_tile_density,  item_tile_density.data(), sizeof(float) * item_tile_density.size(), cudaMemcpyHostToDevice);
    cudaMalloc(&d_i_nz_list, sizeof(int) * item_nz_tile_list.size());
    cudaMalloc(&d_i_nz_ptr,  sizeof(int) * item_nz_tile_ptr.size());
    cudaMemcpy(d_i_nz_list, item_nz_tile_list.data(), sizeof(int) * item_nz_tile_list.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_nz_ptr,  item_nz_tile_ptr.data(),  sizeof(int) * item_nz_tile_ptr.size(),  cudaMemcpyHostToDevice);

    // Dense / giant entity split
    auto compute_n_dense = [&](const vector<int>& offsets, int N) -> int {
        if (K != 16 && K != 32 && K != 48 && K != 64 && K != 96) return 0;
        int cnt = 0;
        for (int u = 0; u < N; u++)
            if (offsets[u + 1] - offsets[u] >= DENSE_NNZ_THRESH) cnt++;
        return (cnt / XB) * XB;
    };
    int n_dense_u = compute_n_dense(h_user_offsets, num_users);
    int n_dense_i = compute_n_dense(h_item_offsets, num_items);
    int tx_split_u = n_dense_u / XB;
    int tx_split_i = n_dense_i / XB;

    auto compute_n_giant = [&](const vector<int>& offsets, int N, int n_dense) -> int {
        if (K != 16 && K != 32 && K != 48 && K != 64 && K != 96) return 0;
        int cnt = 0;
        for (int u = 0; u < N; u++)
            if (offsets[u + 1] - offsets[u] >= GIANT_NNZ_THRESH) cnt++;
        return (cnt < n_dense) ? cnt : n_dense;
    };
    int n_giant_u = compute_n_giant(h_user_offsets, num_users, n_dense_u);
    int n_giant_i = compute_n_giant(h_item_offsets, num_items, n_dense_i);
    printf("Giant split: User giants=%d, Item giants=%d (nnz >= %d, %d warps/giant)\n",
           n_giant_u, n_giant_i, GIANT_NNZ_THRESH, GIANT_WARPS);

    // Job lists for the scalar tile kernel
    auto build_jobs = [](const vector<int>& nz_ptr, int num_tiles_x, int tx_start, int chunk,
                         vector<int>& jtx, vector<int>& jchunk, vector<int>& jnch) {
        jtx.clear(); jchunk.clear(); jnch.clear();
        for (int tx = tx_start; tx < num_tiles_x; tx++) {
            int cnt = nz_ptr[tx + 1] - nz_ptr[tx];
            if (cnt <= 0) continue;
            int nch = (cnt + chunk - 1) / chunk;
            for (int c = 0; c < nch; c++) { jtx.push_back(tx); jchunk.push_back(c); jnch.push_back(nch); }
        }
    };
    int num_tiles_x_u = (num_users + XB - 1) / XB;
    int num_tiles_x_i = (num_items + XB - 1) / XB;
    vector<int> u_job_tx, u_job_chunk, u_job_nch, i_job_tx, i_job_chunk, i_job_nch;
    build_jobs(user_nz_tile_ptr, num_tiles_x_u, 0, TILES_PER_CHUNK, u_job_tx, u_job_chunk, u_job_nch);
    build_jobs(item_nz_tile_ptr, num_tiles_x_i, 0, TILES_PER_CHUNK, i_job_tx, i_job_chunk, i_job_nch);
    int u_num_jobs = (int)u_job_tx.size();
    int i_num_jobs = (int)i_job_tx.size();
    vector<int> u_sjob_tx, u_sjob_chunk, u_sjob_nch, i_sjob_tx, i_sjob_chunk, i_sjob_nch;
    build_jobs(user_nz_tile_ptr, num_tiles_x_u, tx_split_u, TILES_PER_CHUNK, u_sjob_tx, u_sjob_chunk, u_sjob_nch);
    build_jobs(item_nz_tile_ptr, num_tiles_x_i, tx_split_i, TILES_PER_CHUNK, i_sjob_tx, i_sjob_chunk, i_sjob_nch);
    int u_num_sjobs = (int)u_sjob_tx.size();
    int i_num_sjobs = (int)i_sjob_tx.size();
    printf("Load balancing: User jobs=%d (from %d tile-rows), Item jobs=%d (from %d tile-rows)\n",
           u_num_jobs, num_tiles_x_u, i_num_jobs, num_tiles_x_i);
    printf("Hybrid split: User dense=%d entities (%d tile-rows, %d sparse jobs), "
           "Item dense=%d entities (%d tile-rows, %d sparse jobs)\n",
           n_dense_u, tx_split_u, u_num_sjobs, n_dense_i, tx_split_i, i_num_sjobs);

    auto upload_jobs = [](const vector<int>& jt, const vector<int>& jc, const vector<int>& jn,
                          int** dt, int** dc, int** dn) {
        int n = (int)jt.size();
        cudaMalloc(dt, sizeof(int) * max(n, 1));
        cudaMalloc(dc, sizeof(int) * max(n, 1));
        cudaMalloc(dn, sizeof(int) * max(n, 1));
        if (n > 0) {
            cudaMemcpy(*dt, jt.data(), sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(*dc, jc.data(), sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(*dn, jn.data(), sizeof(int) * n, cudaMemcpyHostToDevice);
        }
    };
    int *d_u_job_tx, *d_u_job_chunk, *d_u_job_nch, *d_i_job_tx, *d_i_job_chunk, *d_i_job_nch;
    int *d_u_sjob_tx, *d_u_sjob_chunk, *d_u_sjob_nch, *d_i_sjob_tx, *d_i_sjob_chunk, *d_i_sjob_nch;
    upload_jobs(u_job_tx,  u_job_chunk,  u_job_nch,  &d_u_job_tx,  &d_u_job_chunk,  &d_u_job_nch);
    upload_jobs(i_job_tx,  i_job_chunk,  i_job_nch,  &d_i_job_tx,  &d_i_job_chunk,  &d_i_job_nch);
    upload_jobs(u_sjob_tx, u_sjob_chunk, u_sjob_nch, &d_u_sjob_tx, &d_u_sjob_chunk, &d_u_sjob_nch);
    upload_jobs(i_sjob_tx, i_sjob_chunk, i_sjob_nch, &d_i_sjob_tx, &d_i_sjob_chunk, &d_i_sjob_nch);

    // Plain CSR for the WMMA gather path
    int   *d_u_offsets, *d_u_colidx, *d_i_offsets, *d_i_colidx;
    float *d_u_csr_vals, *d_i_csr_vals;
    cudaMalloc(&d_u_offsets,  sizeof(int)   * h_user_offsets.size());
    cudaMalloc(&d_u_colidx,   sizeof(int)   * h_item_indices.size());
    cudaMalloc(&d_u_csr_vals, sizeof(float) * h_user_ratings.size());
    cudaMalloc(&d_i_offsets,  sizeof(int)   * h_item_offsets.size());
    cudaMalloc(&d_i_colidx,   sizeof(int)   * h_user_indices.size());
    cudaMalloc(&d_i_csr_vals, sizeof(float) * h_item_ratings.size());
    cudaMemcpy(d_u_offsets,  h_user_offsets.data(), sizeof(int)   * h_user_offsets.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_colidx,   h_item_indices.data(), sizeof(int)   * h_item_indices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_csr_vals, h_user_ratings.data(), sizeof(float) * h_user_ratings.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_offsets,  h_item_offsets.data(), sizeof(int)   * h_item_offsets.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_colidx,   h_user_indices.data(), sizeof(int)   * h_user_indices.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_i_csr_vals, h_item_ratings.data(), sizeof(float) * h_item_ratings.size(), cudaMemcpyHostToDevice);

    int shared_mem_size = yb_eff * K * sizeof(float);
    int solve_threads  = 32;
    int solve_smem     = (K * K + K) * sizeof(float);

    // Tiled-solver load scatter map (upper-tri memory index -> tiled smem
    // offset), built once per K on the host — see cholesky_solve_tiled.
    int2* d_chol_map = nullptr;
    int chol_nvec = 0, chol_ntot = 0;
    {
        std::vector<int2> hmap;
        build_cholesky_tile_map(K, hmap, chol_nvec);
        chol_ntot = (int)hmap.size();
        cudaMalloc(&d_chol_map, hmap.size() * sizeof(int2));
        cudaMemcpy(d_chol_map, hmap.data(), hmap.size() * sizeof(int2), cudaMemcpyHostToDevice);
    }

#if CHOL_MP != 0
    // Non-finite-solve counter for the mixed-precision solver (FP16 overflow
    // / breakdown guard — see CHOL_MP in common.cuh). Zeroed per training
    // run, reported in the profile block.
    int* d_chol_fail = nullptr;
    cudaMalloc(&d_chol_fail, sizeof(int));
    cudaMemset(d_chol_fail, 0, sizeof(int));
#endif
#if CHOL_STALE
    // Per-entity tiled-L cache (see CHOL_STALE in common.cuh). BF16 tile
    // image, refreshed on warmup/refresh iterations, read by the stale-
    // refine kernel in between.
    chol_store_t *d_Lcache_u = nullptr, *d_Lcache_i = nullptr;
    {
        size_t eu = (size_t)num_users * chol_cache_elems<K_DIM>();
        size_t ei = (size_t)num_items * chol_cache_elems<K_DIM>();
        cudaError_t r1 = cudaMalloc(&d_Lcache_u, eu * sizeof(chol_store_t));
        cudaError_t r2 = cudaMalloc(&d_Lcache_i, ei * sizeof(chol_store_t));
        if (r1 != cudaSuccess || r2 != cudaSuccess) {
            printf("CHOL_STALE: L-cache alloc FAILED (%.2f GB needed). Rebuild with a smaller -DENTITY_BATCH_SIZE (shrinks the 2 LHS buffers) or a smaller K.\n",
                   (double)(eu + ei) * sizeof(chol_store_t) / 1e9);
            return 1;
        }
        printf("CHOL_STALE: L-cache %.2f GB | warmup=%d full iters | refresh every %d iters | residual gate tau=%.2f\n",
               (double)(eu + ei) * sizeof(chol_store_t) / 1e9,
               (int)CHOL_STALE_WARMUP, (int)CHOL_STALE_REFRESH, (double)CHOL_STALE_TAU);
    }
    // Residual-gate work buffers (per-batch list of entities needing a full
    // re-solve, device-side count, and a per-run total for reporting).
    int *d_stale_cnt, *d_stale_list, *d_stale_total;
    cudaMalloc(&d_stale_cnt,   sizeof(int));
    cudaMalloc(&d_stale_list,  sizeof(int) * lhs_batch);
    cudaMalloc(&d_stale_total, sizeof(int));
    cudaMemset(d_stale_total, 0, sizeof(int));
#endif

#if CHOL_MP != 0
    // Single dispatch site for the mixed-precision tiled solve. CACHE is the
    // per-side L-cache base (d_Lcache_u / d_Lcache_i); its tokens are never
    // expanded when CHOL_STALE=0, so the symbols need not exist there.
#if CHOL_STALE
    // Stale iterations run the residual-gated two-pass: stale-refine flags
    // entities whose stale-L residual fails the τ gate into d_stale_list
    // (leaving their RHS untouched), then a list-driven full mp solve re-does
    // exactly those (device-side count -> no host sync, stream-ordered on sv;
    // it also refreshes their cache slots = per-entity adaptive refresh).
    #define CHOL_MP_SOLVE(KK, NTH, CACHE) do { \
        chol_store_t* Lc_b_ = (CACHE) + (long long)bstart * chol_cache_elems<KK>(); \
        if (chol_fresh) cholesky_solve_tiled_mp<KK, NTH, chol_store_t><<<bn, NTH, cholesky_tiled_smem_mp<KK, chol_store_t>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_chol_fail, Lc_b_, nullptr, nullptr, d_nnzw); \
        else { \
            cudaMemsetAsync(d_stale_cnt, 0, sizeof(int), sv); \
            cholesky_stale_refine<KK, NTH, chol_store_t><<<bn, NTH, cholesky_stale_smem<KK, chol_store_t>(), sv>>>(lhs_b, rhs_b, Lc_b_, bn, lambda, d_chol_fail, d_stale_cnt, d_stale_list, d_stale_total, (float)CHOL_STALE_TAU, d_nnzw); \
            cholesky_solve_tiled_mp<KK, NTH, chol_store_t><<<bn, NTH, cholesky_tiled_smem_mp<KK, chol_store_t>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_chol_fail, Lc_b_, d_stale_list, d_stale_cnt, d_nnzw); \
        } \
    } while (0)
#else
    #define CHOL_MP_SOLVE(KK, NTH, CACHE) \
        cholesky_solve_tiled_mp<KK, NTH, chol_store_t><<<bn, NTH, cholesky_tiled_smem_mp<KK, chol_store_t>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_chol_fail, (chol_store_t*)nullptr, nullptr, nullptr, d_nnzw)
#endif
#endif

#ifdef USE_CUSOLVER
    cusolverDnHandle_t cusolver_h;
    cusolverDnCreate(&cusolver_h);
    float **d_Aptr, **d_Bptr_u, **d_Bptr_i;
    int   *d_info, *d_info_single;
    cudaMalloc(&d_Aptr,        sizeof(float*) * lhs_batch);
    cudaMalloc(&d_Bptr_u,      sizeof(float*) * lhs_batch);
    cudaMalloc(&d_Bptr_i,      sizeof(float*) * lhs_batch);
    cudaMalloc(&d_info,        sizeof(int)    * lhs_batch);
    cudaMalloc(&d_info_single, sizeof(int));
    auto cusolver_solve = [&](float* d_LHS, float* d_RHS, float** d_Aptr_, float** d_Bptr_, int n,
                              const int* d_nnzw_) {
        build_ptr_array<<<(n + 255) / 256, 256>>>(d_Aptr_, d_LHS, (long long)K * K, n);
        build_ptr_array<<<(n + 255) / 256, 256>>>(d_Bptr_, d_RHS, (long long)K, n);
        long long tot = (long long)n * K;
        // cuSOLVER can't fuse the diag add, so this fallback path (unsupported
        // K only) keeps a pre-add kernel: weighted per-entity λ·nnz or plain λ.
#if WEIGHTED_LAMBDA
        add_weighted_lambda_diag<<<(int)((tot + 255) / 256), 256>>>(d_LHS, d_nnzw_, 0, n, K, lambda);
#else
        (void)d_nnzw_;
        add_lambda_diag<<<(int)((tot + 255) / 256), 256>>>(d_LHS, K, n, lambda);
#endif
        cusolverDnSpotrfBatched(cusolver_h, CUBLAS_FILL_MODE_LOWER, K, d_Aptr_, K, d_info, n);
        cusolverDnSpotrsBatched(cusolver_h, CUBLAS_FILL_MODE_LOWER, K, 1, d_Aptr_, K, d_Bptr_, K, d_info_single, n);
    };
#endif

    // Scalar kernel thread geometry
    // Register-tile factor. Pick the SMALLEST RR_C that still fills a full
    // 1024-thread block (plane*DZ=1024). Rationale, confirmed by ptxas -v:
    //   peak regs ~= persistent(RPT*RR_C^2 ~= K^2/32, FIXED at full occupancy)
    //               + inner l_v[RR][RC] block (RR_C^2, live in the hot nnz loop).
    // Persistent is fixed once all 1024 threads are used, so minimizing RR_C
    // minimizes the HOT inner block -> minimizes spill. Measured spill at RPT=2
    // grows fast with RR_C: RR4=288B, RR6=1296B, RR8=3232B, RR12=10136B. So do
    // NOT inflate RR_C (an earlier "K/8" attempt did and made K64/K96 WORSE than
    // the original 2/3).
    // Full-occupancy min-RR_C: K16->2, K32->2, K48->3, K64->2, K96->3.
    //   Coverage TRp*RR_C = K exact for {16,32,48,64,96}.
    #define RR_C ((K_DIM==48 || K_DIM==96) ? 3 : 2)
    #define RC_C ((K_DIM==48 || K_DIM==96) ? 3 : 2)
    int TRp = K / RR_C; if (TRp > 32) TRp = 32;
    int TCp = K / RC_C; if (TCp > 32) TCp = 32;
    int plane = TRp * TCp;
    int DZ = 1;
    if      (plane * 16 <= 1024) DZ = 16;
    else if (plane *  8 <= 1024) DZ =  8;
    else if (plane *  4 <= 1024) DZ =  4;
    else if (plane *  2 <= 1024) DZ =  2;
    else if (plane *  1 <= 1024) DZ =  1;
    else { cout << "Error: K=" << K << " too large." << endl; return 1; }
    dim3 lhs_threads_bals(TRp, TCp, DZ);

    // ROW_SPLIT: split each XB(=32)-row tile-row across gridDim.y sibling
    // blocks so the per-thread persistent accumulator block RPT*RR_C*RC_C
    // (= K^2/32 unsplit) stays register-resident under the 64-reg cap of
    // __launch_bounds__(1024,1). Unsplit, K>=48 spills these accumulators to
    // local memory and re-reads/re-writes them on every per-tile flush ->
    // the FP32 scalar path collapses (~230 -> 63 GFlops/s on Netflix K=48).
    // Full analysis + per-K ptxas numbers: see main_experiment.cu.
    #ifndef ROW_SPLIT
    #define ROW_SPLIT (K_DIM==16 ? 1 : K_DIM==32 ? 2 : K_DIM==48 ? 4 : 8)
    #endif
    if (XB % (DZ * ROW_SPLIT) != 0) {
        cout << "Error: ROW_SPLIT=" << ROW_SPLIT << " incompatible with DZ=" << DZ << endl;
        return 1;
    }
    int rows_per_thread = XB / (DZ * ROW_SPLIT);

    #define LAUNCH_FUSED_KERNEL(blocks, RPT_VAL, num_ents, d_TilePtr, d_TileCol, d_SegPtr, d_SegCol, d_SegVal, d_Feat, d_LHS, d_RHS, d_Dens, d_NzList, d_NzPtr, d_JobTx, d_JobChunk, d_JobNch, bstart, bsize) \
        compute_LHS_RHS_BALS_block<RPT_VAL, RR_C, RC_C><<<dim3((unsigned)(blocks), ROW_SPLIT), lhs_threads_bals, shared_mem_size, sC>>>( \
            num_ents, K, lambda, \
            d_TilePtr, d_TileCol, d_SegPtr, d_SegCol, d_SegVal, \
            d_Feat, d_LHS, d_RHS, d_Dens, d_NzList, d_NzPtr, \
            d_JobTx, d_JobChunk, d_JobNch, bstart, bsize)

    cudaFuncSetAttribute((const void*)compute_LHS_RHS_BALS_block<1, RR_C, RC_C>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    cudaFuncSetAttribute((const void*)compute_LHS_RHS_BALS_block<2, RR_C, RC_C>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    cudaFuncSetAttribute((const void*)compute_LHS_RHS_BALS_block<4, RR_C, RC_C>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    cudaFuncSetAttribute((const void*)compute_LHS_RHS_BALS_block<8, RR_C, RC_C>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    cudaFuncSetAttribute((const void*)compute_LHS_RHS_BALS_block<16, RR_C, RC_C>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    cudaFuncSetAttribute((const void*)compute_LHS_RHS_BALS_block<32, RR_C, RC_C>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);

    // UNIFORM smem carveout (MaxShared) for every pipeline kernel: kernels
    // with different L1/smem carveouts cannot co-execute on Ampere, which
    // would serialize the sC/sS solve-overlap pipeline (see
    // main_experiment.cu SYNC-2026-07-04d note). MaxL1 on the scalar kernel
    // is vestigial since ROW_SPLIT (stack 0-64 B fits any L1 slice).
    auto set_max_shared = [](const void* f) {
        cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    };
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<1, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<2, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<4, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<8, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<16, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<32, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_wmma<K_DIM/16>);
    set_max_shared((const void*)compute_LHS_RHS_wmma_giant<K_DIM/16, GIANT_WARPS>);
    set_max_shared((const void*)convert_fp32_to_fp16);
    set_max_shared((const void*)compute_RMSE_kernel);
#if K_DIM <= BATCHED_CHOLESKY_MAX_K
    // batched solver: zero smem, its per-thread local matrix lives on L1 -
    // leave at driver default (max L1). Forcing MaxShared starved it to a
    // 28 KB L1 slice and cost ~25% solve time at K=32 (measured 07-04).
#elif K_DIM == 16
    set_max_shared((const void*)cholesky_solve_packed<16>);
#elif K_DIM == 32
    set_max_shared((const void*)CHOL_TILED_SOLVER(32, 64));
#elif K_DIM == 48
    set_max_shared((const void*)CHOL_TILED_SOLVER(48, 96));
#elif K_DIM == 64
    set_max_shared((const void*)CHOL_TILED_SOLVER(64, 128));
#else
    set_max_shared((const void*)CHOL_TILED_SOLVER(96, 128));
#endif
#if CHOL_STALE
    set_max_shared((const void*)cholesky_stale_refine<K_DIM, (K_DIM == 32 ? 64 : K_DIM == 48 ? 96 : 128), chol_store_t>);
#endif

    // Training loop
    struct TrainResult { float compute_ms, wall_ms, train_rmse, test_rmse; };

    auto run_training = [&](bool use_mixed, const char* label) -> TrainResult {
        cudaMemcpy(d_X, h_X_init.data(), sizeof(float)*num_users*K, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, h_Y_init.data(), sizeof(float)*num_items*K, cudaMemcpyHostToDevice);

        if (use_mixed) {
            cudaMemcpy(d_u_tile_density, user_tile_density.data(), sizeof(float)*user_tile_density.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_i_tile_density, item_tile_density.data(), sizeof(float)*item_tile_density.size(), cudaMemcpyHostToDevice);
        } else {
            vector<float> zd_u(user_tile_density.size(), 0.0f);
            vector<float> zd_i(item_tile_density.size(), 0.0f);
            cudaMemcpy(d_u_tile_density, zd_u.data(), sizeof(float)*zd_u.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_i_tile_density, zd_i.data(), sizeof(float)*zd_i.size(), cudaMemcpyHostToDevice);
        }

        printf("\n=== %s ===\n", label);
#if CHOL_MP != 0
        cudaMemset(d_chol_fail, 0, sizeof(int));
#endif
#if CHOL_STALE
        cudaMemset(d_stale_total, 0, sizeof(int));
#endif
        float prev_rmse = 1e9f, final_train_rmse = 0, final_test_rmse = 0;
        float total_u_compute=0, total_u_solve=0, total_i_compute=0, total_i_solve=0, total_rmse_t=0;
        int total_iters=0, rmse_calls=0;

        cudaEvent_t ev[10];
        for (int i=0; i<10; i++) cudaEventCreate(&ev[i]);
        cudaEvent_t &t0=ev[0],&t1=ev[1],&t2=ev[2],&t3=ev[3],&t4=ev[4],
                    &t5=ev[5],&t6=ev[6],&t7=ev[7],&ev_start=ev[8],&ev_stop=ev[9];
        cudaEventRecord(ev_start);

        for (int iter = 0; iter < max_iters; iter++) {
            total_iters++;
#if CHOL_STALE
            // Full factorization + cache refresh during warmup and on every
            // REFRESH-th iteration; stale-L + one IR step in between.
            const bool chol_fresh = (iter < CHOL_STALE_WARMUP) || (iter % CHOL_STALE_REFRESH == 0);
#endif
            int  lhs_blocks_u = use_mixed ? u_num_sjobs   : u_num_jobs;
            int* ju_tx        = use_mixed ? d_u_sjob_tx    : d_u_job_tx;
            int* ju_ch        = use_mixed ? d_u_sjob_chunk : d_u_job_chunk;
            int* ju_nch       = use_mixed ? d_u_sjob_nch   : d_u_job_nch;

            cudaEventRecord(t0);
            // --- User-side: entity-batched LHS/RHS + Cholesky ---
            cudaMemsetAsync(d_X, 0, sizeof(float)*num_users*K);
            if (use_mixed && n_dense_u > 0)
                convert_fp32_to_fp16<<<(num_items * K + 255) / 256, 256>>>(d_Y, d_Y_half, num_items * K);
            for (int bstart = 0; bstart < num_users; bstart += lhs_batch) {
                int bend = min(bstart + lhs_batch, num_users);
                int bn   = bend - bstart;
                int bidx = bstart / lhs_batch;
                float* lhs_b = d_LHS_buf[bidx % nbuf];
                int prevb = bidx - nbuf;   // last solve that read this buffer
                if (prevb >= 0 && prevb < MAX_PIPE_EVT)
                    cudaStreamWaitEvent(sC, ev_sol_u[prevb], 0);
                if (!use_mixed) {
                    cudaMemsetAsync(lhs_b, 0, sizeof(float)*(long long)bn*K*K, sC);
                } else {
                    int s0 = max(bstart, n_dense_u);
                    int s1 = min(bend, num_users);
                    if (s1 > s0) {
                        int num_sparse = s1 - s0;
                        long long offset = (long long)(s0 - bstart) * K * K;
                        cudaMemsetAsync(lhs_b + offset, 0, sizeof(float) * num_sparse * K * K, sC);
                    }
                }
                // WMMA path for dense entities in this batch
                // (07-21: K>=64 dense+giant kernels use in-kernel cp.async
                // double-buffered staging; the block-per-entity experiment
                // measured SLOWER — see FIXLOG — and is not dispatched.)
                if (use_mixed && n_dense_u > 0) {
                    // giants: entities [0, n_giant_u) that fall in [bstart, bend)
                    int g0 = max(bstart, 0), g1 = min(bend, n_giant_u);
                    if (g1 > g0) {
                        int ng_b = g1 - g0;
                        if      (K == 16) compute_LHS_RHS_wmma_giant<1, GIANT_WARPS><<<ng_b, GIANT_WARPS*32, 0, sC>>>(ng_b, d_u_offsets, d_u_colidx, d_u_csr_vals, d_Y_half, lhs_b, d_X, g0, bstart);
                        else if (K == 32) compute_LHS_RHS_wmma_giant<2, GIANT_WARPS><<<ng_b, GIANT_WARPS*32, 0, sC>>>(ng_b, d_u_offsets, d_u_colidx, d_u_csr_vals, d_Y_half, lhs_b, d_X, g0, bstart);
                        else if (K == 48) compute_LHS_RHS_wmma_giant<3, GIANT_WARPS><<<ng_b, GIANT_WARPS*32, 0, sC>>>(ng_b, d_u_offsets, d_u_colidx, d_u_csr_vals, d_Y_half, lhs_b, d_X, g0, bstart);
                        else if (K == 64) compute_LHS_RHS_wmma_giant<4, GIANT_WARPS><<<ng_b, GIANT_WARPS*32, 0, sC>>>(ng_b, d_u_offsets, d_u_colidx, d_u_csr_vals, d_Y_half, lhs_b, d_X, g0, bstart);
                        else if (K == 96) compute_LHS_RHS_wmma_giant<6, GIANT_WARPS><<<ng_b, GIANT_WARPS*32, 0, sC>>>(ng_b, d_u_offsets, d_u_colidx, d_u_csr_vals, d_Y_half, lhs_b, d_X, g0, bstart);
                    }
                    // normal dense: entities [n_giant_u, n_dense_u) that fall in [bstart, bend)
                    int d0 = max(bstart, n_giant_u), d1 = min(bend, n_dense_u);
                    if (d1 > d0) {
                        int nn_b = d1 - d0;
                        int wblocks = (nn_b + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
                        if      (K == 16) compute_LHS_RHS_wmma<1><<<wblocks, WARPS_PER_BLOCK*32, 0, sC>>>(d1, d0, d_u_offsets, d_u_colidx, d_u_csr_vals, d_Y_half, lhs_b, d_X, bstart);
                        else if (K == 32) compute_LHS_RHS_wmma<2><<<wblocks, WARPS_PER_BLOCK*32, 0, sC>>>(d1, d0, d_u_offsets, d_u_colidx, d_u_csr_vals, d_Y_half, lhs_b, d_X, bstart);
                        else if (K == 48) compute_LHS_RHS_wmma<3><<<wblocks, WARPS_PER_BLOCK*32, 0, sC>>>(d1, d0, d_u_offsets, d_u_colidx, d_u_csr_vals, d_Y_half, lhs_b, d_X, bstart);
                        else if (K == 64) compute_LHS_RHS_wmma<4><<<wblocks, WARPS_PER_BLOCK*32, 0, sC>>>(d1, d0, d_u_offsets, d_u_colidx, d_u_csr_vals, d_Y_half, lhs_b, d_X, bstart);
                        else if (K == 96) compute_LHS_RHS_wmma<6><<<wblocks, WARPS_PER_BLOCK*32, 0, sC>>>(d1, d0, d_u_offsets, d_u_colidx, d_u_csr_vals, d_Y_half, lhs_b, d_X, bstart);
                    }
                }
                // Scalar tile kernel — writes to local batch row indices
                if      (lhs_blocks_u==0 || (use_mixed && bend <= n_dense_u)) ;   // no sparse tile-rows in batch
                else if (rows_per_thread==1)  LAUNCH_FUSED_KERNEL(lhs_blocks_u, 1,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);
                else if (rows_per_thread==2)  LAUNCH_FUSED_KERNEL(lhs_blocks_u, 2,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);
                else if (rows_per_thread==4)  LAUNCH_FUSED_KERNEL(lhs_blocks_u, 4,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);
                else if (rows_per_thread==8)  LAUNCH_FUSED_KERNEL(lhs_blocks_u, 8,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);
                else if (rows_per_thread==16) LAUNCH_FUSED_KERNEL(lhs_blocks_u,16,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);
                else if (rows_per_thread==32) LAUNCH_FUSED_KERNEL(lhs_blocks_u,32,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);

#if WEIGHTED_LAMBDA
                const int* d_nnzw = d_u_nnz + bstart;   // λ·nnz fused into solver diag pass (07-21)
#else
                const int* d_nnzw = nullptr;
#endif
                // Cholesky solve on sS, overlapping next batch's LHS+RHS
                float* rhs_b = d_X + bstart * K;
                bool piped = (bidx < MAX_PIPE_EVT);
                cudaStream_t sv = piped ? sS : sC;
                if (piped) {
                    cudaEventRecord(ev_lhs_u[bidx], sC);
                    cudaStreamWaitEvent(sS, ev_lhs_u[bidx], 0);
                }
                if      (K == 16 && K <= BATCHED_CHOLESKY_MAX_K) cholesky_solve_batched<16><<<(bn + 127) / 128, 128, 0, sv>>>(lhs_b, rhs_b, bn, lambda, d_nnzw);
                else if (K == 32 && K <= BATCHED_CHOLESKY_MAX_K) cholesky_solve_batched<32><<<(bn + 127) / 128, 128, 0, sv>>>(lhs_b, rhs_b, bn, lambda, d_nnzw);
                else if (K == 16) cholesky_solve_packed<16><<<bn, 32, (16*17/2+16)*sizeof(float), sv>>>(lhs_b, rhs_b, bn, lambda, d_nnzw);
#if CHOL_MP == 0
                else if (K == 32) cholesky_solve_tiled<32, 64><<<bn, 64, cholesky_tiled_smem<32>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_nnzw);
                else if (K == 48) cholesky_solve_tiled<48, 96><<<bn, 96, cholesky_tiled_smem<48>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_nnzw);
                else if (K == 64) cholesky_solve_tiled<64, 128><<<bn, 128, cholesky_tiled_smem<64>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_nnzw);
                else if (K == 96) cholesky_solve_tiled<96, 128><<<bn, 128, cholesky_tiled_smem<96>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_nnzw);
#else
                else if (K == 32) CHOL_MP_SOLVE(32, 64, d_Lcache_u);
                else if (K == 48) CHOL_MP_SOLVE(48, 96, d_Lcache_u);
                else if (K == 64) CHOL_MP_SOLVE(64, 128, d_Lcache_u);
                else if (K == 96) CHOL_MP_SOLVE(96, 128, d_Lcache_u);   // NTH=192 bench-won but app-LOST (07-21 nsys 187->199 ms/iter); 128 stays
#endif
#ifdef USE_CUSOLVER
                else cusolver_solve(lhs_b, rhs_b, d_Aptr, d_Bptr_u, bn, d_nnzw);
#else
                else cholesky_solve_cooperative<<<bn, solve_threads, solve_smem, sv>>>(lhs_b,rhs_b,K,bn,lambda,d_nnzw);
#endif
                if (piped) cudaEventRecord(ev_sol_u[bidx], sS);
            }
            cudaEventRecord(t1);
            // user cholesky timing merged into t0-t1
            cudaEventRecord(t2);

#if MOMENTUM
            // Extrapolate X before the item phase reads it (d_X + the fp16
            // convert below). beta=0 on iter 0 just seeds d_Xprev.
            momentum_extrapolate<<<(num_users * K + 255) / 256, 256>>>(
                d_X, d_Xprev, (long long)num_users * K, iter == 0 ? 0.0f : mom_beta);
#endif

            int  lhs_blocks_i = use_mixed ? i_num_sjobs   : i_num_jobs;
            int* ji_tx        = use_mixed ? d_i_sjob_tx    : d_i_job_tx;
            int* ji_ch        = use_mixed ? d_i_sjob_chunk : d_i_job_chunk;
            int* ji_nch       = use_mixed ? d_i_sjob_nch   : d_i_job_nch;

            cudaEventRecord(t3);
            // --- Item-side: entity-batched LHS/RHS + Cholesky ---
            cudaMemsetAsync(d_Y, 0, sizeof(float)*num_items*K);
            if (use_mixed && n_dense_i > 0)
                convert_fp32_to_fp16<<<(num_users * K + 255) / 256, 256>>>(d_X, d_X_half, num_users * K);
            for (int bstart = 0; bstart < num_items; bstart += lhs_batch) {
                int bend = min(bstart + lhs_batch, num_items);
                int bn   = bend - bstart;
                int bidx = bstart / lhs_batch;
                float* lhs_b = d_LHS_buf[bidx % nbuf];
                int prevb = bidx - nbuf;
                if (prevb >= 0 && prevb < MAX_PIPE_EVT)
                    cudaStreamWaitEvent(sC, ev_sol_i[prevb], 0);
                if (!use_mixed) {
                    cudaMemsetAsync(lhs_b, 0, sizeof(float)*(long long)bn*K*K, sC);
                } else {
                    int s0 = max(bstart, n_dense_i);
                    int s1 = min(bend, num_items);
                    if (s1 > s0) {
                        int num_sparse = s1 - s0;
                        long long offset = (long long)(s0 - bstart) * K * K;
                        cudaMemsetAsync(lhs_b + offset, 0, sizeof(float) * num_sparse * K * K, sC);
                    }
                }
                if (use_mixed && n_dense_i > 0) {
                    int g0 = max(bstart, 0), g1 = min(bend, n_giant_i);
                    if (g1 > g0) {
                        int ng_b = g1 - g0;
                        if      (K == 16) compute_LHS_RHS_wmma_giant<1, GIANT_WARPS><<<ng_b, GIANT_WARPS*32, 0, sC>>>(ng_b, d_i_offsets, d_i_colidx, d_i_csr_vals, d_X_half, lhs_b, d_Y, g0, bstart);
                        else if (K == 32) compute_LHS_RHS_wmma_giant<2, GIANT_WARPS><<<ng_b, GIANT_WARPS*32, 0, sC>>>(ng_b, d_i_offsets, d_i_colidx, d_i_csr_vals, d_X_half, lhs_b, d_Y, g0, bstart);
                        else if (K == 48) compute_LHS_RHS_wmma_giant<3, GIANT_WARPS><<<ng_b, GIANT_WARPS*32, 0, sC>>>(ng_b, d_i_offsets, d_i_colidx, d_i_csr_vals, d_X_half, lhs_b, d_Y, g0, bstart);
                        else if (K == 64) compute_LHS_RHS_wmma_giant<4, GIANT_WARPS><<<ng_b, GIANT_WARPS*32, 0, sC>>>(ng_b, d_i_offsets, d_i_colidx, d_i_csr_vals, d_X_half, lhs_b, d_Y, g0, bstart);
                        else if (K == 96) compute_LHS_RHS_wmma_giant<6, GIANT_WARPS><<<ng_b, GIANT_WARPS*32, 0, sC>>>(ng_b, d_i_offsets, d_i_colidx, d_i_csr_vals, d_X_half, lhs_b, d_Y, g0, bstart);
                    }
                    int d0 = max(bstart, n_giant_i), d1 = min(bend, n_dense_i);
                    if (d1 > d0) {
                        int nn_b = d1 - d0;
                        int wblocks = (nn_b + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
                        if      (K == 16) compute_LHS_RHS_wmma<1><<<wblocks, WARPS_PER_BLOCK*32, 0, sC>>>(d1, d0, d_i_offsets, d_i_colidx, d_i_csr_vals, d_X_half, lhs_b, d_Y, bstart);
                        else if (K == 32) compute_LHS_RHS_wmma<2><<<wblocks, WARPS_PER_BLOCK*32, 0, sC>>>(d1, d0, d_i_offsets, d_i_colidx, d_i_csr_vals, d_X_half, lhs_b, d_Y, bstart);
                        else if (K == 48) compute_LHS_RHS_wmma<3><<<wblocks, WARPS_PER_BLOCK*32, 0, sC>>>(d1, d0, d_i_offsets, d_i_colidx, d_i_csr_vals, d_X_half, lhs_b, d_Y, bstart);
                        else if (K == 64) compute_LHS_RHS_wmma<4><<<wblocks, WARPS_PER_BLOCK*32, 0, sC>>>(d1, d0, d_i_offsets, d_i_colidx, d_i_csr_vals, d_X_half, lhs_b, d_Y, bstart);
                        else if (K == 96) compute_LHS_RHS_wmma<6><<<wblocks, WARPS_PER_BLOCK*32, 0, sC>>>(d1, d0, d_i_offsets, d_i_colidx, d_i_csr_vals, d_X_half, lhs_b, d_Y, bstart);
                    }
                }
                
                if      (lhs_blocks_i==0 || (use_mixed && bend <= n_dense_i)) ;   // no sparse tile-rows in batch
                else if (rows_per_thread==1)  LAUNCH_FUSED_KERNEL(lhs_blocks_i, 1,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,ji_tx,ji_ch,ji_nch,bstart,bn);
                else if (rows_per_thread==2)  LAUNCH_FUSED_KERNEL(lhs_blocks_i, 2,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,ji_tx,ji_ch,ji_nch,bstart,bn);
                else if (rows_per_thread==4)  LAUNCH_FUSED_KERNEL(lhs_blocks_i, 4,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,ji_tx,ji_ch,ji_nch,bstart,bn);
                else if (rows_per_thread==8)  LAUNCH_FUSED_KERNEL(lhs_blocks_i, 8,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,ji_tx,ji_ch,ji_nch,bstart,bn);
                else if (rows_per_thread==16) LAUNCH_FUSED_KERNEL(lhs_blocks_i,16,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,ji_tx,ji_ch,ji_nch,bstart,bn);
                else if (rows_per_thread==32) LAUNCH_FUSED_KERNEL(lhs_blocks_i,32,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,ji_tx,ji_ch,ji_nch,bstart,bn);
                
#if WEIGHTED_LAMBDA
                const int* d_nnzw = d_i_nnz + bstart;   // λ·nnz fused into solver diag pass (07-21)
#else
                const int* d_nnzw = nullptr;
#endif
                float* rhs_b = d_Y + bstart * K;
                bool piped = (bidx < MAX_PIPE_EVT);
                cudaStream_t sv = piped ? sS : sC;
                if (piped) {
                    cudaEventRecord(ev_lhs_i[bidx], sC);
                    cudaStreamWaitEvent(sS, ev_lhs_i[bidx], 0);
                }
                if      (K == 16 && K <= BATCHED_CHOLESKY_MAX_K) cholesky_solve_batched<16><<<(bn + 127) / 128, 128, 0, sv>>>(lhs_b, rhs_b, bn, lambda, d_nnzw);
                else if (K == 32 && K <= BATCHED_CHOLESKY_MAX_K) cholesky_solve_batched<32><<<(bn + 127) / 128, 128, 0, sv>>>(lhs_b, rhs_b, bn, lambda, d_nnzw);
                else if (K == 16) cholesky_solve_packed<16><<<bn, 32, (16*17/2+16)*sizeof(float), sv>>>(lhs_b, rhs_b, bn, lambda, d_nnzw);
#if CHOL_MP == 0
                else if (K == 32) cholesky_solve_tiled<32, 64><<<bn, 64, cholesky_tiled_smem<32>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_nnzw);
                else if (K == 48) cholesky_solve_tiled<48, 96><<<bn, 96, cholesky_tiled_smem<48>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_nnzw);
                else if (K == 64) cholesky_solve_tiled<64, 128><<<bn, 128, cholesky_tiled_smem<64>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_nnzw);
                else if (K == 96) cholesky_solve_tiled<96, 128><<<bn, 128, cholesky_tiled_smem<96>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, lambda, d_nnzw);
#else
                else if (K == 32) CHOL_MP_SOLVE(32, 64, d_Lcache_i);
                else if (K == 48) CHOL_MP_SOLVE(48, 96, d_Lcache_i);
                else if (K == 64) CHOL_MP_SOLVE(64, 128, d_Lcache_i);
                else if (K == 96) CHOL_MP_SOLVE(96, 128, d_Lcache_i);
#endif
#ifdef USE_CUSOLVER
                else cusolver_solve(lhs_b, rhs_b, d_Aptr, d_Bptr_i, bn, d_nnzw);
#else
                else cholesky_solve_cooperative<<<bn, solve_threads, solve_smem, sv>>>(lhs_b,rhs_b,K,bn,lambda,d_nnzw);
#endif
                if (piped) cudaEventRecord(ev_sol_i[bidx], sS);
            }
            cudaEventRecord(t4);
            // item cholesky timing merged into t3-t4
            cudaEventRecord(t5);

#if MOMENTUM
            // Extrapolate Y after its solve, before the RMSE check / next
            // iter's user phase reads it.
            momentum_extrapolate<<<(num_items * K + 255) / 256, 256>>>(
                d_Y, d_Yprev, (long long)num_items * K, iter == 0 ? 0.0f : mom_beta);
#endif

            cudaEventSynchronize(t5);
            float ms;
            cudaEventElapsedTime(&ms,t0,t1); total_u_compute+=ms;
            cudaEventElapsedTime(&ms,t1,t2); total_u_solve+=ms;
            cudaEventElapsedTime(&ms,t3,t4); total_i_compute+=ms;
            cudaEventElapsedTime(&ms,t4,t5); total_i_solve+=ms;

            if ((iter+1)%5==0 || iter==max_iters-1) {
                cudaEventRecord(t6);
#if FAST_RMSE
                // fp16 factor images may be stale (Y_half converts at user-
                // phase start, X_half at item-phase start) — refresh both.
                convert_fp32_to_fp16<<<(num_users * K + 255) / 256, 256>>>(d_X, d_X_half, num_users * K);
                convert_fp32_to_fp16<<<(num_items * K + 255) / 256, 256>>>(d_Y, d_Y_half, num_items * K);
                cudaMemset(d_sq_err, 0, sizeof(double));
                compute_RMSE_csr_half<K_DIM><<<(num_users * 32 + 255) / 256, 256>>>(
                    d_u_offsets, d_u_colidx, d_u_csr_vals, num_users, d_X_half, d_Y_half, d_sq_err);
                double h_sq_train;
                cudaMemcpy(&h_sq_train, d_sq_err, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemset(d_sq_err, 0, sizeof(double));
                compute_RMSE_csr_half<K_DIM><<<(num_users * 32 + 255) / 256, 256>>>(
                    d_t_offsets, d_t_colidx, d_t_vals, num_users, d_X_half, d_Y_half, d_sq_err);
                double h_sq_test;
                cudaMemcpy(&h_sq_test, d_sq_err, sizeof(double), cudaMemcpyDeviceToHost);
#else
                cudaMemset(d_sq_err, 0, sizeof(double));
                int rb_tr = (nnz_train + 255) / 256;
                compute_RMSE_kernel<<<rb_tr,256>>>(d_train_users,d_train_items,d_train_ratings,nnz_train,d_X,d_Y,K,d_sq_err);
                double h_sq_train;
                cudaMemcpy(&h_sq_train, d_sq_err, sizeof(double), cudaMemcpyDeviceToHost);
                cudaMemset(d_sq_err, 0, sizeof(double));
                int rb_te = (nnz_test + 255) / 256;
                compute_RMSE_kernel<<<rb_te,256>>>(d_test_users,d_test_items,d_test_ratings,nnz_test,d_X,d_Y,K,d_sq_err);
                double h_sq_test;
                cudaMemcpy(&h_sq_test, d_sq_err, sizeof(double), cudaMemcpyDeviceToHost);
#endif
                cudaEventRecord(t7); cudaEventSynchronize(t7);
                cudaEventElapsedTime(&ms, t6, t7); total_rmse_t += ms;
                rmse_calls++;
                float curr_train = sqrtf((float)(h_sq_train / nnz_train));
                float curr_test  = sqrtf((float)(h_sq_test  / nnz_test));
                printf("Iter %2d | Train RMSE: %.6f | Test RMSE: %.6f\n", iter+1, curr_train, curr_test);
                final_train_rmse = curr_train;
                final_test_rmse  = curr_test;
                if (fabsf(prev_rmse - curr_train) < tol) {
                    printf("Converged at iteration %d (train delta=%.6f < tol=%.3f)\n",
                           iter+1, fabsf(prev_rmse - curr_train), tol);
                    break;
                }
                prev_rmse = curr_train;
            }
        }

#if FAST_RMSE
        {
            // Reported finals are fp32-exact: one COO pass after convergence
            // (the per-check values above used fp16 factor gathers).
            cudaEventRecord(t6);
            cudaMemset(d_sq_err, 0, sizeof(double));
            compute_RMSE_kernel<<<(nnz_train + 255) / 256, 256>>>(d_train_users, d_train_items, d_train_ratings, nnz_train, d_X, d_Y, K, d_sq_err);
            double h_sq_train;
            cudaMemcpy(&h_sq_train, d_sq_err, sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemset(d_sq_err, 0, sizeof(double));
            compute_RMSE_kernel<<<(nnz_test + 255) / 256, 256>>>(d_test_users, d_test_items, d_test_ratings, nnz_test, d_X, d_Y, K, d_sq_err);
            double h_sq_test;
            cudaMemcpy(&h_sq_test, d_sq_err, sizeof(double), cudaMemcpyDeviceToHost);
            cudaEventRecord(t7); cudaEventSynchronize(t7);
            float ms2; cudaEventElapsedTime(&ms2, t6, t7); total_rmse_t += ms2; rmse_calls++;
            final_train_rmse = sqrtf((float)(h_sq_train / nnz_train));
            final_test_rmse  = sqrtf((float)(h_sq_test  / nnz_test));
            printf("Final fp32-exact | Train RMSE: %.6f | Test RMSE: %.6f\n", final_train_rmse, final_test_rmse);
        }
#endif

        double fps=(double)nnz_train*(K*(K+1)/2.0+K)*2.0;
        double gft=fps*2.0*total_iters/1e9;
        float ums=total_u_compute/total_iters, ims=total_i_compute/total_iters;
        printf("\n=== Profiling: %s (%d iters) ===\n",label,total_iters);
        printf("User LHS+RHS: %7.2f ms total | %5.2f ms/iter | %6.1f GFlops/s\n",total_u_compute,ums,fps/(ums*1e-3)/1e9);
        printf("User Cholesky:%7.2f ms total | %5.2f ms/iter\n",total_u_solve,total_u_solve/total_iters);
        printf("Item LHS+RHS: %7.2f ms total | %5.2f ms/iter | %6.1f GFlops/s\n",total_i_compute,ims,fps/(ims*1e-3)/1e9);
        printf("Item Cholesky:%7.2f ms total | %5.2f ms/iter\n",total_i_solve,total_i_solve/total_iters);
#if CHOL_MP != 0
        {
            int h_chol_fail = 0;
            cudaMemcpy(&h_chol_fail, d_chol_fail, sizeof(int), cudaMemcpyDeviceToHost);
            printf("CHOL_MP [%s]: non-finite solves zeroed = %d entity-iters%s\n",
                   CHOL_MP_STR, h_chol_fail,
                   h_chol_fail > 0 ? "  <-- overflow/breakdown; if >0.1 pct of solves, abandon this precision" : "");
#if CHOL_STALE
            int h_stale_total = 0;
            cudaMemcpy(&h_stale_total, d_stale_total, sizeof(int), cudaMemcpyDeviceToHost);
            printf("CHOL_STALE gate (tau=%.2f): %d entity-iters failed the residual gate and were full-re-solved\n",
                   (double)CHOL_STALE_TAU, h_stale_total);
#endif
        }
#endif
        if (rmse_calls>0) printf("RMSE:         %7.2f ms total | %5.2f ms/call\n",total_rmse_t,total_rmse_t/rmse_calls);
        float tc=total_u_compute+total_u_solve+total_i_compute+total_i_solve;
        float gt=tc+total_rmse_t;
        printf("Total FLOPs: %.2f GFLOPs | Throughput: %.1f GFlops/s (compute: %.1f)\n",
               gft, gft/(gt*1e-3), gft/(tc*1e-3));
        cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
        float wall_ms; cudaEventElapsedTime(&wall_ms,ev_start,ev_stop);
        printf("Training Complete in %.3f seconds!\n",wall_ms/1000.0f);
        for (int i=0; i<10; i++) cudaEventDestroy(ev[i]);
        return {tc, wall_ms, final_train_rmse, final_test_rmse};
    };

    TrainResult apr = run_training(true, "APR-BALS (Mixed Precision)");

    printf("\n=== APR-BALS final ===\n");
    printf("Train RMSE: %.6f | Test RMSE: %.6f | compute=%.2f ms | wall=%.2f ms\n",
           apr.train_rmse, apr.test_rmse, apr.compute_ms, apr.wall_ms);
    printf("Params: K_DIM=%d, lambda=%.2f, DENSE_NNZ_THRESH=%d, WARPS_PER_BLOCK=%d, XB=%d, YB=%d\n",
           K_DIM, lambda, DENSE_NNZ_THRESH, WARPS_PER_BLOCK, XB, YB);

    cudaMemcpy(h_X.data(), d_X, sizeof(float) * num_users * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y.data(), d_Y, sizeof(float) * num_items * K, cudaMemcpyDeviceToHost);

    cudaFree(d_train_users); cudaFree(d_train_items); cudaFree(d_train_ratings);
    cudaFree(d_test_users);  cudaFree(d_test_items);  cudaFree(d_test_ratings);
#if FAST_RMSE
    cudaFree(d_t_offsets);   cudaFree(d_t_colidx);    cudaFree(d_t_vals);
#endif
    cudaFree(d_sq_err);
    cudaStreamDestroy(sC);   cudaStreamDestroy(sS);
    for (int i = 0; i < MAX_PIPE_EVT; i++) {
        cudaEventDestroy(ev_lhs_u[i]); cudaEventDestroy(ev_lhs_i[i]);
        cudaEventDestroy(ev_sol_u[i]); cudaEventDestroy(ev_sol_i[i]);
    }
    cudaFree(d_X);           cudaFree(d_Y);
    cudaFree(d_LHS_buf[0]);  if (nbuf == 2) cudaFree(d_LHS_buf[1]);
    cudaFree(d_X_half);      cudaFree(d_Y_half);
    cudaFree(d_u_tile_ptr);  cudaFree(d_u_tile_colidx);
    cudaFree(d_u_seg_ptr);   cudaFree(d_u_seg_colidx); cudaFree(d_u_seg_values);
    cudaFree(d_i_tile_ptr);  cudaFree(d_i_tile_colidx);
    cudaFree(d_i_seg_ptr);   cudaFree(d_i_seg_colidx); cudaFree(d_i_seg_values);
    cudaFree(d_u_tile_density); cudaFree(d_i_tile_density);
    cudaFree(d_u_nz_list);   cudaFree(d_u_nz_ptr);
    cudaFree(d_i_nz_list);   cudaFree(d_i_nz_ptr);
    cudaFree(d_u_job_tx);    cudaFree(d_u_job_chunk);  cudaFree(d_u_job_nch);
    cudaFree(d_i_job_tx);    cudaFree(d_i_job_chunk);  cudaFree(d_i_job_nch);
#if CHOL_MP != 0
    cudaFree(d_chol_fail);
#endif
#if CHOL_STALE
    cudaFree(d_Lcache_u);    cudaFree(d_Lcache_i);
#endif
    return 0;
}
