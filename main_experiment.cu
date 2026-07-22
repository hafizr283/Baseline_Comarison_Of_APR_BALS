#include "common.cuh"
#include "data_utils.cuh"
#include "fused_kernels.cuh"
#include "wmma_kernels.cuh"
#include "cholesky_kernels.cuh"

// Regularization mode (same switch as only_scalar_fp32.cu):
//   -DWEIGHTED_LAMBDA=1 (default) cumf_als-style weighted-λ (ALS-WR, Zhou et
//                   al. 2008): diag += nnz_e·λ, exactly what cumf_als's
//                   get_hermitian* kernels do. Implemented as a tiny
//                   diag-update kernel after each batch's LHS accumulation
//                   (BOTH legs — baseline and APR — so the comparison stays
//                   apples-to-apples); the Cholesky solvers then receive λ=0.
//                   Entities with zero train ratings get plain λ so the
//                   system stays SPD. Good λ here is ~0.02-0.06 (cumf_als
//                   Netflix tune: 0.048).
//   -DWEIGHTED_LAMBDA=0  plain λ·I (the pre-2026-07-12 behavior; λ=0.1 tune).
#ifndef WEIGHTED_LAMBDA
#define WEIGHTED_LAMBDA 1
#endif
// Initial-factor scale:
//   -DCUMF_INIT=1  factors ~ U(0,0.2) like released cuMF — converges in
//                  ~1/3 fewer ALS iterations AND reaches slightly better
//                  test RMSE (use for any comparison against cuMF).
//   -DCUMF_INIT=0  (default) legacy U(0.1,1.1) init — preserved so all
//                  pre-2026-07-12 validated trajectories stay reproducible.
#ifndef CUMF_INIT
#define CUMF_INIT 0
#endif

#if WEIGHTED_LAMBDA
// cumf_als parity (als.cu "weighted-lambda regularization"):
//   tt[diag] += (end - start) * lambda   with (end-start) = entity's train nnz.
// Runs on the compute stream after each batch's LHS accumulation; the solvers
// then get λ=0 so nothing is added twice. nnz=0 entities (test-only users/
// items) fall back to plain λ — otherwise their LHS is all-zero and Cholesky
// produces NaNs that poison the RMSE.
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

#if CHOL_STALE
#error "CHOL_STALE is wired in justapr.cu only (the lean experiment binary). Validate it there first, then port the wiring here (cache alloc x2 legs + chol_fresh cadence + dispatch)."
#endif

int main(int argc, char* argv[]) {
    printf("=== CODE VERSION: SYNC-2026-07-20 [CHOL-MP 16-bit tiled Cholesky + optional IR — UNVALIDATED ON GPU; CHOL_MP=0 path = 07-06 code] ===\n");
    printf("Tiled Cholesky precision: %s\n", CHOL_MP_STR);
    printf("Regularization: %s\n", WEIGHTED_LAMBDA
        ? "WEIGHTED lambda (ALS-WR, cumf_als-style: diag += nnz_e*lambda)"
        : "PLAIN lambda (diag += lambda)");
    printf("Init: %s\n", CUMF_INIT
        ? "cuMF-scale U(0,0.2) (CUMF_INIT=1)"
        : "legacy U(0.1,1.1) (CUMF_INIT=0)");
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
    // What the Cholesky solvers add to the diagonal themselves. In weighted
    // mode the full λ·nnz_e term is pre-added by add_weighted_lambda_diag,
    // so the solvers must add nothing.
    const float solver_lambda = WEIGHTED_LAMBDA ? 0.0f : lambda;
    int max_iters = 50;
    float tol     = 0.001f;
    int max_ent   = max(num_users, num_items);
    int yb_eff    = (K > 32) ? 128 : YB;

    srand(42);
    vector<float> h_X(num_users * K), h_Y(num_items * K);
#if CUMF_INIT
    // cuMF-scale init: factors ~ U(0,0.2) (released cuMF: thetaT ~ U(0,0.2),
    // XT = 0 — X's value is irrelevant here too, the user solve overwrites it).
    // The legacy U(0.1,1.1) init is ~5x too large and costs ~10 extra ALS
    // iterations to shrink out of (Netflix K=64: 30 -> 20 iters, and test
    // RMSE IMPROVES; measured 2026-07-12). Both legs share h_X_init/h_Y_init,
    // so the baseline-vs-APR comparison is unaffected by the choice.
    for (int i = 0; i < num_users * K; i++) h_X[i] = 0.2f * ((float)rand() / (float)RAND_MAX);
    for (int i = 0; i < num_items * K; i++) h_Y[i] = 0.2f * ((float)rand() / (float)RAND_MAX);
#else
    // legacy init — keep as default: every validated record (thesis tables,
    // FIXLOG trajectories) was produced with this exact sequence.
    for (int i = 0; i < num_users * K; i++) h_X[i] = 0.1f + (rand() % 100) / 100.0f;
    for (int i = 0; i < num_items * K; i++) h_Y[i] = 0.1f + (rand() % 100) / 100.0f;
#endif
    vector<float> h_X_init = h_X, h_Y_init = h_Y;

    float *d_X, *d_Y;
    cudaMalloc(&d_X,       sizeof(float) * num_users * K);
    cudaMalloc(&d_Y,       sizeof(float) * num_items * K);
    // Persistent FP16 feature copies for the WMMA gather path (converted once
    // per phase; halves the gather read bandwidth vs FP32 + per-element convert)
    half *d_X_half, *d_Y_half;
    cudaMalloc(&d_X_half,  sizeof(half) * num_users * K);
    cudaMalloc(&d_Y_half,  sizeof(half) * num_items * K);
    int lhs_batch = min(ENTITY_BATCH_SIZE, max_ent);
    // Double-buffered LHS: the Cholesky solve of batch b (stream sS) only
    // touches its own LHS buffer, so LHS+RHS of batch b+1 (stream sC) can run
    // concurrently in the other buffer. At K>=48 the solve is 61-80% of the
    // APR iteration (FIXLOG 07-04c), so hiding LHS+RHS behind it is the top
    // remaining lever. Falls back to a single buffer (serial, old behavior)
    // if the second allocation doesn't fit.
    float* d_LHS_buf[2];
    int    nbuf = 1;
    cudaMalloc(&d_LHS_buf[0], sizeof(float) * (long long)lhs_batch * K * K);
    if (cudaMalloc(&d_LHS_buf[1], sizeof(float) * (long long)lhs_batch * K * K) == cudaSuccess) {
        nbuf = 2;
    } else {
        cudaGetLastError();   // clear the OOM
        d_LHS_buf[1] = d_LHS_buf[0];
        printf("WARN: 2nd LHS buffer OOM -> solve/LHS+RHS overlap disabled (serial fallback)\n");
    }
    // Pipeline streams: sC = LHS+RHS accumulation, sS = batched Cholesky
    // (higher priority: at K>=48 the solve is the critical path, so its blocks
    // should win SM slots as they free up). Both are BLOCKING streams: every
    // phase-level op left on the legacy NULL stream (feature memset, fp16
    // convert, t0..t5 event records, RMSE) is a full join of both pipes —
    // that alone provides all cross-phase and cross-iteration ordering.
    cudaStream_t sC, sS;
    {
        int prLo, prHi;
        cudaDeviceGetStreamPriorityRange(&prLo, &prHi);
        cudaStreamCreate(&sC);
        cudaStreamCreateWithPriority(&sS, cudaStreamDefault, prHi);
    }
    cudaMemcpy(d_X, h_X.data(), sizeof(float) * num_users * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y.data(), sizeof(float) * num_items * K, cudaMemcpyHostToDevice);

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

#if WEIGHTED_LAMBDA
    // Per-entity train nnz for weighted-λ. The CSR offsets come straight from
    // the frozen .bin (train-set only, frequency-sorted order = the batching
    // order), so nnz_e = offsets[e+1]-offsets[e] is exact for both legs.
    int *d_u_nnz, *d_i_nnz;
    {
        vector<int> h_u_nnz(num_users), h_i_nnz(num_items);
        int u_empty = 0, i_empty = 0;
        for (int u = 0; u < num_users; u++) {
            h_u_nnz[u] = h_user_offsets[u + 1] - h_user_offsets[u];
            if (h_u_nnz[u] == 0) u_empty++;
        }
        for (int i = 0; i < num_items; i++) {
            h_i_nnz[i] = h_item_offsets[i + 1] - h_item_offsets[i];
            if (h_i_nnz[i] == 0) i_empty++;
        }
        cudaMalloc(&d_u_nnz, sizeof(int) * num_users);
        cudaMalloc(&d_i_nnz, sizeof(int) * num_items);
        cudaMemcpy(d_u_nnz, h_u_nnz.data(), sizeof(int) * num_users, cudaMemcpyHostToDevice);
        cudaMemcpy(d_i_nnz, h_i_nnz.data(), sizeof(int) * num_items, cudaMemcpyHostToDevice);
        printf("Weighted-lambda: empty entities (get plain lambda): users=%d, items=%d\n",
               u_empty, i_empty);
    }
#endif

    int shared_mem_size = yb_eff * K * sizeof(float);
    int solve_threads  = 32;
    int solve_smem     = (K * K + K) * sizeof(float);

    // Tiled-solver load scatter map (upper-tri memory index -> tiled smem
    // offset), built once per K on the host — see cholesky_solve_tiled.
    // float4-vectorized since SYNC-2026-07-06: first chol_nvec entries are
    // 16B-chunk loads, the rest scalar head/tail.
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
    // run (both legs), reported in each profile block. NOTE: BOTH legs
    // (baseline and APR) use the mp solver, so the comparison stays
    // apples-to-apples — but the FP32-vs-FP32 headline of record must come
    // from a CHOL_MP=0 build.
    int* d_chol_fail = nullptr;
    cudaMalloc(&d_chol_fail, sizeof(int));
    cudaMemset(d_chol_fail, 0, sizeof(int));
    // Single dispatch site for the mixed-precision tiled solve.
    #define CHOL_MP_SOLVE(KK, NTH) \
        cholesky_solve_tiled_mp<KK, NTH, chol_store_t><<<bn, NTH, cholesky_tiled_smem_mp<KK, chol_store_t>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda, d_chol_fail, (chol_store_t*)nullptr)
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
    auto cusolver_solve = [&](float* d_LHS, float* d_RHS, float** d_Aptr_, float** d_Bptr_, int n) {
        build_ptr_array<<<(n + 255) / 256, 256>>>(d_Aptr_, d_LHS, (long long)K * K, n);
        build_ptr_array<<<(n + 255) / 256, 256>>>(d_Bptr_, d_RHS, (long long)K, n);
        long long tot = (long long)n * K;
        add_lambda_diag<<<(int)((tot + 255) / 256), 256>>>(d_LHS, K, n, solver_lambda);
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
    // blocks. Without it, one block covers all 32 rows and each thread must
    // persistently hold RPT*RR_C*RC_C = K^2/32 accumulators, but
    // __launch_bounds__(1024,1) caps threads at 64 regs -> at K>=48 the
    // accumulators spill to local memory and are re-read/re-written on EVERY
    // per-tile flush. Measured on Netflix K=48 (sm_86): 280B stack/thread =
    // 280KB/SM working set >> 96KB L1 -> ~0.5MB of L2/DRAM spill traffic per
    // tile visit -> FP32 baseline collapsed 230 -> 63 GFlops/s (the fake
    // "30-44x" speedups). yb_eff halving at K>32 doubled tile count and made
    // it worse. Splitting rows shrinks RPT by ROW_SPLIT so accumulators stay
    // in registers (ptxas: 0-64B stack on all running templates). Cost: only
    // the cooperative smem feature staging repeats per sibling block (L2-
    // cached); nnz segment reads are per-row (not duplicated) and total
    // atomic output traffic is unchanged.
    // Per-K choice (ptxas-verified): persistent floats = RPT*RR_C*RC_C
    //   K16: RPT=2  already reg-resident          -> G=1
    //   K32: RPT=8->4  (56B  -> 0B stack)         -> G=2
    //   K48: RPT=8->2  (280B -> 0B stack)         -> G=4
    //   K64: RPT=32->4 (656B -> 0B stack)         -> G=8
    //   K96: RPT=32->4 (1432B-> 64B stack, L1-ok) -> G=8
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

    // UNIFORM smem carveout (MaxShared) for every kernel in the batch
    // pipeline. On Ampere, kernels launched with DIFFERENT L1/smem carveout
    // configs cannot run concurrently — the per-SM config has to drain and
    // switch — which silently serialized the sC/sS solve-overlap pipeline:
    // the scalar kernel ran with MaxL1 (07-04 spill mitigation, vestigial
    // since ROW_SPLIT cut its stack to 0-64 B) between the WMMA and solve
    // kernels (both driver-defaulted to max-shared for occupancy) in every
    // batch. One shared config removes the reconfig barriers; the scalar
    // kernel keeps the ~28 KB L1 slice, ample for the post-ROW_SPLIT spill.
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
    // batched solver: zero smem, its per-thread local matrix lives on L1 —
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
        float prev_rmse = 1e9f, final_train_rmse = 0, final_test_rmse = 0;
        float total_u_compute=0, total_u_solve=0, total_i_compute=0, total_i_solve=0, total_rmse_t=0;
        int total_iters=0, rmse_calls=0;

        cudaEvent_t ev[10];
        for (int i=0; i<10; i++) cudaEventCreate(&ev[i]);
        cudaEvent_t &t0=ev[0],&t1=ev[1],&t2=ev[2],&t3=ev[3],&t4=ev[4],
                    &t5=ev[5],&t6=ev[6],&t7=ev[7],&ev_start=ev[8],&ev_stop=ev[9];
        // Per-batch event pool for true Cholesky timing (solves run inside the
        // batch loops, so t0-t1/t3-t4 alone can't separate them from LHS+RHS).
        // The solve-end events [2b+1] double as "LHS buffer b%nbuf is free
        // again" markers for the double-buffer pipeline.
        const int MAX_SOLVE_EVT = 32;
        cudaEvent_t evs_u[2*MAX_SOLVE_EVT], evs_i[2*MAX_SOLVE_EVT];
        for (int i = 0; i < 2*MAX_SOLVE_EVT; i++) { cudaEventCreate(&evs_u[i]); cudaEventCreate(&evs_i[i]); }
        // sC -> sS handoff events ("batch b's LHS+RHS is complete")
        cudaEvent_t ev_lhs_u[MAX_SOLVE_EVT], ev_lhs_i[MAX_SOLVE_EVT];
        for (int i = 0; i < MAX_SOLVE_EVT; i++) {
            cudaEventCreateWithFlags(&ev_lhs_u[i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&ev_lhs_i[i], cudaEventDisableTiming);
        }
        int n_sevt_u = 0, n_sevt_i = 0;
        cudaEventRecord(ev_start);

        for (int iter = 0; iter < max_iters; iter++) {
            total_iters++;
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
                // Before touching this LHS buffer, wait (on the compute
                // stream) for the solve that last read it: batch bidx-nbuf.
                // Batches past the event pool run their solve on sC (in-order)
                // and need no wait.
                int prevb = bidx - nbuf;
                if (prevb >= 0 && prevb < MAX_SOLVE_EVT)
                    cudaStreamWaitEvent(sC, evs_u[2*prevb+1], 0);
                // Zero only the scalar-path region: WMMA/giant kernels fully
                // overwrite their entities' LHS (plain stores, no accumulation)
                if (!use_mixed) {
                    cudaMemsetAsync(lhs_b, 0, sizeof(float)*(long long)bn*K*K, sC);
                } else {
                    int s0 = max(bstart, n_dense_u);
                    if (bend > s0) {
                        long long offset = (long long)(s0 - bstart) * K * K;
                        cudaMemsetAsync(lhs_b + offset, 0, sizeof(float) * (long long)(bend - s0) * K * K, sC);
                    }
                }
                // WMMA path for dense entities in this batch
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
                // Scalar tile kernel — writes to local batch row indices.
                // Mixed mode: sparse jobs only exist for tile-rows >= tx_split
                // (entities >= n_dense_u), so batches fully inside the dense
                // region have zero live jobs — skip the launch entirely
                // instead of letting every block early-exit.
                if      (lhs_blocks_u==0 || (use_mixed && bend <= n_dense_u)) ;
                else if (rows_per_thread==1)  LAUNCH_FUSED_KERNEL(lhs_blocks_u, 1,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);
                else if (rows_per_thread==2)  LAUNCH_FUSED_KERNEL(lhs_blocks_u, 2,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);
                else if (rows_per_thread==4)  LAUNCH_FUSED_KERNEL(lhs_blocks_u, 4,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);
                else if (rows_per_thread==8)  LAUNCH_FUSED_KERNEL(lhs_blocks_u, 8,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);
                else if (rows_per_thread==16) LAUNCH_FUSED_KERNEL(lhs_blocks_u,16,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);
                else if (rows_per_thread==32) LAUNCH_FUSED_KERNEL(lhs_blocks_u,32,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,ju_tx,ju_ch,ju_nch,bstart,bn);

#if WEIGHTED_LAMBDA
                // diag += nnz_e * lambda (cumf_als parity); on sC after ALL
                // of this batch's LHS accumulation (WMMA/giant/scalar all run
                // on sC, so stream order covers them), before the sC->sS
                // solve handoff.
                add_weighted_lambda_diag<<<(int)(((long long)bn * K + 255) / 256), 256, 0, sC>>>(
                    lhs_b, d_u_nnz, bstart, bn, K, lambda);
#endif

                // Cholesky solve for this batch — handed to the solve stream
                // so it overlaps the NEXT batch's LHS+RHS (reads only lhs_b +
                // its own d_X rows; next batch touches the other LHS buffer
                // and disjoint d_X rows). Timing pair evs_u[2b,2b+1] records
                // on sS AFTER the LHS-done wait, so it still measures pure
                // solve busy time. Batches past the event pool (never hit:
                // pool=32, Netflix user=9) solve in-order on sC instead.
                float* rhs_b = d_X + bstart * K;
                bool piped = (bidx < MAX_SOLVE_EVT);
                cudaStream_t sv = piped ? sS : sC;
                if (piped) {
                    cudaEventRecord(ev_lhs_u[bidx], sC);
                    cudaStreamWaitEvent(sS, ev_lhs_u[bidx], 0);
                    cudaEventRecord(evs_u[2*bidx], sS);
                }
                if      (K == 16 && K <= BATCHED_CHOLESKY_MAX_K) cholesky_solve_batched<16><<<(bn + 127) / 128, 128, 0, sv>>>(lhs_b, rhs_b, bn, solver_lambda);
                else if (K == 32 && K <= BATCHED_CHOLESKY_MAX_K) cholesky_solve_batched<32><<<(bn + 127) / 128, 128, 0, sv>>>(lhs_b, rhs_b, bn, solver_lambda);
                else if (K == 16) cholesky_solve_packed<16><<<bn, 32, (16*17/2+16)*sizeof(float), sv>>>(lhs_b, rhs_b, bn, solver_lambda);
#if CHOL_MP == 0
                else if (K == 32) cholesky_solve_tiled<32, 64><<<bn, 64, cholesky_tiled_smem<32>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 48) cholesky_solve_tiled<48, 96><<<bn, 96, cholesky_tiled_smem<48>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 64) cholesky_solve_tiled<64, 128><<<bn, 128, cholesky_tiled_smem<64>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 96) cholesky_solve_tiled<96, 128><<<bn, 128, cholesky_tiled_smem<96>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
#else
                else if (K == 32) CHOL_MP_SOLVE(32, 64);
                else if (K == 48) CHOL_MP_SOLVE(48, 96);
                else if (K == 64) CHOL_MP_SOLVE(64, 128);
                else if (K == 96) CHOL_MP_SOLVE(96, 128);
#endif
#ifdef USE_CUSOLVER
                else cusolver_solve(lhs_b, rhs_b, d_Aptr, d_Bptr_u, bn);   // dead for supported K (compile guard)
#else
                else cholesky_solve_cooperative<<<bn, solve_threads, solve_smem, sv>>>(lhs_b,rhs_b,K,bn,solver_lambda);
#endif
                if (piped) {
                    cudaEventRecord(evs_u[2*bidx+1], sS);
                    n_sevt_u = min(bidx + 1, MAX_SOLVE_EVT);
                }
            }
            cudaEventRecord(t1);
            // user cholesky timing accumulated from evs_u after sync
            cudaEventRecord(t2);

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
                if (prevb >= 0 && prevb < MAX_SOLVE_EVT)
                    cudaStreamWaitEvent(sC, evs_i[2*prevb+1], 0);
                // Zero only the scalar-path region (see user-side note)
                if (!use_mixed) {
                    cudaMemsetAsync(lhs_b, 0, sizeof(float)*(long long)bn*K*K, sC);
                } else {
                    int s0 = max(bstart, n_dense_i);
                    if (bend > s0) {
                        long long offset = (long long)(s0 - bstart) * K * K;
                        cudaMemsetAsync(lhs_b + offset, 0, sizeof(float) * (long long)(bend - s0) * K * K, sC);
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
                // diag += nnz_e * lambda (see user-side note)
                add_weighted_lambda_diag<<<(int)(((long long)bn * K + 255) / 256), 256, 0, sC>>>(
                    lhs_b, d_i_nnz, bstart, bn, K, lambda);
#endif

                // Solve on sS, overlapping the next batch's LHS+RHS (see
                // user-side note; Netflix items fit one batch, so this side
                // degenerates to the old serial order there).
                float* rhs_b = d_Y + bstart * K;
                bool piped = (bidx < MAX_SOLVE_EVT);
                cudaStream_t sv = piped ? sS : sC;
                if (piped) {
                    cudaEventRecord(ev_lhs_i[bidx], sC);
                    cudaStreamWaitEvent(sS, ev_lhs_i[bidx], 0);
                    cudaEventRecord(evs_i[2*bidx], sS);
                }
                if      (K == 16 && K <= BATCHED_CHOLESKY_MAX_K) cholesky_solve_batched<16><<<(bn + 127) / 128, 128, 0, sv>>>(lhs_b, rhs_b, bn, solver_lambda);
                else if (K == 32 && K <= BATCHED_CHOLESKY_MAX_K) cholesky_solve_batched<32><<<(bn + 127) / 128, 128, 0, sv>>>(lhs_b, rhs_b, bn, solver_lambda);
                else if (K == 16) cholesky_solve_packed<16><<<bn, 32, (16*17/2+16)*sizeof(float), sv>>>(lhs_b, rhs_b, bn, solver_lambda);
#if CHOL_MP == 0
                else if (K == 32) cholesky_solve_tiled<32, 64><<<bn, 64, cholesky_tiled_smem<32>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 48) cholesky_solve_tiled<48, 96><<<bn, 96, cholesky_tiled_smem<48>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 64) cholesky_solve_tiled<64, 128><<<bn, 128, cholesky_tiled_smem<64>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 96) cholesky_solve_tiled<96, 128><<<bn, 128, cholesky_tiled_smem<96>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
#else
                else if (K == 32) CHOL_MP_SOLVE(32, 64);
                else if (K == 48) CHOL_MP_SOLVE(48, 96);
                else if (K == 64) CHOL_MP_SOLVE(64, 128);
                else if (K == 96) CHOL_MP_SOLVE(96, 128);
#endif
#ifdef USE_CUSOLVER
                else cusolver_solve(lhs_b, rhs_b, d_Aptr, d_Bptr_i, bn);   // dead for supported K (compile guard)
#else
                else cholesky_solve_cooperative<<<bn, solve_threads, solve_smem, sv>>>(lhs_b,rhs_b,K,bn,solver_lambda);
#endif
                if (piped) {
                    cudaEventRecord(evs_i[2*bidx+1], sS);
                    n_sevt_i = min(bidx + 1, MAX_SOLVE_EVT);
                }
            }
            cudaEventRecord(t4);
            // item cholesky timing accumulated from evs_i after sync
            cudaEventRecord(t5);

            cudaEventSynchronize(t5);
            float ms;
            // True solve times from the per-batch event pool; subtract from the
            // side totals so "LHS+RHS" really is accumulation-only. (The old
            // t1-t2 / t4-t5 "Cholesky" rows measured empty stream gaps: 0.00.)
            float u_sol = 0, i_sol = 0;
            for (int b = 0; b < n_sevt_u; b++) { cudaEventElapsedTime(&ms, evs_u[2*b], evs_u[2*b+1]); u_sol += ms; }
            for (int b = 0; b < n_sevt_i; b++) { cudaEventElapsedTime(&ms, evs_i[2*b], evs_i[2*b+1]); i_sol += ms; }
            total_u_solve += u_sol;
            total_i_solve += i_sol;
            cudaEventElapsedTime(&ms,t0,t1); total_u_compute += ms - u_sol;
            cudaEventElapsedTime(&ms,t3,t4); total_i_compute += ms - i_sol;

            if ((iter+1)%5==0 || iter==max_iters-1) {
                cudaEventRecord(t6);
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
        }
#endif
        if (rmse_calls>0) printf("RMSE:         %7.2f ms total | %5.2f ms/call\n",total_rmse_t,total_rmse_t/rmse_calls);
        if (nbuf == 2) printf("(Cholesky overlaps next batch's LHS+RHS on a 2nd stream; LHS+RHS rows = EXPOSED time = phase span minus solve busy time, so their GFlops/s overstates pure kernel throughput. Totals/speedups remain exact.)\n");
        float tc=total_u_compute+total_u_solve+total_i_compute+total_i_solve;
        float gt=tc+total_rmse_t;
        printf("Total FLOPs: %.2f GFLOPs | Throughput: %.1f GFlops/s (compute: %.1f)\n",
               gft, gft/(gt*1e-3), gft/(tc*1e-3));
        cudaEventRecord(ev_stop); cudaEventSynchronize(ev_stop);
        float wall_ms; cudaEventElapsedTime(&wall_ms,ev_start,ev_stop);
        printf("Training Complete in %.3f seconds!\n",wall_ms/1000.0f);
        for (int i=0; i<10; i++) cudaEventDestroy(ev[i]);
        for (int i=0; i<2*MAX_SOLVE_EVT; i++) { cudaEventDestroy(evs_u[i]); cudaEventDestroy(evs_i[i]); }
        for (int i=0; i<MAX_SOLVE_EVT; i++) { cudaEventDestroy(ev_lhs_u[i]); cudaEventDestroy(ev_lhs_i[i]); }
        return {tc, wall_ms, final_train_rmse, final_test_rmse};
    };

    TrainResult base = run_training(false, WEIGHTED_LAMBDA
        ? "FP32 Baseline (BALS — all tiles FP32, WEIGHTED lambda like cumf_als)"
        : "FP32 Baseline (BALS — all tiles FP32)");
    TrainResult apr  = run_training(true,  WEIGHTED_LAMBDA
        ? "APR-BALS (Mixed Precision, WEIGHTED lambda like cumf_als)"
        : "APR-BALS (Mixed Precision)");

    float train_delta = fabsf(base.train_rmse - apr.train_rmse);
    float test_delta  = fabsf(base.test_rmse  - apr.test_rmse);
    float speedup_c   = base.compute_ms / apr.compute_ms;
    float speedup_w   = base.wall_ms    / apr.wall_ms;
    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║         APR-BALS vs FP32 Baseline  (80/20 split)           ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Compute speedup:        %5.2fx                             ║\n", speedup_c);
    printf("║  Wall-time speedup:      %5.2fx                             ║\n", speedup_w);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Train RMSE  baseline:   %.6f                         ║\n", base.train_rmse);
    printf("║  Train RMSE  APR-BALS:   %.6f  delta=%+.6f          ║\n", apr.train_rmse, apr.train_rmse - base.train_rmse);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Test  RMSE  baseline:   %.6f                         ║\n", base.test_rmse);
    printf("║  Test  RMSE  APR-BALS:   %.6f  delta=%+.6f  %-12s║\n",
           apr.test_rmse, apr.test_rmse - base.test_rmse,
           apr.test_rmse <= base.test_rmse ? "(APR better)" : "(APR worse)");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Train delta=%.6f (%.4f%%)  Test delta=%.6f (%.4f%%)\n",
           train_delta, 100.f*train_delta/base.train_rmse,
           test_delta,  100.f*test_delta /base.test_rmse);
    printf("Params: K_DIM=%d, lambda=%.4f (%s), DENSE_NNZ_THRESH=%d, WARPS_PER_BLOCK=%d, XB=%d, YB=%d\n",
           K_DIM, lambda,
           WEIGHTED_LAMBDA ? "weighted ALS-WR: diag += nnz_e*lambda" : "plain: diag += lambda",
           DENSE_NNZ_THRESH, WARPS_PER_BLOCK, XB, YB);

    cudaMemcpy(h_X.data(), d_X, sizeof(float) * num_users * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y.data(), d_Y, sizeof(float) * num_items * K, cudaMemcpyDeviceToHost);

    cudaFree(d_train_users); cudaFree(d_train_items); cudaFree(d_train_ratings);
    cudaFree(d_test_users);  cudaFree(d_test_items);  cudaFree(d_test_ratings);
    cudaFree(d_sq_err);
    cudaStreamDestroy(sC);   cudaStreamDestroy(sS);
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
#if WEIGHTED_LAMBDA
    cudaFree(d_u_nnz);       cudaFree(d_i_nnz);
#endif
    return 0;
}
