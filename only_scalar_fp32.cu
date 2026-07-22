// ============================================================================
// only_scalar_fp32.cu — scalar-FP32-only debugging pipeline
//
// This is main_experiment.cu with the mixed-precision (WMMA/FP16) run removed.
// With -DWEIGHTED_LAMBDA=0 it executes EXACTLY the "FP32 Baseline (BALS — all
// tiles FP32)" leg of main_experiment.cu (default is cumf_als-style weighted
// regularization instead — see below): same data loading, same BALS tile
// format, same job
// lists, same entity batching, same double-buffered LHS + sC/sS solve overlap,
// same Cholesky dispatch, same timing events, same RMSE cadence and
// convergence rule, same srand(42) init order. The device density arrays are
// zeroed before training (same as run_training(false)), so every tile takes
// the FP32 branch of compute_LHS_RHS_BALS_block.
//
// Deliberately KEPT although unused by the scalar path, so the memory
// footprint and allocation ORDER match main_experiment.cu (this decides
// whether the 2nd LHS buffer fits -> nbuf=1 vs 2 -> different pipelining):
//   - d_X_half / d_Y_half (FP16 feature copies)
//   - plain-CSR uploads (d_u_offsets/.../d_i_csr_vals, WMMA gather inputs)
// Delete those two blocks if you want a lean build instead of a faithful one.
//
// Removed (mixed-path only, no effect on the baseline leg):
//   - wmma_kernels.cuh launches and their cudaFuncSetAttribute calls
//   - dense/giant entity split + sparse-only (sjob) job lists
//   - precision-tier / NNZ-per-tier printouts
//   - the second run_training(true) call and the comparison table
//
// Regularization (differs from main_experiment.cu BY DEFAULT):
//   -DWEIGHTED_LAMBDA=1 (default) cumf_als-style weighted-λ (ALS-WR,
//                   Zhou et al. 2008): diag += nnz_e·λ, exactly what
//                   cumf_als's get_hermitian* kernels do ("weighted-lambda
//                   regularization", als.cu). Implemented as a tiny
//                   diag-update kernel after each batch's LHS accumulation;
//                   the Cholesky solvers then receive λ=0. Entities with
//                   zero train ratings get plain λ so the system stays SPD.
//                   Use this mode when comparing RMSE against cumf_als at
//                   the same λ (note: good λ values differ! weighted-λ
//                   wants ~0.02-0.06, plain λ was tuned at 0.1).
//   -DWEIGHTED_LAMBDA=0  plain λ·I like main_experiment.cu — use this when
//                   comparing against main_experiment's baseline leg.
//
// Extra debug switches (all OFF by default):
//   -DDEBUG_SYNC    device-sync + error check after each phase of every
//                   iteration (catches async kernel faults at the source;
//                   distorts timing — do not benchmark with this on)
//   -DITER_TIMING   print per-iteration LHS+RHS / Cholesky times per side
//   -DMAX_ITERS=n   cap ALS iterations (default 50, same as main) — use a
//                   small n for quick sanity sweeps
//
// Build (same as main_experiment.cu):
//   nvcc -O3 -arch=sm_86 -std=c++14 -DK_DIM=16 only_scalar_fp32.cu -o scalar_k16
// Run:
//   ./scalar_k16 <dataset.bin> [lambda]
// ============================================================================

#include "common.cuh"
#include "data_utils.cuh"
#include "fused_kernels.cuh"
#include "wmma_kernels.cuh"   // kept so the translation unit matches main_experiment.cu; no WMMA kernel is launched
#include "cholesky_kernels.cuh"

#ifndef WEIGHTED_LAMBDA
#define WEIGHTED_LAMBDA 1
#endif
#ifndef MAX_ITERS
#define MAX_ITERS 50
#endif
// Initial-factor scale (same switch as main_experiment.cu):
//   -DCUMF_INIT=1  factors ~ U(0,0.2) like released cuMF — ~1/3 fewer ALS
//                  iterations and slightly better test RMSE (use for any
//                  comparison against cuMF).
//   -DCUMF_INIT=0  (default) legacy U(0.1,1.1) init — keeps all validated
//                  trajectories reproducible.
#ifndef CUMF_INIT
#define CUMF_INIT 0
#endif
// BALS_SYMTILE (2026-07-21): swap the FP32 Gram kernel for the BALS/cuMF-style
// symmetric register-tiled mapping (dx=dy=4, only the lower-triangular K*K
// tiles computed, symmetric write-out). This is the mapping the BALS paper uses
// to run FASTER than cuMF_ALS; the default kernel (compute_LHS_RHS_BALS_block)
// is the faithful-but-slow 2x2/full-matrix baseline. -DBALS_SYMTILE=1 to enable.
#ifndef BALS_SYMTILE
#define BALS_SYMTILE 0
#endif

#if WEIGHTED_LAMBDA
// cumf_als parity (als.cu "weighted-lambda regularization"):
//   tt[diag] += (end - start) * lambda   with (end-start) = entity's train nnz.
// Runs on the compute stream after each batch's LHS accumulation; the solvers
// then get λ=0 so nothing is added twice. nnz=0 entities (test-only users/
// items) fall back to plain λ — otherwise their LHS is all-zero and Cholesky
// produces NaNs that poison the RMSE (cumf_als never guards this because its
// inputs have no empty rows).
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

#if CHOL_MP != 0 || CHOL_STALE
#error "only_scalar_fp32.cu is the FP32 baseline of record — it is deliberately NOT wired for CHOL_MP/CHOL_STALE. Build it without those flags (a CHOL_MP build here would silently keep the FP32 solver and mislabel the run)."
#endif

int main(int argc, char* argv[]) {
    printf("=== CODE VERSION: SCALAR-FP32-ONLY (from SYNC-2026-07-06) [baseline leg of main_experiment.cu, no WMMA/FP16 launches] ===\n");
    printf("Regularization: %s\n", WEIGHTED_LAMBDA
        ? "WEIGHTED lambda (ALS-WR, cumf_als-style: diag += nnz_e*lambda)"
        : "PLAIN lambda (main_experiment.cu parity: diag += lambda)");
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

#ifndef BALS_REORDER
#define BALS_REORDER 0
#endif
#if BALS_REORDER
    {
        // BALS data reordering (Chen et al. TPDS'21, Algorithm 3): sort rows
        // (users) and columns (items) in DESCENDING order of train nonzeros so
        // the nonzeros cluster toward the top-left of R. This makes scattered
        // vacant row-segments coalesce into whole vacant tiles (skipped once
        // instead of probed per segment) and raises column-vector reuse inside
        // dense tiles. The permutation is a bijection over users/items, so the
        // learned factors and the test RMSE are unchanged — only the tile
        // structure (and thus speed) changes. Toggle with -DBALS_REORDER=1.
        vector<int> u_nnz(num_users, 0), i_nnz(num_items, 0);
        for (int k = 0; k < nnz_train; k++) { u_nnz[raw_users[k]]++; i_nnz[raw_items[k]]++; }
        vector<int> newToOldU(num_users), newToOldI(num_items);
        for (int i = 0; i < num_users; i++) newToOldU[i] = i;
        for (int i = 0; i < num_items; i++) newToOldI[i] = i;
        stable_sort(newToOldU.begin(), newToOldU.end(),
                    [&](int a, int b){ return u_nnz[a] > u_nnz[b]; });
        stable_sort(newToOldI.begin(), newToOldI.end(),
                    [&](int a, int b){ return i_nnz[a] > i_nnz[b]; });
        vector<int> oldToNewU(num_users), oldToNewI(num_items);
        for (int r = 0; r < num_users; r++) oldToNewU[newToOldU[r]] = r;
        for (int r = 0; r < num_items; r++) oldToNewI[newToOldI[r]] = r;

        // Remap train + test COO into the reordered id space.
        for (int k = 0; k < nnz_train; k++) {
            raw_users[k] = oldToNewU[raw_users[k]];
            raw_items[k] = oldToNewI[raw_items[k]];
        }
        for (int k = 0; k < nnz_test; k++) {
            test_users_h[k] = oldToNewU[test_users_h[k]];
            test_items_h[k] = oldToNewI[test_items_h[k]];
        }

        // Rebuild CSR (user-major) from the remapped train COO.
        h_user_offsets.assign(num_users + 1, 0);
        for (int k = 0; k < nnz_train; k++) h_user_offsets[raw_users[k] + 1]++;
        for (int u = 0; u < num_users; u++) h_user_offsets[u + 1] += h_user_offsets[u];
        h_item_indices.assign(nnz_train, 0); h_user_ratings.assign(nnz_train, 0.0f);
        {
            vector<int> cur(h_user_offsets.begin(), h_user_offsets.end() - 1);
            for (int k = 0; k < nnz_train; k++) {
                int p = cur[raw_users[k]]++;
                h_item_indices[p] = raw_items[k];
                h_user_ratings[p] = raw_ratings[k];
            }
        }
        // Rebuild CSC (item-major).
        h_item_offsets.assign(num_items + 1, 0);
        for (int k = 0; k < nnz_train; k++) h_item_offsets[raw_items[k] + 1]++;
        for (int i = 0; i < num_items; i++) h_item_offsets[i + 1] += h_item_offsets[i];
        h_user_indices.assign(nnz_train, 0); h_item_ratings.assign(nnz_train, 0.0f);
        {
            vector<int> cur(h_item_offsets.begin(), h_item_offsets.end() - 1);
            for (int k = 0; k < nnz_train; k++) {
                int p = cur[raw_items[k]]++;
                h_user_indices[p] = raw_users[k];
                h_item_ratings[p] = raw_ratings[k];
            }
        }
        printf("[BALS_REORDER] Alg.3 applied: rows/cols sorted by descending nnz "
               "(max u_nnz=%d, max i_nnz=%d)\n",
               *max_element(u_nnz.begin(), u_nnz.end()),
               *max_element(i_nnz.begin(), i_nnz.end()));
    }
#endif

    int K         = K_DIM;
    float lambda  = (argc > 2) ? atof(argv[2]) : 0.1f;
    // What the Cholesky solvers add to the diagonal themselves. In weighted
    // mode the full λ·nnz_e term is pre-added by add_weighted_lambda_diag,
    // so the solvers must add nothing.
    const float solver_lambda = WEIGHTED_LAMBDA ? 0.0f : lambda;
    int max_iters = MAX_ITERS;
    float tol     = 0.001f;
    int max_ent   = max(num_users, num_items);
    int yb_eff    = (K > 32) ? 128 : YB;

    srand(42);
    vector<float> h_X(num_users * K), h_Y(num_items * K);
#if CUMF_INIT
    // cuMF-scale init (see main_experiment.cu for the measured rationale)
    for (int i = 0; i < num_users * K; i++) h_X[i] = 0.2f * ((float)rand() / (float)RAND_MAX);
    for (int i = 0; i < num_items * K; i++) h_Y[i] = 0.2f * ((float)rand() / (float)RAND_MAX);
#else
    for (int i = 0; i < num_users * K; i++) h_X[i] = 0.1f + (rand() % 100) / 100.0f;
    for (int i = 0; i < num_items * K; i++) h_Y[i] = 0.1f + (rand() % 100) / 100.0f;
#endif
    vector<float> h_X_init = h_X, h_Y_init = h_Y;

    float *d_X, *d_Y;
    cudaMalloc(&d_X,       sizeof(float) * num_users * K);
    cudaMalloc(&d_Y,       sizeof(float) * num_items * K);
    // UNUSED here (WMMA gather inputs) — allocated only so the memory
    // footprint and allocation order match main_experiment.cu, which decides
    // whether the 2nd LHS buffer below fits (nbuf=1 vs 2).
    half *d_X_half, *d_Y_half;
    cudaMalloc(&d_X_half,  sizeof(half) * num_users * K);
    cudaMalloc(&d_Y_half,  sizeof(half) * num_items * K);
    int lhs_batch = min(ENTITY_BATCH_SIZE, max_ent);
    // Double-buffered LHS: the Cholesky solve of batch b (stream sS) only
    // touches its own LHS buffer, so LHS+RHS of batch b+1 (stream sC) can run
    // concurrently in the other buffer. Falls back to a single buffer
    // (serial, old behavior) if the second allocation doesn't fit.
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
    // (higher priority). Both are BLOCKING streams: every phase-level op left
    // on the legacy NULL stream (feature memset, t0..t5 event records, RMSE)
    // is a full join of both pipes — that alone provides all cross-phase and
    // cross-iteration ordering.
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

    // Job lists for the scalar tile kernel (full lists only — the baseline
    // leg never uses the sparse-only sjob lists of the mixed path)
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
    printf("Load balancing: User jobs=%d (from %d tile-rows), Item jobs=%d (from %d tile-rows)\n",
           u_num_jobs, num_tiles_x_u, i_num_jobs, num_tiles_x_i);

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
    upload_jobs(u_job_tx,  u_job_chunk,  u_job_nch,  &d_u_job_tx,  &d_u_job_chunk,  &d_u_job_nch);
    upload_jobs(i_job_tx,  i_job_chunk,  i_job_nch,  &d_i_job_tx,  &d_i_job_chunk,  &d_i_job_nch);

    // UNUSED here (plain CSR for the WMMA gather path) — kept, like the FP16
    // buffers above, purely for allocation-order/footprint parity with
    // main_experiment.cu. The scalar kernel reads only the tile format.
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
    // Per-entity train nnz for weighted-λ (allocated after everything that
    // exists in main_experiment.cu, so the nbuf decision above is unaffected)
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
    int2* d_chol_map = nullptr;
    int chol_nvec = 0, chol_ntot = 0;
    {
        std::vector<int2> hmap;
        build_cholesky_tile_map(K, hmap, chol_nvec);
        chol_ntot = (int)hmap.size();
        cudaMalloc(&d_chol_map, hmap.size() * sizeof(int2));
        cudaMemcpy(d_chol_map, hmap.data(), hmap.size() * sizeof(int2), cudaMemcpyHostToDevice);
    }

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

    // UNIFORM smem carveout for every kernel in the batch pipeline (see
    // main_experiment.cu for the Ampere reconfig-barrier rationale). Defined
    // before the geometry branch so both Gram kernels can use it.
    auto set_max_shared = [](const void* f) {
        cudaFuncSetAttribute(f, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    };
    int g_row_split;      // gridDim.y (row-split) — shared by launch macro + banner
    int rows_per_thread;  // rows each thread-plane sequences (RPT); dispatch key below

#if BALS_SYMTILE
    // ===== BALS symmetric-tile Gram mapping (dx=dy=4, lower triangle only) =====
    // Each ACTIVE thread owns one 4x4 output sub-tile; only NB*(NB+1)/2 tiles
    // (NB=K/4) are computed (symmetry) and written to both halves. RPT is fixed
    // at 1 (DZ*ROW_SPLIT = XB) so the register accumulator is just 4*4+4 floats
    // — no spill. DZ = largest power of two keeping padded_threads*DZ <= 1024
    // and dividing XB; ROW_SPLIT (gridDim.y) covers the remaining XB rows.
#ifndef SYM_DX_VAL
#define SYM_DX_VAL 4
#endif
    const int SYM_DX = SYM_DX_VAL;
    int sym_nb  = K / SYM_DX;
    int sym_nt  = sym_nb * (sym_nb + 1) / 2;                // lower-tri tiles per row
    int sym_ntp = ((sym_nt + 31) / 32) * 32;                // pad active-thread count to a warp
    int sym_dz  = 1;                                        // rows processed concurrently (z-planes)
    while ((sym_dz * 2) <= XB && (long long)sym_ntp * (sym_dz * 2) <= 1024) sym_dz *= 2;
    while (XB % sym_dz != 0) sym_dz /= 2;
    g_row_split     = XB / sym_dz;                          // gridDim.y; RPT=1 so DZ*ROW_SPLIT=XB
    rows_per_thread = 1;
    dim3 lhs_threads_bals(sym_ntp, 1, sym_dz);
    printf("BALS_SYMTILE mapping: dx=dy=%d  nb=%d  tiles/row=%d  padded_threads=%d  DZ=%d  ROW_SPLIT=%d  block=%d\n",
           SYM_DX, sym_nb, sym_nt, sym_ntp, sym_dz, g_row_split, sym_ntp * sym_dz);

    #define LAUNCH_FUSED_KERNEL(blocks, RPT_VAL, num_ents, d_TilePtr, d_TileCol, d_SegPtr, d_SegCol, d_SegVal, d_Feat, d_LHS, d_RHS, d_Dens, d_NzList, d_NzPtr, d_JobTx, d_JobChunk, d_JobNch, bstart, bsize) \
        compute_LHS_RHS_BALS_symtile<1, SYM_DX_VAL><<<dim3((unsigned)(blocks), g_row_split), lhs_threads_bals, shared_mem_size, sC>>>( \
            num_ents, K, lambda, \
            d_TilePtr, d_TileCol, d_SegPtr, d_SegCol, d_SegVal, \
            d_Feat, d_LHS, d_RHS, d_Dens, d_NzList, d_NzPtr, \
            d_JobTx, d_JobChunk, d_JobNch, bstart, bsize)

    cudaFuncSetAttribute((const void*)compute_LHS_RHS_BALS_symtile<1, SYM_DX_VAL>, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size);
    set_max_shared((const void*)compute_LHS_RHS_BALS_symtile<1, SYM_DX_VAL>);
#else
    // Scalar kernel thread geometry — identical to main_experiment.cu
    // (see its RR_C / ROW_SPLIT rationale comments)
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

    #ifndef ROW_SPLIT
    #define ROW_SPLIT (K_DIM==16 ? 1 : K_DIM==32 ? 2 : K_DIM==48 ? 4 : 8)
    #endif
    if (XB % (DZ * ROW_SPLIT) != 0) {
        cout << "Error: ROW_SPLIT=" << ROW_SPLIT << " incompatible with DZ=" << DZ << endl;
        return 1;
    }
    g_row_split     = ROW_SPLIT;
    rows_per_thread = XB / (DZ * ROW_SPLIT);

    #define LAUNCH_FUSED_KERNEL(blocks, RPT_VAL, num_ents, d_TilePtr, d_TileCol, d_SegPtr, d_SegCol, d_SegVal, d_Feat, d_LHS, d_RHS, d_Dens, d_NzList, d_NzPtr, d_JobTx, d_JobChunk, d_JobNch, bstart, bsize) \
        compute_LHS_RHS_BALS_block<RPT_VAL, RR_C, RC_C><<<dim3((unsigned)(blocks), g_row_split), lhs_threads_bals, shared_mem_size, sC>>>( \
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

    set_max_shared((const void*)compute_LHS_RHS_BALS_block<1, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<2, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<4, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<8, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<16, RR_C, RC_C>);
    set_max_shared((const void*)compute_LHS_RHS_BALS_block<32, RR_C, RC_C>);
#endif
    set_max_shared((const void*)compute_RMSE_kernel);
#if K_DIM <= BATCHED_CHOLESKY_MAX_K
    // batched solver: zero smem, its per-thread local matrix lives on L1 —
    // leave at driver default (max L1).
#elif K_DIM == 16
    set_max_shared((const void*)cholesky_solve_packed<16>);
#elif K_DIM == 32
    set_max_shared((const void*)cholesky_solve_tiled<32, 64>);
#elif K_DIM == 48
    set_max_shared((const void*)cholesky_solve_tiled<48, 96>);
#elif K_DIM == 64
    set_max_shared((const void*)cholesky_solve_tiled<64, 128>);
#else
    set_max_shared((const void*)cholesky_solve_tiled<96, 128>);
#endif

    // Catch any setup error (incl. silent cudaMalloc OOM with ignored return
    // codes — the classic "kernel looks buggy but it was OOM" trap)
    {
        cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            printf("FATAL: setup left a CUDA error pending: %s\n", cudaGetErrorString(e));
            return 1;
        }
        size_t mfree = 0, mtotal = 0;
        cudaMemGetInfo(&mfree, &mtotal);
        printf("GPU memory after setup: %.0f MB free / %.0f MB total | LHS buffers=%d\n",
               mfree / 1048576.0, mtotal / 1048576.0, nbuf);
    }

#ifdef DEBUG_SYNC
    #define PHASE_CHECK(tag) do { \
        cudaDeviceSynchronize(); \
        cudaError_t _e = cudaGetLastError(); \
        if (_e != cudaSuccess) { \
            printf("[DEBUG_SYNC] iter %d, %s: %s\n", iter, tag, cudaGetErrorString(_e)); \
            exit(1); \
        } } while (0)
#else
    #define PHASE_CHECK(tag)
#endif

    // Training loop — the run_training(false) leg of main_experiment.cu
    struct TrainResult { float compute_ms, wall_ms, train_rmse, test_rmse; };

    auto run_training = [&](const char* label) -> TrainResult {
        cudaMemcpy(d_X, h_X_init.data(), sizeof(float)*num_users*K, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, h_Y_init.data(), sizeof(float)*num_items*K, cudaMemcpyHostToDevice);

        // Force the FP32 branch of compute_LHS_RHS_BALS_block for every tile
        // (density < TAU1 everywhere), exactly like run_training(false)
        {
            vector<float> zd_u(user_tile_density.size(), 0.0f);
            vector<float> zd_i(item_tile_density.size(), 0.0f);
            cudaMemcpy(d_u_tile_density, zd_u.data(), sizeof(float)*zd_u.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(d_i_tile_density, zd_i.data(), sizeof(float)*zd_i.size(), cudaMemcpyHostToDevice);
        }

        printf("\n=== %s ===\n", label);
        float prev_rmse = 1e9f, final_train_rmse = 0, final_test_rmse = 0;
        float total_u_compute=0, total_u_solve=0, total_i_compute=0, total_i_solve=0, total_rmse_t=0;
        int total_iters=0, rmse_calls=0;

        cudaEvent_t ev[10];
        for (int i=0; i<10; i++) cudaEventCreate(&ev[i]);
        cudaEvent_t &t0=ev[0],&t1=ev[1],&t2=ev[2],&t3=ev[3],&t4=ev[4],
                    &t5=ev[5],&t6=ev[6],&t7=ev[7],&ev_start=ev[8],&ev_stop=ev[9];
        // Per-batch event pool for true Cholesky timing; solve-end events
        // [2b+1] double as "LHS buffer b%nbuf is free again" markers.
        const int MAX_SOLVE_EVT = 32;
        cudaEvent_t evs_u[2*MAX_SOLVE_EVT], evs_i[2*MAX_SOLVE_EVT];
        for (int i = 0; i < 2*MAX_SOLVE_EVT; i++) { cudaEventCreate(&evs_u[i]); cudaEventCreate(&evs_i[i]); }
        cudaEvent_t ev_lhs_u[MAX_SOLVE_EVT], ev_lhs_i[MAX_SOLVE_EVT];
        for (int i = 0; i < MAX_SOLVE_EVT; i++) {
            cudaEventCreateWithFlags(&ev_lhs_u[i], cudaEventDisableTiming);
            cudaEventCreateWithFlags(&ev_lhs_i[i], cudaEventDisableTiming);
        }
        int n_sevt_u = 0, n_sevt_i = 0;
        cudaEventRecord(ev_start);

        for (int iter = 0; iter < max_iters; iter++) {
            total_iters++;

            cudaEventRecord(t0);
            // --- User-side: entity-batched LHS/RHS + Cholesky ---
            cudaMemsetAsync(d_X, 0, sizeof(float)*num_users*K);
            for (int bstart = 0; bstart < num_users; bstart += lhs_batch) {
                int bend = min(bstart + lhs_batch, num_users);
                int bn   = bend - bstart;
                int bidx = bstart / lhs_batch;
                float* lhs_b = d_LHS_buf[bidx % nbuf];
                // Before touching this LHS buffer, wait (on the compute
                // stream) for the solve that last read it: batch bidx-nbuf.
                int prevb = bidx - nbuf;
                if (prevb >= 0 && prevb < MAX_SOLVE_EVT)
                    cudaStreamWaitEvent(sC, evs_u[2*prevb+1], 0);
                cudaMemsetAsync(lhs_b, 0, sizeof(float)*(long long)bn*K*K, sC);

                if      (u_num_jobs==0) ;
                else if (rows_per_thread==1)  LAUNCH_FUSED_KERNEL(u_num_jobs, 1,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,d_u_job_tx,d_u_job_chunk,d_u_job_nch,bstart,bn);
                else if (rows_per_thread==2)  LAUNCH_FUSED_KERNEL(u_num_jobs, 2,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,d_u_job_tx,d_u_job_chunk,d_u_job_nch,bstart,bn);
                else if (rows_per_thread==4)  LAUNCH_FUSED_KERNEL(u_num_jobs, 4,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,d_u_job_tx,d_u_job_chunk,d_u_job_nch,bstart,bn);
                else if (rows_per_thread==8)  LAUNCH_FUSED_KERNEL(u_num_jobs, 8,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,d_u_job_tx,d_u_job_chunk,d_u_job_nch,bstart,bn);
                else if (rows_per_thread==16) LAUNCH_FUSED_KERNEL(u_num_jobs,16,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,d_u_job_tx,d_u_job_chunk,d_u_job_nch,bstart,bn);
                else if (rows_per_thread==32) LAUNCH_FUSED_KERNEL(u_num_jobs,32,num_users,d_u_tile_ptr,d_u_tile_colidx,d_u_seg_ptr,d_u_seg_colidx,d_u_seg_values,d_Y,lhs_b,d_X,d_u_tile_density,d_u_nz_list,d_u_nz_ptr,d_u_job_tx,d_u_job_chunk,d_u_job_nch,bstart,bn);

#if WEIGHTED_LAMBDA
                // diag += nnz_e * lambda (cumf_als parity); on sC after the
                // LHS accumulation, before the sC->sS solve handoff
                add_weighted_lambda_diag<<<(int)(((long long)bn * K + 255) / 256), 256, 0, sC>>>(
                    lhs_b, d_u_nnz, bstart, bn, K, lambda);
#endif

                // Cholesky solve for this batch — handed to the solve stream
                // so it overlaps the NEXT batch's LHS+RHS.
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
                else if (K == 32) cholesky_solve_tiled<32, 64><<<bn, 64, cholesky_tiled_smem<32>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 48) cholesky_solve_tiled<48, 96><<<bn, 96, cholesky_tiled_smem<48>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 64) cholesky_solve_tiled<64, 128><<<bn, 128, cholesky_tiled_smem<64>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 96) cholesky_solve_tiled<96, 128><<<bn, 128, cholesky_tiled_smem<96>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
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
            PHASE_CHECK("user-side LHS/RHS + Cholesky");
            // user cholesky timing accumulated from evs_u after sync
            cudaEventRecord(t2);

            cudaEventRecord(t3);
            // --- Item-side: entity-batched LHS/RHS + Cholesky ---
            cudaMemsetAsync(d_Y, 0, sizeof(float)*num_items*K);
            for (int bstart = 0; bstart < num_items; bstart += lhs_batch) {
                int bend = min(bstart + lhs_batch, num_items);
                int bn   = bend - bstart;
                int bidx = bstart / lhs_batch;
                float* lhs_b = d_LHS_buf[bidx % nbuf];
                int prevb = bidx - nbuf;
                if (prevb >= 0 && prevb < MAX_SOLVE_EVT)
                    cudaStreamWaitEvent(sC, evs_i[2*prevb+1], 0);
                cudaMemsetAsync(lhs_b, 0, sizeof(float)*(long long)bn*K*K, sC);

                if      (i_num_jobs==0) ;
                else if (rows_per_thread==1)  LAUNCH_FUSED_KERNEL(i_num_jobs, 1,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,d_i_job_tx,d_i_job_chunk,d_i_job_nch,bstart,bn);
                else if (rows_per_thread==2)  LAUNCH_FUSED_KERNEL(i_num_jobs, 2,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,d_i_job_tx,d_i_job_chunk,d_i_job_nch,bstart,bn);
                else if (rows_per_thread==4)  LAUNCH_FUSED_KERNEL(i_num_jobs, 4,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,d_i_job_tx,d_i_job_chunk,d_i_job_nch,bstart,bn);
                else if (rows_per_thread==8)  LAUNCH_FUSED_KERNEL(i_num_jobs, 8,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,d_i_job_tx,d_i_job_chunk,d_i_job_nch,bstart,bn);
                else if (rows_per_thread==16) LAUNCH_FUSED_KERNEL(i_num_jobs,16,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,d_i_job_tx,d_i_job_chunk,d_i_job_nch,bstart,bn);
                else if (rows_per_thread==32) LAUNCH_FUSED_KERNEL(i_num_jobs,32,num_items,d_i_tile_ptr,d_i_tile_colidx,d_i_seg_ptr,d_i_seg_colidx,d_i_seg_values,d_X,lhs_b,d_Y,d_i_tile_density,d_i_nz_list,d_i_nz_ptr,d_i_job_tx,d_i_job_chunk,d_i_job_nch,bstart,bn);

#if WEIGHTED_LAMBDA
                add_weighted_lambda_diag<<<(int)(((long long)bn * K + 255) / 256), 256, 0, sC>>>(
                    lhs_b, d_i_nnz, bstart, bn, K, lambda);
#endif

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
                else if (K == 32) cholesky_solve_tiled<32, 64><<<bn, 64, cholesky_tiled_smem<32>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 48) cholesky_solve_tiled<48, 96><<<bn, 96, cholesky_tiled_smem<48>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 64) cholesky_solve_tiled<64, 128><<<bn, 128, cholesky_tiled_smem<64>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
                else if (K == 96) cholesky_solve_tiled<96, 128><<<bn, 128, cholesky_tiled_smem<96>(), sv>>>(lhs_b, rhs_b, d_chol_map, chol_nvec, chol_ntot, bn, solver_lambda);
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
            PHASE_CHECK("item-side LHS/RHS + Cholesky");
            // item cholesky timing accumulated from evs_i after sync
            cudaEventRecord(t5);

            cudaEventSynchronize(t5);
            float ms;
            // True solve times from the per-batch event pool; subtract from the
            // side totals so "LHS+RHS" really is accumulation-only.
            float u_sol = 0, i_sol = 0;
            for (int b = 0; b < n_sevt_u; b++) { cudaEventElapsedTime(&ms, evs_u[2*b], evs_u[2*b+1]); u_sol += ms; }
            for (int b = 0; b < n_sevt_i; b++) { cudaEventElapsedTime(&ms, evs_i[2*b], evs_i[2*b+1]); i_sol += ms; }
            total_u_solve += u_sol;
            total_i_solve += i_sol;
            float u_cmp, i_cmp;
            cudaEventElapsedTime(&ms,t0,t1); u_cmp = ms - u_sol; total_u_compute += u_cmp;
            cudaEventElapsedTime(&ms,t3,t4); i_cmp = ms - i_sol; total_i_compute += i_cmp;
#ifdef ITER_TIMING
            printf("  [iter %2d] user LHS+RHS %8.2f ms | user chol %8.2f ms | item LHS+RHS %8.2f ms | item chol %8.2f ms\n",
                   iter+1, u_cmp, u_sol, i_cmp, i_sol);
#endif

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
        if (rmse_calls>0) printf("RMSE:         %7.2f ms total | %5.2f ms/call\n",total_rmse_t,total_rmse_t/rmse_calls);
        if (nbuf == 2) printf("(Cholesky overlaps next batch's LHS+RHS on a 2nd stream; LHS+RHS rows = EXPOSED time = phase span minus solve busy time, so their GFlops/s overstates pure kernel throughput. Totals remain exact.)\n");
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

    TrainResult base = run_training(WEIGHTED_LAMBDA
        ? "FP32 Baseline (BALS — all tiles FP32, scalar-only, WEIGHTED lambda like cumf_als)"
        : "FP32 Baseline (BALS — all tiles FP32, scalar-only, plain lambda)");

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║        SCALAR FP32 BASELINE — single run  (80/20 split)     ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Compute time:           %10.2f ms                      ║\n", base.compute_ms);
    printf("║  Wall time:              %10.2f ms                      ║\n", base.wall_ms);
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║  Train RMSE:             %.6f                          ║\n", base.train_rmse);
    printf("║  Test  RMSE:             %.6f                          ║\n", base.test_rmse);
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    printf("Params: K_DIM=%d, lambda=%.4f (%s), XB=%d, YB=%d (yb_eff=%d), ENTITY_BATCH_SIZE=%d, ROW_SPLIT=%d, BATCHED_CHOLESKY_MAX_K=%d, MAX_ITERS=%d\n",
           K_DIM, lambda,
           WEIGHTED_LAMBDA ? "weighted ALS-WR: diag += nnz_e*lambda" : "plain: diag += lambda",
           XB, YB, yb_eff, ENTITY_BATCH_SIZE, g_row_split, BATCHED_CHOLESKY_MAX_K, MAX_ITERS);

    cudaMemcpy(h_X.data(), d_X, sizeof(float) * num_users * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y.data(), d_Y, sizeof(float) * num_items * K, cudaMemcpyDeviceToHost);

    cudaFree(d_train_users); cudaFree(d_train_items); cudaFree(d_train_ratings);
    cudaFree(d_test_users);  cudaFree(d_test_items);  cudaFree(d_test_ratings);
    cudaFree(d_sq_err);
    cudaStreamDestroy(sC);   cudaStreamDestroy(sS);
    cudaFree(d_X);           cudaFree(d_Y);
    cudaFree(d_LHS_buf[0]);  if (nbuf == 2) cudaFree(d_LHS_buf[1]);
    cudaFree(d_X_half);      cudaFree(d_Y_half);
    cudaFree(d_u_offsets);   cudaFree(d_u_colidx);     cudaFree(d_u_csr_vals);
    cudaFree(d_i_offsets);   cudaFree(d_i_colidx);     cudaFree(d_i_csr_vals);
    cudaFree(d_chol_map);
#if WEIGHTED_LAMBDA
    cudaFree(d_u_nnz);       cudaFree(d_i_nnz);
#endif
    cudaFree(d_u_tile_ptr);  cudaFree(d_u_tile_colidx);
    cudaFree(d_u_seg_ptr);   cudaFree(d_u_seg_colidx); cudaFree(d_u_seg_values);
    cudaFree(d_i_tile_ptr);  cudaFree(d_i_tile_colidx);
    cudaFree(d_i_seg_ptr);   cudaFree(d_i_seg_colidx); cudaFree(d_i_seg_values);
    cudaFree(d_u_tile_density); cudaFree(d_i_tile_density);
    cudaFree(d_u_nz_list);   cudaFree(d_u_nz_ptr);
    cudaFree(d_i_nz_list);   cudaFree(d_i_nz_ptr);
    cudaFree(d_u_job_tx);    cudaFree(d_u_job_chunk);  cudaFree(d_u_job_nch);
    cudaFree(d_i_job_tx);    cudaFree(d_i_job_chunk);  cudaFree(d_i_job_nch);
    return 0;
}
