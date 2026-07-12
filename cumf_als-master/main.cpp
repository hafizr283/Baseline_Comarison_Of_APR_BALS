/*
 * main.cpp — cuMF_als driver for the APR-BALS thesis comparison (Pillar 2).
 *
 * Replaces the original multi-file loader: reads the APR-BALS preprocessed
 * .bin (the FROZEN 80/20 train/test split used by justapr.cu /
 * main_experiment.cu) so cuMF trains on the *identical* data. The .bin
 * already contains both CSR orientations, so no conversion is needed:
 *   header: int32 version, num_users, num_items, nnz_train, nnz_test
 *   blocks (uint64 count prefix, then count elements):
 *     1 train_users(i32) 2 train_items(i32) 3 train_ratings(f32)   TRAIN coo
 *     4 test_users(i32)  5 test_items(i32)  6 test_ratings(f32)    TEST  coo
 *     7 user_offsets     8 item_indices     9 user_ratings         user-major CSR
 *    10 item_offsets    11 user_indices    12 item_ratings         item-major CSR
 *
 * cuMF orientation (matches the HPDC16 Netflix setup): R is items x users,
 *   m = num_items rows  -> X  (item factors,  updated first)
 *   n = num_users cols  -> theta (user factors, updated second)
 *   CSR of R  = item-major arrays (10,11,12)
 *   CSC of R  = user-major arrays (7,8,9)
 *
 * ALS solver code (als.cu, cg.cu) is the released cuMF code, CUDA-12 ported.
 */
#include "als.h"
#include "host_utilities.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string>
#include <vector>

static void* read_block(FILE* f, size_t elem_size, uint64_t expect_count, const char* name) {
	uint64_t count = 0;
	if (fread(&count, sizeof(uint64_t), 1, f) != 1) {
		fprintf(stderr, "ERROR: failed reading count of block %s\n", name); exit(1);
	}
	if (count != expect_count) {
		fprintf(stderr, "ERROR: block %s count %llu != expected %llu — .bin layout mismatch, DO NOT trust results\n",
				name, (unsigned long long)count, (unsigned long long)expect_count);
		exit(1);
	}
	void* buf = malloc(count * elem_size);
	if (!buf) { fprintf(stderr, "ERROR: OOM on block %s\n", name); exit(1); }
	if (fread(buf, elem_size, count, f) != count) {
		fprintf(stderr, "ERROR: short read on block %s\n", name); exit(1);
	}
	return buf;
}

static void skip_block(FILE* f, size_t elem_size, const char* name) {
	uint64_t count = 0;
	if (fread(&count, sizeof(uint64_t), 1, f) != 1) {
		fprintf(stderr, "ERROR: failed reading count of block %s\n", name); exit(1);
	}
	if (fseek(f, (long)(count * elem_size), SEEK_CUR) != 0) {
		fprintf(stderr, "ERROR: failed skipping block %s\n", name); exit(1);
	}
}

static void save_float_array(const char* path, const float* a, size_t count) {
	FILE* f = fopen(path, "wb");
	if (!f) { fprintf(stderr, "WARN: cannot open %s for writing\n", path); return; }
	fwrite(a, sizeof(float), count, f);
	fclose(f);
}

int main(int argc, char **argv) {
	if (argc < 2) {
		printf("Usage: %s <dataset.bin> [lambda=0.048] [F=100] [X_BATCH=1] [THETA_BATCH=3] [DEVICE=1] [MAX_ITERS=50] [TOL=0.001]\n", argv[0]);
		printf("  dataset.bin: APR-BALS preprocessed binary (frozen 80/20 split)\n");
		printf("  F must be a multiple of 10 (cuMF kernel constraint); F=100 is cuMF's optimized path\n");
		printf("  lambda is WEIGHTED (ALS-WR): diag += nnz_of_entity * lambda. HPDC16 Netflix value: 0.048\n");
		return 0;
	}
	const char* bin_path = argv[1];
	float lambda    = (argc > 2) ? atof(argv[2]) : 0.048f;
	int   f         = (argc > 3) ? atoi(argv[3]) : 100;
	int   X_BATCH   = (argc > 4) ? atoi(argv[4]) : 1;
	int   THETA_BATCH = (argc > 5) ? atoi(argv[5]) : 3;
	int   DEVICEID  = (argc > 6) ? atoi(argv[6]) : 1;
	int   MAX_ITERS = (argc > 7) ? atoi(argv[7]) : 50;
	float TOL       = (argc > 8) ? atof(argv[8]) : 0.001f;

	if (f % T10 != 0) {
		printf("F has to be a multiple of %d\n", T10);
		return 0;
	}

	printf("=== CODE VERSION: CUMF-ALS-HPDC16 [CUDA-12 port | generic SpMM | CG(96-thread) | frozen-split .bin loader | APR-style output] ===\n");
	printf("Loading binary preprocessed dataset: %s\n", bin_path);

	FILE* file = fopen(bin_path, "rb");
	if (!file) { fprintf(stderr, "ERROR: cannot open %s\n", bin_path); return 1; }
	int32_t version, num_users, num_items, nnz_train_i, nnz_test_i;
	if (fread(&version, 4, 1, file) != 1 || fread(&num_users, 4, 1, file) != 1 ||
	    fread(&num_items, 4, 1, file) != 1 || fread(&nnz_train_i, 4, 1, file) != 1 ||
	    fread(&nnz_test_i, 4, 1, file) != 1) {
		fprintf(stderr, "ERROR: failed reading .bin header\n"); return 1;
	}
	long nnz = nnz_train_i, nnz_test = nnz_test_i;
	printf("%d users, %d items | train=%d, test=%d\n", num_users, num_items, nnz_train_i, nnz_test_i);

	// blocks 1-3: train COO (user-major) — not needed, item-major CSR provides the train set
	skip_block(file, 4, "train_users");
	skip_block(file, 4, "train_items");
	skip_block(file, 4, "train_ratings");
	// blocks 4-6: test COO
	int*   test_users   = (int*)   read_block(file, 4, (uint64_t)nnz_test, "test_users");
	int*   test_items   = (int*)   read_block(file, 4, (uint64_t)nnz_test, "test_items");
	float* test_ratings = (float*) read_block(file, 4, (uint64_t)nnz_test, "test_ratings");
	// blocks 7-9: user-major CSR == CSC of R (R = items x users)
	int*   user_offsets = (int*)   read_block(file, 4, (uint64_t)num_users + 1, "user_offsets");
	int*   item_indices = (int*)   read_block(file, 4, (uint64_t)nnz, "item_indices");
	float* user_ratings = (float*) read_block(file, 4, (uint64_t)nnz, "user_ratings");
	// blocks 10-12: item-major CSR == CSR of R
	int*   item_offsets = (int*)   read_block(file, 4, (uint64_t)num_items + 1, "item_offsets");
	int*   user_indices = (int*)   read_block(file, 4, (uint64_t)nnz, "user_indices");
	float* item_ratings = (float*) read_block(file, 4, (uint64_t)nnz, "item_ratings");
	fclose(file);
	if (item_offsets[num_items] != nnz || user_offsets[num_users] != nnz) {
		fprintf(stderr, "ERROR: CSR offset tails (%d, %d) != nnz %ld — layout mismatch\n",
				item_offsets[num_items], user_offsets[num_users], nnz);
		return 1;
	}

	const int m = num_items;   // rows of R -> X (item factors)
	const int n = num_users;   // cols of R -> theta (user factors)
	printf("cuMF mapping: R = items x users (m=%d, n=%d) | F=%d | lambda=%.4f (weighted ALS-WR) | X_BATCH=%d THETA_BATCH=%d | CG_ITER=%d | max_iters=%d tol=%.3f | device=%d\n",
			m, n, f, lambda, X_BATCH, THETA_BATCH, 6, MAX_ITERS, TOL, DEVICEID);

	// train COO row index in CSR order (RMSE kernel needs a row id per nnz)
	int* coo_rows = (int*) malloc(sizeof(int) * (size_t)nnz);
	for (int r = 0; r < m; r++)
		for (int j = item_offsets[r]; j < item_offsets[r+1]; j++)
			coo_rows[j] = r;

	// initialize factors exactly as released cuMF (main.cpp): seed 0, thetaT ~ U(0,0.2), XT = 0
	float* thetaTHost = (float*) malloc(sizeof(float) * (size_t)n * f);
	float* XTHost     = (float*) malloc(sizeof(float) * (size_t)m * f);
	unsigned int seed = 0;
	srand(seed);
	for (long k = 0; k < (long)n * f; k++)
		thetaTHost[k] = 0.2f * ((float) rand() / (float)RAND_MAX);
	for (long k = 0; k < (long)m * f; k++)
		XTHost[k] = 0;

	doALS(item_offsets, user_indices, item_ratings,      /* CSR of R  (item-major) */
	      item_indices, user_offsets, user_ratings,      /* CSC of R  (user-major) */
	      coo_rows, thetaTHost, XTHost,
	      test_items, test_users, test_ratings,          /* test COO: row=item, col=user */
	      m, n, f, nnz, nnz_test, lambda,
	      MAX_ITERS, TOL, X_BATCH, THETA_BATCH, DEVICEID);

	// save the model for external fp64 RMSE validation (bench_eval_rmse.py)
	char pathX[256], pathT[256];
	snprintf(pathX, sizeof(pathX), "cumf_XT_f%d.bin", f);
	snprintf(pathT, sizeof(pathT), "cumf_thetaT_f%d.bin", f);
	save_float_array(pathX, XTHost, (size_t)m * f);
	save_float_array(pathT, thetaTHost, (size_t)n * f);
	printf("Model saved: %s (items %dx%d), %s (users %dx%d)\n", pathX, m, f, pathT, n, f);

	free(test_users); free(test_items); free(test_ratings);
	free(user_offsets); free(item_indices); free(user_ratings);
	free(item_offsets); free(user_indices); free(item_ratings);
	free(coo_rows); free(thetaTHost); free(XTHost);
	printf("\nALS Done.\n");
	return 0;
}
