#!/usr/bin/env python3
"""
bench_eval_rmse.py — read APR-BALS's preprocessed .bin (the FROZEN train/test
split) and compute test RMSE for ANY factor model, so an external baseline
(cuMF, implicit, Surprise, numpy ALS, ...) is scored on the *identical* split
the CUDA code uses. This is Pillar 2's fair-comparison glue.

IMPORTANT: This mirrors the binary layout written by the preprocessing step and
read at the top of justapr.cu / main_experiment.cu. It SELF-VERIFIES: it asserts
the vector lengths equal the header nnz_train / nnz_test. If those asserts fire,
the on-disk format differs from this reader — fix the reader, don't trust the
numbers. Verify once against your .bin before relying on it.

.bin layout (little-endian, x86-64 Linux):
    int32 version, num_users, num_items, nnz_train, nnz_test
    then repeated blocks: uint64 count, then count * (int32 | float32):
      1 raw_users(i32)  2 raw_items(i32)  3 raw_ratings(f32)      # TRAIN coo
      4 test_users(i32) 5 test_items(i32) 6 test_ratings(f32)     # TEST  coo
      (7..12 are the user/item CSR arrays — not needed for scoring)

Usage:
    python bench_eval_rmse.py netflix_ratings.bin           # inspect only
    # or import: from bench_eval_rmse import load_split, rmse
"""
import sys
import struct
import numpy as np


def _read_vec(f, dtype):
    (count,) = struct.unpack("<Q", f.read(8))          # size_t = uint64
    buf = f.read(count * np.dtype(dtype).itemsize)
    return np.frombuffer(buf, dtype=dtype, count=count)


def load_split(path):
    """Return dict with train/test COO arrays and dims. Self-verifying."""
    with open(path, "rb") as f:
        version, num_users, num_items, nnz_train, nnz_test = struct.unpack(
            "<iiiii", f.read(20)
        )
        train_u = _read_vec(f, np.int32)
        train_i = _read_vec(f, np.int32)
        train_r = _read_vec(f, np.float32)
        test_u = _read_vec(f, np.int32)
        test_i = _read_vec(f, np.int32)
        test_r = _read_vec(f, np.float32)

    # Fail loudly if the format assumption is wrong.
    assert len(train_u) == nnz_train == len(train_i) == len(train_r), (
        f"train length mismatch: header nnz_train={nnz_train}, "
        f"got u={len(train_u)} i={len(train_i)} r={len(train_r)} "
        f"(the .bin layout differs from this reader — do not trust results)"
    )
    assert len(test_u) == nnz_test == len(test_i) == len(test_r), (
        f"test length mismatch: header nnz_test={nnz_test}, "
        f"got u={len(test_u)} i={len(test_i)} r={len(test_r)}"
    )
    assert train_i.max(initial=0) < num_items and train_u.max(initial=0) < num_users, (
        "index out of range — layout mismatch"
    )

    return dict(
        version=version, num_users=num_users, num_items=num_items,
        nnz_train=nnz_train, nnz_test=nnz_test,
        train=(train_u, train_i, train_r),
        test=(test_u, test_i, test_r),
    )


def rmse(P, Q, users, items, ratings, batch=1_000_000):
    """RMSE of prediction <P[u], Q[i]> vs ratings, over the given COO.
    P: [num_users, K], Q: [num_items, K]. Batched to bound memory."""
    P = np.ascontiguousarray(P, dtype=np.float64)
    Q = np.ascontiguousarray(Q, dtype=np.float64)
    n = len(ratings)
    se = 0.0
    for s in range(0, n, batch):
        e = min(s + batch, n)
        pred = np.einsum("ij,ij->i", P[users[s:e]], Q[items[s:e]])
        diff = pred - ratings[s:e]
        se += float(np.dot(diff, diff))
    return (se / n) ** 0.5


def _main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    d = load_split(sys.argv[1])
    tu, ti, tr = d["train"]
    print(f"version={d['version']}  users={d['num_users']}  items={d['num_items']}")
    print(f"train nnz={d['nnz_train']}  test nnz={d['nnz_test']}")
    print(f"train rating: min={tr.min():.3f} max={tr.max():.3f} mean={tr.mean():.4f}")
    print("SELF-CHECK PASSED — lengths match header; split is the frozen one the "
          "CUDA code uses. Plug your baseline's P,Q into rmse() to score fairly.")


if __name__ == "__main__":
    _main()
