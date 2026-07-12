#!/usr/bin/env python3
"""
baseline_als_gpu.py — GPU Explicit ALS Baseline (PyTorch / cuBLAS)

Implements the identical closed-form ALS update as justapr.cu / main_experiment.cu:
    x_u = (Y_{I_u}^T Y_{I_u} + λI)^{-1} Y_{I_u}^T r_u

Uses PyTorch standard GPU ops (cuBLAS matmul, cuSolver batch-solve) — no custom
CUDA kernels, no mixed precision, no sparse-structure exploitation. Represents
"vanilla GPU ALS with standard library operations" — the prior art baseline that
APR-BALS improves upon.

COMPARISON CLAIM: APR-BALS achieves the same test RMSE as this baseline but runs
X× faster, because it exploits the tiled sparse structure of the rating matrix and
uses FP16 tensor cores (WMMA) for the dense-entity outer products.

Usage:
    python3 baseline_als_gpu.py netflix_ratings.bin --K 16 --lam 0.1
    python3 baseline_als_gpu.py netflix_ratings.bin --K 32 --lam 0.1 --iters 50
    python3 baseline_als_gpu.py ratings.bin --K 64 --lam 0.1 --device cuda:0

Requirements:
    pip install torch numpy scipy
    (needs CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/)
"""
import argparse
import struct
import time
import sys

import numpy as np
import scipy.sparse as sp

try:
    import torch
except ImportError:
    print("ERROR: PyTorch not found. Install with: pip install torch")
    sys.exit(1)

if not torch.cuda.is_available():
    print("WARNING: CUDA not available — will run on CPU (very slow for large datasets)")


# ─── Binary split reader ────────────────────────────────────────────────────

def load_split(path: str) -> dict:
    """Read the frozen .bin split (same format as justapr.cu / bench_eval_rmse.py)."""
    with open(path, "rb") as f:
        version, num_users, num_items, nnz_train, nnz_test = struct.unpack(
            "<iiiii", f.read(20)
        )

        def _read(dtype):
            (n,) = struct.unpack("<Q", f.read(8))
            return np.frombuffer(f.read(n * np.dtype(dtype).itemsize),
                                 dtype=dtype, count=n).copy()

        train_u = _read(np.int32)
        train_i = _read(np.int32)
        train_r = _read(np.float32)
        test_u  = _read(np.int32)
        test_i  = _read(np.int32)
        test_r  = _read(np.float32)

    assert len(train_u) == nnz_train and len(test_u) == nnz_test, \
        "Length mismatch — .bin format differs from expected layout"

    return dict(
        num_users=num_users, num_items=num_items,
        nnz_train=nnz_train, nnz_test=nnz_test,
        train=(train_u, train_i, train_r),
        test=(test_u, test_i, test_r),
    )


# ─── RMSE ───────────────────────────────────────────────────────────────────

def compute_rmse(P: torch.Tensor, Q: torch.Tensor,
                 users: np.ndarray, items: np.ndarray, ratings: np.ndarray,
                 batch: int = 2_000_000) -> float:
    """CPU-side RMSE evaluation (avoids OOM for large test sets)."""
    P_np = P.detach().cpu().numpy().astype(np.float64)
    Q_np = Q.detach().cpu().numpy().astype(np.float64)
    n, se = len(ratings), 0.0
    for s in range(0, n, batch):
        e = min(s + batch, n)
        pred = np.einsum("ij,ij->i", P_np[users[s:e]], Q_np[items[s:e]])
        diff = pred - ratings[s:e].astype(np.float64)
        se += float(np.dot(diff, diff))
    return (se / n) ** 0.5


# ─── ALS update (vectorized COO scatter) ────────────────────────────────────

def als_step(Y: torch.Tensor,
             entities_np: np.ndarray,  # (nnz,) int32 sorted by entity id (stays on CPU)
             features_np: np.ndarray,  # (nnz,) int32 sorted by entity id (stays on CPU)
             ratings_np: np.ndarray,   # (nnz,) float32 sorted by entity id (stays on CPU)
             offsets: np.ndarray,      # (n_entities+1,) CSR row-pointer into sorted arrays
             n_entities: int,
             K: int,
             lambda_reg: float,
             device: torch.device,
             entity_batch: int = 0) -> torch.Tensor:
    """
    Update X (n_entities × K) via entity-batched ALS solve, given Y (n_features × K).

    COO data lives on CPU. Each entity-batch slice is transferred to GPU, accumulated
    into A/b, solved, then freed — GPU peak usage = P + Q + A_batch + b_batch.
    For K=64, entity_batch=50000: peak extra = 50K×64×64×4 ≈ 800 MB.

    entity_batch=0 → auto: fit within 2 GB on top of factors.
    """
    # Auto-size: keep A_batch ≤ 2 GB
    if entity_batch == 0:
        entity_batch = max(1024, (2 * (1 << 30)) // (K * K * 4))
        entity_batch = min(entity_batch, n_entities)

    # Keep outer-product yr chunk ≤ 256 MB
    CHUNK = max(1024, (256 * 1024 * 1024) // (K * 4))

    X = torch.zeros(n_entities, K, device=device, dtype=torch.float32)
    idx_diag = torch.arange(K, device=device)

    for eb in range(0, n_entities, entity_batch):
        eb_end = min(eb + entity_batch, n_entities)
        s_nnz  = int(offsets[eb])
        e_nnz  = int(offsets[eb_end])
        n_b    = eb_end - eb

        # Transfer only this batch's ratings to GPU (re-zero entity ids within batch)
        u_b = torch.from_numpy((entities_np[s_nnz:e_nnz].astype(np.int64) - eb)).to(device)
        i_b = torch.from_numpy(features_np[s_nnz:e_nnz].astype(np.int64)).to(device)
        r_b = torch.from_numpy(ratings_np[s_nnz:e_nnz]).to(device)

        A = torch.zeros(n_b, K, K, device=device, dtype=torch.float32)
        b = torch.zeros(n_b, K,    device=device, dtype=torch.float32)

        n_chunk = len(u_b)
        for s in range(0, n_chunk, CHUNK):
            e2 = min(s + CHUNK, n_chunk)
            yr = Y[i_b[s:e2]]
            A.index_add_(0, u_b[s:e2], yr.unsqueeze(2) * yr.unsqueeze(1))
            b.index_add_(0, u_b[s:e2], yr * r_b[s:e2].unsqueeze(1))

        A[:, idx_diag, idx_diag] += lambda_reg
        X[eb:eb_end] = torch.linalg.solve(A, b)
        del A, b, u_b, i_b, r_b

    return X


# ─── Main training loop ─────────────────────────────────────────────────────

def run(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    K, lam = args.K, args.lam

    print(f"=== PyTorch GPU ALS Baseline ===")
    print(f"Dataset : {args.bin_path}")
    print(f"Device  : {device} | K={K} | λ={lam} | max_iters={args.iters} | tol={args.tol}")

    # ── Load data ──────────────────────────────────────────────────────────
    t0 = time.time()
    d = load_split(args.bin_path)
    train_u, train_i, train_r = d["train"]
    test_u,  test_i,  test_r  = d["test"]
    num_users, num_items = d["num_users"], d["num_items"]
    # Sort on CPU and build CSR offsets — training COO NEVER goes to GPU as a whole.
    # GPU peak = P + Q + one entity-batch A + b + small yr chunk.
    so_u = np.argsort(train_u, kind="stable")
    u_sorted   = train_u[so_u]           # entity ids sorted by user
    i_us       = train_i[so_u]
    r_us       = train_r[so_u]
    user_off   = np.searchsorted(u_sorted, np.arange(num_users + 1)).astype(np.int64)
    user_off[num_users] = len(u_sorted)

    so_i = np.argsort(train_i, kind="stable")
    i_sorted   = train_i[so_i]           # entity ids sorted by item
    u_is       = train_u[so_i]
    r_is       = train_r[so_i]
    item_off   = np.searchsorted(i_sorted, np.arange(num_items + 1)).astype(np.int64)
    item_off[num_items] = len(i_sorted)

    print(f"Loaded  : {num_users} users, {num_items} items | "
          f"train={d['nnz_train']}, test={d['nnz_test']} ({time.time()-t0:.1f}s load+sort)\n")

    # ── Initialise factors (same seed as CUDA code) ───────────────────────
    rng = np.random.default_rng(42)
    P = torch.from_numpy(
        (0.1 + rng.integers(0, 100, (num_users, K)).astype(np.float32) / 100.0)
    ).to(device)
    Q = torch.from_numpy(
        (0.1 + rng.integers(0, 100, (num_items, K)).astype(np.float32) / 100.0)
    ).to(device)

    # ── Training ──────────────────────────────────────────────────────────
    prev_train_rmse = 1e9
    t_start = time.time()

    for it in range(1, args.iters + 1):
        # User update — entities=users, features=items; COO stays on CPU
        P = als_step(Q, u_sorted, i_us, r_us, user_off, num_users, K, lam, device)
        # Item update — swap roles: entities=items, features=users
        Q = als_step(P, i_sorted, u_is, r_is, item_off, num_items, K, lam, device)

        if it % 5 == 0:
            train_rmse = compute_rmse(P, Q, train_u, train_i, train_r)
            test_rmse  = compute_rmse(P, Q, test_u,  test_i,  test_r)
            wall = time.time() - t_start
            print(f"Iter {it:3d} | Train RMSE: {train_rmse:.6f} | "
                  f"Test RMSE: {test_rmse:.6f} | wall={wall:.1f}s")

            delta = prev_train_rmse - train_rmse
            if delta < args.tol and it >= 10:
                print(f"Converged at iteration {it} "
                      f"(train delta={delta:.6f} < tol={args.tol})")
                break
            prev_train_rmse = train_rmse

    total_wall = time.time() - t_start
    final_train = compute_rmse(P, Q, train_u, train_i, train_r)
    final_test  = compute_rmse(P, Q, test_u,  test_i,  test_r)

    print(f"\n=== PyTorch ALS Final ===")
    print(f"Train RMSE : {final_train:.6f}")
    print(f"Test  RMSE : {final_test:.6f}")
    print(f"Wall time  : {total_wall:.3f}s")
    print(f"Params: K={K}, lambda={lam}, device={device}")
    return final_train, final_test, total_wall


# ─── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="GPU explicit ALS baseline for APR-BALS comparison"
    )
    p.add_argument("bin_path", help="Path to preprocessed .bin file")
    p.add_argument("--K",      type=int,   default=16,    help="Latent factors (default 16)")
    p.add_argument("--lam",    type=float, default=0.1,   help="L2 regularisation λ (default 0.1)")
    p.add_argument("--iters",  type=int,   default=50,    help="Max iterations (default 50)")
    p.add_argument("--tol",    type=float, default=0.001, help="Train-RMSE convergence tol (default 0.001)")
    p.add_argument("--device", type=str,   default="cuda",help="torch device (default 'cuda')")
    run(p.parse_args())
