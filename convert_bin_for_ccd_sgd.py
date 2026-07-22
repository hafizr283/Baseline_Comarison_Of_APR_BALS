#!/usr/bin/env python3
"""
convert_bin_for_ccd_sgd.py — one frozen split, three engines.

Reads a justapr/cuMF-ALS ".bin" (the frozen train/test split used by
justapr.cu and cumf_als) and emits the inputs cuMF_CCD and cuMF_SGD need,
so all four methods (justapr, cuMF-ALS, cuMF_CCD, cuMF_SGD) train and test on
*exactly the same* rows — the fair-comparison protocol used in the BALS paper.

.bin layout (little-endian):
  int32   version, num_users, num_items, nnz_train, nnz_test
  then size-prefixed vectors, each = uint64 count + payload:
    train: users(i32) items(i32) ratings(f32)
    test : users(i32) items(i32) ratings(f32)
    csr  : user_off(i32) item_idx(i32) ratings(f32)   [ignored, rebuilt]
    csc  : item_off(i32) user_idx(i32) ratings(f32)    [ignored, rebuilt]

CCD output (dir <out>/ccd/):  meta_modified_all + 9 bins
  train_csr.rowptr/.colidx/.value, train_csc.colptr/.rowidx/.value,
  test_coo.row/.col/.data
SGD output (dir <out>/sgd/):  train.bin, test.bin   (records: i32 u, i32 v, f32 r)
"""
import sys, os, struct
import numpy as np
from scipy.sparse import coo_matrix


def read_bin(path):
    with open(path, 'rb') as f:
        ver, nu, ni, ntr, nte = struct.unpack('<5i', f.read(20))

        def rvec(dtype):
            (sz,) = struct.unpack('<Q', f.read(8))
            buf = f.read(np.dtype(dtype).itemsize * sz)
            return np.frombuffer(buf, dtype=dtype)

        tr_u = rvec('<i4'); tr_i = rvec('<i4'); tr_r = rvec('<f4')
        te_u = rvec('<i4'); te_i = rvec('<i4'); te_r = rvec('<f4')
    return dict(num_users=nu, num_items=ni, nnz_train=ntr, nnz_test=nte,
                tr_u=tr_u, tr_i=tr_i, tr_r=tr_r, te_u=te_u, te_i=te_i, te_r=te_r)


def write_ccd(d, out):
    os.makedirs(out, exist_ok=True)
    m, n = d['num_users'], d['num_items']
    # Train COO -> CSR + CSC on a fixed (m x n) shape.
    coo = coo_matrix((d['tr_r'].astype(np.float32),
                      (d['tr_u'].astype(np.int32), d['tr_i'].astype(np.int32))),
                     shape=(m, n))
    csr = coo.tocsr(); csc = coo.tocsc()
    csr.indptr.astype(np.int32).tofile(os.path.join(out, 'train_csr.rowptr.bin'))
    csr.indices.astype(np.int32).tofile(os.path.join(out, 'train_csr.colidx.bin'))
    csr.data.astype(np.float32).tofile(os.path.join(out, 'train_csr.value.bin'))
    csc.indptr.astype(np.int32).tofile(os.path.join(out, 'train_csc.colptr.bin'))
    csc.indices.astype(np.int32).tofile(os.path.join(out, 'train_csc.rowidx.bin'))
    csc.data.astype(np.float32).tofile(os.path.join(out, 'train_csc.value.bin'))
    d['te_u'].astype(np.int32).tofile(os.path.join(out, 'test_coo.row.bin'))
    d['te_i'].astype(np.int32).tofile(os.path.join(out, 'test_coo.col.bin'))
    d['te_r'].astype(np.float32).tofile(os.path.join(out, 'test_coo.data.bin'))
    with open(os.path.join(out, 'meta_modified_all'), 'w') as f:
        f.write(f"{m} {n}\n{int(csr.nnz)}\n{d['nnz_test']}\n")
        for name in ('train_csr.rowptr.bin', 'train_csr.colidx.bin', 'train_csr.value.bin',
                     'train_csc.colptr.bin', 'train_csc.rowidx.bin', 'train_csc.value.bin',
                     'test_coo.row.bin', 'test_coo.col.bin', 'test_coo.data.bin'):
            f.write(name + "\n")
    return csr.nnz


def write_sgd(d, out):
    os.makedirs(out, exist_ok=True)
    def dump(path, u, v, r):
        rec = np.empty(u.size, dtype=[('u', '<i4'), ('v', '<i4'), ('r', '<f4')])
        rec['u'] = u; rec['v'] = v; rec['r'] = r
        rec.tofile(path)
    dump(os.path.join(out, 'train.bin'), d['tr_u'], d['tr_i'], d['tr_r'])
    dump(os.path.join(out, 'test.bin'),  d['te_u'], d['te_i'], d['te_r'])


def main():
    if len(sys.argv) < 3:
        print("usage: convert_bin_for_ccd_sgd.py <input.bin> <out_dir>")
        sys.exit(1)
    src, out = sys.argv[1], sys.argv[2]
    d = read_bin(src)
    print(f"{src}: {d['num_users']} users, {d['num_items']} items, "
          f"train={d['nnz_train']}, test={d['nnz_test']}")
    ccd_nnz = write_ccd(d, os.path.join(out, 'ccd'))
    write_sgd(d, os.path.join(out, 'sgd'))
    if ccd_nnz != d['nnz_train']:
        print(f"  note: CSR nnz {ccd_nnz} != header train {d['nnz_train']} "
              f"(duplicate (u,i) pairs summed by COO->CSR)")
    print(f"  wrote CCD -> {os.path.join(out,'ccd')}  and  SGD -> {os.path.join(out,'sgd')}")


if __name__ == '__main__':
    main()
