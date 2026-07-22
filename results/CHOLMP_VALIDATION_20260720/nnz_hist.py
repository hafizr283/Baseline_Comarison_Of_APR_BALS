# Fact-check for the Woodbury proposal: per-entity train-nnz distribution from
# the frozen netflix_ratings.bin. The tiled solve costs the same K^3 per entity
# regardless of nnz, so "share of solve time on entities with nnz < K" ==
# "share of entities with nnz < K" (by count).
import struct, numpy as np, sys

path = sys.argv[1] if len(sys.argv) > 1 else "/home/pc/Desktop/2007080/netflix_ratings.bin"
f = open(path, "rb")
version, num_users, num_items, nnz_train, nnz_test = struct.unpack("<5i", f.read(20))
print(f"users={num_users} items={num_items} train={nnz_train} test={nnz_test}")

def skip_vec(elem_bytes):
    (sz,) = struct.unpack("<Q", f.read(8))
    f.seek(sz * elem_bytes, 1)

def read_vec_i32():
    (sz,) = struct.unpack("<Q", f.read(8))
    return np.fromfile(f, dtype=np.int32, count=sz)

skip_vec(4)  # raw_users
skip_vec(4)  # raw_items
skip_vec(4)  # raw_ratings
skip_vec(4)  # test_users
skip_vec(4)  # test_items
skip_vec(4)  # test_ratings
uoff = read_vec_i32()          # user_offsets (num_users+1)
skip_vec(4)  # item_indices
skip_vec(4)  # user_ratings
ioff = read_vec_i32()          # item_offsets

assert len(uoff) == num_users + 1, len(uoff)
assert len(ioff) == num_items + 1, len(ioff)
unnz = np.diff(uoff); innz = np.diff(ioff)
assert unnz.sum() == nnz_train and innz.sum() == nnz_train

for name, nnz in (("USER", unnz), ("ITEM", innz)):
    n = len(nnz)
    print(f"\n{name}: n={n} mean={nnz.mean():.1f} median={np.median(nnz):.0f} "
          f"p10={np.percentile(nnz,10):.0f} p25={np.percentile(nnz,25):.0f} "
          f"p75={np.percentile(nnz,75):.0f} p90={np.percentile(nnz,90):.0f} max={nnz.max()}")
    for K in (32, 48, 64, 77, 96):
        frac = (nnz < K).mean()
        print(f"  nnz < {K:3d}: {100*frac:5.1f}% of entities  (= share of {name}-solve time)")
    # FLOP-model upper bound if every nnz<t entity solved via Woodbury at cost
    # ratio (nnz/K)^3-ish; real bound needs the kernel bench, this is the MAX.
    for K in (96,):
        light = nnz[nnz < int(0.8 * K)]
        print(f"  nnz < 0.8*K={int(0.8*K)}: {100*len(light)/n:.1f}% of entities, "
              f"their mean nnz={light.mean() if len(light) else 0:.1f}")
