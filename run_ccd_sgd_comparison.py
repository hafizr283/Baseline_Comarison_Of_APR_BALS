#!/usr/bin/env python3
"""
run_ccd_sgd_comparison.py — justapr (APR-BALS) vs cuMF_CCD and cuMF_SGD,
BALS-paper style: every method trains and is scored on the SAME frozen
train/test split (converted from each justapr .bin), and we report
wall-clock-to-converged-RMSE as the headline plus throughput as support.

Protocol
  justapr : ship recipe per K in {16,32,48,64,96}, lambda 0.048, RMSE%5 tol1e-3.
            (K=96 adds -DCHOL_MP=2.)  metric: "Training Complete in" wall + compute.
  cuMF_CCD: same K sweep, l per dataset, t=15 iters, tiles a=b=100000.
            metric: final per-iter cumulative compute time + "Total seconds" wall.
  cuMF_SGD: k=128 (its only supported rank), iter sweep, tuned lr; test RMSE via
            libmf mf-predict on the frozen test.bin.  Compared to justapr K=96.

Outputs a per-dataset table + grand summary + a "vs BALS paper" relative-speedup
section (BALS's own reported speedups over the same baselines, hardcoded from
Chen et al. TPDS'21 Table 2 / Sec 6.2), all written under results/.
"""
import os, re, sys, subprocess, time, shutil, tempfile
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
CONV_SCRIPT = os.path.join(HERE, "convert_bin_for_ccd_sgd.py")
CCD_BIN = os.path.join(HERE, "cumf_ccd-master", "ccdp_gpu")
SGD_BIN = os.path.join(HERE, "cumf_sgd-master", "singleGPU", "cumf_sgd")
SGD_PRED = os.path.join(HERE, "cumf_sgd-master", "test", "mf-predict")
JUSTAPR = os.path.join(HERE, "justapr.cu")

GPU = os.environ.get("GPU", "1")
ARCH = os.environ.get("ARCH", "sm_86")


def _detect_gpu_name():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", GPU],
            capture_output=True, text=True, timeout=10)
        name = out.stdout.strip().splitlines()[0].strip()
        return name if name else "unknown GPU"
    except Exception:
        return "unknown GPU"


GPU_NAME = os.environ.get("GPU_NAME") or _detect_gpu_name()
LAMBDA_APR = os.environ.get("LAMBDA", "0.048")
K_VALUES = [int(x) for x in os.environ.get("K_VALUES", "16 32 48 64 96").split()]
CCD_ITERS = int(os.environ.get("CCD_ITERS", "15"))
SGD_ITER_SWEEP = [int(x) for x in os.environ.get("SGD_ITERS", "10 20 30 40").split()]
SGD_K = 128
WORKDIR = os.environ.get("WORKDIR",
    "/tmp/claude-1000/-home-pc-Desktop-2007080-latent-factorr-experimen/"
    "21b4a089-b675-4fc0-96a8-1a179a2a72ae/scratchpad/conv")

# name -> (.bin path, ccd_lambda, sgd_workers, sgd_lambda, sgd_alpha, sgd_beta)
REGISTRY = {
    "ml10m":   ("/home/pc/Desktop/2007080/ratings10.bin",     0.05,  256, 0.05, 0.08, 0.3),
    "ml20m":   ("/home/pc/Desktop/2007080/ratings.bin",       0.05,  384, 0.05, 0.08, 0.3),
    "ml32m":   ("/home/pc/Desktop/2007080/ratings32.bin",     0.05,  512, 0.05, 0.08, 0.3),
    "netflix": ("/home/pc/Desktop/2007080/netflix_ratings.bin",0.058, 512, 0.05, 0.08, 0.3),
}

# BALS paper (Chen et al., TPDS 2021) reported speedups over the SAME baselines.
# Table 2 = BALS-over-cuMF_CCD, averaged over K, per GPU. Sec 6.2 = SGD (GFlops).
BALS_VS_CCD = {  # TITAN RTX row (closest consumer GPU to our RTX 3060)
    "ml10m": 2.09, "ml20m": 3.86, "netflix": 3.22,
}
BALS_VS_SGD_GFLOPS = 5.3   # avg over six datasets, TITAN RTX, f=128 (throughput)

env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU)


def sh(cmd, log=None, timeout=3600):
    r = subprocess.run(cmd, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, timeout=timeout)
    out = r.stdout.decode("utf-8", "replace")
    if log:
        with open(log, "w") as f:
            f.write(out)
    return r.returncode, out


def find(pat, text, cast=float, last=True):
    m = re.findall(pat, text)
    if not m:
        return None
    try:
        return cast(m[-1] if last else m[0])
    except Exception:
        return m[-1] if last else m[0]


def ensure_converted(name, binpath):
    out = os.path.join(WORKDIR, name)
    meta = os.path.join(out, "ccd", "meta_modified_all")
    test = os.path.join(out, "sgd", "test.bin")
    if os.path.exists(meta) and os.path.exists(test):
        return out
    print(f"  [convert] {name} ...", flush=True)
    rc, o = sh(["python3", CONV_SCRIPT, binpath, out], timeout=1800)
    if rc != 0:
        print(o)
        raise RuntimeError(f"convert failed for {name}")
    return out


def build_justapr(bindir):
    ok = {}
    for K in K_VALUES:
        flags = ["-DK_DIM=%d" % K, "-DWEIGHTED_LAMBDA=1", "-DCUMF_INIT=1", "-DMOMENTUM=1"]
        if K == 96:
            flags.append("-DCHOL_MP=2")
        outb = os.path.join(bindir, "apr_k%d" % K)
        print(f"  [build justapr K={K}] ...", flush=True)
        rc, o = sh(["nvcc", "-O3", "-arch=" + ARCH, "-std=c++14", *flags,
                    JUSTAPR, "-o", outb], timeout=600)
        ok[K] = (rc == 0)
        if rc != 0:
            print(o[-1500:])
    return ok


def run_justapr(bindir, K, binpath, log):
    rc, o = sh([os.path.join(bindir, "apr_k%d" % K), binpath, LAMBDA_APR], log=log)
    it = find(r'Converged at iteration (\d+)', o, int)
    if it is None:
        it = find(r'=== Profiling: .*\((\d+) iters\)', o, int)
    wall = find(r'Training Complete in ([\d.]+) seconds', o)
    comp = find(r'compute=([\d.]+) ms', o)
    tr = find(r'Train RMSE: ([\d.]+) \| Test', o)
    te = find(r'Test RMSE: ([\d.]+)', o)
    thr = find(r'Throughput: [\d.]+ GFlops/s \(compute: ([\d.]+)\)', o)
    gflops = find(r'Total FLOPs: ([\d.]+) GFLOPs', o)
    return dict(iters=it, wall=wall, compute_ms=comp, train=tr, test=te,
                thr_gflops=thr, total_gflops=gflops)


def run_ccd(ccd_dir, K, lam, log):
    rc, o = sh([CCD_BIN, "-T", "1", "-a", "100000", "-b", "100000",
                "-l", str(lam), "-k", str(K), "-t", str(CCD_ITERS), ccd_dir], log=log)
    # last "iter N time X RMSE Y" -> cumulative compute time X, test rmse Y
    rows = re.findall(r'iter\s+(\d+)\s+time\s+([\d.]+)\s+RMSE\s+([\d.eE+-]+)', o)
    comp = float(rows[-1][1]) if rows else None
    te = float(rows[-1][2]) if rows else None
    wall = find(r'Total seconds: ([\d.]+)', o)
    return dict(iters=CCD_ITERS, compute_s=comp, wall=wall, test=te)


def run_sgd(sgd_dir, work_model, name, lam, workers, alpha, beta, log_prefix):
    results = []
    for t in SGD_ITER_SWEEP:
        model = work_model + f".t{t}"
        log = log_prefix + f"_t{t}.txt"
        rc, o = sh([SGD_BIN, "-g", "0", "-l", str(lam), "-a", str(alpha),
                    "-b", str(beta), "-s", str(workers), "-k", str(SGD_K),
                    "-t", str(t), os.path.join(sgd_dir, "train.bin"), model], log=log)
        train_s = find(r'SGD_TRAIN_SECONDS: ([\d.]+)', o)
        rc2, o2 = sh([SGD_PRED, "-e", "0", os.path.join(sgd_dir, "test.bin"), model],
                     log=log + ".pred")
        te = find(r'RMSE = ([\d.]+)', o2)
        try:
            os.remove(model)
        except OSError:
            pass
        results.append(dict(t=t, train=train_s, test=te))
        print(f"    SGD k=128 t={t}: train={train_s}s test={te}", flush=True)
    return results


def fmt(x, nd=3):
    return "?" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def main():
    names = sys.argv[1:] or ["ml10m", "ml20m", "ml32m", "netflix"]
    for n in names:
        if n not in REGISTRY:
            print(f"unknown dataset {n}; known: {list(REGISTRY)}"); sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(HERE, "results")
    logdir = os.path.join(outdir, f"logs_ccd_sgd_{ts}")
    os.makedirs(logdir, exist_ok=True)
    summ_path = os.path.join(outdir, f"JUSTAPR_VS_CCD_SGD_{ts}.txt")
    bindir = tempfile.mkdtemp(prefix="aprccdsgd_")

    lines = []
    def show(s=""):
        print(s, flush=True); lines.append(s)

    show("=" * 78)
    show(" justapr (APR-BALS) vs cuMF_CCD vs cuMF_SGD — frozen-split, BALS protocol")
    show(f" GPU={GPU} arch={ARCH}  datasets: {names}")
    show(f" justapr K={K_VALUES} lambda={LAMBDA_APR} | CCD t={CCD_ITERS} | SGD k=128 t-sweep={SGD_ITER_SWEEP}")
    show(f" Started: {datetime.now()}")
    show("=" * 78)

    japr_ok = build_justapr(bindir)

    per_ds = {}
    for name in names:
        binpath, ccd_lam, workers, sgd_lam, sgd_a, sgd_b = REGISTRY[name]
        if not os.path.exists(binpath):
            show(f"\n-- SKIP {name}: missing {binpath}"); continue
        show(f"\n{'─'*70}\n── dataset: {name}  ({binpath}) ──")
        conv = ensure_converted(name, binpath)
        ccd_dir = os.path.join(conv, "ccd")
        sgd_dir = os.path.join(conv, "sgd")

        apr = {}
        for K in K_VALUES:
            if not japr_ok.get(K):
                continue
            r = run_justapr(bindir, K, binpath, os.path.join(logdir, f"{name}_apr_k{K}.txt"))
            apr[K] = r
            show(f"  justapr K={K:<3} test {fmt(r['test'],6)}  "
                 f"wall {fmt(r['wall'])}s  compute {fmt((r['compute_ms'] or 0)/1000.0)}s  ({fmt(r['iters'],0)} it)")

        ccd = {}
        for K in K_VALUES:
            r = run_ccd(ccd_dir, K, ccd_lam, os.path.join(logdir, f"{name}_ccd_k{K}.txt"))
            ccd[K] = r
            show(f"  cuMF_CCD K={K:<3} test {fmt(r['test'],6)}  "
                 f"compute {fmt(r['compute_s'])}s  wall {fmt(r['wall'])}s  ({CCD_ITERS} it, l={ccd_lam})")

        show(f"  cuMF_SGD (k=128, workers={workers}, l={sgd_lam}, a={sgd_a}, b={sgd_b}):")
        sgd = run_sgd(sgd_dir, os.path.join(bindir, f"{name}.model"), name,
                      sgd_lam, workers, sgd_a, sgd_b, os.path.join(logdir, f"{name}_sgd"))

        per_ds[name] = dict(apr=apr, ccd=ccd, sgd=sgd, ccd_lam=ccd_lam)

    # ---- per-dataset tables ----
    for name in names:
        if name not in per_ds:
            continue
        d = per_ds[name]
        show(f"\n{'='*70}\n TABLE — {name}\n{'='*70}")
        show(f" {'method':<24} {'rank':<6} {'iters':<6} {'train s':<9} {'test RMSE':<10}")
        show(" " + "-" * 62)
        for K in K_VALUES:
            r = d["apr"].get(K)
            if r:
                show(f" {'justapr APR-BALS':<24} {'K=%d'%K:<6} {fmt(r['iters'],0):<6} "
                     f"{fmt(r['wall']):<9} {fmt(r['test'],6):<10}")
        for K in K_VALUES:
            r = d["ccd"].get(K)
            if r:
                show(f" {'cuMF_CCD':<24} {'K=%d'%K:<6} {fmt(r['iters'],0):<6} "
                     f"{fmt(r['compute_s']):<9} {fmt(r['test'],6):<10}")
        for r in d["sgd"]:
            show(f" {'cuMF_SGD':<24} {'k=128':<6} {fmt(r['t'],0):<6} "
                 f"{fmt(r['train']):<9} {fmt(r['test'],6):<10}")

    # ---- speedups + vs-BALS relative section ----
    show(f"\n{'='*70}\n SPEEDUPS — justapr over baselines (this machine, {GPU_NAME})\n{'='*70}")
    show(" cuMF_CCD: wall(CCD compute) / wall(justapr) at matched K, avg over K")
    for name in names:
        if name not in per_ds:
            continue
        d = per_ds[name]
        ratios = []
        for K in K_VALUES:
            a, c = d["apr"].get(K), d["ccd"].get(K)
            if a and c and a["wall"] and c["compute_s"] and a["wall"] > 0:
                ratios.append(c["compute_s"] / a["wall"])
        avg = sum(ratios) / len(ratios) if ratios else None
        bals = BALS_VS_CCD.get(name)
        extra = f"   (BALS paper vs cuMF_CCD: {bals:.2f}x)" if bals else "   (not in BALS paper)"
        show(f"  {name:<9} justapr {fmt(avg,2)}x faster than cuMF_CCD{extra}")

    show("\n cuMF_SGD (k=128) vs justapr (K=96): wall-to-reach-justapr-RMSE + throughput")
    for name in names:
        if name not in per_ds:
            continue
        d = per_ds[name]
        a96 = d["apr"].get(96)
        if not a96 or not a96["test"]:
            continue
        target = a96["test"]
        # smallest SGD t whose test RMSE <= justapr K=96 RMSE
        hit = next((r for r in d["sgd"] if r["test"] and r["test"] <= target), None)
        conv = d["sgd"][-1] if d["sgd"] else None
        msg = f"  {name:<9} justapr K=96: {fmt(target,4)} in {fmt(a96['wall'])}s | "
        if hit:
            spd = hit["train"] / a96["wall"] if (hit["train"] and a96["wall"]) else None
            msg += (f"SGD matches at t={hit['t']} in {fmt(hit['train'])}s "
                    f"(justapr {fmt(spd,2)}x vs SGD-to-same-RMSE)")
        elif conv:
            msg += (f"SGD best {fmt(conv['test'],4)} at t={conv['t']} in {fmt(conv['train'])}s "
                    f"(SGD never reaches justapr RMSE)")
        show(msg)
    show(f"\n  NOTE: BALS paper's headline 5.3x-over-cuMF_SGD is a GFlops (throughput) metric")
    show(f"        at f=128, not wall-to-convergence. ALS does ~f/3x more FLOPs/iter than")
    show(f"        SGD, so ALS wins big on throughput even when wall-time is competitive.")

    show(f"\nFinished: {datetime.now()}")
    show(f"Summary : {summ_path}")
    show(f"Logs    : {logdir}/")
    with open(summ_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    shutil.rmtree(bindir, ignore_errors=True)


if __name__ == "__main__":
    main()
