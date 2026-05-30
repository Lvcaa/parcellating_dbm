#!/usr/bin/env python3
"""
Benchmark a dense-matrix MiniBatch K-Means alternative for Phase 1.

This is a computational feasibility test for clustering connectivity profiles,
not a graph community-detection method. The existing filename intentionally
uses the spelling ``feasability``.

Usage:
    python scripts/benchmarks/phase1_feasability_mini_k_means.py [options]

Parameters:
    --nodes INT              Parcels (default: 36000).
    --voxels INT             Accepted for CLI compatibility; unused.
    --r-threshold FLOAT      Accepted for CLI compatibility; unused.
    --leiden-density FLOAT   Direct dense-matrix density (default: 0.10).
    --seed INT               Random seed (default: 42).
    --n-clusters INT         Fixed K-Means cluster count; inferred when omitted.
    --save-report            Write a timestamped text report.

Examples:
    python scripts/benchmarks/phase1_feasability_mini_k_means.py --nodes 1000 --leiden-density 0.10
    python scripts/benchmarks/phase1_feasability_mini_k_means.py --nodes 36000 --leiden-density 0.30 --n-clusters 100 --save-report
"""

import argparse
import gc
import math
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) > 2 else Path.cwd()
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "benchmarks"

THRESHOLD_CLUSTER_S = 5 * 60
THRESHOLD_RAM_GB = 64.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1 feasibility: full correlation matrix + dense adjacency + MiniBatch K-Means.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--nodes",
        type=int,
        default=36_000,
        help="Number of parcels (default: 36000, Section 2.2 target).",
    )
    p.add_argument(
        "--voxels",
        type=int,
        default=27,
        help="Kept for CLI compatibility with phase1_feasibility.py; unused in this worst-case benchmark.",
    )
    p.add_argument(
        "--r-threshold",
        type=float,
        default=0.5,
        help="Kept for CLI compatibility with phase1_feasibility.py; unused in this worst-case benchmark.",
    )
    p.add_argument(
        "--leiden-density",
        type=float,
        default=0.10,
        help="Dense-matrix edge density for the worst-case benchmark. "
             "Name kept for CLI compatibility with the existing benchmark (default: 0.10).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    p.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Fixed number of MiniBatch K-Means clusters. "
             "If omitted, use round(sqrt(nodes)) capped to [2, 512].",
    )
    p.add_argument(
        "--save-report",
        action="store_true",
        help="Write plain-text report to outputs/benchmarks/.",
    )
    return p.parse_args()


def rss_gb():
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024**2 if kb < 10**10 else kb / 1024**3


def total_ram_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024**2
    except Exception:
        pass
    return None


def fmt_int(v):
    return f"{v:,}"


def fmt_s(v):
    return f"{v / 60:.1f} min" if v >= 60 else f"{v:.3f} s"


def fmt_gb(v):
    return f"{v:.2f} GB"


def check(ok):
    return "PASS" if ok else "FAIL"


def resolve_outputs_dir() -> Path:
    """
    Pick a writable output directory. This matters inside Apptainer/Singularity
    where /app may be read-only after the image is built.
    """
    candidates = []
    env_dir = os.getenv("PARCELLATING_DBM_OUTPUTS_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            DEFAULT_OUTPUTS_DIR,
            Path.cwd() / "outputs" / "benchmarks",
            Path("/tmp") / "parcellating_dbm" / "benchmarks",
        ]
    )

    seen = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("")
            probe.unlink()
            return path
        except OSError:
            continue

    raise OSError(
        "No writable output directory found. "
        "Set PARCELLATING_DBM_OUTPUTS_DIR to a writable path."
    )


def choose_cluster_count(n: int, requested_clusters: int | None) -> tuple[int, str]:
    if requested_clusters is not None:
        return requested_clusters, "user"
    return max(2, min(512, int(round(math.sqrt(n))))), "auto"


def choose_batch_size(n: int, n_clusters: int) -> int:
    return min(n, max(4096, 16 * n_clusters))


def stage_a_correlation(n: int, voxels: int, threshold: float, seed: int) -> dict:
    """
    Simulate the per-subject correlation matrix computation from the current
    Phase 1 script so we can compare Stage B alternatives on the same setup.
    """
    rng = np.random.default_rng(seed)
    max_pairs = n * (n - 1) // 2
    corr_matrix_gb = n * n * 4 / 1024**3

    print(f"\n{'-' * 62}")
    print(f"  Stage A - Correlation matrix  ({fmt_int(n)} parcels x {voxels} voxels)")
    print(f"  Pearson r threshold : {threshold}")
    print(f"  C matrix size       : {fmt_gb(corr_matrix_gb)}  (float32, n x n, dense)")
    print(f"{'-' * 62}")

    res = {"n": n, "voxels": voxels, "threshold": threshold}

    print("  [A1] Generating X (n x voxels, float32) ...")
    t0 = time.perf_counter()
    X = rng.standard_normal((n, voxels)).astype(np.float32)
    t1 = time.perf_counter()
    print(f"       {fmt_s(t1 - t0)}  |  RSS={fmt_gb(rss_gb())}")

    print("  [A2] Row-normalising X (mean-centre + L2-norm) ...")
    t2 = time.perf_counter()
    X -= X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X /= np.where(norms > 0, norms, 1.0)
    t3 = time.perf_counter()
    print(f"       {fmt_s(t3 - t2)}  |  RSS={fmt_gb(rss_gb())}")

    print(f"  [A3] Computing C = X @ X.T  ({fmt_gb(corr_matrix_gb)} float32 matrix) ...")
    t4 = time.perf_counter()
    C = X @ X.T
    del X
    gc.collect()
    t5 = time.perf_counter()
    res["corr_time_s"] = t5 - t4
    res["rss_after_corr_gb"] = rss_gb()
    print(f"       {fmt_s(t5 - t4)}  |  RSS={fmt_gb(rss_gb())}")

    print(f"  [A4] Thresholding (r > {threshold}, upper triangle, row-by-row) ...")
    t6 = time.perf_counter()

    rows_list, cols_list, weights_list = [], [], []

    for i in range(n):
        row = C[i, i + 1:]
        above = np.where(row > threshold)[0]
        if above.size:
            rows_list.append(np.full(above.size, i, dtype=np.int32))
            cols_list.append((above + i + 1).astype(np.int32))
            weights_list.append(row[above].astype(np.float32, copy=False))

    del C
    gc.collect()

    if rows_list:
        rows_arr = np.concatenate(rows_list).astype(np.int32, copy=False)
        cols_arr = np.concatenate(cols_list).astype(np.int32, copy=False)
        weights_arr = np.concatenate(weights_list).astype(np.float32, copy=False)
    else:
        rows_arr = np.array([], dtype=np.int32)
        cols_arr = np.array([], dtype=np.int32)
        weights_arr = np.array([], dtype=np.float32)

    t7 = time.perf_counter()
    edge_count = len(rows_arr)
    observed_density = edge_count / max_pairs if max_pairs else 0.0

    res["threshold_time_s"] = t7 - t6
    res["stage_a_time_s"] = t7 - t0
    res["edge_count"] = edge_count
    res["observed_density"] = observed_density
    res["peak_rss_gb"] = rss_gb()
    res["rows"] = rows_arr
    res["cols"] = cols_arr
    res["weights"] = weights_arr

    print(f"       {fmt_s(t7 - t6)}  |  RSS={fmt_gb(rss_gb())}")
    print(f"  Edges surviving threshold : {fmt_int(edge_count)}")
    print(f"  Observed density          : {observed_density:.4%}")
    if edge_count == 0:
        print("  (Expected ~0 on random data; real Jacobian data should yield structure.)")

    return res


def build_dense_adjacency_from_edges(n: int, rows_arr: np.ndarray, cols_arr: np.ndarray, weights_arr: np.ndarray) -> np.ndarray:
    A = np.zeros((n, n), dtype=np.float32)
    if rows_arr.size:
        A[rows_arr, cols_arr] = weights_arr
        A[cols_arr, rows_arr] = weights_arr
    np.fill_diagonal(A, 0.0)
    return A


def build_synthetic_dense_adjacency(n: int, density: float, seed: int) -> tuple[np.ndarray, int]:
    """
    Build a dense symmetric weighted adjacency matrix row-by-row so we avoid
    materialising huge upper-triangle index lists for the 30% worst-case test.
    """
    rng = np.random.default_rng(seed)
    A = np.zeros((n, n), dtype=np.float32)
    edge_count = 0

    for i in range(n - 1):
        row_len = n - i - 1
        keep = rng.random(row_len, dtype=np.float32) < density
        if not np.any(keep):
            continue

        cols = np.flatnonzero(keep) + i + 1
        vals = rng.random(cols.size, dtype=np.float32)
        A[i, cols] = vals
        A[cols, i] = vals
        edge_count += cols.size

    return A, edge_count


def stage_b_minibatch_kmeans(n: int, leiden_density: float, seed: int, requested_clusters: int | None) -> dict:
    """Cluster a synthetic dense weighted adjacency matrix with MiniBatch K-Means."""
    max_pairs = n * (n - 1) // 2

    print(f"\n{'-' * 62}")
    print("  Stage B - Worst-case dense adjacency + MiniBatch K-Means")
    print(f"{'-' * 62}")

    res = {"using_fallback": False}

    print("  [B1] Building dense weighted adjacency matrix A ...")
    t0 = time.perf_counter()

    print(f"  Building synthetic dense matrix at ~{leiden_density:.0%} density.")
    A, actual_edges = build_synthetic_dense_adjacency(n, leiden_density, seed)
    matrix_source = f"synthetic dense random matrix  (density~{leiden_density:.0%})"

    t1 = time.perf_counter()
    observed_density = actual_edges / max_pairs if max_pairs else 0.0

    res["matrix_source"] = matrix_source
    res["actual_edges"] = actual_edges
    res["observed_density"] = observed_density
    res["matrix_build_time_s"] = t1 - t0

    print(f"       {fmt_s(t1 - t0)}  |  edges={fmt_int(actual_edges)}  |  RSS={fmt_gb(rss_gb())}")
    print(f"  Dense matrix density      : {observed_density:.4%}")

    n_clusters, cluster_mode = choose_cluster_count(n, requested_clusters)
    batch_size = choose_batch_size(n, n_clusters)

    print("  [B2] Running MiniBatch K-Means on node connectivity profiles ...")
    print(f"       requested clusters={fmt_int(n_clusters)}  |  batch_size={fmt_int(batch_size)}")
    t2 = time.perf_counter()

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        random_state=seed,
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    model.fit(A)

    labels = model.labels_
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    nonempty = cluster_sizes[cluster_sizes > 0]

    t3 = time.perf_counter()
    cluster_time = t3 - t2

    res["kmeans_time_s"] = cluster_time
    res["stage_b_time_s"] = t3 - t0
    res["n_clusters_requested"] = n_clusters
    res["cluster_selection"] = cluster_mode
    res["clusters_found"] = int(nonempty.size)
    res["inertia"] = float(model.inertia_)
    res["largest_cluster"] = int(nonempty.max()) if nonempty.size else 0
    res["smallest_nonempty_cluster"] = int(nonempty.min()) if nonempty.size else 0

    del A
    del labels
    del cluster_sizes
    gc.collect()

    res["peak_rss_gb"] = rss_gb()
    res["pass_time"] = res["kmeans_time_s"] < THRESHOLD_CLUSTER_S
    res["pass_ram"] = res["peak_rss_gb"] < THRESHOLD_RAM_GB
    res["pass"] = res["pass_time"] and res["pass_ram"]

    print(f"       {fmt_s(cluster_time)}  |  clusters_found={fmt_int(res['clusters_found'])}  "
          f"|  inertia={res['inertia']:.4e}")
    print(f"       Peak RSS: {fmt_gb(rss_gb())}")
    print()
    print(f"  [{check(res['pass_time'])}] MiniBatch K-Means < 5 min  -> {fmt_s(res['kmeans_time_s'])}")
    print(f"  [{check(res['pass_ram'])}] RAM < 64 GB                -> {fmt_gb(res['peak_rss_gb'])}")
    print(f"  {'GO' if res['pass'] else 'NO-GO'}  -  Proceed to Phase 2: {'YES' if res['pass'] else 'NO'}")

    return res


def build_report(sys_info: dict, args: argparse.Namespace, sb: dict) -> str:
    width = 68
    lines = []

    def hr():
        lines.append("=" * width)

    def subhr():
        lines.append("-" * width)

    def row(k, v):
        lines.append(f"  {k:<34}{v}")

    hr()
    lines.append("  PHASE 1 FEASIBILITY REPORT - MINIBATCH K-MEANS")
    lines.append(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    hr()
    lines.append("")

    lines.append("  SYSTEM")
    subhr()
    row("Platform:", sys_info["platform"])
    row("CPU cores:", str(sys_info["cpu_cores"]))
    ram = sys_info.get("total_ram_gb")
    row("Total RAM:", fmt_gb(ram) if ram else "unknown")
    lines.append("")

    lines.append("  PARAMETERS")
    subhr()
    row("Parcels (nodes):", fmt_int(args.nodes))
    row("Dense matrix target size:", f"{fmt_int(args.nodes)} x {fmt_int(args.nodes)}")
    row("Dense float32 matrix size:", fmt_gb(args.nodes ** 2 * 4 / 1024**3))
    row("Requested density:", f"{args.leiden_density:.0%}")
    row("Seed:", str(args.seed))
    if args.n_clusters is None:
        row("n-clusters arg:", "auto  (round(sqrt(nodes)), capped to [2, 512])")
    else:
        row("n-clusters arg:", str(args.n_clusters))
    row("voxels arg:", f"{args.voxels}  (accepted, unused)")
    row("r-threshold arg:", f"{args.r_threshold}  (accepted, unused)")
    lines.append("")

    lines.append("  STAGE B - Worst-case dense adjacency + MiniBatch K-Means")
    subhr()
    row("Matrix source:", sb["matrix_source"])
    row("Actual edges:", fmt_int(sb["actual_edges"]))
    row("Observed density:", f"{sb['observed_density']:.4%}")
    row("Dense matrix build time:", fmt_s(sb["matrix_build_time_s"]))
    row("Requested clusters:", fmt_int(sb["n_clusters_requested"]))
    row("Cluster selection:", sb["cluster_selection"])
    row("Clusters found:", fmt_int(sb["clusters_found"]))
    row("Largest cluster:", fmt_int(sb["largest_cluster"]))
    row("Smallest non-empty cluster:", fmt_int(sb["smallest_nonempty_cluster"]))
    row("MiniBatch K-Means time:", fmt_s(sb["kmeans_time_s"]))
    row("Inertia:", f"{sb['inertia']:.4e}")
    row("Peak RSS:", fmt_gb(sb["peak_rss_gb"]))
    row("MiniBatch K-Means < 5 min:", check(sb["pass_time"]))
    row("RAM < 64 GB:", check(sb["pass_ram"]))
    row("Overall:", "GO - proceed to Phase 2" if sb["pass"] else "NO-GO")
    lines.append("")

    lines.append("  NOTES")
    subhr()
    lines.append("  This script now skips the Pearson-correlation Stage A entirely and")
    lines.append("  builds the dense weighted matrix directly, so --leiden-density 0.30")
    lines.append("  is a true 30% worst-case benchmark.")
    lines.append("")
    lines.append("  MiniBatch K-Means clusters nodes by Euclidean similarity of their")
    lines.append("  dense connectivity profiles. This is a scalability test, not a direct")
    lines.append("  modularity-based community detection result like Leiden.")
    lines.append("")
    lines.append("  This benchmark answers a practical question: can we replace graph")
    lines.append("  community detection with a cheaper profile-clustering step when the")
    lines.append("  dense matrix becomes the dominant computational bottleneck?")

    hr()
    return "\n".join(lines)


def main():
    args = parse_args()

    if args.nodes < 2:
        sys.exit("--nodes must be >= 2")
    if args.n_clusters is not None and args.n_clusters < 2:
        sys.exit("--n-clusters must be >= 2")
    if args.voxels < 2:
        sys.exit("--voxels must be >= 2")
    if not (0.0 < args.r_threshold < 1.0):
        sys.exit("--r-threshold must be in (0, 1)")
    if not (0.0 < args.leiden_density <= 1.0):
        sys.exit("--leiden-density must be in (0, 1]")

    wall_start = time.perf_counter()

    print("=" * 62)
    print("  PHASE 1 FEASIBILITY - Dense Matrix + MiniBatch K-Means")
    print("=" * 62)

    sys_info = {
        "platform": platform.platform(),
        "cpu_cores": os.cpu_count() or "unknown",
        "total_ram_gb": total_ram_gb(),
    }
    print(f"  Platform  : {sys_info['platform']}")
    print(f"  CPU cores : {sys_info['cpu_cores']}")
    ram = sys_info.get("total_ram_gb")
    if ram:
        print(f"  Total RAM : {fmt_gb(ram)}")
    print(f"  Init RSS  : {fmt_gb(rss_gb())}")

    print("\n  Note: this worst-case benchmark skips Stage A entirely.")
    print("        --voxels and --r-threshold are accepted for CLI compatibility only.")

    dense_matrix_gb = args.nodes ** 2 * 4 / 1024**3
    if dense_matrix_gb > 16:
        print(f"\n  [!] Each dense float32 n x n matrix will occupy ~{fmt_gb(dense_matrix_gb)}.")
        print("      This run allocates that dense matrix directly for the K-Means benchmark.")

    sb = stage_b_minibatch_kmeans(
        n=args.nodes,
        leiden_density=args.leiden_density,
        seed=args.seed,
        requested_clusters=args.n_clusters,
    )

    total_s = time.perf_counter() - wall_start
    print(f"\nTotal wall time: {fmt_s(total_s)}")

    if args.save_report:
        try:
            output_dir = resolve_outputs_dir()
            ts = time.strftime("%Y%m%d_%H%M%S")
            report_path = output_dir / f"phase1_feasability_mini_k_means_{ts}.txt"
            report_text = build_report(sys_info, args, sb)
            report_path.write_text(report_text)
            print(f"Report written to: {report_path}")
        except OSError as exc:
            print(f"Warning: could not write report automatically: {exc}")
            print("Set PARCELLATING_DBM_OUTPUTS_DIR to a writable path and rerun with --save-report.")


if __name__ == "__main__":
    main()
