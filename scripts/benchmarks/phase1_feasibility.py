#!/usr/bin/env python3
"""
Benchmark dense parcel correlation, Pearson thresholding, and Leiden timing.

This Phase 1 feasibility script simulates per-subject graph construction. It
builds the full float32 correlation matrix so every pairwise relationship is
evaluated, then thresholds edges and runs Leiden. The full target can allocate
substantial RAM; begin with a smaller node count.

Usage:
    python scripts/benchmarks/phase1_feasibility.py [options]

Parameters:
    --nodes INT              Parcels (default: 36000).
    --voxels INT             Voxels per parcel (default: 27).
    --r-threshold FLOAT      Pearson edge threshold in (0, 1) (default: 0.5).
    --leiden-density FLOAT   Fallback synthetic graph density (default: 0.10).
    --seed INT               Random seed (default: 42).
    --save-report            Write a timestamped report to outputs/benchmarks.

Examples:
    python scripts/benchmarks/phase1_feasibility.py --nodes 1000 --voxels 27 --r-threshold 0.5 --leiden-density 0.02
    python scripts/benchmarks/phase1_feasibility.py --save-report
"""

import argparse
import gc
import os
import platform
import resource
import sys
import time
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2] if len(SCRIPT_PATH.parents) > 2 else Path.cwd()
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "benchmarks"


THRESHOLD_LEIDEN_S = 5 * 60   # 5 minutes  (project plan accept criterion)
THRESHOLD_RAM_GB   = 64.0      # 64 GB      (project plan accept criterion)

MIN_EDGES_FOR_LEIDEN = 1_000   # below this we fall back to a synthetic graph


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1 feasibility: full correlation matrix + r-threshold + Leiden.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--nodes",    type=int,   default=36_000,
                   help="Number of parcels (default: 36000, Section 2.2 target).")
    p.add_argument("--voxels",   type=int,   default=27,
                   help="Voxels per parcel = Jacobian vector length (default: 27).")
    p.add_argument("--r-threshold", type=float, default=0.5,
                   help="Pearson r threshold for edge inclusion (default: 0.5).")
    p.add_argument("--leiden-density", type=float, default=0.10,
                   help="Fallback graph density for Leiden when Stage A yields too few "
                        "edges from random data (default: 0.10, expected real-data density).")
    p.add_argument("--seed",     type=int,   default=42,
                   help="Random seed.")
    p.add_argument("--save-report", action="store_true",
                   help="Write plain-text report to outputs/benchmarks/.")
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


def fmt_int(v):  return f"{v:,}"
def fmt_s(v):    return f"{v / 60:.1f} min" if v >= 60 else f"{v:.3f} s"
def fmt_gb(v):   return f"{v:.2f} GB"
def check(ok):   return "PASS" if ok else "FAIL"


# ── Stage A: correlation matrix ───────────────────────────────────────────────

def stage_a_correlation(n: int, voxels: int, threshold: float, seed: int) -> dict:
    """
    Simulate the per-subject correlation matrix computation.

    Memory layout:
      X  :  n × voxels  float32  →  n × voxels × 4 bytes  (negligible)
      C  :  n × n       float32  →  n² × 4 bytes  (5.2 GB at n=36k)

    C is kept DENSE so that every pairwise correlation is evaluated.
    After thresholding, only the surviving edges are kept; C is freed.
    """
    rng = np.random.default_rng(seed)
    max_pairs = n * (n - 1) // 2
    corr_matrix_gb = n * n * 4 / 1024**3

    print(f"\n{'─' * 62}")
    print(f"  Stage A – Correlation matrix  ({fmt_int(n)} parcels × {voxels} voxels)")
    print(f"  Pearson r threshold : {threshold}")
    print(f"  C matrix size       : {fmt_gb(corr_matrix_gb)}  (float32, n×n, DENSE)")
    print(f"{'─' * 62}")

    res = {"n": n, "voxels": voxels, "threshold": threshold}

    # 1. Generate synthetic Jacobian feature matrix X ─────────────────────────
    # In the real pipeline X[i, :] = Jacobian values for all voxels in parcel i.
    # Here we use random normal values; structure doesn't matter for timing.
    print("  [A1] Generating X (n × voxels, float32) ...")
    t0 = time.perf_counter()
    X = rng.standard_normal((n, voxels)).astype(np.float32)
    t1 = time.perf_counter()
    print(f"       {fmt_s(t1 - t0)}  |  RSS={fmt_gb(rss_gb())}")

    # 2. Row-normalise: mean-centre then L2-norm ───────────────────────────────
    # After this, X[i] is a unit vector and X @ X.T gives exact Pearson r.
    print("  [A2] Row-normalising X (mean-centre + L2-norm) ...")
    t2 = time.perf_counter()
    X -= X.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X /= np.where(norms > 0, norms, 1.0)
    t3 = time.perf_counter()
    print(f"       {fmt_s(t3 - t2)}  |  RSS={fmt_gb(rss_gb())}")

    # 3. Full correlation matrix C = X @ X.T ──────────────────────────────────
    # This is the most expensive step: O(n² × voxels) FLOPs.
    # Uses BLAS sgemm (float32) — benefits from multi-core BLAS on the cluster.
    # Peak RAM here = corr_matrix_gb (C) + tiny X already in memory.
    print(f"  [A3] Computing C = X @ X.T  ({fmt_gb(corr_matrix_gb)} float32 matrix) ...")
    t4 = time.perf_counter()

    # C is the full dense Pearson r matrix: C[i, j] = r between parcels i and j.
    # Every single pairwise relationship is evaluated, so no risk of missing edges.
    C = X @ X.T        # shape: (n, n), dtype: float32 (BLAS sgemm)

    del X              # free immediately — C is all we need from here
    gc.collect()
    t5 = time.perf_counter()
    res["corr_time_s"] = t5 - t4
    res["rss_after_corr_gb"] = rss_gb()
    print(f"       {fmt_s(t5 - t4)}  |  RSS={fmt_gb(rss_gb())}")

    # 4. Apply threshold row-by-row (upper triangle only) ─────────────────────
    # Row-by-row avoids creating a second n×n boolean array (would add another
    # ~1.3 GB) and avoids materialising all n*(n-1)/2 index pairs at once.
    print(f"  [A4] Thresholding (r > {threshold}, upper triangle, row-by-row) ...")
    t6 = time.perf_counter()

    rows_list, cols_list, weights_list = [], [], []

    # Threshold C row-by-row, and collect surviving edges in lists to concatenate at the end.
    for i in range(n):
        row = C[i, i + 1:]                       # view into C, zero-copy
        above = np.where(row > threshold)[0]
        if above.size:
            rows_list.append(np.full(above.size, i, dtype=np.int32))
            cols_list.append((above + i + 1).astype(np.int32))
            weights_list.append(row[above])      # copy only surviving values

    del C
    gc.collect()

    if rows_list:
        rows_arr    = np.concatenate(rows_list).astype(np.int32)
        cols_arr    = np.concatenate(cols_list).astype(np.int32)
        weights_arr = np.concatenate(weights_list).astype(np.float32)
    else:
        rows_arr = np.array([], dtype=np.int32)
        cols_arr = np.array([], dtype=np.int32)
        weights_arr = np.array([], dtype=np.float32)

    t7 = time.perf_counter()
    edge_count = len(rows_arr)
    observed_density = edge_count / max_pairs if max_pairs else 0.0

    res["threshold_time_s"]   = t7 - t6
    res["stage_a_time_s"]     = t7 - t0
    res["edge_count"]         = edge_count
    res["observed_density"]   = observed_density
    res["peak_rss_gb"]        = rss_gb()
    res["rows"]               = rows_arr
    res["cols"]               = cols_arr
    res["weights"]            = weights_arr

    print(f"       {fmt_s(t7 - t6)}  |  RSS={fmt_gb(rss_gb())}")
    print(f"  Edges surviving threshold : {fmt_int(edge_count)}")
    print(f"  Observed density          : {observed_density:.4%}")
    if edge_count == 0:
        print(f"  (Expected ~0 on random data — real Jacobian data will yield ~5–20%)")

    return res


# ── Stage B: Leiden community detection ───────────────────────────────────────

def stage_b_leiden(stage_a: dict, leiden_density: float, seed: int) -> dict:
    """
    Run Leiden on the graph produced by Stage A.

    If Stage A yielded too few edges (random data → ~0 edges), fall back to
    a synthetic graph at leiden_density so Leiden is always benchmarked.
    This fallback simulates expected real-data conditions.
    """
    n           = stage_a["n"]
    rows_arr    = stage_a["rows"]
    cols_arr    = stage_a["cols"]
    weights_arr = stage_a["weights"]
    edge_count  = stage_a["edge_count"]

    print(f"\n{'─' * 62}")
    print(f"  Stage B – Leiden community detection")
    print(f"{'─' * 62}")

    using_fallback = edge_count < MIN_EDGES_FOR_LEIDEN
    res = {"using_fallback": using_fallback}

    if using_fallback:
        max_pairs = n * (n - 1) // 2
        target_edges = int(max_pairs * leiden_density)
        print(f"  Stage A produced {fmt_int(edge_count)} edges (< {fmt_int(MIN_EDGES_FOR_LEIDEN)} minimum).")
        print(f"  Using synthetic graph at {leiden_density:.0%} density ({fmt_int(target_edges)} edges)")
        print(f"  to benchmark Leiden at expected real-data scale.")
        rng = np.random.default_rng(seed)
        print("  [B1] Building synthetic graph (igraph Erdos-Renyi) ...")
        t0 = time.perf_counter()
        graph = ig.Graph.Erdos_Renyi(n=n, m=target_edges, directed=False, loops=False)
        graph.es["weight"] = rng.random(graph.ecount(), dtype=np.float32).tolist()
        t1 = time.perf_counter()
        res["graph_source"]  = f"synthetic Erdos-Renyi  (density={leiden_density:.0%})"
        res["actual_edges"]  = graph.ecount()
        res["graph_time_s"]  = t1 - t0
        print(f"       {fmt_s(t1 - t0)}  |  edges={fmt_int(graph.ecount())}  |  RSS={fmt_gb(rss_gb())}")
    else:
        print(f"  Using {fmt_int(edge_count)} edges from Stage A  "
              f"(density={stage_a['observed_density']:.4%})")
        print("  [B1] Building igraph from Stage A edge list ...")
        t0 = time.perf_counter()

        # Build adjacency list for igraph: list of (source, target) pairs and list of weights.
        edge_pairs = list(zip(rows_arr.tolist(), cols_arr.tolist()))
        graph = ig.Graph(n=n, edges=edge_pairs, directed=False)
        graph.es["weight"] = weights_arr.tolist()
        del edge_pairs
        t1 = time.perf_counter()
        res["graph_source"]  = f"Stage A threshold  (r > {stage_a['threshold']})"
        res["actual_edges"]  = graph.ecount()
        res["graph_time_s"]  = t1 - t0
        print(f"       {fmt_s(t1 - t0)}  |  edges={fmt_int(graph.ecount())}  |  RSS={fmt_gb(rss_gb())}")

    # Run Leiden ───────────────────────────────────────────────────────────────
    print("  [B2] Running Leiden (ModularityVertexPartition, weighted) ...")
    t2 = time.perf_counter()
    partition = leidenalg.find_partition(
        graph,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        seed=seed,
    )
    t3 = time.perf_counter()

    res["leiden_time_s"]    = t3 - t2
    res["communities"]      = len(partition)
    res["modularity"]       = partition.modularity
    res["peak_rss_gb"]      = rss_gb()
    res["pass_time"]        = res["leiden_time_s"] < THRESHOLD_LEIDEN_S
    res["pass_ram"]         = res["peak_rss_gb"] < THRESHOLD_RAM_GB
    res["pass"]             = res["pass_time"] and res["pass_ram"]

    print(f"       {fmt_s(t3 - t2)}  |  communities={fmt_int(len(partition))}  "
          f"|  modularity={partition.modularity:.4f}")
    print(f"       Peak RSS: {fmt_gb(rss_gb())}")
    print()
    print(f"  [{check(res['pass_time'])}] Leiden < 5 min  → {fmt_s(res['leiden_time_s'])}")
    print(f"  [{check(res['pass_ram'])}] RAM < 64 GB     → {fmt_gb(res['peak_rss_gb'])}")
    print(f"  {'✓ GO' if res['pass'] else '✗ NO-GO'}  –  Proceed to Phase 2: {'YES' if res['pass'] else 'NO'}")

    return res


# ── Report ────────────────────────────────────────────────────────────────────

def build_report(sys_info: dict, sa: dict, sb: dict) -> str:
    W = 64
    lines = []
    def hr():        lines.append("=" * W)
    def subhr():     lines.append("─" * W)
    def row(k, v):   lines.append(f"  {k:<32}{v}")

    hr()
    lines.append("  PHASE 1 FEASIBILITY REPORT")
    lines.append(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    hr()
    lines.append("")

    lines.append("  SYSTEM")
    subhr()
    row("Platform:",   sys_info["platform"])
    row("CPU cores:",  str(sys_info["cpu_cores"]))
    ram = sys_info.get("total_ram_gb")
    row("Total RAM:",  fmt_gb(ram) if ram else "unknown")
    lines.append("")

    lines.append("  STAGE A – Correlation matrix")
    subhr()
    row("Parcels (nodes):",          fmt_int(sa["n"]))
    row("Voxels per parcel:",        str(sa["voxels"]))
    row("r threshold:",              str(sa["threshold"]))
    row("C matrix (dense float32):", fmt_gb(sa["n"] ** 2 * 4 / 1024**3))
    row("X generation:",             fmt_s(sa["stage_a_time_s"] - sa["corr_time_s"] - sa["threshold_time_s"]))
    row("C = X @ X.T  (BLAS):",      fmt_s(sa["corr_time_s"]))
    row("Threshold + edge extract:", fmt_s(sa["threshold_time_s"]))
    row("Stage A total:",            fmt_s(sa["stage_a_time_s"]))
    row("Edges surviving r > thr:",  fmt_int(sa["edge_count"]))
    row("Observed density:",         f"{sa['observed_density']:.4%}")
    row("RSS after Stage A:",        fmt_gb(sa["peak_rss_gb"]))
    lines.append("")

    lines.append("  STAGE B – Leiden")
    subhr()
    row("Graph source:",             sb["graph_source"])
    row("Actual edges:",             fmt_int(sb["actual_edges"]))
    row("igraph build time:",        fmt_s(sb["graph_time_s"]))
    row("Leiden time:",              fmt_s(sb["leiden_time_s"]))
    row("Communities found:",        fmt_int(sb["communities"]))
    row("Modularity:",               f"{sb['modularity']:.4f}")
    row("Peak RSS:",                 fmt_gb(sb["peak_rss_gb"]))
    row("Leiden < 5 min:",           check(sb["pass_time"]))
    row("RAM < 64 GB:",              check(sb["pass_ram"]))
    row("Overall:",                  "GO – proceed to Phase 2" if sb["pass"] else "NO-GO")
    lines.append("")

    lines.append("  APPENDIX B CHECKLIST")
    subhr()
    lines.append(f"  [x] Infrastructure specs documented (see SYSTEM above)")
    lines.append(f"  [x] Synthetic benchmark completed")
    lines.append(f"  [{'x' if sb['pass_time'] else ' '}] Leiden runtime < 5 min")
    lines.append(f"  [{'x' if sb['pass_ram']  else ' '}] Peak memory < 64 GB")
    lines.append(f"  [ ] Single-subject pilot  (Phase 1 Task 3 – requires real MRI data)")
    lines.append(f"  [ ] Segmentation timing   (run sub_parcels_equal_size.py per ROI label)")
    lines.append(f"  [ ] Go/No-Go decision     (see Stage B result above)")
    lines.append("")

    lines.append("  NOTES")
    subhr()
    if sb["using_fallback"]:
        lines.append("  Stage B used a synthetic random graph because Stage A found ~0 edges")
        lines.append("  on random input data.  This is expected: random 27-dim vectors have")
        lines.append("  r ≈ 0 by construction.  On real Jacobian data, biological structure")
        lines.append("  will produce edges at ~5–20% density — Leiden will run on those.")
        lines.append("")
    lines.append("  Modularity near 0 on random data is correct: it means Leiden correctly")
    lines.append("  finds no spurious structure.  Real deformation data will yield higher")
    lines.append("  modularity reflecting genuine atrophy networks.")
    lines.append("")
    lines.append("  Stage A timing is the critical new bottleneck vs. the old approach.")
    lines.append("  C = X @ X.T is O(n² × voxels) and uses all CPU cores via BLAS.")
    lines.append("  On the iRBio cluster with many cores it should be much faster.")

    hr()
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.nodes < 2:
        sys.exit("--nodes must be >= 2")
    if args.voxels < 2:
        sys.exit("--voxels must be >= 2")
    if not (0.0 < args.r_threshold < 1.0):
        sys.exit("--r-threshold must be in (0, 1)")
    if not (0.0 < args.leiden_density <= 1.0):
        sys.exit("--leiden-density must be in (0, 1]")

    wall_start = time.perf_counter()

    print("=" * 62)
    print("  PHASE 1 FEASIBILITY – Full Corr. Matrix + r-Threshold")
    print("=" * 62)

    sys_info = {
        "platform":    platform.platform(),
        "cpu_cores":   os.cpu_count() or "unknown",
        "total_ram_gb": total_ram_gb(),
    }
    print(f"  Platform  : {sys_info['platform']}")
    print(f"  CPU cores : {sys_info['cpu_cores']}")
    ram = sys_info.get("total_ram_gb")
    if ram:
        print(f"  Total RAM : {fmt_gb(ram)}")
    print(f"  Init RSS  : {fmt_gb(rss_gb())}")

    # Upfront RAM warning for the correlation matrix
    corr_gb = args.nodes ** 2 * 4 / 1024**3
    if corr_gb > 16:
        print(f"\n  [!] C matrix will occupy ~{fmt_gb(corr_gb)} RAM.")
        print(f"      Ensure the cluster has at least {fmt_gb(corr_gb * 1.5)} free.")

    sa = stage_a_correlation(args.nodes, args.voxels, args.r_threshold, args.seed)
    sb = stage_b_leiden(sa, args.leiden_density, args.seed)

    report = build_report(sys_info, sa, sb)
    print("\n" + report)

    if args.save_report:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        ts   = time.strftime("%Y%m%d_%H%M%S")
        path = OUTPUTS_DIR / f"phase1_feasibility_{ts}.txt"
        path.write_text(report)
        print(f"\n  Report saved → {path}")

    print(f"\n  Total wall time: {fmt_s(time.perf_counter() - wall_start)}")


if __name__ == "__main__":
    main()
