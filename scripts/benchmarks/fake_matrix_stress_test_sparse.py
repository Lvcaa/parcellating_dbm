import argparse
import gc
import math
import resource
import time

import igraph as ig
import leidenalg
import numpy as np
from scipy import sparse


def parse_args():
    """
    Read command-line arguments so we can easily change
    graph size, density, and random seed without editing the code.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark Leiden community detection on a synthetic sparse graph."
    )

    # Number of graph nodes (regions / parcels in your analogy)
    parser.add_argument(
        "--nodes",
        type=int,
        default=10000,
        help="Number of graph nodes."
    )

    # Desired graph density:
    # 0.30 means we want 30% of all possible undirected edges
    parser.add_argument(
        "--density",
        type=float,
        default=0.001,
        help="Target undirected edge density in [0, 1] for a sparse graph without self-loops.",
    )

    # Random seed for reproducibility:
    # same seed -> same random graph and same random weights
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for graph structure and edge weights.",
    )

    # Two ways to generate the graph:
    # 1) exact edge count (m = target_edges)
    # 2) probability mode (each edge appears with probability p = density)
    parser.add_argument(
        "--use-probability",
        action="store_true",
        help="Use Erdos-Renyi probability p=density instead of an exact edge count.",
    )

    # In the real pipeline, edges represent similarity between parcel-level
    # morphometric profiles, so the graph is naturally weighted.
    # Keep weighted mode as the default benchmark, but allow an unweighted
    # baseline for quicker feasibility checks.
    parser.add_argument(
        "--unweighted",
        action="store_true",
        help="Skip edge weights and run Leiden on an unweighted graph.",
    )

    return parser.parse_args()


def rss_gb():
    """
    Return peak memory usage (RSS = resident set size) in GB.

    Note:
    - Linux usually reports ru_maxrss in KiB
    - macOS usually reports ru_maxrss in bytes
    This function tries to normalize that into GB.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Heuristic to distinguish Linux vs macOS units
    if usage > 10**10:
        return usage / 1024**3   # assume bytes -> GB
    return usage / 1024**2       # assume KiB -> GB


def format_int(value):
    """Format large integers with commas for readability."""
    return f"{value:,}"


def format_seconds(seconds):
    """Format elapsed time in seconds with 3 decimals."""
    return f"{seconds:.3f} s"


def sample_upper_triangle_edges(n, edge_count, rng):
    """
    Sample unique undirected edges from the upper triangle of an adjacency matrix.

    Returns two arrays `rows` and `cols` such that every edge satisfies rows[i] < cols[i].
    """
    if edge_count == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty

    max_edges = n * (n - 1) // 2
    edge_ids = np.sort(rng.choice(max_edges, size=edge_count, replace=False))

    row_lengths = np.arange(n - 1, 0, -1, dtype=np.int64)
    row_starts = np.concatenate(([0], np.cumsum(row_lengths[:-1], dtype=np.int64)))

    rows = np.searchsorted(row_starts, edge_ids, side="right") - 1
    cols = edge_ids - row_starts[rows] + rows + 1
    return rows.astype(np.int64, copy=False), cols.astype(np.int64, copy=False)


def build_sparse_adjacency(n, rows, cols, values):
    """Build a symmetric sparse adjacency matrix from upper-triangle edges."""
    upper = sparse.coo_matrix((values, (rows, cols)), shape=(n, n), dtype=np.float32)
    adjacency = (upper + upper.T).tocsr()
    adjacency.eliminate_zeros()
    return adjacency


def main():
    # Start total benchmark timer
    overall_start = time.time()

    # Read user parameters
    args = parse_args()

    # Basic sanity checks
    if args.nodes < 2:
        raise ValueError("--nodes must be at least 2")
    if not 0.0 <= args.density <= 1.0:
        raise ValueError("--density must be between 0 and 1")
    if args.density == 1.0:
        raise ValueError("--density=1.0 creates a dense graph; use the dense benchmark instead")

    # Random number generator for reproducibility
    rng = np.random.default_rng(args.seed)

    # Number of nodes
    n = args.nodes

    # Maximum possible number of edges in an undirected graph with no self-loops:
    # n * (n - 1) / 2
    max_edges = n * (n - 1) // 2

    # Target number of undirected edges based on requested density
    if args.use_probability:
        target_edges = int(rng.binomial(max_edges, args.density))
    else:
        target_edges = math.floor(max_edges * args.density)
    if target_edges > max_edges:
        raise ValueError("Requested number of edges exceeds the maximum possible graph size")

    # Print benchmark configuration
    print("Synthetic sparse graph benchmark")
    print("Nodes:", format_int(n))
    print("Target density:", args.density)
    print("Max undirected edges:", format_int(max_edges))
    print("Target edges:", format_int(target_edges))
    print("Random seed:", args.seed)
    print("Graph mode:", "probability p" if args.use_probability else "exact edge count")
    print("Weighted graph:", not args.unweighted)
    print("Initial RSS (GB):", round(rss_gb(), 3))

    # ---------------------------------------------------------
    # STEP 1: Build a sparse adjacency matrix
    # ---------------------------------------------------------
    print("Sampling sparse undirected adjacency...")
    t0 = time.time()

    rows, cols = sample_upper_triangle_edges(n, target_edges, rng)
    if args.unweighted:
        edge_values = np.ones(target_edges, dtype=np.float32)
    else:
        edge_values = rng.random(target_edges, dtype=np.float32)

    adjacency = build_sparse_adjacency(n, rows, cols, edge_values)

    t1 = time.time()
    sparse_build_time = t1 - t0

    print("Sparse matrix build time:", format_seconds(sparse_build_time))
    print("Sampled undirected edges:", format_int(target_edges))
    print("Sparse matrix nnz (symmetric entries):", format_int(adjacency.nnz))

    # Actual density may differ slightly in probability mode
    observed_density = target_edges / max_edges
    print("Observed density:", round(observed_density, 6))
    print("RSS after sparse matrix build (GB):", round(rss_gb(), 3))

    # ---------------------------------------------------------
    # STEP 2: Convert the sparse adjacency into an igraph graph
    # ---------------------------------------------------------
    print("Converting sparse adjacency to igraph...")
    t2 = time.time()

    edge_pairs = list(zip(rows.tolist(), cols.tolist()))
    graph = ig.Graph(n=n, edges=edge_pairs, directed=False)
    if not args.unweighted:
        graph.es["weight"] = edge_values.tolist()

    t3 = time.time()
    graph_build_time = t3 - t2

    print("igraph conversion time:", format_seconds(graph_build_time))
    print("igraph edge count:", format_int(graph.ecount()))
    print("RSS after igraph conversion (GB):", round(rss_gb(), 3))

    # Free the sparse matrix before community detection so the benchmark
    # focuses on the graph object that Leiden actually consumes.
    del adjacency
    del edge_pairs
    del rows
    del cols
    del edge_values
    gc.collect()

    # ---------------------------------------------------------
    # STEP 3: Run Leiden community detection
    # ---------------------------------------------------------
    print("Running Leiden community detection...")
    t4 = time.time()

    partition_kwargs = {
        "seed": args.seed,
    }
    if not args.unweighted:
        partition_kwargs["weights"] = "weight"

    partition = leidenalg.find_partition(
        graph,
        leidenalg.ModularityVertexPartition,  # optimize modularity
        **partition_kwargs,
    )

    t5 = time.time()
    leiden_time = t5 - t4
    total_time = t5 - overall_start

    # ---------------------------------------------------------
    # STEP 4: Report results
    # ---------------------------------------------------------
    print("Community detection time:", format_seconds(leiden_time))
    print("Number of communities found:", len(partition))
    print("Modularity:", round(partition.modularity, 6))
    print("Peak RSS (GB):", round(rss_gb(), 3))
    print("Stage summary:")
    print("  sparse matrix build:", format_seconds(sparse_build_time))
    print("  igraph conversion:", format_seconds(graph_build_time))
    print("  Leiden:", format_seconds(leiden_time))
    print("Total runtime:", format_seconds(total_time))


if __name__ == "__main__":
    main()
