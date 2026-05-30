"""
Benchmark Leiden community detection on a synthetic igraph graph.

Edge count and RAM grow quickly with node count and density. Start with a
small run before using the defaults.

Usage:
    python scripts/benchmarks/fake_matrix_stress_test.py [options]

Parameters:
    --nodes INT          Graph nodes (default: 10000).
    --density FLOAT      Undirected edge density in [0, 1] (default: 0.30).
    --seed INT           Random seed (default: 42).
    --use-probability    Use Erdos-Renyi probability mode instead of exact edges.
    --unweighted         Skip edge weights.

Examples:
    python scripts/benchmarks/fake_matrix_stress_test.py --nodes 1000 --density 0.05
    python scripts/benchmarks/fake_matrix_stress_test.py --nodes 5000 --density 0.01 --use-probability --unweighted
"""

import argparse
import math
import time
import resource

import igraph as ig
import leidenalg
import numpy as np


def parse_args():
    """
    Read command-line arguments so we can easily change
    graph size, density, and random seed without editing the code.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark Leiden community detection on a large synthetic weighted graph."
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
        default=0.30,
        help="Target edge density in [0, 1] for an undirected graph without self-loops.",
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

    # Random number generator for reproducibility
    rng = np.random.default_rng(args.seed)

    # Number of nodes
    n = args.nodes

    # Maximum possible number of edges in an undirected graph with no self-loops:
    # n * (n - 1) / 2
    max_edges = n * (n - 1) // 2

    # Target number of edges based on requested density
    target_edges = math.floor(max_edges * args.density)

    # Print benchmark configuration
    print("Synthetic graph benchmark")
    print("Nodes:", format_int(n))
    print("Target density:", args.density)
    print("Max undirected edges:", format_int(max_edges))
    print("Target edges:", format_int(target_edges))
    print("Random seed:", args.seed)
    print("Graph mode:", "probability p" if args.use_probability else "exact edge count")
    print("Weighted graph:", not args.unweighted)
    print("Initial RSS (GB):", round(rss_gb(), 3))

    # ---------------------------------------------------------
    # STEP 1: Build the synthetic graph
    # ---------------------------------------------------------
    print("Building graph in igraph...")
    t0 = time.time()

    if args.use_probability:
        # Probability mode:
        # every possible edge is included independently with probability = density
        graph = ig.Graph.Erdos_Renyi(
            n=n,
            p=args.density,
            directed=False,
            loops=False
        )
    else:
        # Exact-edge mode:
        # generate a graph with exactly target_edges edges
        graph = ig.Graph.Erdos_Renyi(
            n=n,
            m=target_edges,
            directed=False,
            loops=False
        )

    t1 = time.time()
    graph_build_time = t1 - t0

    print("Graph build time:", format_seconds(graph_build_time))
    print("Observed edges:", format_int(graph.ecount()))

    # Actual density may differ slightly in probability mode
    observed_density = graph.ecount() / max_edges
    print("Observed density:", round(observed_density, 6))
    print("RSS after graph build (GB):", round(rss_gb(), 3))

    # ---------------------------------------------------------
    # STEP 2: Assign random edge weights
    # ---------------------------------------------------------
    weight_time = 0.0
    if args.unweighted:
        print("Skipping weight assignment (unweighted baseline)...")
    else:
        # The final parcel graph should be weighted because an edge stores
        # the strength of similarity between two parcel profiles.
        print("Assigning random similarity weights...")
        t2 = time.time()

        # Generate one random float32 weight per edge.
        # Using float32 keeps the synthetic benchmark closer to a realistic
        # weighted similarity graph without paying float64 memory costs.
        graph.es["weight"] = rng.random(graph.ecount(), dtype=np.float32).tolist()

        t3 = time.time()
        weight_time = t3 - t2

        print("Weight assignment time:", format_seconds(weight_time))
        print("RSS after weights (GB):", round(rss_gb(), 3))

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
    print("Total runtime:", format_seconds(total_time))


if __name__ == "__main__":
    main()
