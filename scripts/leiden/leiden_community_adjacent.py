import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import monotonic

import igraph as ig
import leidenalg
import numpy as np
from scipy.stats import wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = PROJECT_ROOT / "outputs/cohorts/oasis3/runs/connected-atlas-8cf3dc8f"
GRAPH_PATH = PROJECT_ROOT / (
    "outputs/atlases/atlas-8cf3dc8f/"
    "parcel_volume_cache/parcel_adjacency.graphml"
)
PARCEL_VECTORS_DIR = RUN_ROOT / "parcel_vectors"
LEIDEN_DIR = RUN_ROOT / "leiden"
SPLITS_PATH = PROJECT_ROOT / "data" / "splits.json"
SUBJECTS_PATH = PROJECT_ROOT / "data" / "leiden_subjects_267.txt"

SEED = 7
GAMMA_VALUES = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
PROGRESS_EVERY = 50_000
MATCHED_SPLIT_GROUPS = ("test_healthy", "test_unhealthy")
MATCHED_COHORT_FILES = (
    "direct_mean.dat",
    "direct_median.dat",
    "parcel_order.txt",
    "complete.json",
)


def resolve_subject_dirs() -> tuple[list[Path], dict[str, int]]:
    """Reproduce the matched test cohort used by regional_mean.ipynb."""
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    if not isinstance(splits, dict):
        raise ValueError(f"Expected a JSON object in {SPLITS_PATH}")

    available_subject_dirs = {
        subject_dir.name: subject_dir
        for subject_dir in PARCEL_VECTORS_DIR.glob("sub-*")
        if subject_dir.is_dir()
        and all((subject_dir / name).is_file() for name in MATCHED_COHORT_FILES)
    }

    selected_by_group: dict[str, set[str]] = {}
    for group in MATCHED_SPLIT_GROUPS:
        subject_ids = splits.get(group)
        
        if not isinstance(subject_ids, list) or not all(
            isinstance(subject_id, str) for subject_id in subject_ids
        ):
            raise ValueError(f"Split {group!r} must contain a list of subject IDs")
        selected_by_group[group] = {
            f"sub-{subject_id.removeprefix('sub-')}"
            for subject_id in subject_ids
        } & available_subject_dirs.keys()

    if selected_by_group["test_healthy"] & selected_by_group["test_unhealthy"]:
        raise ValueError("Healthy and unhealthy test cohorts overlap")

    subject_names = [
        subject_name
        for group in MATCHED_SPLIT_GROUPS
        for subject_name in sorted(selected_by_group[group])
    ]
    if not subject_names:
        raise ValueError(
            f"No matched test subjects were found under {PARCEL_VECTORS_DIR}"
        )

    group_counts = {
        group: len(subject_names) for group, subject_names in selected_by_group.items()
    }
    return [available_subject_dirs[name] for name in subject_names], group_counts



def resolve_subject_ids() -> list[str]:
    """Load and validate the transferred subject manifest."""
    subject_ids = SUBJECTS_PATH.read_text(encoding="utf-8").splitlines()
    if not subject_ids or len(subject_ids) != len(set(subject_ids)):
        raise ValueError(f"Invalid subject manifest: {SUBJECTS_PATH}")
    for subject_id in subject_ids:
        subject_dir = PARCEL_VECTORS_DIR / subject_id
        if not subject_id.startswith("sub-") or not all(
            (subject_dir / name).is_file()
            for name in ("parcel_vectors.dat", "parcel_offsets.dat", "parcel_order.txt")
        ):
            raise ValueError(f"Missing subject inputs: {subject_dir}")
    return subject_ids


def load_subject_data(
    subject_dir: Path,
    expected_parcel_ids: list[str],
) -> tuple[np.memmap, np.ndarray]:
    """Load and validate one subject's concatenated parcel vectors."""

    parcel_ids = (subject_dir / "parcel_order.txt").read_text(
        encoding="utf-8"
    ).splitlines()

    if parcel_ids != expected_parcel_ids:
        raise ValueError(f"Parcel order does not match the graph: {subject_dir}")

    all_values = np.memmap(
        subject_dir / "parcel_vectors.dat",
        dtype=np.float32,
        mode="r",
    )
    offsets = np.fromfile(
        subject_dir / "parcel_offsets.dat",
        dtype=np.int64,
    )

    if len(offsets) != len(parcel_ids) + 1:
        raise ValueError(f"Wrong number of parcel offsets: {subject_dir}")
    if offsets[0] != 0 or offsets[-1] != len(all_values):
        raise ValueError(f"Parcel offsets do not match the vector data: {subject_dir}")
    if np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError(f"Parcel offsets are not strictly increasing: {subject_dir}")
    if not np.isfinite(all_values).all():
        raise ValueError(f"Parcel vectors contain non-finite values: {subject_dir}")

    return all_values, offsets


def compute_edge_weights(
    edges: list[tuple[int, int]],
    all_values: np.memmap,
    offsets: np.ndarray,
) -> np.ndarray:
    """Compute exp(-Wasserstein distance) for every adjacent parcel pair."""
    weights = np.empty(len(edges), dtype=np.float32)
    started_at = monotonic()

    for edge_index, (parcel_a, parcel_b) in enumerate(edges, start=1):

        # Extract the parcel vectors for two parcels from indexes.
        values_a = all_values[offsets[parcel_a]:offsets[parcel_a + 1]]
        values_b = all_values[offsets[parcel_b]:offsets[parcel_b + 1]]

        # Compute the wasserstein distance between the two parcels' vectors
        distance = wasserstein_distance(values_a, values_b)

        # Compute the similarity weight as exp(-distance)
        weights[edge_index - 1] = np.exp(-distance)

        # Print elapsed time and print progress 
        if edge_index % PROGRESS_EVERY == 0 or edge_index == len(edges):
            elapsed = monotonic() - started_at
            print(
                f"  Weighted {edge_index:,}/{len(edges):,} edges "
                f"({100 * edge_index / len(edges):.1f}%) in {elapsed:.1f}s",
                flush=True,
            )

    return weights


def process_subject(
    subject_dir: Path,
    graph: ig.Graph,
    edges: list[tuple[int, int]],
    parcel_ids: list[str],
) -> None:
    """Weight the shared graph and run Leiden for one subject."""
    print(f"Processing {subject_dir.name}", flush=True)

    # Load the subject's parcel vectors and offsets
    all_values, offsets = load_subject_data(subject_dir, parcel_ids)
    weights = compute_edge_weights(edges, all_values, offsets)

    # Create a copy of the graph and assign the computed weights to the edges
    subject_graph = graph.copy()
    subject_graph.es["weight"] = weights.tolist()

    output_dir = LEIDEN_DIR / subject_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run leiden community detection for each gamma.
    for gamma in GAMMA_VALUES:

        partition = leidenalg.find_partition(
            subject_graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=gamma,
            n_iterations=-1,
            seed=SEED,
        )
        membership = np.asarray(partition.membership, dtype=np.int32)
        output_path = output_dir / f"membership_gamma_{gamma:.2f}.npy"
        np.save(output_path, membership)
        print(
            f"  gamma={gamma:.2f}: {len(partition):,} communities, "
            f"quality={partition.quality():.4f}",
            flush=True,
        )



WORKER_GRAPH = None
WORKER_EDGES = None
WORKER_PARCEL_IDS = None


def initialize_worker() -> None:
    global WORKER_GRAPH, WORKER_EDGES, WORKER_PARCEL_IDS
    WORKER_GRAPH = ig.Graph.Read_GraphML(str(GRAPH_PATH))
    if "parcel_id" not in WORKER_GRAPH.vs.attributes():
        raise ValueError("The adjacency graph has no parcel_id attribute")
    if WORKER_GRAPH.is_directed() or not WORKER_GRAPH.is_simple():
        raise ValueError("The adjacency graph must be simple and undirected")
    WORKER_EDGES = WORKER_GRAPH.get_edgelist()
    WORKER_PARCEL_IDS = WORKER_GRAPH.vs["parcel_id"]


def process_subject_id(subject_id: str) -> None:
    """Run the shared-path pipeline for one subject ID."""
    process_subject(
        PARCEL_VECTORS_DIR / subject_id,
        WORKER_GRAPH,
        WORKER_EDGES,
        WORKER_PARCEL_IDS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=1)
    args = parser.parse_args()
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")

    subject_ids = resolve_subject_ids()
    print(f"Processing {len(subject_ids):,} subjects with {args.num_workers} workers")
    if args.num_workers == 1:
        initialize_worker()
        for subject_id in subject_ids:
            process_subject_id(subject_id)
        return

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=initialize_worker,
    ) as executor:
        list(executor.map(process_subject_id, subject_ids, chunksize=1))


if __name__ == "__main__":
    main()
