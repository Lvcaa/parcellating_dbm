from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:  # pragma: no cover - import guard for optional plotting deps
    raise SystemExit(
        "This script requires matplotlib and seaborn. "
        "Install them before running this tool."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH_DIR = PROJECT_ROOT / "outputs" / "wasserstein_graph_label_10"
DEFAULT_VECTOR_ROOT = PROJECT_ROOT / "outputs" / "jacobian_parcel_vectors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect a Wasserstein graph output folder, print sanity checks, and "
            "save visualization figures for the adjacency matrix and weighted degree."
        )
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=DEFAULT_GRAPH_DIR,
        help=f"Directory containing adjacency_matrix.dat, weighted_degree.dat, and metadata.npy (default: {DEFAULT_GRAPH_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where figures will be written. "
            "If omitted, figures are saved inside --graph-dir/inspection."
        ),
    )
    parser.add_argument(
        "--labels",
        type=int,
        nargs="+",
        default=None,
        help=(
            "One or more ROI labels to merge into a single Wasserstein matrix. "
            "When provided, parcel vectors are loaded from "
            "--vectors-root/label_<label>/ and all within-label and cross-label "
            "distances are computed together."
        ),
    )
    parser.add_argument(
        "--vectors-root",
        type=Path,
        default=DEFAULT_VECTOR_ROOT,
        help=(
            "Root directory containing exported parcel vectors organized as "
            "label_<label>/roi_*.npy "
            f"(default: {DEFAULT_VECTOR_ROOT})."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figures interactively after saving them.",
    )
    parser.add_argument(
        "--top-k-edges",
        type=int,
        default=10,
        help="Print the top K strongest off-diagonal edges.",
    )
    return parser.parse_args()


def load_graph(graph_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata_path = graph_dir / "metadata.npy"
    matrix_path = graph_dir / "adjacency_matrix.dat"
    degree_path = graph_dir / "weighted_degree.dat"

    if not metadata_path.is_file():
        raise ValueError(f"Missing metadata file: {metadata_path}")
    if not matrix_path.is_file():
        raise ValueError(f"Missing adjacency matrix file: {matrix_path}")
    if not degree_path.is_file():
        raise ValueError(f"Missing weighted degree file: {degree_path}")

    metadata = np.load(metadata_path)
    if metadata.size < 1:
        raise ValueError(f"Metadata file does not contain the graph size: {metadata_path}")

    n_parcels = int(metadata[0])
    if n_parcels < 1:
        raise ValueError(f"Invalid graph size in metadata: {n_parcels}")

    matrix = np.memmap(matrix_path, dtype="float64", mode="r", shape=(n_parcels, n_parcels))
    weighted_degree = np.memmap(degree_path, dtype="float64", mode="r", shape=(n_parcels,))
    return matrix, weighted_degree


def load_label_vectors(vectors_root: Path, labels: list[int]) -> tuple[list[np.ndarray], list[str]]:
    parcel_vectors: list[np.ndarray] = []
    parcel_names: list[str] = []

    for label in labels:
        label_dir = vectors_root / f"label_{label}"
        if not label_dir.is_dir():
            raise ValueError(f"Missing parcel vector directory for label {label}: {label_dir}")

        parcel_paths = sorted(label_dir.glob("roi_*.npy"))
        if not parcel_paths:
            raise ValueError(f"No parcel vectors found for label {label} in {label_dir}")

        for parcel_path in parcel_paths:
            parcel_vectors.append(np.load(parcel_path))
            parcel_names.append(f"label_{label}/{parcel_path.stem}")

    return parcel_vectors, parcel_names


def build_merged_graph(parcel_vectors: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n_parcels = len(parcel_vectors)
    if n_parcels < 1:
        raise ValueError("At least one parcel vector is required to build a graph.")

    matrix = np.zeros((n_parcels, n_parcels), dtype=np.float64)
    np.fill_diagonal(matrix, 1.0)

    for i in range(n_parcels):
        for j in range(i + 1, n_parcels):
            similarity = float(np.exp(-wasserstein_distance(parcel_vectors[i], parcel_vectors[j])))
            matrix[i, j] = similarity
            matrix[j, i] = similarity

    if n_parcels == 1:
        weighted_degree = np.zeros((1,), dtype=np.float64)
    else:
        weighted_degree = (np.sum(matrix, axis=1) - 1.0) / (n_parcels - 1)

    return matrix, weighted_degree


def format_stats(matrix: np.ndarray, weighted_degree: np.ndarray) -> list[str]:
    diag = np.diag(matrix)
    symmetry_max_abs_diff = float(np.max(np.abs(matrix - matrix.T)))
    matrix_min = float(np.min(matrix))
    matrix_max = float(np.max(matrix))
    degree_min = float(np.min(weighted_degree))
    degree_mean = float(np.mean(weighted_degree))
    degree_max = float(np.max(weighted_degree))

    return [
        f"shape: {matrix.shape}",
        f"symmetry max abs diff: {symmetry_max_abs_diff}",
        f"diagonal min/max: {float(diag.min())} / {float(diag.max())}",
        f"matrix min/max: {matrix_min} / {matrix_max}",
        f"weighted degree min/mean/max: {degree_min} / {degree_mean} / {degree_max}",
        f"all finite: {bool(np.isfinite(matrix).all() and np.isfinite(weighted_degree).all())}",
        f"within [0, 1]: {bool(matrix_min >= 0.0 and matrix_max <= 1.0)}",
    ]


def print_top_edges(matrix: np.ndarray, top_k: int) -> None:
    n_parcels = matrix.shape[0]
    if n_parcels < 2 or top_k < 1:
        return

    tri = np.triu_indices(n_parcels, k=1)
    values = matrix[tri]
    top_k = min(top_k, values.size)
    top_indices = np.argsort(values)[::-1][:top_k]

    print(f"top {top_k} off-diagonal edges:")
    for rank, index in enumerate(top_indices, 1):
        i = int(tri[0][index])
        j = int(tri[1][index])
        value = float(values[index])
        print(f"  {rank:>2}. ({i}, {j}) = {value:.6f}")


def print_label_summary(parcel_names: list[str]) -> None:
    if not parcel_names:
        return

    counts: dict[str, int] = {}
    for parcel_name in parcel_names:
        label_name = parcel_name.split("/", maxsplit=1)[0]
        counts[label_name] = counts.get(label_name, 0) + 1

    print("merged parcel counts by label:")
    for label_name in sorted(counts):
        print(f"  {label_name}: {counts[label_name]} parcels")


def plot_heatmap(matrix: np.ndarray) -> plt.Figure:
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        matrix,
        cmap="mako",
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar_kws={"label": "exp(-Wasserstein)"},
        ax=ax,
    )
    ax.set_title("Wasserstein similarity matrix")
    ax.set_xlabel("Parcel index")
    ax.set_ylabel("Parcel index")
    fig.tight_layout()
    return fig


def plot_weighted_degree(weighted_degree: np.ndarray) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(np.asarray(weighted_degree), bins=20, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Weighted degree distribution")
    ax.set_xlabel("Weighted degree")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    if args.labels:
        labels = list(dict.fromkeys(args.labels))
        parcel_vectors, parcel_names = load_label_vectors(args.vectors_root, labels)
        matrix, weighted_degree = build_merged_graph(parcel_vectors)

        labels_suffix = "_".join(str(label) for label in labels)
        default_output_dir = (
            PROJECT_ROOT / "outputs" / f"wasserstein_graph_labels_{labels_suffix}" / "inspection"
        )
        output_dir = args.output_dir if args.output_dir is not None else default_output_dir

        print(f"Built merged graph from labels: {labels}")
        print_label_summary(parcel_names)
    else:
        graph_dir = args.graph_dir
        if not graph_dir.is_dir():
            raise ValueError(f"Graph directory does not exist: {graph_dir}")

        matrix, weighted_degree = load_graph(graph_dir)
        output_dir = args.output_dir if args.output_dir is not None else graph_dir / "inspection"
        print(f"Loaded graph from {graph_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for line in format_stats(matrix, weighted_degree):
        print(line)
    print_top_edges(matrix, args.top_k_edges)

    heatmap_path = output_dir / "adjacency_heatmap.png"
    degree_path = output_dir / "weighted_degree_hist.png"

    heatmap_fig = plot_heatmap(np.asarray(matrix))
    heatmap_fig.savefig(heatmap_path, dpi=200, bbox_inches="tight")

    degree_fig = plot_weighted_degree(np.asarray(weighted_degree))
    degree_fig.savefig(degree_path, dpi=200, bbox_inches="tight")

    print(f"saved heatmap: {heatmap_path}")
    print(f"saved weighted degree plot: {degree_path}")

    if args.show:
        plt.show()
    else:
        plt.close(heatmap_fig)
        plt.close(degree_fig)


if __name__ == "__main__":
    main()
