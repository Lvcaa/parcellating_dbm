from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:
    raise SystemExit(
        "This script requires matplotlib and seaborn. "
        "Install them before running."
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_LABEL_LOOKUP = PROJECT_ROOT / "docs" / "label_lookup.csv"
SIM_FORMULA_OUTPUT_NAMES = ("expW", "inv1pW")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Wasserstein weighted-degree vectors between two subjects "
            "(healthy vs atrophy). Operates on the 1-D degree files for all "
            "parcel-level analysis; touches the full adjacency matrices only "
            "for a focused top-K submatrix plot."
        )
    )
    parser.add_argument(
        "--healthy-dir",
        type=Path,
        default=None,
        help=(
            "Graph output folder for the healthy subject. If omitted, provide "
            "--healthy-subject and --sim-formula."
        ),
    )
    parser.add_argument(
        "--atrophy-dir",
        type=Path,
        default=None,
        help=(
            "Graph output folder for the atrophy subject. If omitted, provide "
            "--atrophy-subject and --sim-formula."
        ),
    )
    parser.add_argument(
        "--healthy-subject",
        type=str,
        default=None,
        help="Healthy subject id, for example sub-0006.",
    )
    parser.add_argument(
        "--atrophy-subject",
        type=str,
        default=None,
        help="Atrophy subject id, for example sub-OAS30999.",
    )
    parser.add_argument(
        "--sim-formula",
        "--formula",
        dest="sim_formula",
        choices=SIM_FORMULA_OUTPUT_NAMES,
        default=None,
        help=(
            "Formula-specific graph folder suffix to use when resolving subject "
            "ids: expW or inv1pW."
        ),
    )
    parser.add_argument(
        "--label-lookup",
        type=Path,
        default=DEFAULT_LABEL_LOOKUP,
        help=f"CSV mapping label numbers to anatomical names (default: {DEFAULT_LABEL_LOOKUP}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Where to write outputs. "
            "Defaults to <formula_graph_root>/comparison_<healthy>_vs_<atrophy> "
            "when the input graphs share a formula-specific parent folder."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Number of top-drop parcels to list in the text summary (default: 25).",
    )
    parser.add_argument(
        "--submatrix-k",
        type=int,
        default=50,
        help=(
            "Number of top-drop parcels used for the adjacency submatrix "
            "comparison plot (default: 50)."
        ),
    )
    parser.add_argument(
        "--adjacency-dtype",
        choices=("auto", "float32", "float64"),
        default="auto",
        help=(
            "Data type used by adjacency_matrix.dat files. Use auto to infer "
            "each subject's dtype from file size (default: auto)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving.",
    )
    return parser.parse_args()


def default_graph_root(sim_formula: str) -> Path:
    return DEFAULT_OUTPUT_ROOT / f"wasserstein_graphs_{sim_formula}"


def resolve_graph_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.healthy_dir is not None or args.atrophy_dir is not None:
        if args.healthy_dir is None or args.atrophy_dir is None:
            raise ValueError("Provide both --healthy-dir and --atrophy-dir, or neither.")
        return args.healthy_dir, args.atrophy_dir

    if not args.healthy_subject or not args.atrophy_subject or not args.sim_formula:
        raise ValueError(
            "Provide either explicit graph dirs, or --healthy-subject, "
            "--atrophy-subject, and --sim-formula."
        )

    graph_root = default_graph_root(args.sim_formula)
    return graph_root / args.healthy_subject, graph_root / args.atrophy_subject


def default_output_dir(healthy_dir: Path, atrophy_dir: Path) -> Path:
    h_name = healthy_dir.name
    a_name = atrophy_dir.name
    if healthy_dir.parent == atrophy_dir.parent:
        return healthy_dir.parent / f"comparison_{h_name}_vs_{a_name}"
    return DEFAULT_OUTPUT_ROOT / "wasserstein_graph_comparisons" / f"comparison_{h_name}_vs_{a_name}"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_parcel_order(graph_dir: Path) -> list[str]:
    path = graph_dir / "parcel_order.txt"
    if not path.is_file():
        raise ValueError(f"Missing parcel_order.txt: {path}")
    return path.read_text(encoding="utf-8").splitlines()


def load_weighted_degree(graph_dir: Path, n_parcels: int) -> np.ndarray:
    path = graph_dir / "weighted_degree.dat"
    if not path.is_file():
        raise ValueError(f"Missing weighted_degree.dat: {path}")
    return np.memmap(path, dtype="float64", mode="r", shape=(n_parcels,))


def resolve_adjacency_dtype(path: Path, n_parcels: int, requested_dtype: str) -> str:
    if requested_dtype != "auto":
        return requested_dtype

    file_size = path.stat().st_size
    expected_float32 = n_parcels * n_parcels * np.dtype("float32").itemsize
    expected_float64 = n_parcels * n_parcels * np.dtype("float64").itemsize

    if file_size == expected_float32:
        return "float32"
    if file_size == expected_float64:
        return "float64"

    raise ValueError(
        f"Cannot infer adjacency dtype for {path}: file has {file_size} bytes, "
        f"expected {expected_float32} for float32 or {expected_float64} for float64"
    )


def load_label_lookup(csv_path: Path) -> dict[int, str]:
    lookup: dict[int, str] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lookup[int(row["label_number"])] = row["label_name"]
    return lookup


def load_submatrix(
    graph_dir: Path,
    n_parcels: int,
    row_indices: np.ndarray,
    dtype: str,
) -> np.ndarray:
    """Load only the specified rows/columns from the adjacency memmap.

    Reads K full rows (K × N) then column-slices to K × K.
    Peak extra RAM ≈ K × N × 8 bytes — ~24 MB for K=50, N=60k.
    """
    path = graph_dir / "adjacency_matrix.dat"
    if not path.is_file():
        raise ValueError(f"Missing adjacency_matrix.dat: {path}")
    resolved_dtype = resolve_adjacency_dtype(path, n_parcels, dtype)
    mm = np.memmap(path, dtype=resolved_dtype, mode="r", shape=(n_parcels, n_parcels))
    submatrix = np.asarray(mm[np.ix_(row_indices, row_indices)])
    del mm
    return submatrix


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def align_parcel_orders(
    order_a: list[str],
    order_b: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (idx_a, idx_b, common_ids) aligned to the same parcel order.

    If both orders are identical the function short-circuits without
    building any dictionaries.
    """

    if order_a == order_b:
        idx = np.arange(len(order_a), dtype=np.int64)
        return idx, idx, list(order_a)

    # Build lookup dictionaries to find positions of common parcels in each order
    # Example: pos_a["label_24/roi_42724"] = 1234 means that this parcel is at index 1234 in order_a
    pos_a = {pid: i for i, pid in enumerate(order_a)}
    pos_b = {pid: i for i, pid in enumerate(order_b)}
    common = sorted(set(pos_a) & set(pos_b))

    if not common:
        raise ValueError("No parcels in common between the two subjects.")

    # Build aligned index arrays for the common parcels
    # Example: idx_a[0] = 1234 means that the first common parcel in order_a is at index 1234
    idx_a = np.array([pos_a[pid] for pid in common], dtype=np.int64)
    idx_b = np.array([pos_b[pid] for pid in common], dtype=np.int64)
    return idx_a, idx_b, common


# ---------------------------------------------------------------------------
# Label-level statistics
# ---------------------------------------------------------------------------

def _label_number(parcel_id: str) -> int:
    """Extract integer label from a parcel id like 'label_24/roi_42724'."""
    return int(parcel_id.split("/")[0].removeprefix("label_"))


def compute_label_stats(
    parcel_ids: list[str],
    delta: np.ndarray,
    lookup: dict[int, str],
) -> list[dict]:
    groups: dict[int, list[float]] = defaultdict(list)
    for pid, d in zip(parcel_ids, delta):
        groups[_label_number(pid)].append(float(d))

    stats = []
    for label_num, values in groups.items():
        arr = np.array(values)
        stats.append(
            {
                "label_num": label_num,
                "label_name": lookup.get(label_num, f"label_{label_num}"),
                "count": len(arr),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "min": float(arr.min()),
                "max": float(arr.max()),
            }
        )
    stats.sort(key=lambda s: s["mean"])
    return stats


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

def write_summary(
    output_dir: Path,
    healthy_dir: Path,
    atrophy_dir: Path,
    parcel_ids: list[str],
    degree_h: np.ndarray,
    degree_a: np.ndarray,
    delta: np.ndarray,
    label_stats: list[dict],
    top_k: int,
) -> None:
    corr = float(np.corrcoef(degree_h, degree_a)[0, 1])
    lines = [
        f"Healthy graph: {healthy_dir}",
        f"Diseased graph: {atrophy_dir}",
        f"Parcel count: {len(parcel_ids)}",
        f"Weighted-degree correlation: {corr:.6f}",
        f"Healthy degree mean: {float(degree_h.mean()):.6f}",
        f"Diseased degree mean: {float(degree_a.mean()):.6f}",
        f"Mean degree delta (diseased - healthy): {float(delta.mean()):.6f}",
        "",
        f"Top {top_k} parcels by weighted-degree drop:",
    ]
    top_indices = np.argsort(delta)[:top_k]
    for rank, idx in enumerate(top_indices, 1):
        lines.append(
            f" {rank:>2}. {parcel_ids[idx]}"
            f" | healthy={degree_h[idx]:.6f}"
            f" | diseased={degree_a[idx]:.6f}"
            f" | delta={delta[idx]:.6f}"
        )

    lines += ["", "Label summary ordered by mean degree delta:"]
    for s in label_stats:
        lines.append(
            f"{s['label_name']}: count={s['count']}"
            f" mean={s['mean']:.6f}"
            f" median={s['median']:.6f}"
            f" min={s['min']:.6f}"
            f" max={s['max']:.6f}"
        )

    (output_dir / "comparison_summary.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_degree_scatter(
    degree_h: np.ndarray,
    degree_a: np.ndarray,
    output_dir: Path,
) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        degree_h, degree_a,
        s=1, alpha=0.3, color="#4C72B0", rasterized=True,
    )
    lo = min(float(degree_h.min()), float(degree_a.min())) - 0.02
    hi = max(float(degree_h.max()), float(degree_a.max())) + 0.02
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="identity")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Weighted degree (healthy)")
    ax.set_ylabel("Weighted degree (atrophy)")
    ax.set_title("Parcel weighted degree: healthy vs atrophy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "degree_scatter.png", dpi=200, bbox_inches="tight")
    return fig


def plot_delta_hist(delta: np.ndarray, output_dir: Path) -> plt.Figure:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(np.asarray(delta), bins=60, kde=True, ax=ax, color="#DD8452")
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Weighted degree delta (diseased − healthy)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of per-parcel degree delta")
    fig.tight_layout()
    fig.savefig(output_dir / "degree_delta_hist.png", dpi=200, bbox_inches="tight")
    return fig


def plot_label_bar(label_stats: list[dict], output_dir: Path) -> plt.Figure:
    names = [s["label_name"] for s in label_stats]
    means = [s["mean"] for s in label_stats]

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.4)))
    colors = ["#DD8452" if m < 0 else "#4C72B0" for m in means]
    ax.barh(names, means, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean degree delta (diseased − healthy)")
    ax.set_title("Per-label mean weighted-degree drop")
    fig.tight_layout()
    fig.savefig(output_dir / "label_mean_degree_delta.png", dpi=200, bbox_inches="tight")
    return fig


def plot_submatrix_comparison(
    sub_h: np.ndarray,
    sub_a: np.ndarray,
    n_parcels: int,
    output_dir: Path,
) -> plt.Figure:
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    # Use a shared color scale for the two submatrices to make differences more visually apparent
    vmin = min(float(sub_h.min()), float(sub_a.min()))
    vmax = max(float(sub_h.max()), float(sub_a.max()))
    shared_kws = dict(
        cmap="mako", vmin=vmin, vmax=vmax,
        square=True, xticklabels=False, yticklabels=False,
    )
    # The diff plot uses a diverging colormap centered at zero, with symmetric limits based on the max absolute difference
    sns.heatmap(sub_h, ax=axes[0], **shared_kws)
    axes[0].set_title("Healthy — top-K submatrix")

    sns.heatmap(sub_a, ax=axes[1], **shared_kws)
    axes[1].set_title("Atrophy — top-K submatrix")

    # The difference is computed as healthy minus atrophy,
    # so negative values indicate a drop in similarity.
    diff = sub_h - sub_a
    
    abs_max = float(np.abs(diff).max()) or 1.0
    sns.heatmap(
        diff, ax=axes[2],
        cmap="RdBu", vmin=-abs_max, vmax=abs_max,
        square=True, xticklabels=False, yticklabels=False,
    )
    axes[2].set_title("Diff (healthy − atrophy)")

    fig.suptitle(
        f"Top-{n_parcels} degree-drop parcels: adjacency submatrix comparison",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "top_degree_drop_submatrix.png",
        dpi=200, bbox_inches="tight",
    )
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    healthy_dir, atrophy_dir = resolve_graph_dirs(args)

    # Step 1 — load parcel orders and build a shared aligned index
    order_h = load_parcel_order(healthy_dir)
    order_a = load_parcel_order(atrophy_dir)
    idx_h, idx_a, parcel_ids = align_parcel_orders(order_h, order_a)

    n_h = len(order_h)
    n_a = len(order_a)
    n_common = len(parcel_ids)

    if n_common < n_h or n_common < n_a:
        print(
            f"Warning: parcel orders differ. "
            f"Using {n_common} common parcels (dropped {n_h - n_common} from healthy, "
            f"{n_a - n_common} from atrophy).",
            flush=True,
        )

    # Step 2 — load weighted degrees (memmap, no full matrix in RAM)
    degree_h = np.asarray(load_weighted_degree(healthy_dir, n_h)[idx_h], dtype=np.float64)
    degree_a = np.asarray(load_weighted_degree(atrophy_dir, n_a)[idx_a], dtype=np.float64)

    # Step 3 — delta: negative means atrophy parcel has lower degree (expected signal)
    delta = degree_a - degree_h

    # Step 4 — per-label statistics
    lookup = load_label_lookup(args.label_lookup) if args.label_lookup.is_file() else {}
    label_stats = compute_label_stats(parcel_ids, delta, lookup)

    # Step 5 — rank by largest drop (most negative delta first)
    top_k = min(args.top_k, n_common)
    submatrix_k = min(args.submatrix_k, n_common)
    top_drop_order = np.argsort(delta)  # ascending: most negative first

    # Resolve common-space indices back to per-subject positions for memmap slicing
    top_common_indices = top_drop_order[:submatrix_k]
    top_idx_h = idx_h[top_common_indices]
    top_idx_a = idx_a[top_common_indices]

    # Output directory
    out_dir = args.output_dir if args.output_dir is not None else default_output_dir(healthy_dir, atrophy_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Healthy:  {healthy_dir}  ({n_h} parcels)", flush=True)
    print(f"Atrophy:  {atrophy_dir}  ({n_a} parcels)", flush=True)
    print(f"Common parcels: {n_common}", flush=True)
    print(f"Healthy degree mean:  {degree_h.mean():.6f}", flush=True)
    print(f"Atrophy degree mean:  {degree_a.mean():.6f}", flush=True)
    print(f"Mean delta (diseased − healthy): {delta.mean():.6f}", flush=True)

    write_summary(
        out_dir, healthy_dir, atrophy_dir,
        parcel_ids, degree_h, degree_a, delta, label_stats, top_k,
    )

    figs: list[plt.Figure] = []
    figs.append(plot_degree_scatter(degree_h, degree_a, out_dir))
    figs.append(plot_delta_hist(delta, out_dir))
    figs.append(plot_label_bar(label_stats, out_dir))

    # Submatrix comparison — the only point where the full adjacency matrices
    # are touched; only K rows are read from each memmap
    print(
        f"Loading top-{submatrix_k} submatrix rows from adjacency matrices...",
        flush=True,
    )
    sub_h = load_submatrix(healthy_dir, n_h, top_idx_h, args.adjacency_dtype)
    sub_a = load_submatrix(atrophy_dir, n_a, top_idx_a, args.adjacency_dtype)
    figs.append(plot_submatrix_comparison(sub_h, sub_a, submatrix_k, out_dir))

    print(f"Outputs saved to {out_dir}", flush=True)

    if args.show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
