"""
Group-level weighted-degree visualisation: healthy vs unhealthy.

Group membership is read directly from data/healthy/ and data/unhealthy/:
the subject ID is the filename prefix before the first _ses- token.
For each subject the matching graph folder is looked up under
outputs/wasserstein_graphs_<formula>/. Subjects with no graph yet are skipped
with a warning.

Produces two figures:
  1. Side-by-side heatmap  — subjects × top-K parcels, one panel per group.
  2. Ribbon plot           — group mean ± std over the same top-K parcels.

Parcels are ranked by mean healthy-group degree (descending).

Usage:
    python scripts/graph_building/plot_group_weighted_degree.py --formula expW
    python scripts/graph_building/plot_group_weighted_degree.py --formula inv1pW --top-k 30
"""

import argparse
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as exc:
    raise SystemExit("matplotlib and seaborn are required.") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_HEALTHY   = PROJECT_ROOT / "data" / "healthy"
DATA_UNHEALTHY = PROJECT_ROOT / "data" / "unhealthy"
GRAPH_ROOT     = PROJECT_ROOT / "outputs"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "group_degree_plots"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def subject_ids_from_data_dir(data_dir: Path) -> list[str]:
    """Extract sub-XXXX from filenames like sub-0001_ses-V01_T1w-enhanced.nii.gz."""
    ids = sorted(
        f.name.split("_ses-")[0]
        for f in data_dir.glob("*.nii.gz")
    )
    return ids


def load_subject(graph_dir: Path) -> tuple[list[str], np.ndarray]:
    order = (graph_dir / "parcel_order.txt").read_text(encoding="utf-8").splitlines()
    degree = np.array(
        np.memmap(graph_dir / "weighted_degree.dat", dtype="float64", mode="r", shape=(len(order),))
    )
    return order, degree


def align_to_common(
    subjects: list[tuple[str, list[str], np.ndarray]]
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return (common_parcel_ids, {sub_id: aligned_degree_array})."""
    common = set(subjects[0][1])
    for _, order, _ in subjects[1:]:
        common &= set(order)
    common_ids = sorted(common)

    aligned: dict[str, np.ndarray] = {}
    for sub_id, order, degree in subjects:
        pos = {pid: i for i, pid in enumerate(order)}
        idx = np.array([pos[pid] for pid in common_ids], dtype=np.int64)
        aligned[sub_id] = degree[idx]

    return common_ids, aligned


def load_group(
    sub_ids: list[str],
    formula_root: Path,
    label: str,
) -> list[tuple[str, list[str], np.ndarray]]:
    loaded = []
    for sub_id in sub_ids:
        graph_dir = formula_root / sub_id
        if not (graph_dir / "weighted_degree.dat").exists():
            print(f"  [WARN] {sub_id} ({label}): no graph found at {graph_dir}, skipping")
            continue
        order, degree = load_subject(graph_dir)
        loaded.append((sub_id, order, degree))
        print(f"  loaded {sub_id}  ({label}, {len(order)} parcels)")
    return loaded


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_heatmaps(
    healthy_matrix: np.ndarray,
    unhealthy_matrix: np.ndarray,
    healthy_labels: list[str],
    unhealthy_labels: list[str],
    top_k: int,
    output_dir: Path,
) -> plt.Figure:
    n_healthy   = len(healthy_labels)
    n_unhealthy = len(unhealthy_labels)

    vmin = min(healthy_matrix.min(), unhealthy_matrix.min())
    vmax = max(healthy_matrix.max(), unhealthy_matrix.max())
    shared_kws = dict(cmap="mako", vmin=vmin, vmax=vmax,
                      xticklabels=False, cbar=False)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(max(12, top_k * 0.25), 3 + (n_healthy + n_unhealthy) * 0.45),
        gridspec_kw={"height_ratios": [n_healthy, n_unhealthy]},
        constrained_layout=True,
    )

    sns.heatmap(healthy_matrix, ax=axes[0], yticklabels=healthy_labels, **shared_kws)
    axes[0].set_title(f"Healthy — top-{top_k} parcels by mean weighted degree", fontsize=11)
    axes[0].set_ylabel("Subject")
    axes[0].tick_params(axis="y", labelsize=8)

    sns.heatmap(unhealthy_matrix, ax=axes[1], yticklabels=unhealthy_labels, **shared_kws)
    axes[1].set_title("Unhealthy — same parcel order", fontsize=11)
    axes[1].set_ylabel("Subject")
    axes[1].set_xlabel("Parcel rank (1 = highest mean healthy degree)")
    axes[1].tick_params(axis="y", labelsize=8)

    fig.colorbar(
        axes[0].collections[0], ax=axes,
        orientation="vertical", fraction=0.015, pad=0.02,
        label="Weighted degree",
    )
    fig.suptitle("Group weighted-degree heatmap", fontsize=13)

    out = output_dir / "group_heatmap.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  saved {out}")
    return fig


def plot_ribbon(
    healthy_matrix: np.ndarray,
    unhealthy_matrix: np.ndarray,
    top_k: int,
    output_dir: Path,
) -> plt.Figure:
    x      = np.arange(1, top_k + 1)
    h_mean = healthy_matrix.mean(axis=0)
    h_std  = healthy_matrix.std(axis=0)
    u_mean = unhealthy_matrix.mean(axis=0)
    u_std  = unhealthy_matrix.std(axis=0)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.22), 5))

    ax.plot(x, h_mean, color="#4C72B0", label="Healthy (mean)", linewidth=2)
    ax.fill_between(x, h_mean - h_std, h_mean + h_std,
                    color="#4C72B0", alpha=0.2, label="Healthy ±1 SD")

    ax.plot(x, u_mean, color="#DD8452", label="Unhealthy (mean)", linewidth=2)
    ax.fill_between(x, u_mean - u_std, u_mean + u_std,
                    color="#DD8452", alpha=0.2, label="Unhealthy ±1 SD")

    ax.set_xlabel(f"Parcel rank (1 = highest mean healthy degree, top {top_k} shown)")
    ax.set_ylabel("Weighted degree")
    ax.set_title("Group mean ± SD weighted degree — healthy vs unhealthy")
    ax.legend()
    fig.tight_layout()

    out = output_dir / "group_ribbon.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"  saved {out}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot group-level weighted degree for healthy vs unhealthy subjects."
    )
    parser.add_argument(
        "--formula", choices=("expW", "inv1pW"), required=True,
        help="Similarity formula; selects outputs/wasserstein_graphs_<formula>/.",
    )
    parser.add_argument(
        "--top-k", type=int, default=50, metavar="K",
        help="Number of top parcels to display (default: 50).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Where to save figures (default: outputs/group_degree_plots/<formula>/).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Display figures interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formula_root = GRAPH_ROOT / f"wasserstein_graphs_{args.formula}"
    out_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.formula
    out_dir.mkdir(parents=True, exist_ok=True)

    healthy_ids   = subject_ids_from_data_dir(DATA_HEALTHY)
    unhealthy_ids = subject_ids_from_data_dir(DATA_UNHEALTHY)

    print(f"Healthy subjects in data/healthy/   ({len(healthy_ids)}): {healthy_ids}")
    print(f"Unhealthy subjects in data/unhealthy/ ({len(unhealthy_ids)}): {unhealthy_ids}\n")

    healthy_subjects   = load_group(healthy_ids,   formula_root, "Healthy")
    unhealthy_subjects = load_group(unhealthy_ids, formula_root, "Unhealthy")

    if not healthy_subjects:
        raise SystemExit("No healthy subjects have graphs yet.")
    if not unhealthy_subjects:
        raise SystemExit("No unhealthy subjects have graphs yet.")

    # Align all subjects to a common parcel set
    common_ids, aligned = align_to_common(healthy_subjects + unhealthy_subjects)
    print(f"\nCommon parcels: {len(common_ids)}")

    # Rank by mean healthy degree (descending), take top-K
    healthy_stack = np.stack([aligned[s[0]] for s in healthy_subjects])
    mean_healthy  = healthy_stack.mean(axis=0)
    top_k         = min(args.top_k, len(common_ids))
    top_idx       = np.argsort(mean_healthy)[::-1][:top_k]

    healthy_matrix   = healthy_stack[:, top_idx]
    unhealthy_matrix = np.stack([aligned[s[0]] for s in unhealthy_subjects])[:, top_idx]

    healthy_labels   = [s[0] for s in healthy_subjects]
    unhealthy_labels = [s[0] for s in unhealthy_subjects]

    figs = []
    figs.append(plot_heatmaps(
        healthy_matrix, unhealthy_matrix,
        healthy_labels, unhealthy_labels,
        top_k, out_dir,
    ))
    figs.append(plot_ribbon(healthy_matrix, unhealthy_matrix, top_k, out_dir))

    print(f"\nAll outputs saved to {out_dir}")

    if args.show:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
