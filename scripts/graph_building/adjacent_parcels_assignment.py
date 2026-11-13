from pathlib import Path
from time import monotonic
import igraph as ig
import nibabel as nib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROIS_DIR = PROJECT_ROOT / "outputs/atlases/atlas-8cf3dc8f/rois"
PARCEL_ORDER_PATH = PROJECT_ROOT / (
    "outputs/cohorts/oasis3/runs/connected-atlas-8cf3dc8f/"
    "parcel_vectors/sub-0001/parcel_order.txt"
)
REFERENCE_PATH = ROIS_DIR / "3/roi_0000.nii.gz"
CACHE_DIR = PROJECT_ROOT / "outputs/atlases/atlas-8cf3dc8f/parcel_volume_cache"
GRAPH_PATH = CACHE_DIR / "parcel_adjacency.graphml"
PROGRESS_EVERY = 1000


def format_duration(seconds: float) -> str:
    """Format a duration as HH:MM:SS for progress output."""
    minutes, seconds = divmod(max(0, round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parcel_mask_path(rois_dir: Path, parcel_id: str) -> Path:
    """Translate 'label_3/roi_0000' into '<rois_dir>/3/roi_0000.nii.gz'."""
    label_name, roi_name = parcel_id.split("/", maxsplit=1)
    label = label_name.removeprefix("label_")
    return rois_dir / label / f"{roi_name}.nii.gz"


def load_parcel_volume(
    rois_dir: Path,
    parcel_order_path: Path,
    reference_path: Path,
    progress_every: int = 1000,
) -> tuple[list[str], np.ndarray]:
    """Load the ordered parcel masks into one integer atlas volume."""
    parcel_ids = parcel_order_path.read_text(encoding="utf-8").splitlines()
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative")

    total_parcels = len(parcel_ids)
    started_at = monotonic()
    if progress_every:
        print(f"Loading {total_parcels:,} parcel masks...", flush=True)
    parcel_paths = [parcel_mask_path(rois_dir, parcel_id) for parcel_id in parcel_ids]
    missing_path = next((path for path in parcel_paths if not path.is_file()), None)
    if missing_path is not None:
        raise FileNotFoundError(
            "parcel_order.txt does not match the selected ROI atlas; "
            f"missing mask: {missing_path}"
        )

    reference = nib.load(reference_path)
    parcel_volume = np.zeros(reference.shape, dtype=np.int32)

    for parcel_index, (parcel_id, mask_path) in enumerate(
        zip(parcel_ids, parcel_paths), start=1
    ):
        mask_image = nib.load(mask_path)

        if mask_image.shape != reference.shape:
            raise ValueError(f"Wrong mask shape: {parcel_id}")
        if not np.allclose(mask_image.affine, reference.affine):
            raise ValueError(f"Wrong mask affine: {parcel_id}")

        parcel_mask = np.asanyarray(mask_image.dataobj) != 0
        if not np.any(parcel_mask):
            raise ValueError(f"Empty parcel mask: {parcel_id}")
        if np.any(parcel_volume[parcel_mask] != 0):
            raise ValueError(f"Parcel overlap detected: {parcel_id}")

        # Zero remains background; stored ID N + 1 matches subject-vector row N.
        parcel_volume[parcel_mask] = parcel_index

        if progress_every and (
            parcel_index % progress_every == 0 or parcel_index == total_parcels
        ):
            elapsed = monotonic() - started_at
            rate = parcel_index / elapsed if elapsed else 0.0
            remaining = total_parcels - parcel_index
            eta = remaining / rate if rate else 0.0
            percentage = 100.0 * parcel_index / total_parcels
            print(
                f"Loaded {parcel_index:,}/{total_parcels:,} "
                f"({percentage:5.1f}%) | elapsed {format_duration(elapsed)} "
                f"| {rate:.1f} masks/s | ETA {format_duration(eta)}",
                flush=True,
            )

    return parcel_ids, parcel_volume


def save_parcel_volume(
    parcel_ids: list[str],
    parcel_volume: np.ndarray,
    reference_path: Path,
    cache_dir: Path,
) -> tuple[Path, Path]:
    """Save a combined parcel volume and its label-to-parcel mapping."""
    if parcel_volume.ndim != 3:
        raise ValueError("parcel_volume must be a 3D array")
    if not np.issubdtype(parcel_volume.dtype, np.integer):
        raise ValueError("parcel_volume must contain integer parcel labels")
    if any(not parcel_id or "\n" in parcel_id for parcel_id in parcel_ids):
        raise ValueError("parcel_ids cannot be empty or contain newlines")
    if len(set(parcel_ids)) != len(parcel_ids):
        raise ValueError("parcel_ids must be unique")

    parcel_labels = np.unique(parcel_volume)
    parcel_labels = parcel_labels[parcel_labels != 0]
    expected_labels = np.arange(1, len(parcel_ids) + 1)

    if not np.array_equal(parcel_labels, expected_labels):
        raise ValueError("parcel_volume labels must match parcel_ids in order")

    reference = nib.load(reference_path)
    if parcel_volume.shape != reference.shape:
        raise ValueError("parcel_volume shape does not match the reference image")

    cache_dir.mkdir(parents=True, exist_ok=True)
    volume_path = cache_dir / "parcel_volume.nii.gz"
    order_path = cache_dir / "parcel_order.txt"

    header = reference.header.copy()
    header.set_data_dtype(np.int32)
    cached_image = nib.Nifti1Image(
        parcel_volume.astype(np.int32, copy=False),
        reference.affine,
        header,
    )
    nib.save(cached_image, volume_path)
    order_path.write_text("\n".join(parcel_ids) + "\n", encoding="utf-8")
    return volume_path, order_path


def load_saved_parcel_volume(cache_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load and validate a parcel volume previously saved in ``cache_dir``."""
    volume_path = cache_dir / "parcel_volume.nii.gz"
    order_path = cache_dir / "parcel_order.txt"

    parcel_ids = order_path.read_text(encoding="utf-8").splitlines()
    cached_image = nib.load(volume_path)
    if not np.issubdtype(cached_image.get_data_dtype(), np.integer):
        raise ValueError("Saved parcel volume must contain integer labels")

    parcel_volume = np.asarray(cached_image.dataobj, dtype=np.int32)
    if parcel_volume.ndim != 3:
        raise ValueError("Saved parcel volume must be a 3D array")

    parcel_labels = np.unique(parcel_volume)
    parcel_labels = parcel_labels[parcel_labels != 0]
    expected_labels = np.arange(1, len(parcel_ids) + 1)
    if not np.array_equal(parcel_labels, expected_labels):
        raise ValueError("Saved parcel labels do not match parcel_order.txt")

    return parcel_ids, parcel_volume


def build_parcel_adjacency_graph(parcel_volume: np.ndarray) -> ig.Graph:
    """Build an undirected graph for parcels sharing a voxel face."""
    if parcel_volume.ndim != 3:
        raise ValueError("parcel_volume must be a 3D array")
    if not np.issubdtype(parcel_volume.dtype, np.integer):
        raise ValueError("parcel_volume must contain integer parcel labels")
    if np.any(parcel_volume < 0):
        raise ValueError("parcel_volume cannot contain negative parcel labels")

    parcel_labels = np.unique(parcel_volume)
    parcel_labels = parcel_labels[parcel_labels != 0]

    # load_parcel_volume assigns consecutive labels 1, 2, ..., N. Enforcing
    # that invariant makes vertex index N - 1 unambiguously represent label N.
    expected_labels = np.arange(1, len(parcel_labels) + 1)

    if not np.array_equal(parcel_labels, expected_labels):
        raise ValueError("Parcel labels must be consecutive integers starting at 1")

    edges: set[tuple[int, int]] = set()

    # Only the positive directions are needed for an undirected graph. Each
    # shared voxel face is inspected once instead of once from each side.
    neighbor_offsets = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
    ]

    size_x, size_y, size_z = parcel_volume.shape

    for x in range(size_x):
        for y in range(size_y):
            for z in range(size_z):
                current_parcel = int(parcel_volume[x, y, z])

                if current_parcel == 0:
                    continue

                for offset_x, offset_y, offset_z in neighbor_offsets:
                    neighbor_x = x + offset_x
                    neighbor_y = y + offset_y
                    neighbor_z = z + offset_z

                    if (
                        neighbor_x >= size_x
                        or neighbor_y >= size_y
                        or neighbor_z >= size_z
                    ):
                        continue

                    neighbor_parcel = int(
                        parcel_volume[neighbor_x, neighbor_y, neighbor_z]
                    )

                    if (
                        neighbor_parcel == 0
                        or neighbor_parcel == current_parcel
                    ):
                        continue

                    # Normalize the endpoint order and convert labels 1...N to
                    # igraph vertex indices 0...N-1.
                    edge = tuple(
                        sorted((current_parcel - 1, neighbor_parcel - 1))
                    )
                    edges.add(edge)

    # n= retains parcels that have no adjacency edges.
    graph = ig.Graph(n=len(parcel_labels), edges=sorted(edges), directed=False)
    graph.vs["parcel_label"] = parcel_labels.tolist()
    return graph


def main() -> None:
    """Build or reuse the parcel volume, then save its adjacency graph."""
    cached_volume_path = CACHE_DIR / "parcel_volume.nii.gz"
    cached_order_path = CACHE_DIR / "parcel_order.txt"
    volume_is_cached = cached_volume_path.is_file()
    order_is_cached = cached_order_path.is_file()

    if volume_is_cached and order_is_cached:
        print(f"Loading cached parcel volume from {CACHE_DIR}...", flush=True)
        parcel_ids, parcel_volume = load_saved_parcel_volume(CACHE_DIR)
    elif volume_is_cached or order_is_cached:
        raise RuntimeError(
            f"Incomplete parcel cache in {CACHE_DIR}: expected both "
            "parcel_volume.nii.gz and parcel_order.txt"
        )
    else:
        parcel_ids, parcel_volume = load_parcel_volume(
            ROIS_DIR,
            PARCEL_ORDER_PATH,
            REFERENCE_PATH,
            progress_every=PROGRESS_EVERY,
        )
        print(f"Saving parcel-volume cache to {CACHE_DIR}...", flush=True)
        volume_path, order_path = save_parcel_volume(
            parcel_ids,
            parcel_volume,
            REFERENCE_PATH,
            CACHE_DIR,
        )
        print(f"Saved {volume_path}", flush=True)
        print(f"Saved {order_path}", flush=True)

    print("Building parcel adjacency graph...", flush=True)
    graph = build_parcel_adjacency_graph(parcel_volume)
    graph.vs["parcel_id"] = parcel_ids

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    graph.write_graphml(str(GRAPH_PATH))

    print(f"Parcels: {graph.vcount():,}", flush=True)
    print(f"Adjacency edges: {graph.ecount():,}", flush=True)
    print(f"Saved graph to {GRAPH_PATH}", flush=True)


if __name__ == "__main__":
    main()
