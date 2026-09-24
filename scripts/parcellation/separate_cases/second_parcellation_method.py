"""
Build a connected approximately equal-size parcel atlas.

Each approximately equal-sized parcel is written as a NIfTI mask under
<output-root>/<roi-label>/.
Labels are processed sequentially; NIfTI writing within each label is parallel.

Usage:
    python scripts/parcellation/separate_cases/second_roi_test.py [options]

Parameters:
    --roi-label INT          Optional label; repeatable (default: all retained labels).
    --parcel-size INT        Target voxels per parcel (default: 15).
    --segmentation PATH      Input segmentation image.
    --output-root PATH       Root (default: outputs/test_parcellation/rois).
    --workers INT            Parallel writers (default: 75% of detected CPUs).
    --skip-neighbor-check    Skip connectivity diagnostics.

Examples:
    python scripts/parcellation/separate_cases/second_roi_test.py
    python scripts/parcellation/separate_cases/second_roi_test.py --roi-label 24 --validate-only
"""

import argparse
import heapq
import math
import os
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from atlas_labels import KEEP_LABELS, LABEL_DICT
from pipeline_integrity import atomic_write_json, file_signature, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SEGMENTATION = PROJECT_ROOT / "data" / "reference" / "MNI152_T1_1mm_seg.nii.gz"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "test_parcellation" / "rois"
DEFAULT_WORKERS = max(1, math.floor((os.cpu_count() or 1) * 0.75))
NEIGHBOR_DELTAS = np.array(
    [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ],
    dtype=np.int16,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a validated 6-connected parcel atlas."
    )
    parser.add_argument(
        "--roi-label",
        type=int,
        action="append",
        default=None,
        help=(
            "Segmentation label to parcelize. Repeat to select several labels. "
            "When omitted, every retained label is processed sequentially."
        ),
    )
    parser.add_argument(
        "--parcel-size",
        type=int,
        default=15,
        help="Target number of voxels per sub-parcel.",
    )
    parser.add_argument(
        "--segmentation",
        type=Path,
        default=DEFAULT_SEGMENTATION,
        help="Path to the input segmentation image.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where ROI-specific parcel folders will be written.",
    )
    parser.add_argument(
        "--skip-neighbor-check",
        action="store_true",
        help="Compatibility option; mandatory final connectivity validation still runs.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Parallel NIfTI writer processes used within each label "
            f"(default: 75%% of detected CPUs = {DEFAULT_WORKERS})."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build and validate parcels in memory without writing NIfTI masks.",
    )
    return parser.parse_args()


def six_connectivity_structure():
    """Build the 3×3×3 binary structuring element for 6-connectivity (face-adjacent only)."""
    structure = np.zeros((3, 3, 3), dtype=np.uint8)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = 1
    structure[2, 1, 1] = 1
    structure[1, 0, 1] = 1
    structure[1, 2, 1] = 1
    structure[1, 1, 0] = 1
    structure[1, 1, 2] = 1
    return structure


def load_segmentation(segmentation_path):
    """Load a segmentation NIfTI; casts float data to int64 when values are integer-valued."""
    template_img = nib.load(str(segmentation_path))
    seg_image = np.asanyarray(template_img.dataobj)

    if np.issubdtype(seg_image.dtype, np.floating):
        rounded = np.rint(seg_image)
        if np.allclose(seg_image, rounded, atol=1e-6):
            seg_image = rounded.astype(np.int64, copy=False)

    return template_img, seg_image


def write_nifti(roi_indices, sub_roi_index, template_img, output_dir):
    """Write voxel coordinates of one sub-parcel as a uint8 binary NIfTI mask."""
    roi_indices = np.asarray(roi_indices, dtype=np.int32)
    if roi_indices.size == 0:
        raise ValueError(f"Sub-parcel {sub_roi_index} is empty and will not be written")

    empty_image = np.zeros(template_img.shape, dtype=np.uint8)
    empty_image[tuple(roi_indices.T)] = 1

    header = template_img.header.copy()
    header.set_data_dtype(np.uint8)
    nii_ = nib.Nifti1Image(empty_image, affine=template_img.affine, header=header)
    output_path = output_dir / f"roi_{sub_roi_index:04d}.nii.gz"
    temporary = output_dir / f".{output_path.name}.{os.getpid()}.tmp.nii.gz"
    try:
        nii_.to_filename(str(temporary))
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path


_WRITER_TEMPLATE_IMAGE = None
_WRITER_OUTPUT_DIR = None


def initialize_writer(segmentation_path, output_dir):
    """Load immutable NIfTI geometry once in each writer process."""
    global _WRITER_TEMPLATE_IMAGE, _WRITER_OUTPUT_DIR
    _WRITER_TEMPLATE_IMAGE = nib.load(str(segmentation_path))
    _WRITER_OUTPUT_DIR = Path(output_dir)


def write_parcel_worker(task):
    """Write one parcel in a worker and return its validated manifest record."""
    parcel_index, parcel_coords, target_size = task
    output_path = write_nifti(
        parcel_coords,
        parcel_index,
        _WRITER_TEMPLATE_IMAGE,
        _WRITER_OUTPUT_DIR,
    )
    return {
        "file": output_path.name,
        "voxel_count": int(len(parcel_coords)),
        "target_voxel_count": int(target_size),
        "component_count_6": 1,
        "sha256": sha256_file(output_path),
    }


def write_parcels_parallel(all_parcels, all_targets, segmentation_path, output_dir, workers):
    """Write one label with a bounded process pool, preserving parcel order."""
    tasks = (
        (parcel_index, parcel_coords, target_size)
        for parcel_index, (parcel_coords, target_size) in enumerate(
            zip(all_parcels, all_targets)
        )
    )
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_writer,
        initargs=(segmentation_path, output_dir),
    ) as executor:
        return list(executor.map(write_parcel_worker, tasks, chunksize=8))


def parcel_targets(n_voxels, n_parcels):
    """Distribute n_voxels as evenly as possible across n_parcels, giving remainder to the first parcels."""
    base_size = n_voxels // n_parcels
    remainder = n_voxels % n_parcels
    targets = np.full(n_parcels, base_size, dtype=np.int32)
    targets[:remainder] += 1
    return targets


def allocate_parcels_to_components(component_sizes, parcel_size):
    """Assign parcels to connected components proportionally, ensuring at least one per component."""
    n_components = len(component_sizes)
    total_voxels = int(component_sizes.sum())
    total_parcels = max(n_components, int(np.ceil(total_voxels / parcel_size)))

    allocation = np.ones(n_components, dtype=np.int32)
    for _ in range(total_parcels - n_components):
        splittable = allocation < component_sizes
        scores = np.where(splittable, component_sizes / allocation, -np.inf)
        best_component = int(np.argmax(scores))
        if not np.isfinite(scores[best_component]):
            break
        allocation[best_component] += 1

    return allocation


def choose_seed_indices(coords, n_seeds):
    """Select n_seeds voxel indices using a maximin spread: each seed maximises its distance from all prior seeds."""
    if n_seeds == 1:
        return np.array([0], dtype=np.int32)

    centroid = coords.mean(axis=0, dtype=np.float64)
    distances_to_centroid = np.sum((coords - centroid) ** 2, axis=1)
    first_seed = int(np.argmin(distances_to_centroid))

    seeds = [first_seed]
    min_distances = np.sum((coords - coords[first_seed]) ** 2, axis=1)

    for _ in range(1, n_seeds):
        next_seed = int(np.argmax(min_distances))
        seeds.append(next_seed)
        next_distances = np.sum((coords - coords[next_seed]) ** 2, axis=1)
        min_distances = np.minimum(min_distances, next_distances)

    return np.asarray(seeds, dtype=np.int32)


def build_adjacency(coords):
    """Build a per-voxel adjacency list using 6-connectivity (face neighbors only)."""
    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(coords.tolist())}
    adjacency = [[] for _ in range(len(coords))]

    for idx, (x, y, z) in enumerate(coords):
        for dx, dy, dz in NEIGHBOR_DELTAS:
            neighbor_index = coord_to_index.get((x + dx, y + dy, z + dz))
            if neighbor_index is not None:
                adjacency[idx].append(neighbor_index)

    return adjacency


def grow_connected_parcels(adjacency, targets, seeds):
    """BFS region-growing from seeds: first respects per-parcel targets, then fills overflow voxels."""
    n_parcels = len(targets)
    owners = np.full(len(adjacency), -1, dtype=np.int32)
    parcel_sizes = np.zeros(n_parcels, dtype=np.int32)
    frontiers = [deque() for _ in range(n_parcels)]

    for parcel_id, seed_index in enumerate(seeds):
        owners[seed_index] = parcel_id
        parcel_sizes[parcel_id] = 1

    for parcel_id, seed_index in enumerate(seeds):
        for neighbor_index in adjacency[seed_index]:
            if owners[neighbor_index] == -1:
                frontiers[parcel_id].append(neighbor_index)

    def expand_once(parcel_id, respect_target):
        if respect_target and parcel_sizes[parcel_id] >= targets[parcel_id]:
            return False

        frontier = frontiers[parcel_id]
        while frontier and owners[frontier[0]] != -1:
            frontier.popleft()
        if not frontier:
            return False

        voxel_index = frontier.popleft()
        if owners[voxel_index] != -1:
            return False

        owners[voxel_index] = parcel_id
        parcel_sizes[parcel_id] += 1
        for neighbor_index in adjacency[voxel_index]:
            if owners[neighbor_index] == -1:
                frontiers[parcel_id].append(neighbor_index)
        return True

    # Phase 1: expand parcels in order of their fill ratio (most underfilled first)
    # until every parcel has reached its target or no further progress is possible.
    while np.any(owners == -1):
        progress = False
        deficit_order = np.argsort(parcel_sizes / targets)

        for parcel_id in deficit_order:
            progress = expand_once(int(parcel_id), respect_target=True) or progress

        if not progress:
            break

        if np.all(parcel_sizes >= targets):
            break

    # Phase 2: claim remaining unassigned voxels without size constraints,
    # prioritising the smallest parcel to balance sizes as much as possible.
    while np.any(owners == -1):
        progress = False
        size_order = np.argsort(parcel_sizes)

        for parcel_id in size_order:
            progress = expand_once(int(parcel_id), respect_target=False) or progress

        if not progress:
            break

    if np.any(owners == -1):
        n_unassigned = int(np.count_nonzero(owners == -1))
        raise RuntimeError(
            f"Failed to assign {n_unassigned} voxels while growing {n_parcels} parcels"
        )

    return owners


def build_parcel_members(owners, n_parcels):
    """Return a list of sets where each set contains the voxel indices owned by that parcel."""
    return [set(np.flatnonzero(owners == parcel_id)) for parcel_id in range(n_parcels)]


def parcel_adjacency(owners, adjacency, n_parcels):
    """Return a list of sets of neighbouring parcel IDs for each parcel."""
    parcel_neighbors = [set() for _ in range(n_parcels)]

    for voxel_index, neighbors in enumerate(adjacency):
        source_parcel = int(owners[voxel_index])
        for neighbor_index in neighbors:
            target_parcel = int(owners[neighbor_index])
            if source_parcel != target_parcel:
                parcel_neighbors[source_parcel].add(target_parcel)

    return parcel_neighbors


def is_connected_after_removal(members, remove_index, adjacency):
    """Return True if the parcel stays connected when remove_index is removed (BFS check)."""
    if len(members) <= 1:
        return False

    remaining = members - {remove_index}
    start = next(iter(remaining))
    queue = deque([start])
    seen = {start}

    while queue:
        voxel_index = queue.popleft()
        for neighbor_index in adjacency[voxel_index]:
            if neighbor_index == remove_index:
                continue
            if neighbor_index in remaining and neighbor_index not in seen:
                seen.add(neighbor_index)
                queue.append(neighbor_index)

    return len(seen) == len(remaining)


def find_transfer_voxel(source_parcel, target_parcel, owners, parcel_members, adjacency):
    """Find a boundary voxel in source_parcel that touches target_parcel and can be moved without disconnecting source."""
    candidates = []
    for voxel_index in parcel_members[source_parcel]:
        touch_target = any(owners[neighbor] == target_parcel for neighbor in adjacency[voxel_index])
        if touch_target:
            target_contacts = sum(owners[neighbor] == target_parcel for neighbor in adjacency[voxel_index])
            candidates.append((target_contacts, voxel_index))

    candidates.sort(reverse=True)
    for _, voxel_index in candidates:
        if is_connected_after_removal(parcel_members[source_parcel], voxel_index, adjacency):
            return voxel_index

    return None


def shortest_surplus_path(start_parcel, parcel_neighbors, parcel_sizes, targets):
    """BFS over the parcel graph from start_parcel to the nearest parcel that has surplus voxels."""
    queue = deque([(start_parcel, [start_parcel])])
    visited = {start_parcel}

    while queue:
        parcel_id, path = queue.popleft()
        if parcel_sizes[parcel_id] > targets[parcel_id]:
            return path

        for neighbor_id in parcel_neighbors[parcel_id]:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append((neighbor_id, path + [neighbor_id]))

    return None


def members_are_connected(members, adjacency):
    """Return whether a non-empty set of voxel indices is connected."""
    if not members:
        return False
    if len(members) == 1:
        return True

    start = min(members)
    queue = deque([start])
    seen = {start}
    while queue:
        voxel_index = queue.popleft()
        for neighbor_index in adjacency[voxel_index]:
            if neighbor_index in members and neighbor_index not in seen:
                seen.add(neighbor_index)
                queue.append(neighbor_index)
    return len(seen) == len(members)


def connected_prefix(members, start, size, adjacency, preferred, reverse=False):
    """Grow a deterministic connected prefix of the requested size."""
    selected = set()
    queued = {start}
    queue = deque([start])

    while queue and len(selected) < size:
        voxel_index = queue.popleft()
        selected.add(voxel_index)
        neighbors = [
            neighbor
            for neighbor in adjacency[voxel_index]
            if neighbor in members and neighbor not in queued
        ]
        neighbors.sort(
            key=lambda neighbor: (neighbor not in preferred, neighbor),
            reverse=reverse,
        )
        for neighbor in neighbors:
            queued.add(neighbor)
            queue.append(neighbor)

    return selected


def find_connected_bipartition(combined_members, first_members, first_size, adjacency):
    """Split a small connected union into two connected sets of exact sizes."""
    if first_size < 1 or first_size >= len(combined_members):
        return None

    # Starting inside the parcel that receives a voxel keeps repairs local. If
    # that cannot produce a valid cut, try every vertex in the small union.
    starts = sorted(first_members) + sorted(combined_members - first_members)
    for preferred in (first_members, combined_members):
        for reverse in (False, True):
            for start in starts:
                first = connected_prefix(
                    combined_members,
                    start,
                    first_size,
                    adjacency,
                    preferred,
                    reverse=reverse,
                )
                if len(first) != first_size:
                    continue
                second = combined_members - first
                if members_are_connected(second, adjacency):
                    return first, second
    return None


def transfer_voxels(
    source_parcel,
    destination_parcel,
    amount,
    owners,
    parcel_members,
    parcel_sizes,
    adjacency,
):
    """Move several voxels, locally repartitioning the adjacent pair if needed."""
    if amount < 1 or amount >= parcel_sizes[source_parcel]:
        return False

    saved_source = set(parcel_members[source_parcel])
    saved_destination = set(parcel_members[destination_parcel])
    saved_source_size = int(parcel_sizes[source_parcel])
    saved_destination_size = int(parcel_sizes[destination_parcel])

    direct_success = True
    for _ in range(amount):
        voxel_index = find_transfer_voxel(
            source_parcel,
            destination_parcel,
            owners,
            parcel_members,
            adjacency,
        )
        if voxel_index is None:
            direct_success = False
            break
        owners[voxel_index] = destination_parcel
        parcel_members[source_parcel].remove(voxel_index)
        parcel_members[destination_parcel].add(voxel_index)
        parcel_sizes[source_parcel] -= 1
        parcel_sizes[destination_parcel] += 1
    if direct_success:
        return True

    parcel_members[source_parcel] = saved_source
    parcel_members[destination_parcel] = saved_destination
    parcel_sizes[source_parcel] = saved_source_size
    parcel_sizes[destination_parcel] = saved_destination_size
    owners[np.fromiter(saved_source, dtype=np.int64)] = source_parcel
    owners[np.fromiter(saved_destination, dtype=np.int64)] = destination_parcel

    combined = saved_source | saved_destination
    split = find_connected_bipartition(
        combined,
        saved_destination,
        saved_destination_size + amount,
        adjacency,
    )
    if split is None:
        return False

    destination_members, source_members = split
    parcel_members[destination_parcel] = destination_members
    parcel_members[source_parcel] = source_members
    owners[np.fromiter(destination_members, dtype=np.int64)] = destination_parcel
    owners[np.fromiter(source_members, dtype=np.int64)] = source_parcel
    parcel_sizes[destination_parcel] = len(destination_members)
    parcel_sizes[source_parcel] = len(source_members)
    return True


def candidate_surplus_paths(
    start_parcel,
    parcel_neighbors,
    parcel_sizes,
    targets,
    max_paths=32,
):
    """Return shortest candidate paths from a deficit to nearby surplus parcels."""
    queue = deque([start_parcel])
    predecessors = {start_parcel: None}
    paths = []

    while queue and len(paths) < max_paths:
        parcel_id = queue.popleft()
        if parcel_id != start_parcel and parcel_sizes[parcel_id] > targets[parcel_id]:
            path = []
            cursor = parcel_id
            while cursor is not None:
                path.append(cursor)
                cursor = predecessors[cursor]
            paths.append(list(reversed(path)))
            continue

        for neighbor_id in sorted(parcel_neighbors[parcel_id]):
            if neighbor_id not in predecessors:
                predecessors[neighbor_id] = parcel_id
                queue.append(neighbor_id)

    return paths


def try_transfer_path(path, amount, owners, parcel_members, parcel_sizes, adjacency):
    """Transactionally propagate an amount from a surplus to a deficit."""
    affected = set(path)
    saved_members = {
        parcel_id: set(parcel_members[parcel_id])
        for parcel_id in affected
    }
    saved_sizes = {
        parcel_id: int(parcel_sizes[parcel_id])
        for parcel_id in affected
    }

    for path_index in range(len(path) - 1, 0, -1):
        source_parcel = path[path_index]
        destination_parcel = path[path_index - 1]
        if transfer_voxels(
            source_parcel,
            destination_parcel,
            amount,
            owners,
            parcel_members,
            parcel_sizes,
            adjacency,
        ):
            continue

        for parcel_id, members in saved_members.items():
            parcel_members[parcel_id] = members
            parcel_sizes[parcel_id] = saved_sizes[parcel_id]
            owners[np.fromiter(members, dtype=np.int64)] = parcel_id
        return False

    return True


def rebalance_connected_parcels(owners, adjacency, targets):
    """Reach exact targets using transactional, connectivity-preserving repairs."""
    n_parcels = len(targets)
    parcel_members = build_parcel_members(owners, n_parcels)
    parcel_sizes = np.array([len(members) for members in parcel_members], dtype=np.int32)

    while True:
        deficits = np.flatnonzero(parcel_sizes < targets)
        if deficits.size == 0:
            break

        progress = False
        parcel_neighbors = parcel_adjacency(owners, adjacency, n_parcels)
        deficit_order = sorted(deficits, key=lambda parcel_id: targets[parcel_id] - parcel_sizes[parcel_id], reverse=True)

        for target_parcel in deficit_order:
            if parcel_sizes[target_parcel] >= targets[target_parcel]:
                continue
            while parcel_sizes[target_parcel] < targets[target_parcel]:
                paths = candidate_surplus_paths(
                    int(target_parcel),
                    parcel_neighbors,
                    parcel_sizes,
                    targets,
                )
                moved = False
                deficit = int(targets[target_parcel] - parcel_sizes[target_parcel])
                for path in paths:
                    donor = path[-1]
                    donor_surplus = int(parcel_sizes[donor] - targets[donor])
                    max_amount = min(deficit, donor_surplus)
                    for amount in range(max_amount, 0, -1):
                        if try_transfer_path(
                            path,
                            amount,
                            owners,
                            parcel_members,
                            parcel_sizes,
                            adjacency,
                        ):
                            moved = True
                            break
                    if moved:
                        break
                if not moved:
                    break
                progress = True

        if not progress:
            break

    if not np.array_equal(parcel_sizes, targets):
        deficits = np.flatnonzero(parcel_sizes < targets)
        surpluses = np.flatnonzero(parcel_sizes > targets)
        largest_deficit = int(np.max(targets[deficits] - parcel_sizes[deficits])) if len(deficits) else 0
        largest_surplus = int(np.max(parcel_sizes[surpluses] - targets[surpluses])) if len(surpluses) else 0
        raise RuntimeError(
            "Could not reach connected parcel targets: "
            f"{len(deficits)} deficits (largest={largest_deficit}), "
            f"{len(surpluses)} surpluses (largest={largest_surplus})"
        )

    return owners


def coarsen_connected_graph(adjacency, n_parcels):
    """Merge adjacent voxel clusters until exactly n_parcels remain."""
    n_voxels = len(adjacency)
    parents = np.arange(n_voxels, dtype=np.int32)
    sizes = np.ones(n_voxels, dtype=np.int32)
    rng = np.random.default_rng(4)
    masses = rng.lognormal(mean=0.0, sigma=0.025, size=n_voxels)
    neighbors = [set(voxel_neighbors) for voxel_neighbors in adjacency]
    heap = [
        (masses[left] + masses[right], left, right)
        for left, voxel_neighbors in enumerate(adjacency)
        for right in voxel_neighbors
        if left < right
    ]
    heapq.heapify(heap)
    cluster_count = n_voxels

    def find_root(voxel_index):
        while parents[voxel_index] != voxel_index:
            parents[voxel_index] = parents[parents[voxel_index]]
            voxel_index = int(parents[voxel_index])
        return voxel_index

    while cluster_count > n_parcels:
        while heap:
            stored_mass, left, right = heapq.heappop(heap)
            left = find_root(left)
            right = find_root(right)
            current_mass = masses[left] + masses[right]
            if (
                left != right
                and right in neighbors[left]
                and abs(stored_mass - current_mass) <= 1e-12
            ):
                break
        else:
            raise RuntimeError(
                f"Connected coarsening stopped at {cluster_count} clusters; "
                f"requested {n_parcels}"
            )

        if sizes[left] < sizes[right]:
            left, right = right, left
        parents[right] = left
        sizes[left] += sizes[right]
        masses[left] += masses[right]
        cluster_count -= 1

        merged_neighbors = (neighbors[left] | neighbors[right]) - {left, right}
        neighbors[left] = set()
        for old_neighbor in merged_neighbors:
            neighbor = find_root(old_neighbor)
            if neighbor == left:
                continue
            neighbors[neighbor].discard(left)
            neighbors[neighbor].discard(right)
            neighbors[neighbor].add(left)
            neighbors[left].add(neighbor)
            heapq.heappush(
                heap,
                (
                    masses[left] + masses[neighbor],
                    min(left, neighbor),
                    max(left, neighbor),
                ),
            )
        neighbors[right].clear()

    roots = np.array([find_root(index) for index in range(n_voxels)], dtype=np.int32)
    _, owners = np.unique(roots, return_inverse=True)
    return owners.astype(np.int32, copy=False)


def balance_adjacent_parcels(owners, adjacency, n_parcels):
    """Reduce local size differences by repartitioning adjacent parcel pairs."""
    parcel_members = build_parcel_members(owners, n_parcels)
    parcel_sizes = np.array([len(members) for members in parcel_members], dtype=np.int32)

    while True:
        parcel_neighbors = parcel_adjacency(owners, adjacency, n_parcels)
        edges = sorted(
            (
                abs(int(parcel_sizes[left]) - int(parcel_sizes[right])),
                left,
                right,
            )
            for left, adjacent in enumerate(parcel_neighbors)
            for right in adjacent
            if left < right and abs(int(parcel_sizes[left]) - int(parcel_sizes[right])) >= 2
        )
        moves = 0
        for _, left, right in reversed(edges):
            difference = int(parcel_sizes[left]) - int(parcel_sizes[right])
            if abs(difference) < 2:
                continue
            source, destination = (left, right) if difference > 0 else (right, left)
            amount = abs(difference) // 2
            if transfer_voxels(
                source,
                destination,
                amount,
                owners,
                parcel_members,
                parcel_sizes,
                adjacency,
            ):
                moves += 1
        if moves == 0:
            return owners, parcel_members, parcel_sizes


def candidate_receiver_paths(start_parcel, parcel_neighbors, parcel_sizes, upper_bound):
    """Return paths from an overfull parcel to nearby parcels with spare capacity."""
    queue = deque([start_parcel])
    predecessors = {start_parcel: None}
    paths = []

    while queue and len(paths) < 128:
        parcel_id = queue.popleft()
        if parcel_id != start_parcel and parcel_sizes[parcel_id] < upper_bound:
            path = []
            cursor = parcel_id
            while cursor is not None:
                path.append(cursor)
                cursor = predecessors[cursor]
            paths.append(list(reversed(path)))
            continue
        for neighbor_id in sorted(parcel_neighbors[parcel_id]):
            if neighbor_id not in predecessors:
                predecessors[neighbor_id] = parcel_id
                queue.append(neighbor_id)
    return paths


def balance_to_size_bounds(
    owners,
    adjacency,
    parcel_members,
    parcel_sizes,
    lower_bound,
    upper_bound,
):
    """Move mass along parcel paths until every parcel is inside fixed bounds."""
    n_parcels = len(parcel_sizes)
    lower_targets = np.full(n_parcels, lower_bound, dtype=np.int32)

    while np.any(parcel_sizes < lower_bound):
        progress = False
        parcel_neighbors = parcel_adjacency(owners, adjacency, n_parcels)
        deficits = sorted(
            np.flatnonzero(parcel_sizes < lower_bound),
            key=lambda parcel_id: parcel_sizes[parcel_id],
        )
        for destination in deficits:
            if parcel_sizes[destination] >= lower_bound:
                continue
            paths = candidate_surplus_paths(
                int(destination),
                parcel_neighbors,
                parcel_sizes,
                lower_targets,
                max_paths=128,
            )
            for path in paths:
                source = path[-1]
                amount = min(
                    lower_bound - int(parcel_sizes[destination]),
                    int(parcel_sizes[source]) - lower_bound,
                )
                if amount > 0 and try_transfer_path(
                    path,
                    amount,
                    owners,
                    parcel_members,
                    parcel_sizes,
                    adjacency,
                ):
                    progress = True
                    break
        if not progress:
            break

    while np.any(parcel_sizes > upper_bound):
        progress = False
        parcel_neighbors = parcel_adjacency(owners, adjacency, n_parcels)
        for source in np.flatnonzero(parcel_sizes > upper_bound):
            if parcel_sizes[source] <= upper_bound:
                continue
            for outward_path in candidate_receiver_paths(
                int(source), parcel_neighbors, parcel_sizes, upper_bound
            ):
                destination = outward_path[-1]
                amount = min(
                    int(parcel_sizes[source]) - upper_bound,
                    upper_bound - int(parcel_sizes[destination]),
                )
                transfer_path = list(reversed(outward_path))
                if amount > 0 and try_transfer_path(
                    transfer_path,
                    amount,
                    owners,
                    parcel_members,
                    parcel_sizes,
                    adjacency,
                ):
                    progress = True
                    break
        if not progress:
            break

    return owners


def grow_component(coords, n_parcels):
    """Parcellate one component with connected coarsening and bounded repair."""
    if n_parcels == 1:
        return np.zeros(len(coords), dtype=np.int32)

    adjacency = build_adjacency(coords)
    targets = parcel_targets(len(coords), n_parcels)
    owners = coarsen_connected_graph(adjacency, n_parcels)
    owners, parcel_members, parcel_sizes = balance_adjacent_parcels(
        owners, adjacency, n_parcels
    )
    preferred_lower = max(1, int(targets.min()) - 1)
    preferred_upper = int(targets.max()) + 1
    owners = balance_to_size_bounds(
        owners,
        adjacency,
        parcel_members,
        parcel_sizes,
        preferred_lower,
        preferred_upper,
    )

    sizes = np.bincount(owners, minlength=n_parcels)
    hard_lower = max(1, int(targets.min()) - 4)
    hard_upper = int(targets.max()) + 3
    if sizes.min() < hard_lower or sizes.max() > hard_upper:
        raise RuntimeError(
            f"Could not satisfy connected hard size bounds {hard_lower}-{hard_upper}; "
            f"observed {int(sizes.min())}-{int(sizes.max())}"
        )
    old_ids = sorted(
        range(n_parcels),
        key=lambda parcel_id: (-sizes[parcel_id], parcel_id),
    )
    new_id_for_old = np.empty(n_parcels, dtype=np.int32)
    new_id_for_old[np.asarray(old_ids, dtype=np.int32)] = np.arange(
        n_parcels, dtype=np.int32
    )
    return new_id_for_old[owners]


def connected_subcomponents(coords):
    """Count 6-connected components in a set of voxel coordinates (used for post-hoc QC)."""
    if len(coords) <= 1:
        return 1

    coord_set = {tuple(coord) for coord in coords.tolist()}
    seen = set()
    n_components = 0

    for start in coord_set:
        if start in seen:
            continue

        n_components += 1
        queue = deque([start])
        seen.add(start)

        while queue:
            x, y, z = queue.popleft()
            for dx, dy, dz in NEIGHBOR_DELTAS:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in coord_set and neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)

    return n_components


def check_neigh(roi_coords_in_voxel):
    """Print a warning if the parcel voxels form more than one disconnected piece."""
    n_components = connected_subcomponents(roi_coords_in_voxel)
    if n_components > 1:
        print(
            f"Connectivity warning: parcel with {len(roi_coords_in_voxel)} voxels "
            f"contains {n_components} disconnected pieces"
        )


def parcellate_label(args, label, template_img, seg_image):
    """Build, validate, and optionally write one anatomical label."""
    roi_mask = seg_image == label

    if not np.any(roi_mask):
        raise ValueError(f"ROI label {label} not found in {args.segmentation}")

    structure = six_connectivity_structure()
    component_map, n_components = ndimage.label(roi_mask.astype(np.uint8), structure=structure)

    roi_coords = np.argwhere(roi_mask)
    roi_component_ids = component_map[roi_mask]
    component_ids, component_sizes = np.unique(roi_component_ids, return_counts=True)
    component_allocation = allocate_parcels_to_components(component_sizes, args.parcel_size)

    print(f"Connected components: {n_components}")
    print(f"Largest component size: {int(component_sizes.max())}")
    print(f"Total ROI voxels: {int(len(roi_coords))}")
    print(f"Target parcel size: {args.parcel_size}")
    print(f"Total output parcels: {int(component_allocation.sum())}")

    all_parcels = []
    all_targets = []
    start_time = time.time()

    for component_id, n_component_parcels in zip(component_ids, component_allocation):
        component_coords = roi_coords[roi_component_ids == component_id]
        component_owners = grow_component(component_coords, int(n_component_parcels))
        component_targets = parcel_targets(len(component_coords), int(n_component_parcels))

        for parcel_id in range(int(n_component_parcels)):
            parcel_coords = component_coords[component_owners == parcel_id]
            if len(parcel_coords) == 0:
                raise RuntimeError(
                    f"Generated an empty parcel inside component {int(component_id)}"
                )
            all_parcels.append(parcel_coords)
            all_targets.append(int(component_targets[parcel_id]))

    print(f"Parcellation time: {time.time() - start_time:.3f} s")

    parcel_sizes = np.array([len(parcel) for parcel in all_parcels], dtype=np.int32)
    parcel_sizes_match_targets = np.array_equal(
        parcel_sizes, np.asarray(all_targets, dtype=np.int32)
    )
    print(f"Parcel size range: {int(parcel_sizes.min())} - {int(parcel_sizes.max())}")

    component_counts = [connected_subcomponents(parcel) for parcel in all_parcels]
    disconnected = [index for index, count in enumerate(component_counts) if count != 1]
    if disconnected:
        raise RuntimeError(f"Generated disconnected parcels: {disconnected[:10]}")
    stacked = np.vstack(all_parcels)
    if len(np.unique(stacked, axis=0)) != len(roi_coords):
        raise RuntimeError("Parcels overlap or do not cover the source ROI exactly")

    summary = {
        "label": int(label),
        "source_voxel_count": int(len(roi_coords)),
        "parcel_count": len(all_parcels),
        "parcel_size_min": int(parcel_sizes.min()),
        "parcel_size_max": int(parcel_sizes.max()),
        "parcel_sizes_match_targets": bool(parcel_sizes_match_targets),
    }
    if args.validate_only:
        print(f"Validated label {label} in memory; no files written.")
        return summary

    output_dir = args.output_root / str(label)
    existing = list(output_dir.glob("roi_*.nii.gz")) if output_dir.exists() else []
    if existing:
        raise FileExistsError(
            f"Refusing to mix a new parcellation with {len(existing)} existing masks in {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing {len(all_parcels)} masks with {args.workers} workers")
    parcel_records = write_parcels_parallel(
        all_parcels,
        all_targets,
        args.segmentation,
        output_dir,
        args.workers,
    )

    manifest_path = output_dir / "label_manifest.json"
    atomic_write_json(manifest_path, {
        "schema_version": 2,
        "validation_status": "passed",
        "algorithm": "adjacency_constrained_weighted_coarsening",
        "label": int(label),
        "label_name": LABEL_DICT[label],
        "source_segmentation": file_signature(args.segmentation, include_sha256=True),
        "source_voxel_count": int(len(roi_coords)),
        "parcel_count": len(parcel_records),
        "parcel_size_target": int(args.parcel_size),
        "parcel_size_min": int(parcel_sizes.min()),
        "parcel_size_max": int(parcel_sizes.max()),
        "parcel_sizes_match_targets": bool(parcel_sizes_match_targets),
        "size_policy": {
            "preferred_tolerance_voxels": 1,
            "hard_lower_tolerance_voxels": 4,
            "hard_upper_tolerance_voxels": 3,
            "small_source_components_remain_single_connected_parcels": True,
        },
        "connectivity": 6,
        "coverage_exact": True,
        "overlap_voxels": 0,
        "workers": int(args.workers),
        "parcels": parcel_records,
    })
    summary["manifest_sha256"] = sha256_file(manifest_path)
    print(f"Validated and wrote label {label}: {output_dir}")
    return summary


def main():
    args = parse_args()
    if args.parcel_size < 1:
        raise ValueError("--parcel-size must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    production_rois = (PROJECT_ROOT / "outputs" / "rois").resolve()
    if not args.validate_only and args.output_root.resolve() == production_rois:
        raise ValueError(
            "Refusing to write this test atlas into outputs/rois; "
            f"use the isolated default {DEFAULT_OUTPUT_ROOT}"
        )

    labels = list(args.roi_label) if args.roi_label else list(KEEP_LABELS)
    if len(labels) != len(set(labels)):
        raise ValueError(f"Duplicate --roi-label values: {labels}")
    unknown_labels = sorted(set(labels) - set(KEEP_LABELS))
    if unknown_labels:
        raise ValueError(f"Labels are not retained atlas labels: {unknown_labels}")

    detected_cpus = os.cpu_count() or 1
    print(f"Detected CPUs: {detected_cpus}")
    print(f"Writer workers: {args.workers} ({args.workers / detected_cpus:.0%} of CPUs)")
    print(f"Output root: {args.output_root.resolve()}")
    print(f"Labels processed sequentially: {labels}")
    if args.validate_only:
        print("Validation-only mode: no output files will be written")
    else:
        args.output_root.mkdir(parents=True, exist_ok=True)

    template_img, seg_image = load_segmentation(args.segmentation)
    summaries = []
    for index, label in enumerate(labels, start=1):
        print(
            f"\n[{index}/{len(labels)}] Label {label}: {LABEL_DICT[label]}",
            flush=True,
        )
        summaries.append(parcellate_label(args, label, template_img, seg_image))

    if not args.validate_only and labels == list(KEEP_LABELS):
        manifest_path = args.output_root / "atlas_manifest.json"
        atomic_write_json(manifest_path, {
            "schema_version": 2,
            "validation_status": "passed",
            "algorithm": "adjacency_constrained_weighted_coarsening",
            "labels_processed_sequentially": True,
            "writer_workers": int(args.workers),
            "source_segmentation": file_signature(args.segmentation, include_sha256=True),
            "parcel_size_target": int(args.parcel_size),
            "labels": summaries,
        })
        print(f"Atlas manifest: {manifest_path}")

    print("All requested labels validated successfully.")


if __name__ == "__main__":
    main()
