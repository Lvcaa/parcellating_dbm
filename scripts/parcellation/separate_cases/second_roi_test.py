"""
Split one segmentation ROI with the alternative connected region-growing path.

Each approximately equal-sized parcel is written as a NIfTI mask under
``<output-root>/<roi-label>/``.

Usage:
    python scripts/parcellation/separate_cases/second_roi_test.py [options]

Parameters:
    --roi-label INT          Segmentation label to split (default: 10).
    --parcel-size INT        Target voxels per parcel (default: 27).
    --segmentation PATH      Input segmentation image.
    --output-root PATH       Root for ROI-specific folders (default: outputs/rois).
    --skip-neighbor-check    Skip connectivity diagnostics.

Examples:
    python scripts/parcellation/separate_cases/second_roi_test.py
    python scripts/parcellation/separate_cases/second_roi_test.py --roi-label 42 --parcel-size 15 --skip-neighbor-check
"""

import argparse
import time
from collections import deque
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTATION = PROJECT_ROOT / "data" / "reference" / "MNI152_T1_1mm_seg.nii.gz"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "rois"
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
        description="Split a segmentation ROI into approximately equal-sized sub-parcels."
    )
    parser.add_argument(
        "--roi-label",
        type=int,
        default=10,
        help="Segmentation label to parcelize.",
    )
    parser.add_argument(
        "--parcel-size",
        type=int,
        default=27,
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
        help="Skip the local connectivity diagnostic for each sub-parcel.",
    )
    return parser.parse_args()


def six_connectivity_structure():
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
    template_img = nib.load(str(segmentation_path))
    seg_image = np.asanyarray(template_img.dataobj)

    if np.issubdtype(seg_image.dtype, np.floating):
        rounded = np.rint(seg_image)
        if np.allclose(seg_image, rounded, atol=1e-6):
            seg_image = rounded.astype(np.int64, copy=False)

    return template_img, seg_image


def write_nifti(roi_indices, sub_roi_index, template_img, output_dir):
    roi_indices = np.asarray(roi_indices, dtype=np.int32)
    if roi_indices.size == 0:
        raise ValueError(f"Sub-parcel {sub_roi_index} is empty and will not be written")

    empty_image = np.zeros(template_img.shape, dtype=np.uint8)
    empty_image[tuple(roi_indices.T)] = 1

    header = template_img.header.copy()
    header.set_data_dtype(np.uint8)
    nii_ = nib.Nifti1Image(empty_image, affine=template_img.affine, header=header)
    output_path = output_dir / f"roi_{sub_roi_index:04d}.nii.gz"
    nii_.to_filename(str(output_path))


def parcel_targets(n_voxels, n_parcels):
    base_size = n_voxels // n_parcels
    remainder = n_voxels % n_parcels
    targets = np.full(n_parcels, base_size, dtype=np.int32)
    targets[:remainder] += 1
    return targets


def allocate_parcels_to_components(component_sizes, parcel_size):
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
    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(coords.tolist())}
    adjacency = [[] for _ in range(len(coords))]

    for idx, (x, y, z) in enumerate(coords):
        for dx, dy, dz in NEIGHBOR_DELTAS:
            neighbor_index = coord_to_index.get((x + dx, y + dy, z + dz))
            if neighbor_index is not None:
                adjacency[idx].append(neighbor_index)

    return adjacency


def grow_connected_parcels(adjacency, targets, seeds):
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

    while np.any(owners == -1):
        progress = False
        deficit_order = np.argsort(parcel_sizes / targets)

        for parcel_id in deficit_order:
            progress = expand_once(int(parcel_id), respect_target=True) or progress

        if not progress:
            break

        if np.all(parcel_sizes >= targets):
            break

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
    return [set(np.flatnonzero(owners == parcel_id)) for parcel_id in range(n_parcels)]


def parcel_adjacency(owners, adjacency, n_parcels):
    parcel_neighbors = [set() for _ in range(n_parcels)]

    for voxel_index, neighbors in enumerate(adjacency):
        source_parcel = int(owners[voxel_index])
        for neighbor_index in neighbors:
            target_parcel = int(owners[neighbor_index])
            if source_parcel != target_parcel:
                parcel_neighbors[source_parcel].add(target_parcel)

    return parcel_neighbors


def is_connected_after_removal(members, remove_index, adjacency):
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


def rebalance_connected_parcels(owners, adjacency, targets):
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
            path = shortest_surplus_path(int(target_parcel), parcel_neighbors, parcel_sizes, targets)
            if not path or len(path) == 1:
                continue

            moved = True
            for path_index in range(len(path) - 1, 0, -1):
                source_parcel = path[path_index]
                destination_parcel = path[path_index - 1]
                voxel_index = find_transfer_voxel(
                    source_parcel,
                    destination_parcel,
                    owners,
                    parcel_members,
                    adjacency,
                )
                if voxel_index is None:
                    moved = False
                    break

                owners[voxel_index] = destination_parcel
                parcel_members[source_parcel].remove(voxel_index)
                parcel_members[destination_parcel].add(voxel_index)
                parcel_sizes[source_parcel] -= 1
                parcel_sizes[destination_parcel] += 1

            if moved:
                progress = True

        if not progress:
            break

    return owners


def grow_component(coords, n_parcels):
    if n_parcels == 1:
        return np.zeros(len(coords), dtype=np.int32)

    adjacency = build_adjacency(coords)
    targets = parcel_targets(len(coords), n_parcels)
    seeds = choose_seed_indices(coords, n_parcels)
    owners = grow_connected_parcels(adjacency, targets, seeds)
    owners = rebalance_connected_parcels(owners, adjacency, targets)
    return owners


def connected_subcomponents(coords):
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
    n_components = connected_subcomponents(roi_coords_in_voxel)
    if n_components > 1:
        print(
            f"Connectivity warning: parcel with {len(roi_coords_in_voxel)} voxels "
            f"contains {n_components} disconnected pieces"
        )


def main():
    args = parse_args()

    if args.parcel_size < 1:
        raise ValueError("--parcel-size must be at least 1")

    template_img, seg_image = load_segmentation(args.segmentation)
    roi_mask = seg_image == args.roi_label

    if not np.any(roi_mask):
        raise ValueError(f"ROI label {args.roi_label} not found in {args.segmentation}")

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
    start_time = time.time()

    for component_id, n_component_parcels in zip(component_ids, component_allocation):
        component_coords = roi_coords[roi_component_ids == component_id]
        component_owners = grow_component(component_coords, int(n_component_parcels))

        for parcel_id in range(int(n_component_parcels)):
            parcel_coords = component_coords[component_owners == parcel_id]
            if len(parcel_coords) == 0:
                raise RuntimeError(
                    f"Generated an empty parcel inside component {int(component_id)}"
                )
            all_parcels.append(parcel_coords)

    print(f"Parcellation time: {time.time() - start_time:.3f} s")

    parcel_sizes = np.array([len(parcel) for parcel in all_parcels], dtype=np.int32)
    print(f"Parcel size range: {int(parcel_sizes.min())} - {int(parcel_sizes.max())}")

    output_dir = args.output_root / str(args.roi_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    for parcel_index, parcel_coords in enumerate(all_parcels):
        if not args.skip_neighbor_check:
            check_neigh(parcel_coords)
        write_nifti(parcel_coords, parcel_index, template_img, output_dir)


if __name__ == "__main__":
    main()
