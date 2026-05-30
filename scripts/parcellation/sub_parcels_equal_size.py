"""
Split one segmentation ROI into approximately equal-sized sub-parcel masks.

The clustering-based implementation constrains parcels spatially and writes
one NIfTI mask per output parcel under ``<output-root>/<roi-label>/``.

Usage:
    python scripts/parcellation/sub_parcels_equal_size.py [options]

Parameters:
    --roi-label INT          Segmentation label to split (default: 10).
    --parcel-size INT        Target voxels per parcel (default: 27).
    --segmentation PATH      Input segmentation image.
    --output-root PATH       Root for ROI-specific folders (default: outputs/rois).
    --skip-neighbor-check    Skip local connectivity diagnostics.

Examples:
    python scripts/parcellation/sub_parcels_equal_size.py
    python scripts/parcellation/sub_parcels_equal_size.py --roi-label 49 --parcel-size 15
    python scripts/parcellation/sub_parcels_equal_size.py --roi-label 10 --output-root outputs/custom_rois --skip-neighbor-check
"""

import argparse
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction import image
from sklearn.neighbors import NearestCentroid


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEGMENTATION = PROJECT_ROOT / "data" / "reference" / "MNI152_T1_1mm_seg.nii.gz"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "rois"

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


def write_nifti(roi_indices, sub_roi_index, template_img, output_dir):

    # Create an empty image and set the voxels corresponding to the current sub-parcel to 1
    empty_image = np.zeros(template_img.shape, dtype=np.uint8)

    # roi_indices is in (x, y, z) format, so we can directly index into the empty image
    for ii in roi_indices:
        empty_image[ii[0], ii[1], ii[2]] = 1

    # Create a NIfTI image and save it
    nii_ = nib.Nifti1Image(empty_image, affine=template_img.affine, header=template_img.header)
    output_path = output_dir / f"roi_{sub_roi_index:04d}.nii.gz"
    nii_.to_filename(str(output_path))


def check_neigh(roi_coords_in_voxel):
    for ii_index, ii in enumerate(roi_coords_in_voxel):
        roi_coords_in_voxel_copy = roi_coords_in_voxel.copy()
        roi_coords_in_voxel_copy = np.delete(roi_coords_in_voxel_copy, ii_index, axis=0)
        distances_neigh = (np.abs(ii - roi_coords_in_voxel_copy) > 1).sum(axis=1)
        if np.where(distances_neigh > 3)[0].shape[0] != 0:
            print(f"{ii} has no direct neigh")


def main():
    args = parse_args()

    if args.parcel_size < 1:
        raise ValueError("--parcel-size must be at least 1")

    # Load the segmentation image and extract the coordinates of the voxels 
    # belonging to the specified ROI label
    template_img = nib.load(str(args.segmentation))
    
    # Get the segmentation data as a numpy array
    seg_image = template_img.get_fdata()

    # Extract the coordinates of the voxels that belong to the specified ROI label
    roi_coord = np.vstack(np.where(seg_image == args.roi_label)).T
    if roi_coord.size == 0:
        raise ValueError(f"ROI label {args.roi_label} not found in {args.segmentation}")

    # Calculate the number of clusters needed to achieve the desired parcel size
    n_clust = int(np.ceil(len(roi_coord) / args.parcel_size))

    # Create a connectivity graph for the voxels 
    # in the ROI to ensure that sub-parcels are spatially contiguous
    
    # Create a binary mask for the ROI and use it to compute the connectivity graph
    roi_mask = (seg_image == args.roi_label).astype(np.uint8)
    
    # The grid_to_graph function creates a sparse matrix 
    # where each row corresponds to a voxel in the ROI and 
    # each column corresponds to a neighboring voxel.
    conn = image.grid_to_graph(
        n_x=roi_mask.shape[0],
        n_y=roi_mask.shape[1],
        n_z=roi_mask.shape[2],
        mask=roi_mask,
    )

    graph = csr_matrix(conn)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    component_sizes = np.bincount(labels)
    print(f"Connected components: {n_components}")
    print(f"Largest component size: {int(component_sizes.max())}")

    start_time = time.time()
    labels_clust = AgglomerativeClustering(
        n_clusters=n_clust,
        connectivity=conn,
    ).fit_predict(roi_coord)
    print(f"Agglomerative clustering time: {time.time() - start_time:.3f} s")

    clf = NearestCentroid()
    clf.fit(roi_coord, labels_clust)
    centroids = clf.centroids_

    centers = (
        centroids.reshape(-1, 1, roi_coord.shape[-1])
        .repeat(args.parcel_size, 1)
        .reshape(-1, roi_coord.shape[-1])
    )
    distance_matrix = cdist(roi_coord, centers)

    start_time = time.time()
    clusters = linear_sum_assignment(distance_matrix)[1] // args.parcel_size
    print(f"Linear assignment time: {time.time() - start_time:.3f} s")

    output_dir = args.output_root / str(args.roi_label)
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_labels = np.unique(clusters)
    for ii_index, ii in enumerate(unique_labels):
        dummy = np.where(clusters == ii)[0]
        if not args.skip_neighbor_check:
            check_neigh(roi_coord[dummy])
        write_nifti(roi_coord[dummy], ii_index, template_img, output_dir)


if __name__ == "__main__":
    main()
