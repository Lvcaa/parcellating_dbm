"""
Inspect retained segmentation labels and write the binary keep-label mask.

The script prints voxel counts and their greatest common divisor. Paths and
label configuration are defined in const.py.

Usage:
    python scripts/preprocessing/inspect_labels.py
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import const


def main():
    img = nib.load(str(const.SEGMENTATION_PATH))
    data = img.get_fdata()
    labels, counts = np.unique(data, return_counts=True)
    label_counts = dict(zip(labels.astype(int), counts.astype(int)))

    counts_to_consider = [label_counts.get(label, 0) for label in const.KEEP_LABELS]
    non_zero_counts = [c for c in counts_to_consider if c > 0]
    missing_labels = [
        label for label, c in zip(const.KEEP_LABELS, counts_to_consider) if c == 0
    ]
    gcd = int(np.gcd.reduce(np.asarray(non_zero_counts, dtype=np.int64))) if non_zero_counts else 0

    print(f"Greatest common divisor of counts for labels to keep: {gcd}")
    if missing_labels:
        print(f"Labels not present in the segmentation: {missing_labels}")
    for label in const.KEEP_LABELS:
        count = label_counts.get(label, 0)
        print(f"Label {label} ({const.LABEL_DICT[label]}): {count} voxels")

    mask = np.isin(data, const.KEEP_LABELS).astype(np.uint8)
    kept_voxels = int(mask.sum())
    print(f"Total voxels in the segmentation: {int(data.size)}")
    print(f"Voxels kept by the mask: {kept_voxels}")
    print(f"Saving mask to {const.MASK_PATH}")
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(const.MASK_PATH))


if __name__ == "__main__":
    main()
