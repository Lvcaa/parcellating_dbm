from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
SEGMENTATION_PATH = REFERENCE_DIR / "MNI152_T1_1mm_seg.nii.gz"
MASK_PATH = REFERENCE_DIR / "MNI152_keep_labels_mask.nii.gz"

label_dict = {
    0: "Background",
    2: "Left cerebral white matter",
    3: "Left cerebral cortex",
    4: "Left lateral ventricle",
    5: "Left inferior lateral ventricle",
    7: "Left cerebellum white matter",
    8: "Left cerebellum cortex",
    10: "Left thalamus",
    11: "Left caudate",
    12: "Left putamen",
    13: "Left pallidum",
    14: "3rd ventricle",
    15: "4th ventricle",
    16: "Brain-stem",
    17: "Left hippocampus",
    18: "Left amygdala",
    24: "CSF",
    26: "Left accumbens area",
    28: "Left ventral DC",
    41: "Right cerebral white matter",
    42: "Right cerebral cortex",
    43: "Right lateral ventricle",
    44: "Right inferior lateral ventricle",
    46: "Right cerebellum white matter",
    47: "Right cerebellum cortex",
    49: "Right thalamus",
    50: "Right caudate",
    51: "Right putamen",
    52: "Right pallidum",
    53: "Right hippocampus",
    54: "Right amygdala",
    58: "Right accumbens area",
    60: "Right ventral DC"
}

# Keep labels that are not white matter
# Labels reference: https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg
keep_labels = [
    3, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28,
    42, 47, 49, 50, 51, 52, 53, 54, 58, 60,
]


def main():
    img = nib.load(str(SEGMENTATION_PATH))
    data = img.get_fdata()

    labels, counts = np.unique(data, return_counts=True)
    label_counts = dict(zip(labels.astype(int), counts.astype(int)))

    for label in keep_labels:
        count = label_counts.get(label, 0)
        print(f"Label {label} ({label_dict[label]}): {count} voxels")

    mask = np.isin(data, keep_labels).astype(np.uint8)
    print("Mask created", mask.shape, int(mask.sum()))

    print(f"Saving mask to {MASK_PATH}")
    out = nib.Nifti1Image(mask, img.affine, img.header)
    nib.save(out, str(MASK_PATH))


if __name__ == "__main__":
    main()
