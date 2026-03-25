import nibabel as nib
import numpy as np

img = nib.load("MNI152_T1_1mm_seg.nii.gz")
data = img.get_fdata()

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

# Count the number of voxels for each label
labels, counts = np.unique(data, return_counts=True)

# Keep labels that are not white matter
# Labels reference: https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg
keep_labels = [3, 8, 10, 11, 12, 13, 16, 17, 18, 26, 28,
               42, 47, 49, 50, 51, 52, 53, 54, 58, 60]

# Print labels and counts for labels to keep
for label in keep_labels:
    count = counts[labels == label][0]
    print(f"Label {label} ({label_dict[label]}): {count} voxels")

# Mask gray-matter only
mask = np.isin(data, keep_labels).astype(np.uint8)
print("Mask created", mask.shape, mask.sum())

# Save mask
print("Saving mask to MNI152_keep_labels_mask.nii.gz")
out = nib.Nifti1Image(mask, img.affine, img.header)
nib.save(out, "MNI152_keep_labels_mask.nii.gz")