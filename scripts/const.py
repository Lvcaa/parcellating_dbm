from pathlib import Path
import os
import random
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
SCRIPTS_DIR   = PROJECT_ROOT / "scripts"
DATA_DIR      = PROJECT_ROOT / "data"
DATASET_DIR   = DATA_DIR / "dataset"
REFERENCE_DIR = DATA_DIR / "reference"
DATABASE_PATH = DATA_DIR / "database_finale_labels_corrette.csv"
SPLITS_PATH   = DATA_DIR / "splits.json"

ROIS_DIR               = PROJECT_ROOT / "outputs" / "rois"
AGGREGATED_ROIS_DIR    = PROJECT_ROOT / "outputs" / "aggregated_parcellations"
HIGH_MEMORY_LABELS_PATH = PROJECT_ROOT / "docs" / "high_memory_labels.txt"
LABEL_LOOKUP_PATH       = PROJECT_ROOT / "docs" / "label_lookup.csv"

# Per-subject filenames inside DATASET_DIR/sub-XXXX/
WARP_FILENAME         = "_SyN1Warp.nii.gz"
LOG_JACOBIAN_FILENAME = "logJacobian.nii.gz"

# Segmentation reference files
SEGMENTATION_PATH = REFERENCE_DIR / "MNI152_T1_1mm_seg.nii.gz"
MASK_PATH         = REFERENCE_DIR / "MNI152_keep_labels_mask.nii.gz"

# FreeSurfer/SynthSeg label map (https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg)
LABEL_DICT = {
    0:  "Background",
    2:  "Left cerebral white matter",
    3:  "Left cerebral cortex",
    4:  "Left lateral ventricle",
    5:  "Left inferior lateral ventricle",
    7:  "Left cerebellum white matter",
    8:  "Left cerebellum cortex",
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
    60: "Right ventral DC",
}

# Labels to keep (non-white-matter grey matter structures)
KEEP_LABELS = [
    3, 4, 5, 8, 10, 11, 12, 13, 16, 17, 18, 24, 26, 28,
    42, 43, 44, 47, 49, 50, 51, 52, 53, 54, 58, 60,
]

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED   = 0
DETERMINISTIC = True
BENCHMARK     = False

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_GPU = torch.cuda.device_count()
print(f"Using {DEVICE} device")

torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

torch.backends.cudnn.deterministic = DETERMINISTIC
torch.backends.cudnn.benchmark     = BENCHMARK
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.profiler.profile(enabled=False)
torch.autograd.set_detect_anomaly(False)

# ---------------------------------------------------------------------------
# Image geometry
# ---------------------------------------------------------------------------
RESOLUTION       = 1.5
INPUT_SHAPE_1mm  = (182, 218, 182)
INPUT_SHAPE_1p5mm = (122, 146, 122)
INPUT_SHAPE_AE   = (120, 144, 120)
