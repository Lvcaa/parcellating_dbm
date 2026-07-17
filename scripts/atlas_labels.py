"""Pure-data anatomical label configuration for atlas building and validation."""

LABEL_DICT = {
    3: "Left cerebral cortex",
    4: "Left lateral ventricle",
    5: "Left inferior lateral ventricle",
    8: "Left cerebellum cortex",
    10: "Left thalamus",
    11: "Left caudate",
    12: "Left putamen",
    13: "Left pallidum",
    16: "Brain-stem",
    17: "Left hippocampus",
    18: "Left amygdala",
    24: "CSF",
    26: "Left accumbens area",
    28: "Left ventral DC",
    42: "Right cerebral cortex",
    43: "Right lateral ventricle",
    44: "Right inferior lateral ventricle",
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

KEEP_LABELS = list(LABEL_DICT)
