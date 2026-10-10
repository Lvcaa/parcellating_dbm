# Network-based statistics with NBS v1.2

The quickest route is the **original NBS v1.2 MATLAB GUI** from [NITRC](https://www.nitrc.org/projects/nbs/). Its [reference manual](https://www.nitrc.org/frs/download.php/5331/Reference_Manual_NBS_v1.2.pdf) specifies the following setup.

1. **Prepare one similarity matrix per subject.** Use the same 944 retained parcels in the same order for everyone. Put the symmetric matrices in one MATLAB variable shaped `944 × 944 × number_of_subjects`, then save it as a `.mat` file containing **only that variable**. Slice 3 must correspond to row 3 of the design matrix. The toolbox also accepts separate text files, but the single array makes ordering easier to verify. Do not feed it the weighted-degree vectors from [t_tests.ipynb](../scripts/analysis/t_tests.ipynb). [NBS input specification](https://www.nitrc.org/frs/download.php/5331/Reference_Manual_NBS_v1.2.pdf)
2. **Make a numeric design-matrix text file.** For example, columns could be `[intercept, Unhealthy(1)/Healthy(0), age, sex, education]`, one row per subject. For **Unhealthy > Healthy**, enter contrast `[0 1 0 0 0]`; reverse its sign for **Healthy > Unhealthy**. Select **t-test**. Do not include an intercept *and* separate indicators for both groups—that makes the model rank deficient. [Toolbox author’s design example](https://www.nitrc.org/forum/forum.php?forum_id=3444&max_rows=25&offset=25&style=nested)
3. **Launch MATLAB:** unzip NBS1.2, add it and its subfolders to the MATLAB path, change into its directory, and type `NBS`. In the GUI, select the matrices and design file; set **Method: NBS**, **Component Size: extent**, a prespecified **t-statistic threshold**, and the permutation count. Node coordinates and labels help display results but are optional for the calculation. [Installation and GUI guide](https://www.nitrc.org/frs/download.php/5331/Reference_Manual_NBS_v1.2.pdf)
4. **Run a small input check first**—say 100 permutations—then a final run with several thousand. The toolbox’s t-test is **one-sided**, so run both contrast signs if you want both directions and account for testing both. Its corrected *p*-value belongs to the **connected edge component**, not to each edge individually. [NBS manual](https://www.nitrc.org/frs/download.php/5331/Reference_Manual_NBS_v1.2.pdf)

**Your immediate missing input is the subject-level 944 × 944 matrices for the new atlas.** The atlas exists, but the current notebook uses older-atlas weighted degrees. For 267 subjects, the double-precision matrix stack alone is about **1.9 GB**, before MATLAB and NBS working memory; the small pilot should establish runtime and peak RAM before the full permutation run.

## Prepare the matrices from the test split

```bash
python scripts/graph_building/prepare_subject_similarity_matrices.py --method was --sim-formula 1 --dry-run
python scripts/graph_building/prepare_subject_similarity_matrices.py --method was --sim-formula 1
```

This uses `test_healthy` and `test_unhealthy` from `data/splits.json`, keeping
subjects with a local `data/warps/sub-XXXX/Reg_/_SyN1Warp.nii.gz`. Currently
that gives 137 healthy and 130 unhealthy subjects. The voxel-built atlas at
`outputs/atlases/atlas-1000-nocsf-from-voxels/rois` supplies the parcel order;
the 56 parcels with fewer than 15 voxels are excluded before extraction.

The script reuses the 15-point quantile representation and Wasserstein
similarity calculation from `wasserstein_distance_graph2.py`. Formula 1 is
`exp(-W)`; for formula 2 (`1/(1+W)`), also choose a separate `--output` path.
Log-Jacobians are computed from the forward warps with ANTs (non-geometric
mode) and cached under the output directory for reuse. Processing is serial
with one ANTs thread. The double-precision stack needs 1.77 GiB; allow about
4.5 GiB for the stack, MAT export and image buffers.

The default output is `outputs/nbs/similarities_was_expW.mat`, containing only
`similarity_matrices`, shaped `944 × 944 × 267`. Its `.subjects.txt` sidecar
lists subjects in slice order, and its `.json` sidecar records that order,
group indicators (healthy 0, unhealthy 1), retained and excluded parcels,
and calculation settings. Build the step-2 design matrix in this exact
subject order. Similarities are unthresholded, symmetric, with diagonal 1.
