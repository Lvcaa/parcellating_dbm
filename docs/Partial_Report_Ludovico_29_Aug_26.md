# OASIS3 Healthy vs Unhealthy Analysis

> **Atlas note.** The earlier parcelisation used global centroid/slot assignment after clustering. Assigning voxels to those global slots did not preserve face-to-face voxel adjacency, so one nominal parcel could contain separated pieces. The issue is documented in [`REPORT_ISSUES.md`](REPORT_ISSUES.md).
>
> The analyses below use `atlas-8cf3dc8f`, whose manifest records **adjacency-constrained weighted coarsening**: parcels are grown and repaired only through 6-connected voxel neighbours, then connectivity is validated.

## 1. Parcel-wise analysis: `t_tests.ipynb`

### 1.1 Objective

The parcel-wise analysis tested whether Wasserstein-derived weighted degree differs between healthy and unhealthy participants at each of the 83,442 microparcels.

Weighted degree was calculated from the unthresholded Wasserstein similarity graph using the `exp(−W)` transformation.

### 1.2 Participants

- Healthy: 137
- Unhealthy: 130
- Total: 267
- Parcels per participant: 83,442

Participants were selected using the predefined OASIS3 cohort split.

### 1.3 Analysis steps

#### Step 1: Load weighted-degree vectors

For every participant, the complete 83,442-element `weighted_degree.dat` vector was loaded using a common parcel order.

#### Step 2: Construct the group matrices

Two matrices were constructed:

- Healthy: `137 × 83,442`
- Unhealthy: `130 × 83,442`

Each column represented one parcel and each row one participant.

#### Step 3: Parcel-wise group comparison

For every parcel, an independent two-sided Welch t-test compared healthy and unhealthy participants.

Welch’s test was used because it does not assume equal group variances.

#### Step 4: Effect-size estimation

Hedges’ g was calculated for every parcel as:

`Unhealthy − Healthy`

Therefore:

- Negative Hedges’ g means lower weighted degree in unhealthy participants.
- Positive Hedges’ g means higher weighted degree in unhealthy participants.

#### Step 5: Multiple-comparison correction

Benjamini–Hochberg FDR correction was applied across all 83,442 parcel-wise tests at `q < 0.05`.

### 1.4 Parcel-wise results

- Parcels tested: 83,442
- BH-FDR survivors: 1,371
- Percentage surviving: 1.64%
- Negative effects: 1,356
- Positive effects: 15
- Median Hedges’ g among significant parcels: `−0.467`
- Strongest absolute effect: `g = −0.733`

Thus, 98.9% of significant parcels had lower weighted degree in unhealthy participants.

### 1.5 Strongest individual parcels

The strongest parcel-level effect was found in the left amygdala:

- Parcel: `label_18/roi_0053`
- Healthy mean: 0.736
- Unhealthy mean: 0.705
- Hedges’ g: `−0.733`
- BH-FDR q: `0.0010`

Other large parcel effects occurred in:

- Left amygdala
- Bilateral thalamus
- Bilateral lateral ventricles
- Left cerebral cortex
- Right ventral diencephalon

Several individual parcels reached effect sizes between approximately `g = −0.64` and `−0.73`.

### 1.6 Anatomical distribution of significant parcels

| Region | FDR parcels | Direction |
|---|---:|---|
| Left lateral ventricle | 467 | All lower in unhealthy |
| Right lateral ventricle | 190 | All lower |
| Right thalamus | 116 | All lower |
| Left thalamus | 72 | All lower |
| Left caudate | 68 | All lower |
| Left putamen | 67 | All lower |
| Right putamen | 60 | All lower |
| Left cerebral cortex | 58 | 56 lower, 2 higher |
| Right cerebral cortex | 47 | All lower |
| Right caudate | 38 | All lower |
| Right ventral diencephalon | 29 | All lower |
| Brain-stem | 25 | 15 lower, 10 higher |
| CSF | 24 | 23 lower, 1 higher |
| Left ventral diencephalon | 20 | All lower |
| Left cerebellar cortex | 20 | All lower |
| Right accumbens | 19 | All lower |

### 1.7 Interpretation

The parcel-wise results indicate a predominantly negative weighted-degree effect in unhealthy participants. This suggests that many affected parcels are less strongly connected to morphologically similar parcels in the Wasserstein graph.

The largest concentrations of significant parcels were found in the lateral ventricles, thalamus, caudate, and putamen.

The thresholded analyses currently stored in the repository were generated using an older atlas run and 121 participants per group. They should be rerun using the current connected atlas before formal reporting.

---

# 2. Regional analysis: `regional_mean.ipynb`

## 2.1 Objective

The regional analysis reduced the 83,442 microparcels to 26 anatomical structures.

Two measurements were examined separately:

1. Direct log-Jacobian values
2. Wasserstein-graph weighted degree

This analysis tested whether parcel-level changes form coherent effects at the level of major anatomical structures.

## 2.2 Participants

The regional analysis used the same groups:

- Healthy: 137
- Unhealthy: 130
- Total: 267
- Anatomical regions: 26

## 2.3 Regional aggregation

### Step 1: Identify parcels belonging to each region

The parcel lookup table defined the beginning and ending parcel indices for each of the 26 anatomical labels.

### Step 2: Calculate one regional value per participant

For each participant and region:

- The regional mean was calculated as a voxel-count-weighted average of its parcel values.
- The regional median was calculated from the parcel medians within that region.

Weighting the mean by parcel size prevents small and large parcels from contributing equally when their voxel counts differ.

### Step 3: Compare regional means

A two-sided independent Welch t-test compared the subject-level regional means between healthy and unhealthy participants.

### Step 4: Compare regional medians

A two-sided Mann–Whitney U test compared the subject-level regional medians.

### Step 5: Calculate effect sizes

Hedges’ g was calculated from the regional means as:

`Unhealthy − Healthy`

### Step 6: Correct for multiple comparisons

Benjamini–Hochberg FDR correction was applied separately to the 26 Welch and 26 Mann–Whitney tests.

---

# 3. Direct regional log-Jacobian results

## 3.1 Overall results

- Welch FDR survivors: 14/26 regions
- Mann–Whitney FDR survivors: 12/26
- Surviving both tests: 11/26

Positive Hedges’ g indicates higher log-Jacobian in unhealthy participants, consistent with relative expansion. Negative Hedges’ g indicates relative contraction.

## 3.2 Strongest effects

| Region | Hedges’ g | Welch q | Interpretation |
|---|---:|---:|---|
| Right inferior lateral ventricle | +0.619 | 0.000027 | Relative expansion |
| Left inferior lateral ventricle | +0.514 | 0.000567 | Relative expansion |
| Right putamen | −0.451 | 0.002385 | Relative contraction |
| Left lateral ventricle | +0.441 | 0.002541 | Relative expansion |
| Left putamen | −0.430 | 0.002750 | Relative contraction |
| Right thalamus | −0.409 | 0.004221 | Relative contraction |
| Left thalamus | −0.382 | 0.007168 | Relative contraction |
| Right lateral ventricle | +0.378 | 0.007413 | Relative expansion |
| Left accumbens | −0.353 | 0.012539 | Relative contraction |
| Left amygdala | −0.337 | 0.017639 | Relative contraction |
| Right accumbens | −0.328 | 0.018569 | Relative contraction |

## 3.3 Interpretation

The strongest direct morphometric finding was bilateral ventricular expansion, particularly in the inferior lateral ventricles.

This was accompanied by bilateral contraction-like effects in:

- Putamen
- Thalamus
- Accumbens
- Amygdala

These results represent relatively direct anatomical changes because they are calculated from the Jacobian values themselves.

---

# 4. Regional weighted-degree results

## 4.1 Overall results

- Welch FDR survivors: 16/26 regions
- Mann–Whitney FDR survivors: 16/26
- Surviving both tests: 16/26
- Regions with `|g| ≥ 0.2`: 20/26
- Regions with `|g| ≥ 0.5`: 2/26

All 26 regional Hedges’ g estimates were negative. Weighted degree was therefore consistently lower in unhealthy participants, although not every region reached statistical significance.

## 4.2 Strongest effects

| Region | Hedges’ g | Welch q |
|---|---:|---:|
| Left lateral ventricle | −0.546 | 0.000328 |
| Left caudate | −0.515 | 0.000515 |
| Right thalamus | −0.468 | 0.001435 |
| Right caudate | −0.455 | 0.001435 |
| Right lateral ventricle | −0.453 | 0.001435 |
| Left thalamus | −0.441 | 0.001691 |
| Left hippocampus | −0.438 | 0.001707 |
| Left putamen | −0.426 | 0.001909 |
| Right putamen | −0.422 | 0.001922 |
| Right hippocampus | −0.391 | 0.004347 |
| Right accumbens | −0.378 | 0.005266 |
| Left accumbens | −0.376 | 0.005266 |

## 4.3 Interpretation

The strongest weighted-degree reductions involved:

- Bilateral lateral ventricles
- Bilateral caudate
- Bilateral thalamus
- Bilateral putamen
- Bilateral hippocampus
- Bilateral accumbens

The bilateral caudate result is particularly interesting because it was strong in weighted degree but not significant in the direct regional Jacobian analysis. This may indicate altered morphometric similarity without a correspondingly large change in average regional Jacobian.

---

# 5. Convergence between parcel-wise and regional findings

## 5.1 Parcel-to-region weighted-degree agreement

Regional weighted-degree effect sizes strongly agreed with the average parcel-level effect sizes:

- Spearman `ρ = 0.902`
- `p = 3.0 × 10⁻¹⁰`
- Matching effect direction in all 26 regions

Every region surviving the regional weighted-degree Welch test contained at least one FDR-significant parcel.

This indicates that regional averaging summarizes an already-present parcel-level pattern rather than producing an unrelated result.

## 5.2 Convergence between direct Jacobian and weighted degree

Eleven regions survived regional Welch FDR for both measures:

- Left and right lateral ventricles
- Left and right thalamus
- Left and right putamen
- Left and right accumbens
- Left and right amygdala
- Right ventral diencephalon

The most consistent combined findings were:

| Structure | Direct Jacobian | Weighted degree |
|---|---|---|
| Lateral ventricles | Bilateral expansion | Bilateral reduction |
| Thalamus | Bilateral contraction | Bilateral reduction |
| Putamen | Bilateral contraction | Bilateral reduction |
| Accumbens | Bilateral contraction | Bilateral reduction |
| Amygdala | Bilateral contraction | Bilateral reduction |

## 5.3 Important distinction between the metrics

Although they overlap anatomically, direct Jacobian and weighted degree do not measure the same property.

Their regional effect sizes were only weakly correlated:

- Spearman `ρ = 0.260`
- `p = 0.199`

Direct Jacobian measures local expansion or contraction. Weighted degree measures a parcel’s similarity-based connectivity to the rest of the morphometric graph.

Accordingly:

- Inferior lateral ventricular expansion was primarily a direct Jacobian finding.
- Bilateral caudate and hippocampal abnormalities were primarily weighted-degree findings.

---

# 6. Overall conclusion

The analyses provide converging evidence for structural and morphometric-network differences between healthy and unhealthy OASIS3 participants.

The strongest direct anatomical findings are:

- Bilateral ventricular expansion
- Bilateral putaminal contraction
- Bilateral thalamic contraction

The strongest morphometric-network findings are:

- Reduced weighted degree in the lateral ventricles
- Bilateral caudate reductions
- Bilateral thalamic reductions
- Bilateral putaminal reductions
- Bilateral hippocampal reductions

The convergence across parcel-wise, regional, parametric, non-parametric, and bilateral analyses strengthens the findings. However, the results should remain described as group associations until covariate-adjusted models are independently reproduced with diagnostics and sensitivity analyses.

---

# 7. Spatially constrained Leiden community analysis

## 7.1 Objective

This analysis asked a different question from the parcel-wise and regional tests: within an individual, which *spatially adjacent* microparcels have similar log-Jacobian distributions? It was used to describe whether the most robust anatomical ROIs form one local morphometric module or are consistently divided across several modules.

## 7.2 Graph construction and community detection

- Nodes were the 83,442 atlas microparcels.
- The graph contained an edge only when two parcels shared a voxel face in the common atlas. Spatially distant parcels could therefore not be assigned to the same community merely because their distributions were similar.
- For each subject and spatial-adjacency edge, the edge weight was `exp(-Wasserstein distance)` between the two parcels' Jacobian-value distributions.
- Leiden was run separately for each of the 267 participants at gamma = 1.00, producing approximately 40--50 communities per participant.

Leiden community numbers are arbitrary for each participant. Consequently, community 0 for one participant was not compared directly with community 0 for another participant.

## 7.3 ROI-to-community allocation

Interpretation was focused on the 11 ROIs that survived both regional direct-Jacobian tests in the current analysis notebook: bilateral lateral and inferior lateral ventricles, bilateral thalamus, bilateral putamen, bilateral accumbens, and brain-stem. For each subject, each ROI's parcels were counted within every Leiden community and converted to an ROI-relative community share:

`ROI community share = parcels from an ROI in a community / total parcels in that ROI`

Assignments representing less than 5% of an ROI's parcels were removed as minor boundary spillover. The largest remaining share is the ROI's dominant-community share; its number of retained communities describes local fragmentation.

## 7.4 Initial descriptive findings

Across the 267 participants, several selected ROIs were highly cohesive:

| ROI | Mean dominant-community share | Mean retained communities |
|---|---:|---:|
| Left putamen | 99.6% | 1.00 |
| Right thalamus | 99.6% | 1.02 |
| Right putamen | 99.8% | 1.00 |
| Left thalamus | 98.5% | 1.07 |

The brain-stem and accumbens regions were more consistently divided across multiple local communities:

| ROI | Mean dominant-community share | Mean second-largest share | Mean retained communities |
|---|---:|---:|---:|
| Brain-stem | 71.4% | 20.4% | 2.57 |
| Left accumbens | 71.1% | 27.4% | 1.93 |
| Right accumbens | 67.7% | 30.9% | 1.85 |

For example, in `sub-0002`, the right thalamus and left thalamus were almost entirely assigned to community 0, whereas the brain-stem was split principally between communities 6 (78.6%) and 0 (20.4%). This is an anatomical allocation description within that participant; it does not imply that the same numeric community labels recur in other participants.

## 7.5 Interpretation and limitation

The community analysis adds spatial-organisation information to the prior results: it identifies which altered anatomical regions behave as compact local morphometric modules and which show reproducible internal subdivision. It does not yet demonstrate a Healthy--Unhealthy difference in community organisation. Exploratory group comparisons of dominant-community share did not survive multiple-comparison correction across the 11 ROIs.