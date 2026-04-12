# Research Plan: Brain Morphometry Networks via Deformation-Based Graph Analysis

**Status**: Pre-implementation feasibility study  
**Last Updated**: March 2026  
**Primary Goal**: Establish a scalable pipeline for detecting atrophy patterns in neurodegenerative disease via deformation field analysis and community detection in morphometric networks.

---

## 1. Scientific Motivation

### Problem Statement
Current voxel-wise morphometric analyses (e.g., Jacobian determinants from nonlinear registration) identify local brain atrophy but fail to capture **functional and structural dependencies** between regions. In neurodegenerative diseases (Parkinson's, Alzheimer's), atrophy and volume changes are not isolated; they follow **common functional networks** with consistent patterns.

In practice:
The usual approach measures where the brain has changed shape or volume after aligning a patient’s MRI to a standard brain. That gives a very detailed map of local expansion or shrinkage, but it treats each location mostly in isolation.

> What we are trying to do  is treat the brain **more like a network**.
First, we split the brain into many very small regions. For each little region, we measure how the tissue has deformed relative to a template brain. Then we compare regions to each other: if two regions show similar deformation patterns, we connect them in a graph. In that graph, regions are the nodes and similarity is the edge weight.

### Hypothesis
By modeling the brain as a **weighted graph** where:
- **Nodes** = anatomically parcellated regions (e.g., 36k microregions)
- **Edges** = morphometric similarity (deformation field profiles within each parcel)

We can apply **community detection algorithms** to reveal network-level atrophy patterns that are robust to noise and more interpretable than voxel-level statistics.

### Key Insight
Coarse parcellations aggregate 2M voxels into 4k regions, losing fine-grained structure. We propose a **hierarchical parcellization + graph-based aggregation** approach:
1. Build deformation field maps (nonlinear registration to MNI template)
2. Hierarchically segment: cortex (L/R) → subcortex → microstructures
3. Compute morphometric similarity within local neighborhoods (15/18/27/36 voxels per parcel)
4. Construct weighted adjacency matrix based on morphometric correlation
5. Apply community detection (Leiden algorithm) to identify robust network modules

---

## 2. Methodology

### 2.1 Data Processing Pipeline

#### Step 1: Nonlinear Registration & Deformation Field Extraction
- **Input**: Subject-specific structural MRI (T1-weighted)
- **Output**: Deformation field warp map (3D array of displacement vectors)
- **Method**: Nonlinear registration to MNI 152 template (e.g., SyN, DARTEL)
- **Measurement**: Jacobian determinant per voxel = morphometric measure of local volume change
  - Values > 1: local expansion (hyperplasia)
  - Values < 1: local atrophy
  - Values ≈ 1: no morphological change
- **Inclusion**: All tissue (grey matter, white matter initially; **exclude white matter in final analysis**)

#### Step 2: Hierarchical Brain Segmentation
Segment native space using MNI-aligned atlases:

**Macroscale** (independent processing):
- Left cortex
- Right cortex
- Subcortical structures (thalamus, striatum, hippocampus, etc.)
- Cerebellum (optional)

**Microscale** (within each macroscale region):
- Apply fine-grained parcellation atlas (e.g., Schaefer 400, Power 264, or custom)
- **Exclude white matter** and ventricles from all analysis

**Indexing**: Each parcel contains approximately 27–36 voxels; validate this assumption empirically.

#### Step 3: Morphometric Feature Extraction per Parcel
For each parcel **i** in the segmentation:
- Extract all Jacobian values from voxels within the parcel (let's say V_i = 27–36 voxels)
- Compute summary statistics:
  - **Mean Jacobian** J̄_i
  - **Std Dev** σ_i
  - **Min/Max** to capture heterogeneity
  - **Voxel-wise correlation profile** (27–36 dimensional vector per voxel)

**Rationale**: Captures both aggregate morphometric change and internal structural heterogeneity.

---

### 2.2 Graph Construction

#### Node Definition
- **Nodes**: All parcels in segmentation (target: ~36k microregions; validate scalability)
- **Exclusion**: White matter, ventricles, cerebrospinal fluid

#### Edge Weight Calculation
For each subject, compute pairwise **morphometric similarity** between parcels:

**Option A** (Recommended for pilot): Pearson correlation of voxel-wise Jacobian profiles
```
W[i,j] = corr(J_voxels[parcel_i], J_voxels[parcel_j])
W[i,j] ∈ [-1, 1]
```

**Option B** (If Option A insufficient): Euclidean distance in morphometric feature space
```
W[i,j] = 1 / (1 + ||feature_i - feature_j||₂)
W[i,j] ∈ [0, 1]
```

**Density Control**:
- Construct adjacency matrix **A** (36k × 36k)
- Apply sparsity threshold to keep only top **k** correlations per node (e.g., k=5–20)
- **Target density**: ~30% (as specified in feasibility study)
- **Justification**: Reduces noise while preserving robust connectivity patterns

#### Graph Properties
- **Type**: Undirected, weighted
- **Sparsity**: ~30% density (reduces computational burden while retaining signal)
- **Self-loops**: Excluded (no node-to-self edges)

---

### 2.3 Community Detection

#### Algorithm: Leiden
- **Rationale**: Superior modularity and computational efficiency vs. Louvain; better handles large, sparse networks
- **Implementation**: NetworkX or `leidenalg` Python package
- **Input**: Weighted, undirected graph (adjacency matrix)
- **Output**: Community assignments for all nodes (parcels)
- **Key Parameters** (to be tuned):
  - Modularity resolution (default: 1.0; adjust if over/under-clustering)
  - Seed (for reproducibility)

#### Community Validation
- **Modularity score**: Verify that detected communities have high internal density
- **Stability**: Re-run with different seeds; check robustness of assignment
- **Interpretation**: Map communities back to anatomical regions; verify biological plausibility

---

### 2.4 Analysis & Interpretation

#### Group-Level Analysis (Cross-Subject)
For diseased vs. control cohorts:
1. Compute community assignments for each subject
2. Consensus clustering: identify "stable" communities present in ≥70% of subjects
3. Statistical test (e.g., permutation test): Do community Jacobian profiles differ between groups?
4. Identify disease-specific atrophy networks

#### Robustness Checks
- **Parcellation sensitivity**: Repeat with 2–3 different atlas definitions
- **Threshold sensitivity**: Vary density (20%, 30%, 40%) and re-detect communities
- **Algorithm sensitivity**: Compare Leiden vs. Louvain vs. Greedy modularity optimization

---

## 3. Implementation Plan

### 3.1 Phase 1: Feasibility Study (Current)
**Duration**: 1–2 weeks  
**Goal**: Validate computational scalability on target infrastructure

#### Tasks
1. **Infrastructure Assessment** (FBK iRBio cluster)
   - CPU cores, RAM, GPU availability
   - Storage capacity for intermediate outputs
   - Job scheduling system (SLURM, etc.)

2. **Synthetic Benchmark**
   - Generate random adjacency matrix (90k × 90k, 30% density)
   - Load into NetworkX; benchmark Leiden community detection
   - **Measure**: Runtime, peak memory usage
   - **Accept/reject**: If runtime < 5 min & memory < 64 GB → proceed to Phase 2

3. **Single-Subject Pilot**
   - Select 1 subject from pilot dataset
   - Run full pipeline: registration → segmentation → graph construction → community detection
   - **Measure**: Total runtime per subject, identify bottlenecks
   - **Document**: Code paths, intermediate file sizes

4. **Segmentation Timing** (separate timing study)
   - Time nonlinear registration + Jacobian computation on 1 subject
   - **Note**: One-time cost (not per-analysis), but important for planning

#### Deliverables
- [ ] Feasibility report (computational requirements vs. available resources)
- [ ] Performance profile (CPU, memory, disk) per component
- [ ] Decision: Proceed to Phase 2 (Yes/No/Conditional)

---

### 3.2 Phase 2: Pilot Cohort (Weeks 3–6)
**Goal**: Develop end-to-end pipeline on 10–20 subjects

#### Tasks
1. **Data Preparation**
   - Collect/organize structural MRI data (diseased + controls)
   - Standardize naming, quality checks (motion, artifacts)

2. **Pipeline Development**
   - Implement registration → segmentation → graph construction
   - Version control all scripts (Git)
   - Document hyperparameters, assumptions

3. **Parameter Tuning**
   - Test 2–3 parcellation atlases (Schaefer 400, Power 264, etc.)
   - Test graph density thresholds (20%, 30%, 40%)
   - Test Leiden resolution parameters (0.5, 1.0, 1.5)

4. **Validation**
   - Check anatomical plausibility of detected communities
   - Cross-check against known functional networks (e.g., default mode, sensorimotor)
   - Stability analysis: robustness to parameter changes

#### Deliverables
- [ ] End-to-end pipeline code (documented, reproducible)
- [ ] Parameter tuning report
- [ ] Preliminary community maps (visualizations)

---

### 3.3 Phase 3: Full Cohort Analysis (Weeks 7+)
**Goal**: Scale to full patient cohort; conduct statistical analysis

#### Tasks
1. **Scale Pipeline**
   - Process all subjects (target N = 50–200, depending on availability)
   - Monitor for failures, re-run as needed

2. **Group-Level Analysis**
   - Disease vs. control comparison
   - Consensus community detection
   - Statistical testing (permutation tests, effect sizes)

3. **Publication-Ready Outputs**
   - Network visualizations (brain renderings with community colors)
   - Community morphometric profiles (radar plots, heatmaps)
   - Supplementary materials (atlas overlaps, community memberships)

#### Deliverables
- [ ] Full cohort results
- [ ] Manuscript-ready figures
- [ ] Supplementary code repository (reproducible analysis)

---

## 4. Technical Specifications

### 4.1 Software Stack
- **Image Processing**: ANTs (registration), FSL (segmentation), SPM (Jacobian computation)
- **Graph Analysis**: NetworkX, `leidenalg` or igraph
- **Data Handling**: Pandas, NumPy, scikit-image
- **Visualization**: Matplotlib, Mayavi (3D brain rendering), Plotly
- **Version Control**: Git
- **Documentation**: Jupyter notebooks for reproducibility

### 4.2 Data Formats
- **Input**: NIFTI (.nii.gz) structural MRI
- **Intermediate**: Deformation fields (NIFTI), parcellation masks (NIFTI)
- **Graph**: Adjacency matrices (HDF5 or .npz for efficiency)
- **Output**: Community assignments (CSV), visualizations (PNG/PDF)

### 4.3 Computational Resources
- **Single-subject pipeline**:
  - Registration: ~10–30 min (GPU-accelerated if available)
  - Segmentation: ~5 min
  - Graph construction: ~1–5 min (depending on atlas resolution)
  - Community detection: ~1–10 min (benchmark in Phase 1)
- **Full cohort** (N=100): ~40–60 CPU-days (parallelizable)

---

## 5. Expected Outcomes & Impact

### Primary Outcome
A **validated, scalable pipeline** for identifying atrophy networks in neurodegenerative disease using morphometric graph analysis.

### Secondary Outcomes
1. **Novel biomarkers**: Community-level atrophy profiles for disease classification
2. **Biological insights**: Network patterns of atrophy (e.g., which functional networks affected first?)
3. **Methodological contribution**: Comparison of parcellation strategies for morphometry

### Clinical Relevance
- Early detection of atrophy patterns (potentially before symptom emergence)
- Disease progression monitoring
- Potential therapeutic targets (communities at risk)

---

## 6. Adaptive Methodology (Crucial Note)

**This plan is a scaffold, not a dogma.** As we progress through Phase 1 and Phase 2, we will encounter empirical constraints and opportunities. **We are open to methodological deviations** if they improve robustness, scalability, or interpretability:

### Common Adaptations
- **Graph Construction**: If Pearson correlation is too noisy, switch to partial correlation, mutual information, or graph neural network embeddings
- **Community Detection**: If Leiden under-/over-clusters, test Louvain, spectral clustering, or hierarchical methods
- **Parcellation**: If 36k regions are intractable, aggregate to coarser atlas; if noise dominates, refine to finer resolution
- **Statistical Testing**: Adapt to actual data distribution (permutation tests, Bayesian hierarchical models, etc.)
- **Visualization**: Choose 2D/3D rendering strategies based on findings (e.g., if strong regional patterns, use glass brains; if network focus, use graph layouts)

**Guiding principle**: Fidelity to the scientific question (detecting atrophy networks) > fidelity to initial technical choices.

---

## 7. Milestones & Timeline

| Phase | Task | Duration | Start | End | Status |
|-------|------|----------|-------|-----|--------|
| 1     | Feasibility Study | 1–2 weeks | Now | Week 2 | 🔄 In Progress |
| 1     | Synthetic Benchmark | 3 days | Week 1 | Week 1 | ⏳ Pending |
| 1     | Single-Subject Pilot | 5 days | Week 1 | Week 2 | ⏳ Pending |
| 2     | Data Preparation | 1 week | Week 2 | Week 3 | ⏳ Pending |
| 2     | Pipeline Development | 3 weeks | Week 3 | Week 6 | ⏳ Pending |
| 2     | Parameter Tuning | 2 weeks | Week 4 | Week 6 | ⏳ Pending |
| 3     | Full Cohort | 4+ weeks | Week 7 | Week 11+ | ⏳ Pending |

---

## 8. Key References & Resources

### Relevant Methods
- **Nonlinear Registration & Jacobian**: Ashburner & Friston (2000) VBM; Yushkevich et al. ANTS
- **Community Detection**: Blondel et al. (2008) Louvain; Traag et al. (2019) Leiden
- **Brain Parcellations**: Schaefer et al. (2018) 400-region atlas; Power et al. (2011) 264-region atlas
- **Morphometry in Disease**: Whitwell (2011) neuroimaging review; Thompson & Toga (2002) brain morphometry

### Tools
- [ANTs](http://stnava.github.io/ANTs/) – Image registration
- [FSL](https://fsl.fmrib.ox.ac.uk/) – Brain segmentation & analysis
- [NetworkX](https://networkx.org/) – Graph analysis in Python
- [leidenalg](https://github.com/vtraag/leidenalg) – Leiden community detection
- [Nilearn](https://nilearn.github.io/) – Neuroimaging + machine learning in Python

---

## 9. Contact & Questions

**Primary Investigator**: [Your Name]  
**Institutional Affiliation**: Fondazione Bruno Kessler – NILab  
**Last Updated**: March 2026

---

## Appendix A: List of Regions of Interest (To be populated)

*Note: Add specific anatomical regions of interest, parcellation scheme details, and disease-specific hypotheses here.*

**Example** (template):
- Cortical regions: [List with L/R hemispheric pairs]
- Subcortical targets: Striatum, Thalamus, Hippocampus, Amygdala, Substantia Nigra (Parkinson's focus)
- Cerebellar regions: [As relevant]
- White matter: EXCLUDED from primary analysis

---

## Appendix B: Feasibility Study Checklist

- [ ] Infrastructure specs documented (CPU, RAM, GPU, storage)
- [ ] Synthetic benchmark completed (90k × 90k matrix, 30% density)
- [ ] Leiden runtime benchmarked (target: < 5 min)
- [ ] Peak memory usage documented (target: < 64 GB)
- [ ] Single-subject pipeline timed (registration, segmentation, graph, community detection)
- [ ] Bottleneck identified and documented
- [ ] Go/No-Go decision made for Phase 2