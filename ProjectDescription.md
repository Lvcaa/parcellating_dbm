# 🧠 Project Overview (Refined Plan)

## 1. Goal: Morphometric Analysis via Deformation Fields

We start with images already registered in **MNI space**.

- Nonlinear registration produces a **deformation (warp) field**
- For each voxel, this encodes:
  - local expansion (inflation)
  - local contraction (shrinkage)

👉 This deformation serves as a **proxy for brain morphology**

### Clinical motivation:
- Neurodegenerative diseases (e.g. Parkinson’s, Alzheimer’s)
- Expect **atrophy patterns** (systematic shrinkage)
- Deformation fields reveal **where structural changes occur**

---

## 2. From Voxels to Brain Regions

### Problem:
- Voxel-level data is too large (~2M voxels)
- Morphological changes are **not independent**, but structured across regions

### Solution:
- Apply **brain parcellation**
- Reduce dimensionality to **region-level representation**

Typical scale:
- ~1k – 10k parcels (explored experimentally)

👉 Each parcel aggregates multiple voxels → first compression step

---

## 3. Brain Partitioning Strategy

1. Split into macro compartments:
   - Left cortex
   - Right cortex
   - Subcortical regions

2. Perform **micro-parcellation** within each compartment

3. Recombine into full-brain parcel set

⚠️ Important:
- Test **multiple resolutions** (low / medium / high)
- Avoid jumping directly to maximum resolution

---

## 4. Parcel-Level Feature Construction (Node Features)

Instead of reducing each parcel to a single summary statistic such as the mean,
we represent each parcel by the **distribution of Jacobian determinant values**
measured across all voxels in that parcel.

### Node definition:

For each parcel:

1. Extract all voxel-wise Jacobian determinant values from the nonlinear warp
2. Build a normalized histogram or probability density estimate
3. Represent the parcel as a probability distribution **P(x)**

👉 Result:
- Each parcel is described by its **full morphometric profile**, not just by one number

🎯 Purpose:
- Preserve the **shape** of the deformation pattern
- Distinguish uniform abnormalities from heterogeneous or patchy abnormalities
- Retain richer information for downstream network construction

---

## 5. Edge Definitions (3 Types)

### Goal:
Build subject-level parcel graphs using multiple definitions of pairwise
morphometric similarity so that downstream validation does not depend on a
single edge metric.

For each pair of parcels A and B, compute edges using:

1. **Wasserstein distance**
2. **KL divergence (Gaussian version)**
3. **Median + IQR**

### Edge construction details

For each parcel pair:

- compute **A conditioned on B**
- compute **B conditioned on A**

This keeps the pairwise comparison explicitly directional at the feature
construction stage, even when later graph summaries are compared across
subjects.

### Similarity convention

- For **Wasserstein distance**, convert distance into a similarity score before
  adding the edge to the graph.
- For **KL divergence (Gaussian version)**, use the divergence-derived pairwise
  comparison defined on Gaussian parcel summaries.
- For **Median + IQR**, use robust summary-statistic comparisons rather than
  full-density matching.

👉 Result:
- Nodes = parcels
- Edge weights = pairwise parcel relationships under three alternative
  morphometric definitions

---

## 6. Graph Construction

For each subject and for each edge definition:

- build a graph using the corresponding parcel-to-parcel edge weights
- start with a **fully dense graph**
- later, evaluate **2-3 different thresholding strategies**

Recommended workflow:

1. construct the full dense weighted graph
2. compute graph statistics on the dense version first
3. then test a small set of thresholded variants for sensitivity analysis

This makes it possible to compare how stable downstream results are with
respect to sparsification.

---

## 7. Graph Metric

For each graph, compute the **weighted degree** of every parcel:

$$
\mathrm{WeightedDegree}_i = \frac{\sum_j w_{ij}}{N}
$$

where:

- $w_{ij}$ is the edge weight between parcel $i$ and parcel $j$
- $N$ is the number of nodes in the graph

👉 Primary graph-level readout:
- weighted degree = sum of edge weights / number of nodes

### Interpretation of weighted degree

- High weighted degree means a parcel is morphometrically similar to many
  other parcels.
- Diseased parcels are expected to be more atrophied and to show deformation
  patterns that deviate from the rest of the brain.
- Therefore, diseased parcels are hypothesized to show **lower weighted
  degree** than healthy parcels.

---

## 8. Datasets for Validation

Validation will be carried out on the following datasets:

- **OASIS**
- **Epilepsy dataset**
- **PPMI**
- **Psychiatric datasets**

These datasets provide the test bed for comparing graph-derived parcel metrics
between healthy and diseased groups.

---

## 9. Validation Procedure

For each subject-level graph:

1. compute the **weighted degree** for every parcel
2. split subjects into:
   - **Healthy**
   - **Diseased (137 subjects)**
3. for each parcel:
   - compare weighted degree between healthy and diseased groups
   - run a **t-test**
   - compute **effect size**

Repeat this analysis for:

- **all parcels** (approximately **90k**)
- **all three edge metrics**

👉 Core validation question:
- which parcels show the strongest group differences in weighted degree, and
  how consistent are those differences across edge definitions?

---

# 🧪 Practical Study Design

## Primary analysis

- Build one graph per subject for each of the three edge definitions
- Compute parcel-wise weighted degree
- Perform parcel-wise healthy vs diseased comparisons
- Record both significance and effect size

## Sensitivity analysis

- Start from fully dense graphs
- Later test **2-3 threshold choices**
- Re-run the weighted-degree validation after thresholding

## Scale

- Perform the analysis across the full parcellation
- Target scale: approximately **90k parcels**

---

# 🕰️ Legacy Methods

The following methods remain part of the project as **legacy approaches** from
earlier exploration, but they are no longer the primary validation pathway:

## A. Leiden-Based Graph Clustering

- Sparse weighted graph construction
- **Leiden community detection**

Historical purpose:
- identify communities of parcels with coordinated deformation patterns

Current status:
- retained for comparison and historical reference
- not the main validation endpoint

## B. Mini-Batch Clustering

- **Mini-batch K-means** on parcel-level features
- optional dimensionality reduction before clustering

Historical purpose:
- fast scalable clustering baseline

Current status:
- retained as a legacy baseline
- not the main analysis for the current project description

---

# 🚫 What to Avoid

- Treating one edge definition as definitive before validation
- Discarding the dense-graph analysis too early
- Relying only on clustering outputs when the current endpoint is
  parcel-wise statistical validation
- Gradient-based / deep learning approaches (for now)

👉 The current emphasis is on validating parcel-wise graph statistics across
multiple edge definitions and datasets.

---

# 📊 Evaluation Metrics

### Statistical:

- t-statistics
- effect sizes
- parcel-wise healthy vs diseased differences

### Graph-derived:

- weighted degree per parcel
- sensitivity to threshold choice
- consistency across the three edge definitions

### Practical:

- runtime
- peak RAM
- scalability to approximately 90k parcels

---

# 🎯 Final Output Goal

- Build parcel graphs under three alternative edge definitions
- Compute parcel-wise weighted degree for each subject
- Identify parcels whose weighted degree differs between healthy and diseased
  groups
- Quantify how robust those findings are across datasets and threshold choices

---

# 🧭 Final Strategy Summary

### Core idea:

> Represent each parcel through morphometric summaries derived from the warp,
> construct dense subject-level graphs using three alternative edge
> definitions, compute parcel-wise weighted degree, and validate group
> differences between healthy and diseased subjects.

---

### Active analysis path:

1. **Wasserstein-based graph**
2. **Gaussian KL-based graph**
3. **Median + IQR-based graph**
4. **Weighted-degree validation across parcels and datasets**

---

### Legacy methods kept in scope:

1. **Leiden community detection** → legacy graph method  
2. **Mini-batch K-means** → legacy clustering baseline
