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

## 5. Graph Construction (JSD-Based Morphometric Connectivity)

### Goal:
Build a graph where parcels are connected when they show **similar deformation
profiles** and are both **strongly abnormal**.

### Step 1: Distribution similarity

For each pair of parcels A and B, compute the **Jensen-Shannon divergence (JSD)**
between their Jacobian distributions:

$$
\mathrm{JSD}(P_A, P_B)
$$

The JSD is preferred over correlation here because we are comparing **entire
probability distributions**, not paired voxel samples. It measures how different
two morphometric profiles are in information-theoretic terms.

Convert this divergence into a similarity term:

$$
S_{AB} = 1 - \mathrm{JSD}(P_A, P_B)
$$

So:
- if two parcels have nearly identical deformation distributions, similarity is close to 1
- if their morphometric profiles differ strongly, similarity approaches 0

### Step 2: Anomaly weighting

To emphasize parcels that are not only similar, but also truly abnormal, weight
each edge by the intensity of anomaly in both parcels:

$$
W_{AB}^{\mathrm{anom}} = |J_A - 1| \times |J_B - 1|
$$

where:
- $J_A$ = representative central Jacobian value for parcel A
- $J_B$ = representative central Jacobian value for parcel B

This term increases the weight of edges linking parcels whose deformation
magnitudes are both far from the healthy reference value of 1.0.

### Step 3: Final edge weight

The final morphometric edge weight is:

$$
W_{AB} = (1 - \mathrm{JSD}(P_A, P_B)) \times |J_A - 1| \times |J_B - 1|
$$

👉 Result:
- Nodes = parcels
- Edge weights = similarity of Jacobian distributions, amplified by anomaly intensity

This produces a graph that highlights parcels that are both:
- morphometrically similar
- meaningfully abnormal

---

## 6. Adaptive Sparsification and Community Detection

### Adaptive sparsification

The pairwise weight matrix is initially dense. To suppress weak background
connections and retain only subject-specific abnormal structure, apply an
adaptive threshold:

$$
\mathrm{Threshold} = \mu_{\mathrm{global}} + \sigma_{\mathrm{global}}
$$

where $\mu_{\mathrm{global}}$ and $\sigma_{\mathrm{global}}$ are computed from the
full distribution of edge weights for that subject.

Keep only edges satisfying:

$$
W_{AB} > \mu_{\mathrm{global}} + \sigma_{\mathrm{global}}
$$

👉 Result:
- weak and noisy connections are removed
- the dense matrix becomes a **sparse weighted graph**
- retained edges represent stronger-than-background abnormal similarity


# Potential Issues
- A distribution loses within-parcel spatial layout. Two parcels can have the same histogram but different spatial organization of abnormal voxels.

- All-to-all JSD is still an O(N^2) step, so this is biologically cleaner than correlation but not automatically cheaper computationally.

- The mu + sigma threshold is reasonable as a first heuristic, but it may create very different graph densities across subjects. That could affect comparability and Leiden stability.
### Community detection

Apply **Leiden community detection** on the resulting sparse weighted graph.

### Output:
- Groups of parcels with **coordinated deformation patterns**

### Interpretation:
- Candidate **atrophy networks**
- Reflect structured morphological dependencies

👉 This is the main biologically meaningful result

### Limitations and safeguards

To make the method more robust and scalable, we explicitly include the
following safeguards:

#### A. Preserve some within-parcel spatial structure

A histogram alone does not encode where abnormal voxels are located inside the
parcel. Two parcels may have similar Jacobian distributions but different
spatial organization.

Mitigation:
- augment each parcel representation with a small set of spatial descriptors
- examples: centroid of abnormal voxels, spatial variance, compactness, or a
  simple fragmentation score
- optional extension: build a small number of sub-histograms within coarse
  parcel subzones rather than using one global histogram only

This keeps the distribution-based idea while reducing the risk of treating
spatially distinct abnormalities as equivalent.

#### B. Avoid full all-to-all JSD when possible

Exact pairwise JSD over all parcels scales as $O(N^2)$, which can become
costly at high resolution.

Mitigation:
- use a two-stage graph construction strategy
- first, compute a cheap coarse similarity on lightweight features
- then, for each parcel, retain only a candidate neighborhood of likely matches
- compute exact JSD only within that reduced candidate set

Candidate screening may use:
- coarse histogram embeddings
- median or mean Jacobian similarity
- anomaly magnitude similarity
- optional anatomical neighborhood constraints

This preserves the biologically meaningful JSD comparison while avoiding
unnecessary pairwise evaluations.

#### C. Stabilize sparsification across subjects

The threshold $\mu_{\mathrm{global}} + \sigma_{\mathrm{global}}$ is a useful
subject-adaptive heuristic, but it may yield different graph densities across
subjects, which can affect comparability and Leiden stability.

Mitigation:
- treat $\mu + \sigma$ as an exploratory threshold, not the only one
- compare it against fixed-density strategies such as top-$k$ neighbors per
  parcel or a fixed percentile threshold
- report resulting graph density per subject as a quality-control variable
- prefer density-matched graphs for cross-subject comparisons and statistical
  analyses

Recommended practical strategy:
- exploratory analysis: adaptive threshold $\mu + \sigma$
- primary analysis: fixed top-$k$ or fixed percentile sparsification

This balances sensitivity to subject-specific signal with reproducibility across
the cohort.

---

# 🧪 Feasibility Study (Revised)

## Objective:
Evaluate computational feasibility on **IRBIO cluster (CPU + RAM)**

---

## Step 1: Distribution-Based Graph Benchmark

Benchmark the proposed method with synthetic parcel distributions:

- Nodes: 10k → 50k → 90k (progressively)
- Each node stores a histogram or probability vector
- Pairwise similarity computed via JSD
- Edge weights modulated by anomaly magnitude
- Adaptive thresholding applied per synthetic subject

Measure:
- Memory usage
- Runtime of pairwise JSD computation
- Runtime of sparsification
- Runtime of Leiden

---

## Step 2: Leiden Benchmark

Run:
LEIDEN_COMMUNITIES(Graph)

Using:
- `igraph`
- weighted, undirected graph

Measure:
- Execution time
- Peak RAM usage
- Scaling with:
  - number of nodes
  - graph sparsity

---

## Step 3: Distribution Extraction Cost

Evaluate cost of:

- extracting voxel-wise Jacobian determinants per parcel
- building parcel histograms / density estimates
- computing anomaly summary values per parcel

⚠️ Expected:
- relatively cheap (linear in voxel count)
- not a bottleneck

---

## Step 4: Resolution Study

Test different parcel scales:

- Low resolution (~1k parcels)
- Medium (~5k–10k)
- High (~20k–50k+)

Measure:
- Graph size
- Runtime
- Memory usage
- Stability of detected communities

👉 Identify **maximum feasible resolution**

---

## Step 5: Data Constraint

- Exclude:
  - ❌ White matter
- Include:
  - ✅ Cortical gray matter
  - ✅ Subcortical gray matter

---

# 🔁 Baseline Comparisons

To contextualize results, implement two additional pipelines:

---

## A. Fast Baseline (Feature Clustering)

- Input: parcel feature vectors (no graph)
- Method: **Mini-batch K-means**

Purpose:
- extremely scalable
- sanity check for morphometric grouping

Limitation:
- ignores network structure

---

## B. Intermediate Baseline (Optional)

- Input: richer parcel representations
- Apply dimensionality reduction (e.g. PCA / SVD)
- Cluster with mini-batch K-means

Purpose:
- bridge between feature clustering and graph methods

---

## C. Main Pipeline (Target Method)

- JSD-weighted anomaly graph + Leiden

Purpose:
- capture **network-level morphometric organization**

---

# 🚫 What to Avoid

- Reducing each parcel to the mean Jacobian only
- Using correlation when the object of comparison is a full probability distribution
- Keeping dense all-to-all graphs without sparsification
- Gradient-based / deep learning approaches (for now)

👉 These choices would either discard too much morphometric information or create unnecessary computational burden

---

# 📊 Evaluation Metrics

For each pipeline:

### Computational:
- Runtime
- Peak RAM
- Scaling behavior

### Structural:
- Number of clusters / communities
- Stability across parameter choices

### Biological:
- Anatomical coherence of regions
- Consistency with known atrophy patterns

---

# 🎯 Final Output Goal

- Identify **networks of coordinated deformation**
- Produce:
  - list of relevant brain regions
  - community assignments per parcel
  - subject-level morphometric network structure

---

# 🧭 Final Strategy Summary

### Core idea:

> Represent each parcel by the distribution of Jacobian determinants from the warp, compare parcels with Jensen-Shannon divergence, weight similarities by anomaly intensity, sparsify adaptively, and apply Leiden community detection.

---

### Pipeline hierarchy:

1. **Mini-batch K-means (simple parcel summaries)** → fastest baseline  
2. **Mini-batch K-means (distribution-derived features)** → intermediate  
3. **JSD-weighted anomaly graph + Leiden** → main method  

---

### Key feasibility principle:

> Preserve the full parcel-level deformation profile where it matters, but enforce sparsity after weighting so the resulting graph remains biologically focused and computationally manageable.
