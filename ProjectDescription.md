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

## 4. Parcel-Level Feature Construction (Key Optimization Step)

Instead of using full voxel distributions or full connectivity rows:

For each parcel compute **compact morphometric descriptors**:

- Mean deformation
- Standard deviation
- Median
- Quantiles (e.g. 25%, 75%)
- Skewness / distribution shape
- Optional: local variability metrics

👉 Result:
- Each parcel becomes a **low-dimensional feature vector (≈5–20 features)**

🎯 Purpose:
- Massive reduction in computational cost
- Avoids constructing large dense matrices

---

## 5. Graph Construction (Sparse Representation)

### Goal:
Model **dependencies between parcels** without building a dense matrix

### Steps:

1. Compute similarity between parcel feature vectors:
   - correlation
   - cosine similarity

2. Apply **sparsification strategy**:
   - keep only **top-k neighbors per parcel**, OR
   - threshold weak similarities

👉 Result:
- **Sparse weighted graph**

Where:
- Nodes = parcels
- Edges = strongest morphometric similarities only

⚠️ Critical:
- Avoid full N × N dense matrix (infeasible at large N)

---

## 6. Network Analysis (Main Method)

Apply **Leiden community detection** on the sparse graph.

### Output:
- Groups of parcels with **coordinated deformation patterns**

### Interpretation:
- Candidate **atrophy networks**
- Reflect structured morphological dependencies

👉 This is the main biologically meaningful result

---

# 🧪 Feasibility Study (Revised)

## Objective:
Evaluate computational feasibility on **IRBIO cluster (CPU + RAM)**

---

## Step 1: Synthetic Graph Benchmark (Sparse, Not Dense)

Instead of dense 90k × 90k matrix:

- Generate **sparse graphs** with:
  - Nodes: 10k → 50k → 90k (progressively)
  - Degree: fixed (e.g. k = 20–100)
  - Weighted edges (random)

👉 Mimics real sparsified morphometric graph

Measure:
- Memory usage
- Runtime of graph construction
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

## Step 3: Feature Extraction Cost

Evaluate cost of:

- computing parcel-level morphometric summaries
- aggregating voxel data → parcel descriptors

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

- Sparse graph + Leiden

Purpose:
- capture **network-level morphometric organization**

---

# 🚫 What to Avoid

- Full dense adjacency matrices (N × N)
- Full row-by-row correlation matrices
- Clustering raw adjacency rows (too high dimensional)
- Gradient-based / deep learning approaches (for now)

👉 These approaches are computationally prohibitive in the feasibility phase

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

> Compress voxel-level deformation into parcel-level features, construct a sparse similarity graph, and apply Leiden community detection.

---

### Pipeline hierarchy:

1. **Mini-batch K-means (features)** → fastest baseline  
2. **Mini-batch K-means (reduced profiles)** → intermediate  
3. **Sparse graph + Leiden** → main method  

---

### Key feasibility principle:

> Avoid dense all-to-all computations. Use compact representations and sparse structures.
