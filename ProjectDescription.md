# 🧠 Project Overview (Rewritten Notes)

## 1. Goal: Morphometric Analysis via Deformation Fields

We start with an image already registered in **MNI space**.

- When performing **nonlinear registration**, we obtain a **warp field (deformation field)**.
- This field tells us, for each voxel:
    - how much it had to **expand (inflate)** or
    - **contract (shrink)**
        
        to match the MNI template.
        

👉 This deformation is a **measure of brain morphology**.

- Example:
    - In diseases like **Parkinson’s** or **Alzheimer’s**, we expect **atrophy**.
    - The deformation field highlights **where tissue has shrunk or expanded**, revealing affected regions.

---

## 2. From Voxels to Brain Regions

- The deformation is defined **per voxel**.
- But brain changes are **not isolated** — they occur in **functionally connected networks**.

👉 Therefore:

We want to model **dependencies between regions**, not just individual voxels.

### Problem:

- Too many voxels (~2 million)

### Solution:

- Use **parcellation**
    - Reduce to ~4,000 regions (or similar scale)
    - Each region aggregates multiple voxels

---

## 3. Brain Partitioning Strategy

From the deformation field:

1. Split the brain into **macro areas**:
    - Left cortex
    - Right cortex
    - Subcortical regions
2. Perform **micro-parcellation**:
    - Divide into smaller regions (parcels)
    - Keep cortical and subcortical structures separate
3. Recombine all parcels into a full brain representation

---

## 4. Graph Construction (Core Idea)

For each subject:

- Each **node** = a parcel (e.g. ~36k parcels)
- Each parcel contains ~27–36 voxels

### Build edges:

- Compare **morphometric profiles** within parcels
- Define similarity between parcels

👉 Result:

A **graph** where:

- Nodes = parcels
- Edges = similarity of deformation patterns

---

## 5. Network Analysis

- Apply **community detection**
- Idea: identify **networks of regions** that share similar atrophy/expansion patterns

👉 Larger matrices → better resolution of patterns

---

# 🧪 Your Proposed Experiment (Clarified)

You want to:

1. Use **subject-specific data**
2. Apply **MNI-based segmentation**
3. Build graphs for each subject
4. Evaluate:
    - scalability
    - computational cost

---

# ⚙️ FIRST STEP (Feasibility Study)

## Objective:

Test if your pipeline is computationally feasible on the **IRBIO cluster (CPU/RAM)**

---

## Step 1: Simulate a Large Graph

- Create a fake adjacency matrix:
    - Size: **90,000 × 90,000**
    - Density: **30%**
    - Values: random

👉 This approximates your real problem size

---

## Step 2: Run Community Detection

- Use:
    - `Igraph`
    - Undirected graph
- Algorithm:
    - **Leiden communities**

Pseudo-call:

```
LEIDEN_COMMUNITIES(Graph)
```

👉 Measure:

- Execution time
- Memory usage

---

## Step 3: Evaluate Segmentation Cost

- Measure time for:
    - Brain segmentation into parcels

⚠️ Note:

- This step is less critical (done once per dataset)

---

## Step 4: Data Selection Constraint

- Exclude:
    - ❌ White matter
- Focus only on:
    - ✅ Gray matter (cortex + subcortex)

---

## 6. Output Goal

- Identify:
    - Which regions (or networks) show **atrophy patterns**
- Produce:
    - A **list of relevant brain regions**

---

# 🧭 What You Should Do Next (Simple Plan)

### ✅ Immediate actions:

1. **Generate synthetic matrix (90k x 90k)**
2. **Run Leiden community detection**
3. **Benchmark:**
    - Time
    - RAM usage
4. Decide:
    - Is this feasible on your cluster?

---

### 🔜 After that:

- Implement real pipeline:
    1. Compute deformation fields
    2. Apply parcellation
    3. Extract parcel-level features
    4. Build graph
    5. Run community detection