# Phase 1 Feasibility — Design Decisions & Simulation Accuracy

**Status**: Pre-implementation review
**Last Updated**: March 2026
**Scope**: Documents the rationale behind `scripts/benchmarks/phase1_feasibility.py` and identifies gaps between the current simulation and the project plan.

---

## 1. What the Simulation Is Actually Testing

The script has two independent stages that answer two different questions:

| Stage | Question | Data-distribution dependent? |
|---|---|---|
| **Stage A** | Can we hold a 36k×36k float32 matrix in RAM and compute it in time? | **No** — BLAS sgemm timing depends on matrix shape, not values |
| **Stage B** | Can Leiden handle the graph that real data would produce? | **Yes** — edge count matters directly |

This distinction is critical: Stage A is fully representative regardless of whether we use random or real data. Stage B is only representative if the graph size matches expected real-data conditions.

---

## 2. Key Design Decisions

### 2.1 Node count: 36,000 (not 90,000)

Appendix B of the project plan mentions a "90k × 90k matrix" — this was an earlier design estimate that predated the hierarchical segmentation methodology. Section 2.2 of the project plan establishes the principled target:

- MNI brain volume: ~1.2M grey-matter voxels (excluding white matter and ventricles)
- Parcel size: 27–36 voxels (3×3×3 neighbourhood in native space)
- Resulting parcels: ~36,000

The 90k figure should be treated as obsolete. **36,000 nodes is the correct target.**

The RAM implication is significant: at 36k nodes the dense C matrix is `36,000² × 4 bytes ≈ 5.2 GB`. At 90k it would be `90,000² × 4 bytes ≈ 32.4 GB` — both are within the 64 GB SLURM allocation, but 5.2 GB is far less risky.

### 2.2 Voxels per parcel: 27

This maps directly to a 3×3×3 voxel neighbourhood in MNI space, which is the minimum granularity that preserves a meaningful Jacobian profile per parcel. Each row of matrix X is a 27-dimensional vector of Jacobian determinant values — one per voxel in the parcel. This is the correlation vector that defines morphometric similarity between two parcels.

**Why 27 and not more**: using larger parcels (e.g., 64 voxels, 4×4×4) would reduce the number of parcels from 36k to ~18k, losing fine-grained spatial resolution. The tradeoff is documented in the project plan as an adaptive option.

### 2.3 Dense matrix first, then threshold

The correlation matrix C is computed as a full n×n float32 matrix (`X @ X.T`) before any thresholding. This is a deliberate choice:

- **Correctness**: every pairwise correlation is evaluated — no morphometric relationship is ever missed due to sparsification shortcuts
- **BLAS efficiency**: `X @ X.T` is a single highly-optimised BLAS sgemm call (multi-threaded, cache-friendly) — computing only selected pairs would require custom sparse matmul with no BLAS benefit
- **Clinical validity**: thresholding after full computation ensures the r > 0.5 cutoff is applied uniformly across all pairs, not biased by pre-selection

The cost is 5.2 GB of RAM for C. This is well within the 64 GB allocation and is a one-time peak that is immediately freed after thresholding.

### 2.4 r > 0.5 threshold

The r > 0.5 cutoff retains only parcel pairs whose Jacobian profiles are strongly correlated. The clinical rationale:

- Parcels with |r| < 0.5 are not sharing a meaningful deformation pattern — keeping them would add noise to the graph
- r > 0.5 is a standard cutoff in functional connectivity literature and has precedent in morphometric correlation studies
- The threshold is applied to the upper triangle only (no double-counting; self-loops excluded)

The row-by-row thresholding implementation avoids allocating a second n×n boolean mask (which would add ~1.3 GB). Only surviving edge indices and weights are materialised.

### 2.5 Leiden with ModularityVertexPartition, weighted

The project plan specifies Leiden over Louvain for its better-guaranteed modularity and convergence properties (Traag et al. 2019). Using `ModularityVertexPartition` with edge weights `r` preserves the strength of morphometric similarity in the community structure — a strongly correlated pair contributes more to module cohesion than a pair near the 0.5 cutoff.

---

## 3. Where the Simulation Is Accurate

### Stage A — Fully representative

- **Timing**: BLAS sgemm on random float32 data takes the same time as on real Jacobians of the same shape. The hardware bottleneck (memory bandwidth + FLOPs) is identical.
- **Memory**: the peak RSS during `X @ X.T` and the subsequent threshold pass are exact.
- **Thresholding logic**: the row-by-row approach correctly collects edges without re-allocating C.

Stage A is the critical new bottleneck in the pipeline (it did not exist in the old k-NN approach) and the simulation measures it correctly.

---

## 4. Where the Simulation Is NOT Representative — Critical Gap

### Stage B — Leiden density is under-tested

**The misleading comment in the script**: the code says *"Expected ~0 edges on random data"*. This is incorrect.

With 27-dimensional random unit vectors, the expected Pearson r between any two rows follows approximately N(0, 1/(d−1)) = N(0, 1/26), giving a standard deviation of ~0.196. By the normal CDF:

```
P(r > 0.5) ≈ 0.54% per pair
```

At n = 36,000 there are ~648 million pairs. Expected surviving edges:

```
648,000,000 × 0.0054 ≈ 3,500,000 edges   (density ≈ 0.54%)
```

This is vastly above the `MIN_EDGES_FOR_LEIDEN = 1,000` fallback threshold. **The fallback never triggers when running with random data at the default parameters.** Stage B runs Leiden on 3.5 million random edges at 0.54% density.

The problem: real Jacobian data from MNI warps has spatial autocorrelation and disease-driven network structure. The expected real-data density after r > 0.5 thresholding is **5–30%**, as established in the project plan. At 30% density that is:

```
648,000,000 × 0.30 ≈ 194,000,000 edges   (55× more than what the current simulation tests)
```

Leiden's runtime scales super-linearly with edge count. Testing at 0.54% density and asserting feasibility at 30% density is not valid.

### Why this matters

The accept criterion from Appendix B is:
- Leiden runtime < 5 min
- Peak memory < 64 GB

Both depend on graph size. A test at 0.54% density passing those criteria says nothing about whether 30% density would also pass. The current simulation does not provide evidence for the hardest feasibility case.

---

## 5. How to Make the Simulation Representative

### For Stage A: no change needed
The matrix computation test is exact as-is.

### For Stage B: explicitly benchmark Leiden at expected real-data density

The project plan's Section 2.2 target density is **30%** (with the k-NN control approach). After switching to an r-threshold approach, the real density is uncertain (5–30% depending on cohort and disease severity). The conservative benchmark should use **30%** to test the worst case.

**Option A — Use the existing sparse stress test:**
```bash
python scripts/benchmarks/fake_matrix_stress_test_sparse.py \
    --nodes 36000 --density 0.30 --seed 42
```
This gives the authoritative Leiden benchmark at n=36k, 30% density and is already in the repo.

**Option B — Force the fallback in phase1_feasibility.py:**
Submit to the cluster with a very high `--leiden-density` and acknowledge that the fallback will be used (random data gives 0.54% edges, well below the synthetic fallback's 30%):
```bash
BENCHMARK_ARGS="--nodes 36000 --voxels 27 --r-threshold 0.5 --leiden-density 0.30 --save-report"
```
Note: this only works if `MIN_EDGES_FOR_LEIDEN` is raised above 3.5M (the expected random edge count). As currently written, the fallback will not trigger and Leiden will run at 0.54% density even if you pass `--leiden-density 0.30`.

**Recommended approach for the cluster test:**
Run **both** scripts as separate SLURM jobs:
1. `phase1_feasibility.py --nodes 36000 --voxels 27 --r-threshold 0.5 --save-report` → benchmarks Stage A (matrix + threshold) exactly
2. `fake_matrix_stress_test_sparse.py --nodes 36000 --density 0.30` → benchmarks Stage B (Leiden) at worst-case real-data density

---

## 6. Summary Table

| Parameter | Current default | Project plan target | Representative? |
|---|---|---|---|
| Nodes | 36,000 | 36,000 | ✓ |
| Voxels per parcel | 27 | 27–36 | ✓ |
| r threshold | 0.5 | 0.5 (recommended) | ✓ |
| Stage A RAM | 5.2 GB | 5.2 GB | ✓ |
| Stage A timing | BLAS exact | BLAS exact | ✓ |
| Stage B density tested | ~0.54% (random data) | 30% (worst case) | ✗ |
| Stage B fallback default | 10% (rarely triggered) | 30% | ✗ |
| Accept criterion: Leiden < 5 min | Tested at 0.54% only | Must pass at 30% | ✗ |
| Accept criterion: RAM < 64 GB | Tested at 0.54% only | Must pass at 30% | ✗ |

---

## 7. Clinical Grounding of the Simulation

The simulation captures the warp structure of MNI space in the following ways:

**What it gets right:**
- Each parcel's feature vector has length 27 (= 3×3×3 voxel neighbourhood), matching the spatial scale of the MNI registration grid
- The full pairwise correlation matrix is computed — no biological relationship is discarded before thresholding
- The r > 0.5 cutoff mirrors what functional connectivity studies use to define meaningful network edges; at this threshold, only parcels with strongly co-varying Jacobian profiles (i.e., anatomically adjacent or functionally linked deformation patterns) remain
- Graph edge weights are Pearson r values, preserving the graded morphometric similarity rather than binarising the network

**What random data misses:**
- Real Jacobians are log-normally distributed around 1.0 (not standard-normal), with positive skew in atrophic regions
- Adjacent parcels share spatial autocorrelation through the deformation field — a property absent in i.i.d. random data
- Disease-specific patterns (e.g., nigro-striatal pathway in Parkinson's) create structured, high-r clusters that drive the real edge density toward 5–30%

None of these biological differences affect Stage A timing or memory — BLAS sees the same shape regardless. They do affect how many edges survive Stage A. This is why the Stage B fallback (synthetic graph at expected density) exists, and why it must be tested at 30% rather than 0.54%.

