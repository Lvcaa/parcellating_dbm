# Phase 1 Feasibility Report

**Date**: March 2026
**Cluster**: FBK iRBio-3 — 144 CPU cores, 2015 GB RAM
**Status**: Stage A validated. Stage B partially validated. Single-subject pilot pending.

---

## What we tested

The pipeline has two computationally distinct stages per subject:

- **Stage A** — build a full 36,000 × 36,000 Pearson correlation matrix from parcel Jacobian profiles, then extract edges with r > 0.5
- **Stage B** — run Leiden community detection on the resulting weighted graph

We ran two benchmarks on the cluster using synthetic data:

1. `phase1_feasibility.py` — end-to-end Stage A + Stage B on random data (n=36k, 27 voxels/parcel, r > 0.5)
2. `fake_matrix_stress_test_sparse.py` — isolated Stage B stress test at 30% graph density (the worst-case estimate from the original project plan)

---

## Results

### Stage A — Correlation matrix pipeline

| Metric | Result |
|---|---|
| Matrix size (float32, dense) | 4.83 GB |
| BLAS compute time (C = X @ X.T) | 5.9s |
| Threshold + edge extraction | 0.8s |
| **Stage A total** | **6.7s** |
| Edges surviving r > 0.5 | 2,560,766 (0.40% density) |
| Peak RAM | 5.08 GB |

**Verdict: GO.** The correlation matrix step — the new bottleneck introduced by this design — is fast and memory-efficient. It is not a concern at any realistic subject count.

Note: random data produces ~0.4% density after r > 0.5 thresholding (by chance, ~0.5% of random 27-dim unit vector pairs will have r > 0.5). Real Jacobian data will produce more edges due to spatial autocorrelation and disease-driven network structure.

### Stage B — Leiden community detection

| Scenario | Edges | Leiden time | RAM | Verdict |
|---|---|---|---|---|
| Random data, 0.4% density | 2.56M | 32.8s | 5.1 GB | **PASS** |
| Synthetic graph, 30% density | 194M | 6.88 hours | 45.7 GB | **FAIL** |

**Accept criterion**: Leiden < 5 min, RAM < 64 GB.

The 30% density case fails by a factor of 83×. Leiden is super-linear in edge count — going from 2.6M to 194M edges (75× more) caused a ~755× slowdown, consistent with O(E log E) or worse scaling in dense regimes.

---

## What this tells us

The 30% density target in the project plan was inherited from an older design that controlled graph density via k-NN (keep only the top-k edges per node). We switched to an r > 0.5 threshold instead, which is more principled biologically — it retains only pairs with genuine morphometric covariation — and naturally suppresses density.

The benchmark confirms that **Leiden is only viable when density is kept low**. Based on the two data points:

- At 0.4% density → 33s (comfortably within budget)
- At ~1–2% density → estimated 2–5 min (borderline)
- At ≥ 5% density → estimated hours (over budget)

Real Jacobian data will land somewhere between 0.4% (random) and potentially 10–20% (strong atrophy networks). We do not yet know where.

---

## Open questions

The single-subject pilot — the remaining item in Appendix B — is now the most critical next step. It will determine:

1. **What edge density real Jacobian data actually produces** after r > 0.5 thresholding
2. Whether the current pipeline is viable end-to-end, or whether we need to adapt

If real density exceeds ~2%, the leading mitigation options are:

- **Raise the r threshold** (e.g., r > 0.7) to reduce edge count at the cost of missing weaker but potentially real covariation
- **Hierarchical community detection** — run Leiden independently within macroscale regions (left cortex, right cortex, subcortex) rather than across all 36k nodes globally; this reduces the per-run graph size dramatically
- **Switch algorithm** — Infomap scales significantly better than Leiden on dense graphs and is a reasonable alternative if modularity-based detection proves intractable

---

## Appendix B Checklist

- [x] Infrastructure specs documented (144 cores, 2015 GB RAM)
- [x] Stage A benchmark completed — GO (6.7s, 5 GB)
- [x] Stage B benchmark at real-data worst case — NO-GO at 30% density
- [x] Stage B benchmark at r-threshold density — PASS at 0.4% (random data)
- [ ] Single-subject pilot — requires real MRI data
- [ ] Segmentation timing
- [ ] Go/No-Go decision — pending real-data density estimate
