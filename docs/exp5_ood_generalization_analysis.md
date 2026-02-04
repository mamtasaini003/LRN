# Experiment 5: OOD Forcing Generalization

**Date:** 2026-02-02  
**Script:** `examples/exp5_ood_forcing.py`  
**Status:** ✅ Completed

---

## 1. Objective

Evaluate LRR-FNO's ability to generalize to unseen domain geometries and forcing distributions. Trained on **Circle** domain, tested on 5 other domains.

Hypothesis: LRR's latent anchoring learns a more generalizable physical manifold that transfers better to OOD scenarios.

---

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Training Domain** | Circle (160 samples) |
| **Test Domains** | Circle (ID), Ellipse (x3), Cone, Semicircle |
| **Resolution** | 64×64 |
| **Epochs** | 100 (2-stage) |

---

## 3. Results (Relative L2 Error)

| Domain | Type | FNO | LRR-FNO | Improvement |
|--------|------|-----|---------|-------------|
| **Circle** | **In-Dist** | 0.1257 | 0.1118 | **+11.1%** |
| **Ellipse (AR=1.5)** | OOD | 0.0354 | 0.0271 | **+23.5%** |
| **Ellipse (AR=2.0)** | OOD | 0.1215 | 0.1045 | **+14.0%** |
| **Ellipse (AR=2.5)** | OOD | 0.0384 | 0.0235 | **+38.8%** |
| **Cone-F** | OOD | 0.1321 | 0.1207 | **+8.6%** |
| **Semicircle-F** | OOD | 0.0707 | 0.0583 | **+17.6%** |

### Key Findings:
1. **Consistent OOD improvement:** LRR outperformed FNO on **all 5** unseen domains
2. **Average OOD improvement:** +20.5%
3. **Strongest generalization:** Ellipse domains (+14% to +39%)

---

## 4. Visualization

![OOD Generalization](images/Circle_ood_generalization.png)

---

## 5. Interpretation

LRR's training process (Stage 1 with InfoNCE) forces the backbone to learn features that align with solution encodings. Since solution encodings capture fundamental physical properties rather than dataset-specific artifacts, the learned features are more robust to domain shifts.

---


---

## 6. Artifacts

| File | Description |
|------|-------------|
| `examples/exp5_ood_forcing.py` | Experiment script |
| `results/exp5_ood_forcing/Circle_ood_generalization.png` | Comparison plot |

---

## 7. Reproducibility

To reproduce these results with the full dataset, refer to [Reproducibility Guide](reproducibility.md).

**Quick Reproduce:**
```bash
python3 examples/exp5_ood_forcing.py --train dataset/Circle.nc --epochs 500 --max_samples -1
```
