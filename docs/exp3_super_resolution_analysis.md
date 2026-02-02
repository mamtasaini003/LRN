# Experiment 3: Zero-Shot Super-Resolution

**Date:** 2026-02-02  
**Script:** `examples/exp3_super_resolution.py`  
**Status:** ✅ Completed

---

## 1. Objective

Test whether LRR-FNO generalizes better across resolutions by training on 64×64 and evaluating on 128×128 without retraining. This tests whether latent anchoring helps models learn a continuous physical manifold.

---

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Training Resolution** | 64×64 |
| **Test Resolutions** | 64×64, 128×128 |
| **Dataset** | Circle.nc |
| **Training Samples** | 160 |
| **Test Samples** | 40 |
| **Epochs** | 50 (2-stage: 36+14) |

---

## 3. Training Protocol

| Stage | Epochs | LR | Loss |
|-------|--------|-----|------|
| Stage 1 | 36 | 1e-3 | NCE + MSE |
| Stage 2 | 14 | 1e-4 | MSE only |

---

## 4. Results

| Model | 64×64 (In-Dist) | 128×128 (Zero-Shot) | Degradation |
|-------|-----------------|---------------------|-------------|
| FNO | 0.126 | 0.370 | **+193.98%** |
| LRR-FNO | 0.112 | 0.262 | **+134.25%** |

### Key Findings:
1. **In-distribution:** LRR improves by +11.1%
2. **Zero-shot 128×128:** LRR improves by +29.3%
3. **Degradation rate:** LRR degrades 31% less than FNO

---

## 5. Visualization

![Super-Resolution](file:///home/mamta/work/LRN/results/exp3_super_resolution/Circle_super_resolution.png)

---

## 6. Interpretation

LRR's latent anchoring encourages the model to learn resolution-invariant representations. The solution encoder captures the fundamental structure of the solution field, which transfers better to higher resolutions.

---

## 7. Artifacts

| File | Description |
|------|-------------|
| `examples/exp3_super_resolution.py` | Experiment script |
| `results/exp3_super_resolution/Circle_super_resolution.png` | Comparison plot |
