# Experiment 4: Noise Robustness

**Date:** 2026-02-02  
**Script:** `examples/exp4_noise_robustness.py`  
**Status:** ✅ Completed

---

## 1. Objective

Test model degradation with increasing Gaussian noise on input forcing. Hypothesis: LRR acts as implicit regularizer, filtering high-frequency noise through latent anchoring.

---

## 2. Experimental Setup

| Parameter | Value |
|-----------|-------|
| **Resolution** | 64×64 |
| **Dataset** | Circle.nc |
| **Training Samples** | 160 |
| **Test Samples** | 40 |
| **Epochs** | 50 (2-stage: 36+14) |
| **Noise Levels (σ)** | 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5 |

---

## 3. Noise Model

Gaussian noise added to input forcing at test time:
```
c_noisy = c + N(0, σ² × std(c)²)
```

---

## 4. Results

| Noise σ | FNO | LRR-FNO | Improvement |
|---------|-----|---------|-------------|
| 0.00 | 0.1257 | 0.1118 | **+11.1%** |
| 0.05 | 0.1263 | 0.1118 | **+11.5%** |
| 0.10 | 0.1281 | 0.1119 | **+12.6%** |
| 0.15 | 0.1312 | 0.1122 | **+14.5%** |
| 0.20 | 0.1352 | 0.1125 | **+16.7%** |
| 0.30 | 0.1463 | 0.1134 | **+22.5%** |
| 0.50 | 0.1790 | 0.1178 | **+34.2%** |

### Key Findings:
1. **LRR improvement increases with noise:** From +11% at σ=0 to +34% at σ=0.5
2. **FNO degradation:** +42.4% error increase from σ=0 to σ=0.5
3. **LRR degradation:** Only +5.4% error increase from σ=0 to σ=0.5

---

## 5. Visualization

![Noise Robustness](file:///home/mamta/work/LRN/results/exp4_noise_robustness/Circle_noise_robustness.png)

---

## 6. Interpretation

LRR's latent anchoring acts as implicit regularization by:
1. Projecting inputs to a learned physical manifold
2. Filtering high-frequency noise through the latent space
3. Maintaining stability through solution-grounded representations

The InfoNCE loss encourages the backbone to extract noise-invariant features that match the solution encoding.

---

## 7. Artifacts

| File | Description |
|------|-------------|
| `examples/exp4_noise_robustness.py` | Experiment script |
| `results/exp4_noise_robustness/Circle_noise_robustness.png` | Comparison plot |
