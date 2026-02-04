# Experiment 2: Latent Space Analysis and Spectral Profiling

**Date:** 2026-02-02  
**Script:** `examples/exp2_latent_analysis_spectral.py`  
**Status:** ✅ Completed

---

## 1. Objective

Analyze the latent space representations in LRR-FNO and compare spectral error profiles with vanilla FNO to understand:
1. How latent representations (z_f, z_u) evolve during training
2. Whether LRR provides better spectral error distribution across frequency bands
3. The alignment quality between backbone features and solution encodings

---

## 2. Dataset

| Property | Value |
|----------|-------|
| **Dataset** | Circle.nc (Poisson equation on circular domain) |
| **Training Samples** | 160 |
| **Test Samples** | 40 |
| **Resolution** | 64×64 (interpolated from point cloud) |
| **Input Channels** | 3 (x, y coordinates + forcing) |
| **Output Channels** | 1 (solution field) |

---

## 3. Training Protocol (2-Stage)

| Stage | Epochs | LR | Loss Function |
|-------|--------|-----|---------------|
| Stage 1 (Combined) | 73 | 1e-3 | NCE + MSE |
| Stage 2 (Distillation) | 27 | 1e-4 | MSE only |
| **Total** | **100** | - | - |

---

## 4. Latent Representation Architecture

```
Input (f) ──→ FNO Backbone ──→ v_K ──→ GAP ──→ MLP ──→ z_f (64-dim)
                                                         ↓
                                                    InfoNCE Loss
                                                         ↑
Solution (u) ──→ CNN Encoder [32,64,128] ──→ z_u (64-dim)
```

---

## 5. Visualizations

### 5.1 t-SNE Latent Space Evolution

Shows how z_f (backbone) and z_u (solution) representations evolve and align during training.

![t-SNE Evolution](images/Circle_tsne_evolution.png)

**Observation:** Progressive overlap between z_f (blue) and z_u (red) clusters indicates increasing alignment as training progresses.

### 5.2 Alignment Metrics Over Training

![Alignment Metrics](images/Circle_alignment_metrics.png)

### 5.3 Spectral Error Comparison (fRMSE)

![Spectral Comparison](images/Circle_spectral_comparison.png)

| Frequency Band | FNO | LRR-FNO | Improvement |
|----------------|-----|---------|-------------|
| Low (<0.1) | 6.98 | 6.91 | **-1.1%** |
| Mid (0.1-0.3) | 2.13 | 1.62 | **-23.8%** |
| High (>0.3) | 1.62 | 1.16 | **-28.4%** |

**Key Finding:** LRR significantly reduces mid and high-frequency errors, indicating better capture of fine-grained solution structure.

### 5.4 Latent Representation Alignment

![Latent Reps](images/Circle_latent_reps.png)

**Visualization Features:**
- **Blue circles:** z_f (backbone features)
- **Red squares:** z_u (solution encodings)  
- **Gray lines:** Connection between corresponding sample pairs
- **Cosine Similarity:** Displayed in the annotation box

---

## 6. Results

| Model | Relative L2 Error | Improvement |
|-------|-------------------|-------------|
| Vanilla FNO | 0.1169 | - |
| LRR-FNO | 0.1123 | **+3.87%** |

---

## 7. Key Findings

1. **Latent Alignment:** t-SNE visualizations confirm progressive alignment between backbone features (z_f) and solution encodings (z_u) during training
2. **Spectral Improvement:** LRR reduces mid-frequency errors by 23.8% and high-frequency errors by 28.4%
3. **2-Stage Protocol:** Stage 1 establishes representation alignment, Stage 2 refines prediction quality
4. **Visualization Validation:** All three metrics (t-SNE, alignment, spectral) consistently show LRR's benefits

---


---

## 8. Artifacts

| File | Description |
|------|-------------|
| `examples/exp2_latent_analysis_spectral.py` | Experiment script |
| `results/exp2_latent_analysis/Circle_tsne_evolution.png` | t-SNE evolution plot |
| `results/exp2_latent_analysis/Circle_alignment_metrics.png` | Alignment metrics |
| `results/exp2_latent_analysis/Circle_spectral_comparison.png` | Spectral comparison |

---

## 9. Reproducibility

To reproduce these results with the full dataset, refer to [Reproducibility Guide](reproducibility.md).

**Quick Reproduce:**
```bash
python3 examples/exp2_latent_analysis_spectral.py --dataset dataset/Circle.nc --epochs 500 --max_samples -1
```
