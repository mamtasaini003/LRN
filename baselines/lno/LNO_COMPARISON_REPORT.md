# LNO vs LRN-FNO: Comprehensive Baseline Comparison

**Date:** January 21, 2026  
**Paper:** "Latent Neural Operator for Solving Forward and Inverse PDE Problems" (NeurIPS 2024)  
**Authors:** Tian Wang, Chuang Wang  
**Reference:** https://github.com/L-I-M-I-T/LatentNeuralOperator

---

## 1. Overview

This document provides a fair comparison between:

1. **Vanilla FNO** - Baseline Fourier Neural Operator
2. **LRN-FNO** - Our Latent Reciprocity Network augmented FNO
3. **LNO** - Latent Neural Operator (NeurIPS 2024)

All models are evaluated on identical datasets with fixed random seeds for reproducibility.

---

## 2. LNO Architecture Summary

### 2.1 Key Components

The Latent Neural Operator (LNO) consists of four main modules:

1. **Embedding Layer**
   - **Trunk-projector**: Embeds positions to D-dimensional space
   - **Branch-projector**: Embeds position+value pairs to D-dimensional space
   
2. **PhCA Encoder** (Physics-Cross-Attention)
   - Transforms from geometric space (N points) to latent space (M tokens, M << N)
   - Learnable latent positions as queries
   - Position embeddings as keys, position+value embeddings as values

3. **Transformer Blocks**
   - L stacked self-attention layers
   - Process representations in the compact latent space
   - Uses scaled dot-product attention (proved better than linear attention)

4. **PhCA Decoder**
   - Transforms from latent space back to geometric space
   - Shares weights with encoder (W₁ = W₂)
   - Enables prediction at arbitrary output positions

### 2.2 Mathematical Formulation

**Encoder:**
```
Z⁰ = softmax(W₁ · X̂ᵀ) · Ŷ · Wᵥ
```

**Decoder:**
```
U = softmax(P · W₂ᵀ) · Zᴸ · W'ᵥ
```

Where W₁ = W₂ (shared attention projector).

### 2.3 Key Innovations

1. **Learnable Latent Space** - Unlike FNO's fixed frequency domain, LNO learns optimal latent representations
2. **Decoupled Positions** - Observation and prediction positions can differ (useful for inverse problems)
3. **Single Transformation** - Unlike Transolver, LNO transforms to/from latent space only once (not at each layer)
4. **Shared Weights** - Encoder and decoder share the attention projector

---

## 3. Experimental Setup

### 3.1 Common Configuration

| Parameter | Value |
|:---|:---|
| Random Seed | 42 |
| Training Epochs | 150 |
| Loss Function | Relative L2 |
| Device | NVIDIA RTX A6000 |

### 3.2 Model Configurations

#### FNO & LRN-FNO
| Parameter | Burgers 2D | Darcy | Navier-Stokes |
|:---|:---:|:---:|:---:|
| Modes (k₁, k₂) | 12, 12 | 12, 12 | 12, 12 |
| Width | 32 | 32 | 32 |
| Layers | 4 | 4 | 4 |
| Latent Dim (LRN) | 64 | 64 | 64 |

#### LNO (Paper Configuration)
| Parameter | Burgers 2D | Darcy | Navier-Stokes |
|:---|:---:|:---:|:---:|
| Embed Dim | 128 | 128 | 256 |
| Latent Size (M) | 256 | 256 | 256 |
| Layers | 4 | 4 | 8 |
| Heads | 8 | 8 | 8 |

### 3.3 Training Protocol

| Model | Optimizer | Scheduler | Learning Rate |
|:---|:---|:---|:---|
| FNO | Adam | CosineAnnealing | 1e-3 |
| LRN-FNO | Adam | 2-Stage (NCE+MSE → MSE) | 1e-3 → 5e-4 |
| LNO | AdamW | OneCycleLR | 1e-3 |

---

## 4. Benchmarks

### 4.1 2D Burgers Equation

**PDE:**
```
∂u/∂t + u·∇u = ν∇²u
```

- Resolution: 64×64
- Channels: 2 (u, v velocity components)
- Train/Test: 300/100 samples

### 4.2 Darcy Flow

**PDE:**
```
-∇·(a(x)∇u) = f
```

- Resolution: 32×32
- Input: Permeability field a(x)
- Output: Pressure field u(x)
- Train/Test: 400/100 samples

### 4.3 2D Navier-Stokes

**PDE:**
```
∂ω/∂t + u·∇ω = ν∇²ω + f
∇·u = 0
```

- Resolution: 64×64
- Channels: 10 (temporal snapshots)
- Train/Test: 200/50 samples

---

## 5. Expected Results

Based on the paper's reported performance:

### 5.1 Paper Benchmarks (for reference)

| Model | Darcy | NS2d |
|:---|:---:|:---:|
| FNO | 1.08% | 15.56% |
| Transolver | 0.58% | 8.79% |
| **LNO** | **0.49%** | **8.45%** |

### 5.2 Our Experimental Setup

**Note:** Our synthetic datasets may differ from the original FNO/GeoFNO benchmarks. Results should be interpreted relative to the FNO baseline within each experiment.

Expected relative performance:
- LNO should show ~15-20% improvement over FNO on Darcy/NS2d
- LRN-FNO shows ~10% improvement over FNO (documented in our previous experiments)

---

## 6. Key Differences: LNO vs LRN-FNO

| Aspect | LNO | LRN-FNO |
|:---|:---|:---|
| **Architecture** | Transformer-based | FNO backbone |
| **Latent Space** | Learned via PhCA | Contrastive alignment |
| **Regularization** | None explicit | InfoNCE loss |
| **Position Handling** | Decoupled (flexible) | Fixed grid |
| **Complexity** | O(MN + LM²) | O(k²HW + Enc/Dec) |
| **Inverse Problems** | Native support | Requires adaptation |
| **Training** | Single-stage | 2-stage |

### 6.1 Complementary Strengths

**LNO Advantages:**
- Better for irregular meshes and inverse problems
- More flexible position handling
- Lower memory on very large grids

**LRN-FNO Advantages:**
- Leverages proven FNO spectral efficiency
- Bidirectional latent alignment provides physical consistency
- Faster training on regular grids
- Smaller model size

---

## 7. Running the Comparisons

### 7.1 Individual Benchmarks

```bash
# Burgers 2D
cd baselines/lno
python lno_burgers2d_demo.py

# Darcy Flow
python lno_darcy_demo.py

# Navier-Stokes
python lno_ns_demo.py
```

### 7.2 All Benchmarks

```bash
python run_all_comparisons.py
```

### 7.3 Output Files

After running, results are saved to:
- `lno_checkpoints/burgers2d_comparison_results.json`
- `lno_checkpoints/darcy_comparison_results.json`
- `lno_checkpoints/navier_stokes_comparison_results.json`
- `lno_checkpoints/all_comparison_results.json`

Visualizations:
- `results/plots/lno_vs_lrn_burgers2d.png`
- `results/plots/lno_vs_lrn_darcy.png`
- `results/plots/lno_vs_lrn_navier_stokes.png`

---

## 8. Fair Comparison Principles

To ensure fair comparison, we follow these principles:

1. **Same Random Seed**: All models use seed=42 for reproducibility
2. **Same Dataset**: Identical training/test splits
3. **Same Training Duration**: 150 epochs for all models
4. **Same Evaluation Metric**: Relative L2 error
5. **Same Hardware**: Single GPU experiments
6. **Paper Configurations**: LNO uses hyperparameters from the paper

---

## 9. Implementation Notes

### 9.1 LNO Implementation

Our LNO implementation follows the paper architecture exactly:

```python
class LNO2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        embed_dim=128,      # D in paper
        latent_size=256,    # M in paper
        num_layers=4,       # L in paper
        num_heads=8,
        ...
    ):
```

### 9.2 PhCA Implementation

```python
class PhysicsCrossAttention(nn.Module):
    def encode(self, query_latent, key_pos, value):
        # Z = softmax(W1 @ X^T) @ Y @ Wv
        ...
    
    def decode(self, query_pos, key_latent, value_latent):
        # U = softmax(P @ W2^T) @ Z @ Wv'
        # W1 = W2 (shared)
        ...
```

---

## 10. References

1. Wang, T., & Wang, C. (2024). Latent Neural Operator for Solving Forward and Inverse PDE Problems. NeurIPS 2024.

2. Li, Z., et al. (2021). Fourier Neural Operator for Parametric Partial Differential Equations. ICLR 2021.

3. Wu, H., et al. (2024). Transolver: A Fast Transformer Solver for PDEs on General Geometries. ICML 2024.

---

## 11. Citation

If using this comparison, please cite:

```bibtex
@inproceedings{wang2024LNO,
    title={Latent Neural Operator for Solving Forward and Inverse PDE Problems},
    author={Tian Wang and Chuang Wang},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2024}
}
```
