# Reproducibility Guide

This document provides all necessary details to replicate the results for the Latent Reciprocity Representation (LRR) project. It covers dataset specifications, model architectures, training protocols, and hyperparameter configurations.

## 1. Environment Setup

### Dependencies
Ensure the following core libraries are installed:
- `torch >= 1.13.0`
- `neuraloperator` (for dataset loading)
- `h5py` (for legacy data support)
- `numpy`, `scipy`, `matplotlib`

### Random Seeds
For deterministic reproduction, a fixed seed is enforced across PyTorch, NumPy, and Python's random module.
- **Default Seed**: `42`
- **Config**: Set in `configs/hyperparameters.py` under `TRAINING_CONFIG['seed']`.

```python
# To enable full determinism in PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## 2. Datasets

### 1D Burgers' Equation
- **Source**: Synthetic generation via `src/data/pde_datasets.py` or `neuraloperator`.
- **Resolution**: 128 spatial points.
- **Equation**: $\partial_t u + u \partial_x u = \nu \partial_{xx} u$
- **Parameters**: $\nu=0.1$ (default synthetic).
- **Data Structure**:
  - Input (`f`): Initial condition at $t=0$, shape `[B, 1, 128]`.
  - Output (`u`): Solution at $t=1$, shape `[B, 1, 128]`.
  - **Implicit Time**: Single-step mapping from IC to final state.

### 2D Darcy Flow
- **Source**: `neuraloperator` library (Zenodo).
- **Resolution**: 85x85 (often subsampled to 64x64 or 32x32).
- **Equation**: $-\nabla \cdot (a(x) \nabla u(x)) = f(x)$
- **Data Structure**:
  - Input (`a`): Permeability field, shape `[B, 1, H, W]`.
  - Output (`u`): Pressure field, shape `[B, 1, H, W]`.

### 2D Navier-Stokes (Steady State / Fixed Mapping)
- **Source**: `datasets/navier_stokes/nsforcing_train_128.pt`
- **Resolution**: 128x128 (often subsampled to 64x64).
- **Equation**: Vorticity formulation of 2D NS.
- **Data Structure**:
  - Input (`x`): Forcing term or Initial Vorticity, shape `[B, 128, 128]` -> `[B, 1, 128, 128]`.
  - Output (`y`): Evolved Vorticity at fixed $T$, shape `[B, 128, 128]` -> `[B, 1, 128, 128]`.
  - **Note**: This dataset represents a fixed time mapping, not a full time-series trajectory.

## 3. Model Architectures

### FNO Backbone (Baseline & LRR)
Standard Fourier Neural Operator architecture.
- **Layers**: 4 Fourier Layers.
- **Modes**: 
  - 1D: 16 modes.
  - 2D: 12 modes (x, y).
- **Hidden Width**: 32 channels.
- **Activation**: GELU.
- **Projection**: 32 -> 128 -> GELU -> 1 (or output channels).

### LRR-FNO (Latent Reciprocity)
Extends FNO with a latent feedback loop.
- **Latent Dimension**: 64.
- **Latent Bridge**: Feature fusion module combining backbone features ($v_K$) with latent context.
- **Zero Context**: By default, the bridge receives a zero-vector context during inference to ensure prediction relies solely on backbone features.
- **Latent Supervision**: 
  - **Input**: Backbone features $v_K$.
  - **Target**: Latent code $z_u$ from `Encoder_u(ground_truth)`.
  - **Loss**: InfoNCE alignment between Projected($v_K$) and $z_u$.

## 4. Training Protocol

### Single-Stage Training (Joint Optimization)
We use a robust 1-stage training process where the manifold alignment and solution mapping are learned simultaneously.

#### Joint Optimization Phase
- **Objective**: Train both the primary mapping (MSE) and the latent alignment (InfoNCE) together from start to finish.
- **Loss**: `L_total = λ_mse * MSE(pred, target) + λ_nce * InfoNCE(z_v_k, z_u)`
- **Epochs**: 100-200.
- **Learning Rate**: `1e-3` (Cosine Annealing).
- **Goal**: The backbone features ($v_K$) are consistently guided by the latent supervision ($z_u$) throughout the entire training process, enforcing semantic alignment without needing a separate refinement stage.

## 5. Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 32 (Burgers), 8-16 (2D Tasks) | Adjusted based on VRAM |
| Optimizer | AdamW | Weight Decay: `1e-4` |
| Scheduler | CosineAnnealingLR | T_max = Epochs |
| $\lambda_{mse}$ | $5.0$ or $10^4$ | Scale dependent (NS/Darcy often need higher weight) |
| $\lambda_{nce}$ | $0.001$ | regularization strength |
| Temperature ($\tau$) | $0.07$ | InfoNCE softmax temperature |

## 6. How to Run

### Standard Training
To run the LRR model on Burgers' equation with the standard curriculum:
```bash
python examples/burgers1d_lrr.py --epochs 100 --batch_size 32
```

### Reproducing Demos
All main experiments are located in `examples/`:
- `examples/darcy_lrr.py`: 2D Darcy Flow.
- `examples/navier_stokes_lrr.py`: 2D Navier-Stokes.
- `examples/burgers1d_lrr.py`: 1D Burgers.

### Using Configuration File
For exact hyperparameter control used in papers/reports:
```bash
python train.py --config configs/default.yaml
```
