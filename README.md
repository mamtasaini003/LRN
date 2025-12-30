# Latent Reciprocity Network (LRN)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

*A Bidirectional Latent-Space Alignment Framework for Neural PDE Solvers*

</div>

---

## 📋 Overview

**Latent Reciprocity Network (LRN)** is a backbone-agnostic framework that augments neural PDE solvers with bidirectional latent space alignment. By enforcing reciprocity between embeddings of source fields and solutions via contrastive learning, LRN constrains operators to evolve within physically admissible manifolds—eliminating spectral aliasing, hallucinated modes, and poor out-of-distribution generalization.

### 🎯 Key Contributions

- **Bidirectional Latent Alignment**: Dual encoders $E_f$ and $E_u$ map inputs and solutions to a shared latent space
- **InfoNCE Contrastive Loss**: Enforces reciprocity for matched (f, u) pairs while distinguishing mismatches
- **Scale-Invariant Optimization**: Incorporates **Relative MSE** loss to handle varied PDE magnitudes
- **Optimized 2-Stage Training**: Simplified protocol (Joint Training → Fine-tuning) for faster and more stable convergence

---

## ✨ Architecture

```
                    ┌─────────────┐
                    │   Input f   │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
   ┌─────────┐       ┌───────────┐      ┌─────────┐
   │  E_f    │       │  FNO      │      │  E_u    │ (training only)
   │(encoder)│       │ Backbone  │      │(encoder)│
   └────┬────┘       └─────┬─────┘      └────┬────┘
        │                  │                  │
        │    z_f           │ v_K              │ z_u
        │                  │                  │
        │      ┌───────────┴───────────┐      │
        └──────►   Latent Bridge       ◄──────┘
               │  v^lat = MLP(v_K ⊕ z_f)│      │
               └───────────┬───────────┘      │
                           │                  │
                           ▼                  │
                    ┌─────────────┐           │
                    │ Projection Π│           │
                    └──────┬──────┘           │
                           │                  │
                           ▼                  │
                    ┌─────────────┐           │
                    │ Prediction ũ├───────────┘
                    └─────────────┘     L_NCE(z_f, z_u)
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/mamtasaini003/LRN.git
cd LRN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
LRN/
├── src/
│   ├── models/
│   │   ├── fno.py              # Fourier Neural Operator backbone
│   │   ├── encoders.py         # Forward/Reverse encoders (E_f, E_u)
│   │   ├── latent_bridge.py    # Latent injection module
│   │   └── lrn_fno.py          # Complete LRN-FNO model
│   ├── losses/
│   │   └── infonce.py          # InfoNCE and RelativeMSE losses
│   ├── data/
│   │   └── pde_datasets.py     # Burgers, Darcy, Navier-Stokes datasets
│   └── utils/
│       └── training.py         # 3-stage and 2-stage trainers
├── checkpoints/                # Model checkpoints organized by task
├── results/
│   ├── plots/                  # Visualizations and loss curves
│   └── logs/                   # Training logs
├── configs/
│   └── default.yaml            # Hyperparameter configuration
├── examples/
│   ├── burgers2d_demo_v2.py    # 2-stage demo for Burgers
│   ├── darcy_demo_v2.py        # 2-stage demo for Darcy
│   └── navier_stokes_demo_v2.py # 2-stage demo for Navier-Stokes
├── train.py                    # Main training script (supports --v2)
├── requirements.txt
└── README.md
```

---

## 💻 Usage

### Quick Start (2-Stage V2)

```python
import torch
from src.models import LRNFNO2d
from src.losses import LRNLoss

# Create LRN-FNO model for 2D Darcy Flow
model = LRNFNO2d(
    in_channels=1,
    out_channels=1,
    modes1=12,
    modes2=12,
    width=32,
    latent_dim=64,
)

# Training Pass (V2 Stage 1: Balanced Combined Optimization)
# Using high lambda_mse and low lambda_nce for scale-stability
loss_fn = LRNLoss(lambda_mse=10000.0, lambda_nce=0.01, use_relative_mse=False)
output = model(f, u)
losses = loss_fn(output['prediction'], u, output['z_f'], output['z_u'], stage=2)
```

### Training Command

```bash
# Optimized 2-stage training with fixed seed (42) for reproducibility
python train.py --v2 --dataset ns --lambda_mse 1.0 --lambda_nce 1.0

# Darcy Flow demo (Reproducible)
python examples/darcy_demo_v2.py
```

### Running the Demos (V2 - Reproducible)

```bash
# Navier-Stokes 2-stage demo (+9% improvement)
python examples/navier_stokes_demo_v2.py

# Burgers 2D 2-stage demo
python examples/burgers2d_demo_v2.py
```

---

## 📊 Training Protocols

### 2-Stage Protocol (V2 - Recommended)
| Stage | Description | Loss |
|:---|:---|:---|
| **I** | Combined Optimization | $\mathcal{L}_{NCE} + \lambda \mathcal{L}_{RelMSE}$ |
| **II** | Autonomous Distillation | $\mathcal{L}_{RelMSE}$ |

### 3-Stage Curriculum (V1)
| Stage | Description | Loss |
|:---|:---|:---|
| **I** | Manifold Alignment | $\mathcal{L}_{NCE}$ |
| **II** | Hybrid Optimization | $\mathcal{L}_{NCE} + \lambda \mathcal{L}_{MSE}$ |
| **III** | Autonomous Distillation | $\mathcal{L}_{MSE}$ |

---

## 📈 Results (V2 Final - Reproducible)

Comparison of FNO vs LRN-FNO V2 (150 epochs total, Fixed Seed 42, Relative L2 Error):

| PDE Task | Resolution | FNO Rel L2 | LRN-FNO V2 | Improvement |
|:--- |:--- |:--- |:--- | :---: |
| **Darcy Flow** | 32x32 | 0.1498 | 0.1340 | **+10.55%** |
| **Navier-Stokes** | 64x64 | 0.2276 | 0.2070 | **+9.09%** |
| **Burgers 2D** | 64x64 | 0.0146 | 0.0141 | **+3.42%** |

---

## ⚙️ Configuration

Use the `configs/default.yaml` or CLI arguments to adjust hyperparameters:
- `lambda_mse`: Weight for reconstruction (recommended: 10,000 for standard MSE on small scales)
- `lambda_nce`: Weight for reciprocity (recommended: 0.01 for small scales)
- `use_relative_mse`: Set to `True` for scale-invariant training
- `use_gated_bridge`: Enabled for better latent injection stability

---

## 📝 Citation

```bibtex
@misc{lrn2025,
  title={Latent Reciprocity Network: A Bidirectional Latent-Space Alignment for Solution Operators},
  author={Saini, Mamta},
  year={2025},
  howpublished={\url{https://github.com/mamtasaini003/LRN}}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if you find it useful!**

</div>
