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
- **Latent Bridge Injection**: Conditions backbone features on latent codes for physically-constrained predictions
- **3-Stage Curriculum Training**: Progressive learning from manifold alignment → hybrid optimization → distillation

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
│   │   └── infonce.py          # InfoNCE contrastive loss
│   ├── data/
│   │   └── pde_datasets.py     # Burgers, Darcy, Navier-Stokes datasets
│   └── utils/
│       └── training.py         # 3-stage curriculum trainer
├── configs/
│   └── default.yaml            # Hyperparameter configuration
├── examples/
│   └── burgers_demo.py         # Burgers equation demo
├── train.py                    # Main training script
├── requirements.txt
└── README.md
```

---

## 💻 Usage

### Quick Start

```python
import torch
from src.models import LRNFNO1d
from src.losses import LRNLoss

# Create LRN-FNO model for 1D PDEs
model = LRNFNO1d(
    in_channels=1,
    out_channels=1,
    modes=16,           # Fourier modes
    width=64,           # Hidden channels
    num_layers=4,       # FNO layers (K)
    latent_dim=64,      # Latent space dimension (d_z)
)

# Forward pass
f = torch.randn(8, 128)  # Input field [batch, spatial]
u = torch.randn(8, 128)  # Solution (for training)

output = model(f, u, return_latents=True)
prediction = output['prediction']  # Predicted solution
z_f = output['z_f']                # Forward latent
z_u = output['z_u']                # Reverse latent

# Compute loss
loss_fn = LRNLoss(lambda_mse=1.0, temperature=0.1)
losses = loss_fn(prediction, u, z_f, z_u, stage=2)
total_loss = losses['total']
```

### Training with 3-Stage Curriculum

```bash
# Full training with default config
python train.py --config configs/default.yaml

# Quick training on Burgers equation
python train.py --dataset burgers --stage1_epochs 20 --stage2_epochs 50 --stage3_epochs 20

# Training on 2D Darcy flow
python train.py --dataset darcy --resolution 64 --batch_size 16
```

### Running the Demo

```bash
# Quick demonstration
python examples/burgers_demo.py --mode demo

# Compare LRN-FNO vs vanilla FNO
python examples/burgers_demo.py --mode compare
```

---

## 📊 Training Protocol

LRN employs a **3-stage curriculum learning** protocol:

| Stage | Components | Loss | Purpose |
|-------|-----------|------|---------|
| **I** | $E_f$, $E_u$ | $\mathcal{L}_{NCE}$ | Manifold alignment |
| **II** | $E_f$, $E_u$, $G_\theta$ | $\mathcal{L}_{NCE} + \lambda \mathcal{L}_{MSE}$ | Hybrid optimization |
| **III** | $E_f$, $G_\theta$ | $\mathcal{L}_{MSE}$ | Autonomous distillation |

**InfoNCE Loss:**
$$\mathcal{L}_{NCE} = -\sum_i \log \frac{\exp(\text{sim}(z_{f,i}, z_{u,i})/\tau)}{\sum_j \exp(\text{sim}(z_{f,i}, z_{u,j})/\tau)}$$

---

## 📈 Results

Comparison of FNO vs LRN-FNO on benchmark PDEs:

| PDE | Resolution | FNO MSE | LRN-FNO MSE | Improvement |
|-----|------------|---------|-------------|-------------|
| Burgers | 16 | 0.003394 | 0.002929 | **-13.70%** |
| Darcy | 16 | 0.142206 | 0.126364 | **-11.14%** |
| Darcy | 32 | 0.188096 | 0.176746 | **-6.03%** |
| Navier-Stokes | 128 | 0.012740 | 0.010340 | **-18.84%** |

---

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  latent_dim: 64          # Latent space dimension
  fno:
    modes: 16             # Fourier modes
    width: 64             # Hidden channels
    num_layers: 4         # Number of FNO layers

training:
  stage1:
    epochs: 50            # Manifold alignment epochs
  stage2:
    epochs: 100           # Hybrid optimization epochs
  stage3:
    epochs: 50            # Distillation epochs

loss:
  lambda_mse: 1.0         # MSE weight (λ)
  temperature: 0.1        # InfoNCE temperature (τ)
```

---

## 📝 Citation

```bibtex
@misc{lrn2024,
  title={Latent Reciprocity Network: A Bidirectional Latent-Space Alignment for Solution Operators},
  author={Saini, Mamta},
  year={2024},
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
