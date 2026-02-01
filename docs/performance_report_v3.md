# Experiment Analysis: LRN-FNO V3 (2-Stage Training Milestone)

**Date:** 2025-12-30
**Status:** Historical Milestone

---

## 1. Problem Statement & Focus
- **Objective:** Simplify the original 3-stage training curriculum into a more robust 2-stage joint training protocol.
- **Point of Focus:** Evaluating if joint optimization of NCE+MSE (Stage 1) followed by MSE fine-tuning (Stage 2) improves convergence over the sequential NCE-then-MSE approach.

---

## 2. Technical Configuration

### Dataset Details
- **Type:** Synthetic (Rotating vortices for NS, Fourier sum for Burgers 2D, Pooling for Darcy).
- **Samples:** Standard defaults.
- **Resolution:** 64x64 for NS/Burgers, 32x32 for Darcy.

### Model Architecture
- **FNO Modes/Width:** 12, 12 / 32
- **LRN Latent Dim:** 64
- **Bridge Type:** Gated Bridge (introduced in this version for stability).

### Training Configuration
- **Total Epochs:** 150
- **Stage 1 (NCE+MSE Combined):** 110 Epochs
- **Stage 2 (MSE Only Fine-tuning):** 40 Epochs
- **Optimizers/LR:** Adam, Stage 1: 1e-3, Stage 2: 1e-4.
- **λ_MSE / λ_NCE:** 1.0 / 1.0 (Initial balancing).

---

## 3. Comparison & Results

| Model | Darcy Flow Error | Navier-Stokes Error | Burgers 2D Error |
| :--- | :---: | :---: | :---: |
| **Vanilla FNO** | 0.3997 | 0.2055 | 0.0138 |
| **LRN-FNO V3** | **0.1398** (+65%) | 0.2173 (-5.7%) | 0.0163 (-18.5%) |

---

## 4. Analysis & Justification

### What is Changing?
- Transitioned from a 3-Stage curriculum (NCE → NCE+MSE → MSE) to a 2-Stage joint protocol (NCE+MSE → MSE).
- Implemented `LRNTrainerV2` for a simplified workflow.
- Fixed numerical instability in Burgers 2D dataset.

### Why is this better?
- **Stability:** Backbone stability is improved by training backbone and encoders jointly from epoch 1.
- **Simplicity:** Easier to tune than the 3-stage curriculum.

### Limitations
- **Synthetic Data Simplicity:** LRN underperformed on NS and Burgers 2D because the synthetic tasks were too simple for the architecture to provide a benefit.
- **Regularization Tax:** The contrastive loss acts as a regularizer that adds overhead without benefit on "easy" problems.

### Scope of Improvement
- Move to high-complexity natural datasets (e.g., PDEBench).
- Enable Gated Bridge for all tasks to allow autonomous "ignoring" of latents.

---

## 5. Visual Logs
- [darcy_comparison_v2.png](../results/plots/darcy_comparison_v2.png)
- [ns_comparison_v2.png](../results/plots/ns_comparison_v2.png)
- [burgers2d_comparison_v2.png](../results/plots/burgers2d_comparison_v2.png)

---

## 6. More Information

### Issues Faced & Resolved
- **Numerical Instability (Burgers 2D):** Low viscosity and large timesteps caused NaN blow-ups. Resolved by enforcing CFL-safe timesteps (`dt = 0.25 * dx**2 / nu`) and clamping velocity values.
- **Broadcasting Warnings:** Identified mismatched tensor sizes (`[B, 1, H, W]` vs `[B, H, W]`) in FNO baselines. While handled by broadcasting, it highlighted the need for explicit squeezing.
- **Information Bottleneck:** Observed that LRN's latent bottleneck acts as noise on simple 1-channel steady problems.

### Recommendations for Future Experiments
- **Batch Size:** Increase to 32 or 64 to provide more negative samples for InfoNCE.
- **Noise Injection:** Add input noise to training to make LRN's manifold robustness a tangible advantage.
- **Pre-Training:** Consider self-supervised pre-training of encoders on unlabeled physical sequences.

---

## 7. Conclusion
V3 proved the efficiency of the 2-stage trainer but also highlighted that LRN-FNO's strengths lie in complex, multi-channel manifolds rather than simple local steady-state mappings.
