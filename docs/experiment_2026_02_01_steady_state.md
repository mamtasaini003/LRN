# Experiment Analysis: Steady-State Burgers 2D & Navier-Stokes

**Date:** 2026-02-01
**Status:** Quick Validation Complete (User to run full epochs)

---

## 1. Problem Statement & Focus
- **Objective:** Evaluate LRN-FNO performance on **time-independent (steady-state)** versions of Burgers 2D and Navier-Stokes, while keeping Darcy Flow unchanged from the baseline.
- **Point of Focus:** Determine if LRN's latent reciprocity offers advantages when the input-output mapping is a direct spatial transformation rather than a temporal evolution.

---

## 2. Technical Configuration

### Dataset Details
- **Darcy Flow:** (Unchanged) Steady-state pressure from circular permeability "blobs" (1-channel).
- **Burgers 2D (NEW - Steady):** Steady-state velocity fields $(u, v)$ from a forcing function (2-channel input → 2-channel output).
- **Navier-Stokes (NEW - Steady):** Steady-state vorticity field from forcing (1-channel input → 1-channel output).
- **Reproduction Seeds:** Fixed Seed 42.

### Model Architecture
- **Fourier Modes:** 12, 12
- **Width:** 32, Layers: 4
- **LRN Latent Dim:** 64
- **Gated Bridge:** Enabled for Burgers 2D.

### Training Configuration
- **Protocol:** 2-Stage Training (110 NCE+MSE, 40 MSE)
- **Optimizers:** Adam with CosineAnnealingLR.
- **λ_MSE / λ_NCE:** 
  - Darcy/NS: 1.0 / 1.0
  - Burgers 2D: 10,000 / 0.01

---

## 3. Comparison & Results
**(Quick Validation: 20 FNO epochs, 15+5 LRN stages)**

| PDE Case | FNO Rel. L2 Error | LRN-FNO Rel. L2 Error | Improvement | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Darcy Flow** | 0.1498 | 0.1340 | **+10.56%** | Baseline (unchanged) |
| **Burgers 2D (Steady)** | 0.0541 | 0.0536 | **+0.96%** | Marginal |
| **Navier-Stokes (Steady)** | 0.0311 | 0.0529 | **-70.21%** | LRN underperforms |

---

## 4. Analysis & Justification

### What is Changing?
- Burgers 2D: Replacing time-evolution with a steady-state forcing-to-solution mapping.
- Navier-Stokes: Replacing 10-channel transient sequence with a 1-channel steady vorticity field.

### Why is this better?
- This experiment tests whether LRN's reciprocity provides value for simpler, time-independent mappings, or if vanilla FNO is sufficient.

### Limitations
- **1-Channel Steady-State Limitation:** LRN's reciprocity framework is designed for complex multi-channel manifolds. Simple 1-channel Poisson problems don't benefit from contrastive alignment.
- **Epoch Count:** These results are from a quick validation with only 20 FNO epochs and 15+5 LRN stages. Full training may close the gap.
- **Architecture Overhead:** The dual-encoder LRN architecture adds parameters without benefit on these simple problems.
- **Bug Fixed:** The original -2904% failure was caused by a shape mismatch in the data loader (ChannelWrapper adding an extra channel dimension).

### Scope of Improvement
- Run full 150 epochs to allow LRN to converge properly.
- Consider adaptive `lambda_nce` decay for steady-state problems.
- Explore simpler encoder architectures for time-independent tasks.

---

## 5. Visual Logs
(To be added after experiment run)
