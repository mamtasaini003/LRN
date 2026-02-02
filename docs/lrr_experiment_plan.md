# LRR-FNO Experiment Plan
# Latent Reciprocity Representation for Steady-State PDEs

## Experiment Overview

This document outlines the experiments to demonstrate that LRR-FNO with Latent Reciprocity Representation provides a superior inductive bias over vanilla FNO for steady-state (time-independent) PDE problems.

## Completed Experiments

### Experiment 1: Steady-State Poisson and Heat Conduction (exp1_steady_state_poisson_lrr.py)
**Status:** ✅ Completed
**Datasets:** Circle, Cone-F, Ellipse-1/2/3, Semicircle-F (from GAOT time_indep)
**Results:**
| Dataset                  | FNO      | LRR-FNO  | Improvement |
|--------------------------|----------|----------|-------------|
| Poisson Circle Domain    | 0.157104 | 0.088544 | +43.64%     |
| Heat Conduction Cone     | 0.066316 | 0.056618 | +14.62%     |
| Poisson Ellipse (AR=1.5) | 0.041202 | 0.031752 | +22.94%     |
| Poisson Ellipse (AR=2.0) | 0.047980 | 0.038312 | +20.15%     |
| Poisson Ellipse (AR=2.5) | 0.039872 | 0.029895 | +25.02%     |
| Forced Semicircle BVP    | 0.045593 | 0.037125 | +18.57%     |

**Key Finding:** LRR-FNO consistently outperforms vanilla FNO on all 6 steady-state domains with an average improvement of ~24%.

---

## Planned Experiments

### Experiment 2: Spectral Error Profiling (fRMSE)
**File:** exp2_spectral_error_profiling.py
**Goal:** Analyze error distribution across frequency bands (low/mid/high)
**Hypothesis:** LRR reduces error in high-frequency modes, mitigating FNO's spectral bias

### Experiment 3: Zero-Shot Super-Resolution
**File:** exp3_super_resolution.py
**Goal:** Train on 64x64, test on 128x128 and 256x256 without retraining
**Hypothesis:** LRR anchors internal representations to continuous physical manifold

### Experiment 4: Out-of-Distribution (OOD) Forcing Generalization
**File:** exp4_ood_forcing.py
**Goal:** Train on smooth Gaussian forcing, test on sharp Poisson-style perturbations
**Hypothesis:** LRR maintains physically admissible solutions under distribution shift

### Experiment 5: Noise Robustness
**File:** exp5_noise_robustness.py
**Goal:** Add varying Gaussian noise to input forcing, measure degradation
**Hypothesis:** LRR acts as regularizer, filtering high-frequency noise

---

## Experiment File Naming Convention

```
exp{N}_{short_name}.py
```

Where:
- N = Experiment number (1, 2, 3, ...)
- short_name = Descriptive name (e.g., steady_state_poisson_lss, spectral_error_profiling)

## Usage

```bash
# Run Experiment 1 on all datasets
python examples/exp1_steady_state_poisson_lrr.py --all --epochs 50

# Run Experiment 1 on a single dataset
python examples/exp1_steady_state_poisson_lrr.py --dataset dataset/Circle.nc --epochs 50
```
