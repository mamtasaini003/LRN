# Final Performance Report: LRN-FNO V2 (2-Stage Training)

**Date:** December 30, 2025  
**Time:** 15:26 IST  
**Framework:** Latent Reciprocity Network (LRN-FNO)  
**Protocol:** 2-Stage Training (Simplified Joint Training → Fine-tuning)

---

## 1. Executive Summary

This report documents the final evaluation of the LRN-FNO model across three benchmark PDE problems using the optimized **2-Stage Training Protocol**. By addressing numerical instabilities and loss scaling imbalances identified in previous iterations, LRN-FNO now consistently matches or significantly outperforms the vanilla FNO baseline.

| Task | Training Protocol | FNO Rel. L2 Error | LRN-FNO Rel. L2 Error | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Darcy Flow** | 2-Stage (150 ep) | 0.3997 | 0.1398 | **+65.02%** |
| **Navier-Stokes** | 2-Stage (150 ep) | 0.2146 | 0.2130 | **+0.73%** |
| **Burgers 2D** | 2-Stage (150 ep) | 0.0162 | 0.0154 | **+4.92%** |

---

## 2. Key Improvements & Fixes

The transition from the original 3-stage curriculum to the final 2-stage optimized protocol involved three critical fixes:

### A. Loss Scaling Balance (Relative MSE)
We identified that for simple problems like Burgers and Navier-Stokes, the absolute MSE was so small (~$10^{-5}$) that it was "drowned out" by the InfoNCE Contrastive Loss (~3.0).
- **Fix:** Switched to **Relative MSE Loss** and increased the reconstruction weight ($\lambda = 20.0$).
- **Result:** Impact on Burgers 2D shifted from **-18%** degradation to **+4.92%** improvement.

### B. Numerical Stability in Burgers 2D
The initial synthetic data generator was unstable and produced NaNs.
- **Fix:** Implemented CFL-safe timestepping and increased numerical viscosity in `src/data/pde_datasets.py`.
- **Result:** Training is now 100% stable without gradients exploding.

### C. 2-Stage Training Refinement
Simplified the curriculum to focus on joint optimization from the start.
- **Stage 1:** Combined Optimization (NCE + MSE) - Jointly aligns the manifold while learning physics.
- **Stage 2:** Autonomous Distillation (MSE only) - Fine-tunes the backbone for inference.

---

## 3. Detailed Results by Task

### 2D Darcy Flow
*   **Input:** Random permutations ($a(x)$)
*   **Output:** Pressure field ($u(x)$)
*   **Observation:** This remains the strongest result for LRN. The latent reciprocity constraint provides an excellent inductive bias for the complex mapping in Darcy flow.
*   **Final Error:** 0.1398 (**65% improvement**)

### 2D Navier-Stokes
*   **Input:** Initial vorticity at $t=0..10$
*   **Output:** Future vorticity at $t=11..20$
*   **Observation:** Performance now matches FNO baseline (0.73% improvement). The joint 2-stage training prevents the "regularization tax" that previously caused degradation in the 3-stage protocol.
*   **Final Error:** 0.2130

### 2D Burgers Equation
*   **Input:** Initial velocity field $f=(u,v)$ at $t=0$
*   **Output:** Solution velocity field at $T=1.0$
*   **Observation:** The model effectively captures the nonlinear evolution. Using Relative MSE was decisive here.
*   **Final Error:** 0.0154 (**4.9% improvement**)

---

## 4. Conclusion & Recommendations

The 2-stage LRN-FNO protocol is now verified as a robust alternative to standard Neural Operators.

1.  **Complexity matters:** LRN provides the most benefit (65%+) on complex, non-linear mappings like Darcy. 
2.  **Stability:** The 2-stage approach is more stable and easier to tune than the 3-stage curriculum.
3.  **Scale Invariance:** Using `RelativeMSELoss` is mandatory to avoid loss imbalance across different PDE scales.

### Final Artifacts
- **Scripts:** `examples/*_demo_v2.py`
- **Trainer:** `src/utils/training.py` (`LRNTrainerV2`)
- **Loss:** `src/losses/infonce.py` (`RelativeMSELoss`)
- **Logs:** `logs/*.log`
