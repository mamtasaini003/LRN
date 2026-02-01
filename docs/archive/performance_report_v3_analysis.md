# LRN-FNO Performance Update: The "Loss Imbalance" Fix

**Date:** December 30, 2025
**Task:** 2D Burgers Equation (Fixing the degradation issue)

---

## 1. The Issue

In previous V2 experiments, the LRN model performed significantly worse than the FNO baseline on the 2D Burgers task (-18% degradation), while performing excellently on Darcy Flow (+65%).

**Observation:**
- Darcy Flow MSE Loss Magnitude: ~`0.007`
- Burgers 2D MSE Loss Magnitude: ~`0.00002`

**Diagnosis:**
The combined loss function is $\mathcal{L} = \mathcal{L}_{NCE} + \lambda \mathcal{L}_{MSE}$.
The Contrastive Loss ($\mathcal{L}_{NCE}$) typically has values around 3.0.
- For Darcy: Ratio NCE/MSE ≈ 400 (Manageable)
- For Burgers: Ratio NCE/MSE ≈ 150,000 (Catastrophic Imbalance)

The optimizer effectively ignored the reconstruction objective because its gradient contribution was negligible compared to the contrastive loss.

---

## 2. The Solution

We implemented two changes to the loss function:

1.  **Switch to Relative MSE**: Instead of absolute MSE (which depends on data scale), we use relative squared error $||\hat{u}-u||^2 / ||u||^2$. This is scale-invariant.
2.  **Increase Reconstruction Weight ($\lambda$)**: Increased $\lambda$ from 1.0 to 20.0 to prioritize pixel-perfect accuracy.

```python
# Old Configuration
loss_fn = LRNLoss(lambda_mse=1.0) # Uses standard MSE

# New Configuration
loss_fn = LRNLoss(lambda_mse=20.0, use_relative_mse=True)
```

---

## 3. Results (Burgers 2D)

| Model | Setup | Rel L2 Error | Improvement |
|-------|-------|--------------|-------------|
| **Vanilla FNO** | Baseline | 0.0153 | - |
| **LRN V2 (Old)** | Standard MSE, λ=1 | 0.0163 | -18.51% (Degradation) |
| **LRN V3 (Fixed)** | Relative MSE, λ=20 | **0.0150** | **+1.95% (Improvement)** |

---

## 4. Conclusion

The "weird" degradation on simple tasks was a **optimization artifact** caused by loss scaling. Simple data often results in very small absolute MSE values, which gets drowned out by the InfoNCE loss in the joint objective.

**Recommendation:** Always use `use_relative_mse=True` and tune $\lambda$ to ensure the reconstruction loss term is of comparable magnitude to the contrastive term (approx 1.0 - 5.0 after weighting).
