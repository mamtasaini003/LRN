# Repository Modification and Experiment Guidelines

This document outlines the strict rules for modifying the LRN-FNO repository and documenting new experiments. Adhering to these rules ensures the project's integrity, traceability, and long-term reproducibility.

---

## 1. General Rules for Modifications

### A. Preservation of History
- **Rule:** Previous results, reports, and analysis documents (e.g., `docs/final_performance_report.md`, `docs/architecture_evolution.md`) must **NEVER** be modified unless explicitly requested by the user.
- **Goal:** To maintain a permanent record of the project's development and previous success baselines.

### B. Documentation of New Work
- **Rule:** Every new experiment, architectural change, or hyperparameter sweep must be recorded in a **new file**.
- **Naming Convention:** Use meaningful, snake_case names: `docs/analysis/experiment_YYYY_MM_DD_description.md`.
- **Reference:** Always link back to the previous baseline report to show context.

---

## 2. Standardized Analysis Format

Every analysis file MUST follow the same structural sequence to ensure peer-reviewability:

1. **Header Information**
   - **Date:** (Current Date)
   - **Experiment Name:** (Clear, descriptive title)
   - **Status:** (Draft / Reproducible / Final)

2. **Problem Statement & Focus**
   - **Goal:** What specific problem are we trying to solve?
   - **Point of Focus:** What is the primary hypothesis being tested?

3. **Technical Configuration**
   - **Dataset:** (Name, complexity level, 1-channel vs 10-channel, Transient vs Steady-state)
   - **Model Details:** (Architecture version, hidden width, number of layers, latent dim)
   - **Hyperparameters:** (λ_MSE, λ_NCE, learning rates for each stage)
   - **Optimizers:** (Optimizer type, scheduler type, total epochs)
   - **Training Details:** (Number of samples, batch size, hardware used)

4. **Detailed Changes**
   - **What is changing?** (List exact code modifications, new script names, or dataset logic changes)
   - **Why is this better?** (Justify the change: higher accuracy, better convergence, faster training, etc.)

5. **Analytical Summary**
   - **Key Findings:** (Quantified improvements: FNO Error vs LRN Error)
   - **Limitations:** (Where does this approach fail? What was not tested?)
   - **Scope of Improvement:** (What are the next logical steps?)

---

## 3. Procedural Requirements

### A. Reproducibility
- All experiments must use `set_seed(42)` (or the current project seed) to ensure weight initialization and data generation are deterministic.
- Use `git` to tag or commit specifically before/after an experiment to preserve the exact code state.

### B. Metric Consistency
- Always report **Relative L2 Error** as the primary metric.
- Include an **Improvement %** calculation relative to the Vanilla FNO baseline.

### C. Visual Validation
- Every experiment must save its comparison plots to `results/plots/` using a unique filename.
- Embed these plots directly in the analysis document.
