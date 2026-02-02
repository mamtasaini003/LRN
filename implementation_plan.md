# Refactoring & Navier-Stokes LRR Implementation Plan

## Goal
Restructure the repository for clarity, generalize the training utilities, and extend the LRR strategy to the Navier-Stokes steady-state problem.

## 1. Refactor Training Utilities
**File**: `src/utils/training.py`
- Rename `LRNTrainer` -> `CurriculumTrainer` (Supports multi-stage training).
- Rename `LRNTrainerV2` -> `Trainer` (Standard trainer used for almost everything now).
- Update class docstrings to be generic (remove "LRN-specific" language where possible, while keeping the staged logic description).

## 2. Restructure Models Directory
Current:
`src/models/` -> `fno.py`, `encoders.py`, `latent_bridge.py`, `lrn_fno.py`

New:
- `src/models/components/`:
    - `fno.py` (Move)
    - `encoders.py` (Move)
    - `latent_bridge.py` (Move)
- `src/models/lrn/`:
    - `model.py` (Created from `LRNFNO*` classes in `lrn_fno.py`)
- `src/models/lrr/`:
    - `model.py` (Created from `LRRFNO*` classes in `lrn_fno.py`)
- `src/models/__init__.py`:
    - Update to expose classes from new locations for backward compatibility or cleaner imports.

## 3. Update Existing Scripts
All files in `examples/` and `src/` that import from `src/models` or `src/utils/training` must be updated.

**Affected Files**:
- `examples/burgers2d_demo_v2.py`
- `examples/darcy_demo_v2.py`
- `examples/navier_stokes_demo_v2.py`
- `examples/burgers2d_steady_demo.py`
- `examples/burgers2d_lrr_steady_demo.py`
- `src/losses/infonce.py` (if applicable)

**Import Changes**:
- `from models.lrn_fno import LRNFNO2d` -> `from models.lrn.model import LRNFNO2d`
- `from models.lrn_fno import LRRFNO2d` -> `from models.lrr.model import LRRFNO2d`
- `from models.fno import FNO2d` -> `from models.components.fno import FNO2d`
- `from utils.training import LRNTrainerV2` -> `from utils.training import Trainer`

## 4. Implement Navier-Stokes LRR Demo
**File**: `examples/navier_stokes_lrr_steady_demo.py`
- Clone `examples/burgers2d_lrr_steady_demo.py`.
- **Dataset**: `NavierStokesSteadyDataset` (check exact name in `src/data/steady_state_datasets.py`).
- **Model Config**:
    - `in_channels=1`, `out_channels=1` (NS is single channel vorticity).
    - `width=32` (Baseline).
    - `use_gated_bridge=False` (Simplified).
    - `lambda_nce=0.01` (Tuned).
- **Training**:
    - Epochs: 100 (60 Stage 1, 40 Stage 2).

## 5. Verification Plan
1. **Regression Testing**: Run `examples/burgers2d_steady_demo.py` (Baseline FNO) -> Should run without import errors.
2. **LRR Validation**: Run `examples/burgers2d_lrr_steady_demo.py` -> Should still achieve ~0.033 error.
3. **New Feature Test**: Run `examples/navier_stokes_lrr_steady_demo.py` -> Should run and ideally outperform FNO baseline.
