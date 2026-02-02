# Repository Restructuring & LRR Expansion

## 1. Refactor Training Utilities in `src/utils/training.py`
- [/] Rename `LRNTrainer` to `CurriculumTrainer` (or similar generic name).
- [/] Rename `LRNTrainerV2` to `Trainer`.
- [/] Ensure docstrings reflect the generic nature (not specific to LRN).

## 2. Restructure `src/models`
- [x] Create `src/models/components/` and move shared files inside:
    - `src/models/fno.py` -> `src/models/components/fno.py`
    - `src/models/encoders.py` -> `src/models/components/encoders.py`
    - `src/models/latent_bridge.py` -> `src/models/components/latent_bridge.py`
- [x] Create `src/models/lrn/` and move `src/models/lrn_fno.py` logic there.
- [x] Create `src/models/lrr/` for LRR-specific models.
- [x] Update `src/models/__init__.py` to export everything cleanly.
- [x] Update import paths in `examples/` and tests.

## 3. Implement LRR Strategy for Steady-State Problems
- [x] Create `examples/burgers2d_lrr_steady_demo.py` (Burgers 2D Steady).
- [x] Create `examples/navier_stokes_lrr_steady_demo.py` (NS Steady).
- [x] Create `examples/darcy_lrr_demo.py` (Darcy Flow).
- [x] Ensure all demos use `argparse` for hyperparameters (no hardcoding).

## 4. Update Imports & Fix Breaking Changes
- [x] Update `src/losses/infonce.py` if it relies on specific imports (check usage).
- [x] Update `src/models/__init__.py` if it exists.
- [x] Update existing examples in `examples/`:
    - [ ] `burgers2d_demo_v2.py`
    - [ ] `darcy_demo_v2.py`
    - [ ] `navier_stokes_demo_v2.py`
    - [ ] `burgers_demo.py` etc.

## 5. Debugging LRR Performance
- [ ] Fix `CurriculumTrainer` parameter groups for LRR (include `vk_projection`, unfreeze backbone?).
- [ ] Verify fix with short training runs on Burgers and NS.
    - [ ] `burgers2d_demo_v2.py`
    - [ ] `darcy_demo_v2.py`
    - [ ] `navier_stokes_demo_v2.py`
    - [ ] `burgers2d_steady_demo.py`
    - [ ] `burgers2d_lrr_steady_demo.py`

## 4. Implement Navier-Stokes LRR Demo
- [ ] Create `examples/navier_stokes_lrr_steady_demo.py`.
- [ ] Use `LRRFNO2d` model from new path.
- [ ] Apply 2-layer MLP projection head (already in model).
- [ ] Use `use_gated_bridge=False`.
- [ ] Use `lambda_nce=0.01` (start with optimized param from Burgers).
- [ ] Train for 100 epochs.

## 5. Verification
- [ ] Run `burgers2d_lrr_steady_demo.py` to ensure refactor didn't break it.
- [ ] Run `navier_stokes_lrr_steady_demo.py` to get results.
