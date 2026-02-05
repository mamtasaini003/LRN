"""
LRR-FNO Experiment Hyperparameters Configuration

This file documents the standard hyperparameters used across all experiments.
These values are validated for research paper reproducibility.

Usage:
    python examples/exp1_steady_state_poisson_lrr.py --all --epochs 100
    python examples/exp2_latent_analysis_spectral.py --epochs 100
    python examples/exp3_super_resolution.py --epochs 100
    python examples/exp4_noise_robustness.py --epochs 100
    python examples/exp5_ood_forcing.py --epochs 100
"""

# =============================================================================
# Model Architecture Hyperparameters (Consistent Across All Experiments)
# =============================================================================

FNO_CONFIG = {
    'modes1': 12,           # Fourier modes in x-direction
    'modes2': 12,           # Fourier modes in y-direction  
    'width': 32,            # Hidden channel width
    'num_layers': 4,        # Number of Fourier layers
}

LRRFNO_CONFIG = {
    # FNO backbone (same as vanilla FNO)
    'modes1': 12,
    'modes2': 12,
    'width': 32,
    'num_layers': 4,
    
    # LRR-specific parameters
    'latent_dim': 64,                    # Latent space dimension
    'encoder_channels': [32, 64, 128],   # Encoder architecture
    'use_gated_bridge': False,           # Whether to use gated bridge
}

# =============================================================================
# Loss Function Configuration
# =============================================================================

LRN_LOSS_CONFIG = {
    'temperature': 0.1,       # InfoNCE temperature
    'lambda_mse': 10000.0,    # MSE loss weight (prediction accuracy)
    'lambda_nce': 0.01,       # InfoNCE loss weight (latent alignment)
}

# =============================================================================
# Training Hyperparameters
# =============================================================================

TRAINING_CONFIG = {
    # Epochs
    'epochs': 100,            # Recommended for final results (50 for quick tests)
    
    # Learning rates
    'stage1_lr': 1e-3,        # Stage 1 (joint training) learning rate
    'stage2_lr': 1e-4,        # Stage 2 (fine-tuning) learning rate
    'fno_lr': 1e-3,           # Vanilla FNO learning rate
    
    # Stage split (LRR two-stage training)
    'stage1_ratio': 0.73,     # 73% of epochs for stage 1
    
    # LR scheduler
    'scheduler': 'CosineAnnealing',  # Learning rate schedule
    
    # Data
    'max_samples': 200,       # Max samples per dataset (None = use all)
    'batch_size': 8,          # Training batch size
    'train_split': 0.8,       # Train/test split ratio
    
    # Reproducibility
    'seed': 42,               # Random seed
}

# =============================================================================
# Experiment-Specific Configurations
# =============================================================================

# Experiment 1: Steady-State PDE Comparison
EXP1_CONFIG = {
    'datasets': [
        'dataset/Circle.nc',
        'dataset/Ellipse-1.nc',
        'dataset/Ellipse-2.nc',
        'dataset/Ellipse-3.nc',
        'dataset/Cone-F.nc',
        'dataset/Semicircle-F.nc',
    ],
    'generate_plots': True,
}

# Experiment 2: Latent Analysis & Spectral Profiling
EXP2_CONFIG = {
    'dataset': 'dataset/Circle.nc',
    'track_epochs': [1, 10, 25, 50],  # Epochs to track latent evolution
}

# Experiment 3: Zero-Shot Super-Resolution
EXP3_CONFIG = {
    'dataset': 'dataset/Circle.nc',
    'train_resolution': 64,
    'test_resolutions': [64, 128],
}

# Experiment 4: Noise Robustness
EXP4_CONFIG = {
    'dataset': 'dataset/Circle.nc',
    'noise_levels': [0.0, 0.01, 0.02, 0.05, 0.1],  # Gaussian noise std
}

# Experiment 5: OOD Generalization
EXP5_CONFIG = {
    'train_dataset': 'dataset/Circle.nc',
    'test_datasets': [
        'dataset/Circle.nc',      # In-distribution
        'dataset/Ellipse-1.nc',   # OOD
        'dataset/Ellipse-2.nc',   # OOD
        'dataset/Ellipse-3.nc',   # OOD
        'dataset/Cone-F.nc',      # OOD
        'dataset/Semicircle-F.nc',# OOD
    ],
}

# =============================================================================
# Quick Run Commands (Copy-Paste Ready)
# =============================================================================

"""
# Run all experiments with recommended settings:

cd /home/mamta/work/LRN

# Experiment 1: All datasets (slowest, ~1 hour with --epochs 100)
python examples/exp1_steady_state_poisson_lrr.py --all --epochs 100

# Experiment 2: Latent analysis
python examples/exp2_latent_analysis_spectral.py --epochs 100

# Experiment 3: Super-resolution
python examples/exp3_super_resolution.py --epochs 100

# Experiment 4: Noise robustness  
python examples/exp4_noise_robustness.py --epochs 100

# Experiment 5: OOD generalization
python examples/exp5_ood_forcing.py --epochs 100

# Quick test run (10 epochs each):
for exp in 1 2 3 4 5; do
    python examples/exp${exp}*.py --epochs 10
done
"""
