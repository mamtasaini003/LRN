#!/usr/bin/env python3
"""
Steady-State Burgers 2D Demo Script
Date: 2026-02-01
Uses time-independent forcing-to-solution mapping.
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.fno import FNO2d
from models.lrn_fno import LRNFNO2d
from losses.infonce import LRNLoss
from utils.training import LRNTrainerV2
from data.steady_state_datasets import Burgers2dSteadyDataset

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_relative_l2(pred, target):
    """Compute relative L2 error."""
    diff = pred - target
    return torch.norm(diff) / torch.norm(target)

def main():
    print("=" * 60)
    print("LRN-FNO 2D Burgers (STEADY-STATE) - Experiment 2026-02-01")
    print("=" * 60)
    
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    
    # Dataset
    print("\nPreparing Steady-State Burgers 2D Dataset...")
    RESOLUTION = 64
    N_TRAIN = 300
    N_TEST = 100
    
    train_dataset = Burgers2dSteadyDataset(resolution=RESOLUTION, num_samples=N_TRAIN, train=True)
    test_dataset = Burgers2dSteadyDataset(resolution=RESOLUTION, num_samples=N_TEST, train=False)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # --- Vanilla FNO ---
    print("\n--- Training Vanilla FNO ---")
    fno = FNO2d(
        in_channels=2,
        out_channels=2,
        modes1=12, modes2=12,
        width=32,
        num_layers=4
    ).to(device)
    
    optimizer = optim.Adam(fno.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    mse_loss = nn.MSELoss()
    
    print("Training FNO for 20 epochs (quick validation)...")
    for epoch in range(1, 101):
        fno.train()
        epoch_loss = 0.0
        for f, u in train_loader:
            f, u = f.to(device), u.to(device)
            optimizer.zero_grad()
            pred = fno(f)
            loss = mse_loss(pred, u)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/20, Loss: {epoch_loss/len(train_loader):.6f}")
    
    # Evaluate FNO
    fno.eval()
    fno_errors = []
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            pred = fno(f)
            for i in range(pred.size(0)):
                fno_errors.append(compute_relative_l2(pred[i], u[i]).item())
    fno_rel_l2 = np.mean(fno_errors)
    
    # --- LRN-FNO ---
    print("\n--- Training LRN-FNO (2-Stage) ---")
    set_seed(42)
    
    lrn_fno = LRNFNO2d(
        in_channels=2,
        out_channels=2,
        modes1=12, modes2=12,
        width=32,
        num_layers=4,
        latent_dim=64,
        encoder_channels=[32, 64, 128],
        use_gated_bridge=True
    ).to(device)
    
    lrn_loss_fn = LRNLoss(temperature=0.1, lambda_mse=10000.0, lambda_nce=0.01)
    
    trainer = LRNTrainerV2(
        model=lrn_fno,
        loss_fn=lrn_loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        stage1_epochs=15,
        stage2_epochs=5,
        stage1_lr=1e-3,
        stage2_lr=1e-4
    )
    trainer.train()
    
    # Evaluate LRN-FNO
    lrn_fno.eval()
    lrn_errors = []
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            output = lrn_fno(f)
            pred = output['prediction']
            for i in range(pred.size(0)):
                lrn_errors.append(compute_relative_l2(pred[i], u[i]).item())
    lrn_rel_l2 = np.mean(lrn_errors)
    
    # Results
    print("\n--- Final Evaluation ---")
    print(f"FNO Test Rel L2:     {fno_rel_l2:.6f}")
    print(f"LRN-FNO Test Rel L2: {lrn_rel_l2:.6f}")
    improvement = (fno_rel_l2 - lrn_rel_l2) / fno_rel_l2 * 100
    print(f"Improvement:         {improvement:.2f}%")
    
    # Save plot
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        f, u = next(iter(test_loader))
        f, u = f.to(device), u.to(device)
        fno_pred = fno(f)
        lrn_output = lrn_fno(f)
        lrn_pred = lrn_output['prediction']
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (label, data) in enumerate([('Ground Truth', u[0, 0].cpu()),
                                        ('FNO', fno_pred[0, 0].cpu()),
                                        ('LRN-FNO', lrn_pred[0, 0].cpu())]):
        axes[0, i].imshow(data, cmap='viridis')
        axes[0, i].set_title(f'{label} (u)')
        axes[0, i].axis('off')
    for i, (label, data) in enumerate([('Ground Truth', u[0, 1].cpu()),
                                        ('FNO', fno_pred[0, 1].cpu()),
                                        ('LRN-FNO', lrn_pred[0, 1].cpu())]):
        axes[1, i].imshow(data, cmap='viridis')
        axes[1, i].set_title(f'{label} (v)')
        axes[1, i].axis('off')
    
    fig.suptitle(f'Burgers 2D (Steady-State) | FNO: {fno_rel_l2:.4f} | LRN: {lrn_rel_l2:.4f} | Δ={improvement:.2f}%')
    plt.tight_layout()
    plt.savefig('results/plots/burgers2d_steady_comparison.png', dpi=150)
    print("Plot saved to results/plots/burgers2d_steady_comparison.png")

if __name__ == '__main__':
    main()
