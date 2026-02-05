#!/usr/bin/env python3
"""
Steady-State Burgers 2D Demo Script - LRR Version (Latent Space Supervision)
Date: 2026-02-02
Uses time-independent forcing-to-solution mapping with LRR-FNO.
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.components.fno import FNO2d
from models.lrn.model import LRNFNO2d
from models.lrr.model import LRRFNO2d
from losses.infonce import LRNLoss
from utils.training import Trainer
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
    parser = argparse.ArgumentParser(description='Burgers 2D LRR Demo')
    parser.add_argument('--lambda_nce', type=float, default=0.01, help='NCE loss weight')
    parser.add_argument('--lambda_mse', type=float, default=10000.0, help='MSE loss weight')
    parser.add_argument('--width', type=int, default=32, help='Model width')
    parser.add_argument('--epochs', type=int, default=150, help='Total epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LRR-FNO 2D Burgers (STEADY-STATE) - Latent Supervision Demo")
    print("Alignment Target: Backbone Features (v_K) <-> Solution Latent (z_u)")
    print(f"Config: {args}")
    print("=" * 60)
    
    set_seed(args.seed)
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
        width=args.width,
        num_layers=4
    ).to(device)
    
    optimizer = optim.Adam(fno.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    mse_loss = nn.MSELoss()
    
    print(f"Training FNO for {args.epochs} epochs...")
    for epoch in range(1, args.epochs+1):
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
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{args.epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
    
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
    
    # --- LRR-FNO ---
    print("\n--- Training LRR-FNO (2-Stage) ---")
    set_seed(args.seed)
    
    # Using the new LRRFNO2d class
    lrr_fno = LRRFNO2d(
        in_channels=2,
        out_channels=2,
        modes1=12, modes2=12,
        width=args.width,
        num_layers=4,
        latent_dim=64,
        encoder_channels=[32, 64, 128],
        use_gated_bridge=False
    ).to(device)
    
    # Using same loss config, but now 'nce' means alignment of v_K and z_u
    lrr_loss_fn = LRNLoss(temperature=0.1, lambda_mse=args.lambda_mse, lambda_nce=args.lambda_nce)
    
    # Calculate stage epochs (Standard 110/40 split for 150 total)
    stage1_epochs = int(110/150 * args.epochs)
    stage2_epochs = args.epochs - stage1_epochs
    
    # Using existing trainer - it works because we mapped v_k_proj to 'z_f' in forward output
    trainer = Trainer(
        model=lrr_fno,
        loss_fn=lrr_loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        stage1_lr=1e-3,
        stage2_lr=1e-4,
        checkpoint_dir='checkpoints_lrr'
    )
    trainer.train()
    
    # Evaluate LRR-FNO
    lrr_fno.eval()
    lrr_errors = []
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            output = lrr_fno(f)
            pred = output['prediction']
            for i in range(pred.size(0)):
                lrr_errors.append(compute_relative_l2(pred[i], u[i]).item())
    lrr_rel_l2 = np.mean(lrr_errors)
    
    # Results
    print("\n--- Final Evaluation ---")
    print(f"FNO Test Rel L2:     {fno_rel_l2:.6f}")
    print(f"LRR-FNO Test Rel L2: {lrr_rel_l2:.6f}")
    improvement = (fno_rel_l2 - lrr_rel_l2) / fno_rel_l2 * 100
    print(f"Improvement:         {improvement:.2f}%")
    
    # Save plot
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        f, u = next(iter(test_loader))
        f, u = f.to(device), u.to(device)
        fno_pred = fno(f)
        lrr_output = lrr_fno(f)
        lrr_pred = lrr_output['prediction']
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, (label, data) in enumerate([('Ground Truth', u[0, 0].cpu()),
                                        ('FNO', fno_pred[0, 0].cpu()),
                                        ('LRR-FNO', lrr_pred[0, 0].cpu())]):
        axes[0, i].imshow(data, cmap='viridis')
        axes[0, i].set_title(f'{label} (u)')
        axes[0, i].axis('off')
    for i, (label, data) in enumerate([('Ground Truth', u[0, 1].cpu()),
                                        ('FNO', fno_pred[0, 1].cpu()),
                                        ('LRR-FNO', lrr_pred[0, 1].cpu())]):
        axes[1, i].imshow(data, cmap='viridis')
        axes[1, i].set_title(f'{label} (v)')
        axes[1, i].axis('off')
    
    fig.suptitle(f'Burgers 2D (Steady-State) | FNO: {fno_rel_l2:.4f} | LRR: {lrr_rel_l2:.4f} | Δ={improvement:.2f}%')
    plt.tight_layout()
    plt.savefig('results/plots/burgers2d_lrr_steady_comparison.png', dpi=150)
    print("Plot saved to results/plots/burgers2d_lrr_steady_comparison.png")

if __name__ == '__main__':
    main()
