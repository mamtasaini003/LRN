#!/usr/bin/env python3
"""
Darcy Flow Demo Script - LRR Version (Latent Space Supervision)
Date: 2026-02-02
Uses LRR-FNO for Darcy Flow problem.
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
from data.pde_datasets import DarcyDataset

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
    # Ensure shapes match and flatten
    diff = pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1)
    norm = torch.norm(diff, dim=1)
    target_norm = torch.norm(target.reshape(target.shape[0], -1), dim=1)
    return (norm / target_norm).mean()

def main():
    parser = argparse.ArgumentParser(description='Darcy LRR Demo')
    parser.add_argument('--lambda_nce', type=float, default=1.0, help='NCE loss weight')
    parser.add_argument('--lambda_mse', type=float, default=1.0, help='MSE loss weight')
    parser.add_argument('--width', type=int, default=32, help='Model width')
    parser.add_argument('--epochs', type=int, default=150, help='Total epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LRR-FNO 2D Darcy Flow Demo")
    print("Alignment Target: Backbone Features (v_K) <-> Solution Latent (z_u)")
    print(f"Config: {args}")
    print("=" * 60)
    
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Dataset
    print("\nPreparing Darcy Dataset...")
    RESOLUTION = 32
    N_TRAIN = 400
    N_TEST = 100
    
    train_dataset = DarcyDataset(resolution=RESOLUTION, num_samples=N_TRAIN, train=True)
    test_dataset = DarcyDataset(resolution=RESOLUTION, num_samples=N_TEST, train=False)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # --- Vanilla FNO ---
    print("\n--- Training Vanilla FNO ---")
    fno = FNO2d(
        in_channels=1,
        out_channels=1,
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
        for a, u in train_loader:
            a, u = a.to(device), u.to(device)
            optimizer.zero_grad()
            pred = fno(a)
            # Prediction [B, 1, H, W] -> [B, H, W] for MSE against u [B, H, W]
            if pred.shape[1] == 1: pred = pred.squeeze(1)
            
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
        for a, u in test_loader:
            a, u = a.to(device), u.to(device)
            pred = fno(a)
            if pred.shape[1] == 1: pred = pred.squeeze(1)
            
            fno_errors.append(compute_relative_l2(pred, u).item())
    fno_rel_l2 = np.mean(fno_errors)
    
    # --- LRR-FNO ---
    print("\n--- Training LRR-FNO (2-Stage) ---")
    set_seed(args.seed)
    
    # Using the new LRRFNO2d class
    lrr_fno = LRRFNO2d(
        in_channels=1,
        out_channels=1,
        modes1=12, modes2=12,
        width=args.width,
        num_layers=4,
        latent_dim=64,
        encoder_channels=[16, 32, 64], # Smaller encoder for Darcy (32x32)
        use_gated_bridge=False
    ).to(device)
    
    # Using LRR Loss
    lrr_loss_fn = LRNLoss(temperature=0.1, lambda_mse=args.lambda_mse, lambda_nce=args.lambda_nce)
    
    # Calculate stage epochs (Standard 110/40 split for 150 total)
    stage1_epochs = int(110/150 * args.epochs)
    stage2_epochs = args.epochs - stage1_epochs
    
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
        checkpoint_dir='checkpoints_darcy_lrr'
    )
    
    trainer.train()
    
    # Evaluate LRR-FNO
    lrr_fno.eval()
    lrr_errors = []
    with torch.no_grad():
        for a, u in test_loader:
            a, u = a.to(device), u.to(device)
            output = lrr_fno(a)
            pred = output['prediction']
            
            lrr_errors.append(compute_relative_l2(pred, u).item())
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
        a, u = next(iter(test_loader))
        a, u = a.to(device), u.to(device)
        fno_pred = fno(a)
        if fno_pred.shape[1] == 1: fno_pred = fno_pred.squeeze(1)
        
        lrr_output = lrr_fno(a)
        lrr_pred = lrr_output['prediction']
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Show first sample in batch
    for i, (label, data) in enumerate([('Ground Truth u(x)', u[0].cpu()),
                                        ('FNO Prediction', fno_pred[0].cpu()),
                                        ('LRR-FNO Prediction', lrr_pred[0, 0].cpu())]):
        im = axes[i].imshow(data, cmap='viridis')
        axes[i].set_title(label)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    fig.suptitle(f'Darcy Flow | FNO: {fno_rel_l2:.4f} | LRR: {lrr_rel_l2:.4f} | Δ={improvement:.2f}%')
    plt.tight_layout()
    plt.savefig('results/plots/darcy_lrr_comparison.png', dpi=150)
    print("Plot saved to results/plots/darcy_lrr_comparison.png")

if __name__ == '__main__':
    main()
