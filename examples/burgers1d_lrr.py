#!/usr/bin/env python3
"""
Burgers 1D Demo Script - LRR Version
Date: 2026-02-06
Uses active 1D Burgers benchmark from NeuralOperator (converted from 2D steady demo).
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

from models.components.fno import FNO1d
from models.lrr.model import LRRFNO1d
from losses.infonce import LRNLoss
from utils.training import Trainer

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
    parser = argparse.ArgumentParser(description='Burgers 1D LRR Demo')
    parser.add_argument('--lambda_nce', type=float, default=0.01, help='NCE loss weight')
    parser.add_argument('--lambda_mse', type=float, default=1.0, help='MSE loss weight')
    parser.add_argument('--width', type=int, default=64, help='Model width')
    parser.add_argument('--epochs', type=int, default=200, help='Total epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--modes', type=int, default=16, help='Fourier modes')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LRR-FNO 1D Burgers Demo (Standard Benchmark)")
    print("Alignment Target: Backbone Features (v_K) <-> Solution Latent (z_u)")
    print(f"Config: {args}")
    print("=" * 60)
    
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Dataset
    print("\nPreparing Burgers 1D Dataset (NeuralOperator)...")
    from data.neuralop_loaders import create_neuralop_dataloaders
    RESOLUTION = 128
    
    try:
        train_loader, test_loader, processor = create_neuralop_dataloaders(
            dataset_name='burgers',
            n_train=args.n_train if hasattr(args, 'n_train') else 1000,
            n_test=args.n_test if hasattr(args, 'n_test') else 200,
            batch_size=32,
            test_batch_size=32,
            resolution=RESOLUTION,
            return_tuple_format=True,
            encode_input=True # Normalize inputs for generalization
        )
    except Exception as e:
        print(f"Failed to load NeuralOperator dataset: {e}")
        print("Using synthetic BurgersDataset (1D) from pde_datasets...")
        from data.pde_datasets import BurgersDataset
        
        # Use on-the-fly synthetic generation (more reliable than pre-generated files)
        train_dataset = BurgersDataset(resolution=RESOLUTION, num_samples=1000, train=True)
        test_dataset = BurgersDataset(resolution=RESOLUTION, num_samples=200, train=False)
            
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # --- Vanilla FNO ---
    print("\n--- Training Vanilla FNO ---")
    fno = FNO1d(
        in_channels=1,
        out_channels=1,
        modes=args.modes,
        width=args.width,
        num_layers=4
    ).to(device)
    
    optimizer = optim.Adam(fno.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    mse_loss = nn.MSELoss()
    
    print(f"Training FNO for {args.epochs} epochs...")
    for epoch in range(1, args.epochs+1):
        fno.train()
        epoch_loss = 0.0
        for batch in train_loader:
            if isinstance(batch, dict):
                x, y = batch['x'], batch['y']
            else:
                x, y = batch
                
            x, y = x.to(device), y.to(device)
            # Ensure correct dims [B, C, L]
            if x.dim() == 2: x = x.unsqueeze(1)
            if y.dim() == 2: y = y.unsqueeze(1)
            
            optimizer.zero_grad()
            pred = fno(x)
            # FNO1d outputs [B, L, C], permute to [B, C, L]
            if pred.dim() == 3 and pred.shape[-1] == 1:
                pred = pred.permute(0, 2, 1)
            
            loss = mse_loss(pred, y)
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
        for batch in test_loader:
            if isinstance(batch, dict):
                x, y = batch['x'], batch['y']
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            if x.dim() == 2: x = x.unsqueeze(1)
            if y.dim() == 2: y = y.unsqueeze(1)
            
            pred = fno(x)
            if pred.dim() == 3 and pred.shape[-1] == 1:
                pred = pred.permute(0, 2, 1)
            for i in range(pred.size(0)):
                fno_errors.append(compute_relative_l2(pred[i], y[i]).item())
    fno_rel_l2 = np.mean(fno_errors)
    fno_rel_l2 = np.mean(fno_errors)
    
    # --- LRR-FNO (Latent Supervision Only) ---
    print("\n--- Training LRR-FNO (Latent Supervision Only) ---")
    set_seed(args.seed)
    
    class LatentSupervisedFNO1d(LRRFNO1d):
        """
        LRR-FNO 1D variant that ignores encoder_f context for prediction.
        """
        def forward(self, f, u=None, return_latents=True):
            output = super().forward(f, u, return_latents)
            
            # Re-run bridge with ZERO context to isolate NCE alignment effect
            v_K = self.fno.backbone_forward(f)
            z_f_input = self.encoder_f(f) # Just for shape
            z_zero = torch.zeros_like(z_f_input) # Zero context
            
            v_latent = self.latent_bridge(v_K, z_zero)
            u_pred = self.fno.project(v_latent)
            u_pred = u_pred.permute(0, 2, 1) # [B, C, S]
            
            output['prediction'] = u_pred
            return output

    lrr_fno = LatentSupervisedFNO1d(
        in_channels=1,
        out_channels=1,
        modes=args.modes,
        width=args.width,
        num_layers=4,
        latent_dim=128, # Increased capacity
        encoder_channels=[16, 32, 64],
        use_gated_bridge=False
    ).to(device)
    
    # Physics-First Loss Weights
    loss_fn = LRNLoss(lambda_nce=0.001, lambda_mse=5.0, temperature=0.07, symmetric_nce=True)
    
    trainer = Trainer(
        model=lrr_fno,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        stage1_epochs=args.epochs, # Full training in Stage 1
        stage2_epochs=0, # Disable distillation
        stage1_lr=1e-3,
        weight_decay=1e-4, # Regularization
        device=str(device),
        checkpoint_dir='checkpoints_burgers_lrr'
    )
    
    trainer.train()
    
    # Evaluate LRR-FNO
    lrr_fno.eval()
    lrr_errors = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                x, y = batch['x'], batch['y']
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            if x.dim() == 2: x = x.unsqueeze(1)
            if y.dim() == 2: y = y.unsqueeze(1)
            
            output = lrr_fno(x)
            pred = output['prediction']
            for i in range(pred.size(0)):
                lrr_errors.append(compute_relative_l2(pred[i], y[i]).item())
    lrr_rel_l2 = np.mean(lrr_errors)
    
    # Results
    print("\n--- Final Evaluation ---")
    print(f"FNO Test Rel L2:     {fno_rel_l2:.6f}")
    print(f"LRR-FNO Test Rel L2: {lrr_rel_l2:.6f}")
    improvement = (fno_rel_l2 - lrr_rel_l2) / fno_rel_l2 * 100
    print(f"Improvement:         {improvement:.2f}%")
    
    # Save plot (Line plot for 1D)
    Path('results/plots').mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(test_loader))
        if isinstance(batch, dict):
            x, y = batch['x'], batch['y']
        else:
            x, y = batch
        x, y = x.to(device), y.to(device)
        if x.dim() == 2: x = x.unsqueeze(1)
        if y.dim() == 2: y = y.unsqueeze(1)
        
        fno_pred = fno(x)
        lrr_output = lrr_fno(x)
        lrr_pred = lrr_output['prediction']
        
    # Plotting first sample
    x_np = x[0, 0].cpu().numpy()
    y_np = y[0, 0].cpu().numpy()
    fno_np = fno_pred[0, 0].cpu().numpy()
    lrr_np = lrr_pred[0, 0].cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_np, 'k-', label='Ground Truth', linewidth=2)
    plt.plot(fno_np, 'b--', label='FNO', alpha=0.8)
    plt.plot(lrr_np, 'g--', label='LRR-FNO', alpha=0.8)
    plt.legend()
    plt.title(f'Burgers 1D | FNO: {fno_rel_l2:.4f} | LRR: {lrr_rel_l2:.4f} | Δ={improvement:.2f}%')
    plt.grid(True, alpha=0.3)
    plt.savefig('results/plots/burgers1d_lrr_comparison.png', dpi=150)
    print("Plot saved to results/plots/burgers1d_lrr_comparison.png")

if __name__ == '__main__':
    main()
