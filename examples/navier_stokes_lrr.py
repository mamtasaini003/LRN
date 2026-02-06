#!/usr/bin/env python3
"""
Steady-State Navier-Stokes Demo Script - LRR Version
Date: 2026-02-02
Uses time-independent forcing-to-vorticity mapping with LRR-FNO.
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
from data.steady_state_datasets import NavierStokesSteadyDataset

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
    parser = argparse.ArgumentParser(description='Navier-Stokes LRR Demo')
    parser.add_argument('--lambda_nce', type=float, default=0.01, help='NCE loss weight')
    parser.add_argument('--lambda_mse', type=float, default=10000.0, help='MSE loss weight')
    parser.add_argument('--width', type=int, default=32, help='Model width')
    parser.add_argument('--epochs', type=int, default=200, help='Total epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LRR-FNO 2D Navier-Stokes (STEADY-STATE) Demo")
    print("Alignment Target: Backbone Features (v_K) <-> Solution Latent (z_u)")
    print(f"Config: {args}")
    print("=" * 60)
    
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Dataset
    print("\nPreparing Navier-Stokes Dataset (NeuralOperator)...")
    from data.neuralop_loaders import create_neuralop_dataloaders
    RESOLUTION = 128  # Standard NS resolution
    
    try:
        train_loader, test_loader, processor = create_neuralop_dataloaders(
            dataset_name='navier_stokes',
            n_train=args.n_train if hasattr(args, 'n_train') else 300,
            n_test=args.n_test if hasattr(args, 'n_test') else 100,
            batch_size=16,
            test_batch_size=16,
            resolution=RESOLUTION,
            return_tuple_format=True,
            encode_input=True # Normalize inputs for generalization
        )
    except Exception as e:
        print(f"Failed to load NeuralOperator dataset: {e}")
        print("Falling back to legacy NavierStokesSteadyDataset...")
        from data.steady_state_datasets import NavierStokesSteadyDataset
        train_dataset = NavierStokesSteadyDataset(resolution=64, num_samples=300, train=True)
        test_dataset = NavierStokesSteadyDataset(resolution=64, num_samples=100, train=False)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # --- Vanilla FNO ---
    print("\n--- Training Vanilla FNO ---")
    fno = FNO2d(
        in_channels=1, # NeuralOp NS is 1 channel (vorticity)
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
        for batch in train_loader:
            if isinstance(batch, dict):
                f, u = batch['x'], batch['y']
            else:
                f, u = batch
            
            f, u = f.to(device), u.to(device)
            # Unsqueeze for single channel if needed
            if f.dim() == 3: f = f.unsqueeze(1)
            if u.dim() == 3: u = u.unsqueeze(1)
            
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
        for batch in test_loader:
            if isinstance(batch, dict):
                f, u = batch['x'], batch['y']
            else:
                f, u = batch
            
            f, u = f.to(device), u.to(device)
            if f.dim() == 3: f = f.unsqueeze(1)
            if u.dim() == 3: u = u.unsqueeze(1)
            
            pred = fno(f)
            for i in range(pred.size(0)):
                fno_errors.append(compute_relative_l2(pred[i], u[i]).item())
    fno_rel_l2 = np.mean(fno_errors)
    
    # --- LRR-FNO (Latent Supervision Only) ---
    print("\n--- Training LRR-FNO (Latent Supervision Only) ---")
    set_seed(args.seed)
    
    class LatentSupervisedFNO(LRRFNO2d):
        """
        LRR-FNO variant that ignores encoder_f context for prediction.
        """
        def forward(self, f, u=None, return_latents=True):
            # Standard forward to get z_f_input (for shape) and v_K
            output = super().forward(f, u, return_latents)
            
            # Re-run bridge with ZERO context to isolate NCE alignment effect
            v_K = self.fno.backbone_forward(f)
            z_f_input = self.encoder_f(f) 
            z_zero = torch.zeros_like(z_f_input) # Zero context
            
            v_latent = self.latent_bridge(v_K, z_zero)
            u_pred = self.fno.project(v_latent)
            u_pred = u_pred.permute(0, 3, 1, 2)
            
            output['prediction'] = u_pred
            return output

    lrr_fno = LatentSupervisedFNO(
        in_channels=1,
        out_channels=1,
        modes1=12, modes2=12,
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
        checkpoint_dir='checkpoints_ns_lrr'
    )
    
    trainer.train()
    
    # Evaluate LRR-FNO
    lrr_fno.eval()
    lrr_errors = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                f, u = batch['x'], batch['y']
            else:
                f, u = batch
            
            f, u = f.to(device), u.to(device)
            if f.dim() == 3: f = f.unsqueeze(1)
            if u.dim() == 3: u = u.unsqueeze(1)
            
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
        batch = next(iter(test_loader))
        if isinstance(batch, dict):
            f, u = batch['x'], batch['y']
        else:
            f, u = batch
            
        f, u = f.to(device), u.to(device)
        if f.dim() == 3: f = f.unsqueeze(1)
        if u.dim() == 3: u = u.unsqueeze(1)
        
        fno_pred = fno(f)
        lrr_output = lrr_fno(f)
        lrr_pred = lrr_output['prediction']
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Show first sample
    for i, (label, data) in enumerate([('Ground Truth', u[0, 0].cpu()),
                                        ('FNO', fno_pred[0, 0].cpu()),
                                        ('LRR-FNO', lrr_pred[0, 0].cpu())]):
        axes[i].imshow(data, cmap='viridis')
        axes[i].set_title(label)
        axes[i].axis('off')
    
    fig.suptitle(f'Navier-Stokes | FNO: {fno_rel_l2:.4f} | LRR: {lrr_rel_l2:.4f} | Δ={improvement:.2f}%')
    plt.tight_layout()
    plt.savefig('results/plots/ns_lrr_steady_comparison.png', dpi=150)
    print("Plot saved to results/plots/ns_lrr_steady_comparison.png")

if __name__ == '__main__':
    main()
