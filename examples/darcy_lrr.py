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

class RelativeMSELoss(nn.Module):
    """
    Relative MSE Loss (Lp Loss with p=2).
    L = ||pred - target||_2 / ||target||_2
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred, target):
        # Flatten [B, ...] -> [B, N]
        diff = pred.reshape(pred.shape[0], -1) - target.reshape(target.shape[0], -1)
        diff_norm = torch.norm(diff, dim=1)
        target_norm = torch.norm(target.reshape(target.shape[0], -1), dim=1)
        
        loss = diff_norm / (target_norm + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def compute_relative_l2(pred, target):
    """Compute relative L2 error (wrapper)."""
    criterion = RelativeMSELoss(reduction='mean')
    return criterion(pred, target).item()

def main():
    parser = argparse.ArgumentParser(description='Darcy LRR Demo')
    parser.add_argument('--lambda_nce', type=float, default=1.0, help='NCE loss weight')
    parser.add_argument('--lambda_mse', type=float, default=1.0, help='MSE loss weight')
    parser.add_argument('--width', type=int, default=64, help='Model width')
    parser.add_argument('--epochs', type=int, default=200, help='Total epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--n_train', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=100, help='Number of test samples')

    args = parser.parse_args()
    
    if args.batch_size < 2:
        print("WARNING: Batch size < 2 implies NCE loss will be zero/undefined. Setting to 2.")
        args.batch_size = 2
    
    print("=" * 60)
    print("LRR-FNO 2D Darcy Flow Demo")
    print("Alignment Target: Backbone Features (v_K) <-> Solution Latent (z_u)")
    print(f"Config: {args}")
    print("=" * 60)
    
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    
    # Dataset
    print("\nPreparing Darcy Dataset (NeuralOperator)...")
    from data.neuralop_loaders import create_neuralop_dataloaders
    
    # Using standard resolution 16/32 from the wrapper or custom if downloaded
    RESOLUTION = 32 # We'll try to request 32, but wrapper defaults to 16 for training if not specified.
    # The wrapper's load_darcy enables 85 too if available. 
    # Let's request what the wrapper supports.
    
    try:
        train_loader, test_loader, processor = create_neuralop_dataloaders(
            dataset_name='darcy',
            n_train=args.n_train if hasattr(args, 'n_train') else 400,
            n_test=args.n_test if hasattr(args, 'n_test') else 100,
            batch_size=args.batch_size,
            test_batch_size=args.batch_size,
            resolution=RESOLUTION,
            encode_output=True,
            return_tuple_format=True,
            encode_input=True # Normalize inputs for better generalization
        )
    except Exception as e:
        print(f"Failed to load NeuralOperator dataset: {e}")
        print("Falling back to legacy DarcyDataset...")
        from data.pde_datasets import DarcyDataset
        train_dataset = DarcyDataset(resolution=RESOLUTION, num_samples=400, train=True)
        test_dataset = DarcyDataset(resolution=RESOLUTION, num_samples=100, train=False)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    print(f"Training FNO for {args.epochs} epochs using MSE Loss...")
    for epoch in range(1, args.epochs+1):
        fno.train()
        epoch_loss = 0.0
        for batch in train_loader:
            # Handle both tuple (x, y) and dict {'x':x, 'y':y} just in case
            if isinstance(batch, dict):
                a, u = batch['x'], batch['y']
            else:
                a, u = batch
            
            a, u = a.to(device), u.to(device)
            optimizer.zero_grad()
            pred = fno(a)
            # Prediction [B, 1, H, W] -> [B, H, W] for MSE against u [B, H, W]
            # NeuralOp dataset returns [B, 1, H, W], so we might start with [B, 1, H, W]
            if pred.shape != u.shape:
                 if pred.shape[1] == 1 and u.dim() == 3: pred = pred.squeeze(1)
                 elif u.shape[1] == 1 and pred.dim() == 3: u = u.squeeze(1)
            
            loss = criterion(pred, u)
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
                a, u = batch['x'], batch['y']
            else:
                a, u = batch
                
            a, u = a.to(device), u.to(device)
            pred = fno(a)
            if pred.shape != u.shape:
                 if pred.shape[1] == 1 and u.dim() == 3: pred = pred.squeeze(1)
                 elif u.shape[1] == 1 and pred.dim() == 3: u = u.squeeze(1)
            
            fno_errors.append(compute_relative_l2(pred, u))
    fno_rel_l2 = np.mean(fno_errors)
    
    # --- LRR-FNO (Latent Supervision Only) ---
    print("\n--- Training LRR-FNO (Latent Supervision Only) ---")
    set_seed(args.seed)
    
    class LatentSupervisedFNO(LRRFNO2d):
        """
        LRR-FNO variant that ignores encoder_f context for prediction.
        
        This isolates the effect of Latent Supervision (NCE on v_K <-> z_u).
        The bridge sees zero context, so prediction relies solely on v_K.
        """
        def forward(self, f, u=None, return_latents=True):
            # Standard forward to get z_f_input (for shape) and v_K
            output = super().forward(f, u, return_latents)
            
            # Re-run bridge with ZERO context to remove encoder_f influence
            v_K = self.fno.backbone_forward(f)
            z_f_input = self.encoder_f(f) 
            z_zero = torch.zeros_like(z_f_input) # Zero context
            
            # v_latent = bridge(v_K, 0)
            # We need to manually re-project because we can't easily intercept the internal call
            # properly without copying code.
            # But wait, super().forward() already computed 'prediction' using z_f_input.
            # We need to OVERWRITE 'prediction'.
            
            v_latent = self.latent_bridge(v_K, z_zero)
            u_pred = self.fno.project(v_latent)
            u_pred = u_pred.permute(0, 3, 1, 2)
            
            output['prediction'] = u_pred
            # output['z_v_k'] remains the same (aligned via NCE)
            return output

    # Using the new LatentSupervisedFNO class
    lrr_fno = LatentSupervisedFNO(
        in_channels=1,
        out_channels=1,
        modes1=12, modes2=12,
        width=args.width, # 64
        num_layers=4,
        latent_dim=128, 
        encoder_channels=[16, 32, 64], 
        use_gated_bridge=False
    ).to(device)
    
    # Using LRR Loss with Symmetric NCE and standard MSE (per user request)
    # Using LRR Loss with Symmetric NCE and standard MSE (per user request)
    lrr_loss_fn = LRNLoss(
        temperature=0.1, 
        lambda_mse=5.0, # Boost MSE signal
        lambda_nce=0.001, # Minimal regularizer
        symmetric_nce=True,
        use_relative_mse=False # distinct from eval metric
    )
    
    # 1-Stage Training (Hybrid NCE+MSE only)
    stage1_epochs = args.epochs
    stage2_epochs = 0
    
    trainer = Trainer(
        model=lrr_fno,
        loss_fn=lrr_loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        stage1_epochs=stage1_epochs, # Hybrid
        stage2_epochs=stage2_epochs, # Distillation (Disabled)
        stage1_lr=1e-3,
        stage2_lr=0.0, # Not used
        weight_decay=1e-4, # Regularization to reduce overfitting
        checkpoint_dir='checkpoints_darcy_lrr'
    )
    
    trainer.train()
    
    # Evaluate LRR-FNO
    lrr_fno.eval()
    lrr_errors = []
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                a, u = batch['x'], batch['y']
            else:
                a, u = batch
                
            a, u = a.to(device), u.to(device)
            output = lrr_fno(a)
            pred = output['prediction']
            
            lrr_errors.append(compute_relative_l2(pred, u))
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
            a, u = batch['x'], batch['y']
        else:
            a, u = batch
            
        a, u = a.to(device), u.to(device)
        fno_pred = fno(a)
        if fno_pred.shape != u.shape:
             if fno_pred.shape[1] == 1 and u.dim() == 3: fno_pred = fno_pred.squeeze(1)
        
        lrr_output = lrr_fno(a)
        lrr_pred = lrr_output['prediction']
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Show first sample in batch
    for i, (label, data) in enumerate([('Ground Truth u(x)', u[0].cpu()),
                                        ('FNO Prediction', fno_pred[0].cpu()),
                                        ('LRR-FNO Prediction', lrr_pred[0, 0].cpu())]):
        # Handle channel dim
        if data.ndim == 3: data = data[0]
        
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
