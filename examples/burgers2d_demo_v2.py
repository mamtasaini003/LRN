"""
2D Coupled Burgers Equation Demo for LRN-FNO - VERSION 2 (2-Stage Training)

This version uses a simplified 2-stage training protocol:
    - Stage 1: Combined Optimization (NCE + MSE)
    - Stage 2: Autonomous Distillation (MSE only)

Usage:
    python examples/burgers2d_demo_v2.py
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from src.models import LRNFNO2d, FNO2d
from src.losses import LRNLoss
from src.data import Burgers2dDataset
from src.utils import Trainer, get_device


def visualize_burgers2d(model_fno, model_lrn, dataset, device='cpu', filename='burgers2d_comparison_v2.png'):
    """Visualize u and v fields."""
    model_fno.eval()
    model_lrn.eval()
    
    # Get sample
    idx = 0
    f, u_gt = dataset[idx]
    
    f = f.unsqueeze(0).to(device)   # [1, 2, H, W]
    u_gt = u_gt.to(device)          # [2, H, W]
    
    with torch.no_grad():
        pred_fno = model_fno(f).squeeze(0)          # [2, H, W]
        output_lrn = model_lrn(f, return_latents=False)
        pred_lrn = output_lrn['prediction'].squeeze(0)
    
    # Calculate Metrics
    norm_gt = torch.norm(u_gt.reshape(-1)).item()
    l2_fno = torch.norm((pred_fno - u_gt).reshape(-1)).item() / norm_gt
    l2_lrn = torch.norm((pred_lrn - u_gt).reshape(-1)).item() / norm_gt
    
    # Convert to numpy
    u_0 = f[0, 0].cpu().numpy()
    v_0 = f[0, 1].cpu().numpy()
    
    u_gt_np = u_gt[0].cpu().numpy()
    v_gt_np = u_gt[1].cpu().numpy()
    
    u_fno = pred_fno[0].cpu().numpy()
    v_fno = pred_fno[1].cpu().numpy()
    
    u_lrn = pred_lrn[0].cpu().numpy()
    v_lrn = pred_lrn[1].cpu().numpy()
    
    # Plot u-component
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Row 1: Ground Truth
    im0 = axes[0, 0].imshow(u_0, cmap='viridis')
    axes[0, 0].set_title('Forcing f_x')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(u_gt_np, cmap='magma')
    axes[0, 1].set_title('Truth u_steady')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(v_gt_np, cmap='magma')
    axes[0, 2].set_title('Truth v_steady')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: FNO
    im3 = axes[1, 0].imshow(u_fno, cmap='magma')
    axes[1, 0].set_title(f'FNO u (Rel L2: {l2_fno:.4f})')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(v_fno, cmap='magma')
    axes[1, 1].set_title(f'FNO v')
    plt.colorbar(im4, ax=axes[1, 1])
    
    err_fno = np.abs(u_gt_np - u_fno)
    im5 = axes[1, 2].imshow(err_fno, cmap='inferno')
    axes[1, 2].set_title('Error |u_steady - u_FNO|')
    plt.colorbar(im5, ax=axes[1, 2])
    
    # Row 3: LRN
    im6 = axes[2, 0].imshow(u_lrn, cmap='magma')
    axes[2, 0].set_title(f'LRN u (Rel L2: {l2_lrn:.4f})')
    plt.colorbar(im6, ax=axes[2, 0])
    
    im7 = axes[2, 1].imshow(v_lrn, cmap='magma')
    axes[2, 1].set_title(f'LRN v')
    plt.colorbar(im7, ax=axes[2, 1])
    
    err_lrn = np.abs(u_gt_np - u_lrn)
    im8 = axes[2, 2].imshow(err_lrn, cmap='inferno')
    axes[2, 2].set_title('Error |u_steady - u_LRN|')
    plt.colorbar(im8, ax=axes[2, 2])
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def compare_burgers2d_v2():
    print("="*60)
    print("LRN-FNO 2D Burgers Comparison - VERSION 2 (2-Stage Training)")
    print("="*60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Ensure results directory exists
    os.makedirs('results/plots', exist_ok=True)
    
    device = get_device()
    print(f"Device: {device}")
    
    # 1. Data
    print("\nPreparing 2D Burgers Dataset...")
    train_dataset = Burgers2dDataset(resolution=64, num_samples=300, train=True)
    test_dataset = Burgers2dDataset(resolution=64, num_samples=100, train=False)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 2. Train FNO
    print("\n--- Training Vanilla FNO ---")
    fno = FNO2d(
        in_channels=2,
        out_channels=2,
        modes1=12,
        modes2=12,
        width=32,
        num_layers=4
    ).to(device)
    
    optimizer = torch.optim.Adam(fno.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
    mse_loss = nn.MSELoss()
    
    print("Training FNO for 150 epochs...")
    fno_losses = []
    
    for epoch in range(150):
        fno.train()
        epoch_loss = 0
        for f, u in train_loader:
            f, u = f.to(device), u.to(device)
            optimizer.zero_grad()
            pred = fno(f)
            loss = mse_loss(pred, u)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        fno_losses.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/150, Loss: {avg_loss:.6f}")
            
    # 3. Train LRN with V2 (2-stage)
    print("\n--- Training LRN-FNO (V2: 2-Stage Protocol) ---")
    lrn = LRNFNO2d(
        in_channels=2,
        out_channels=2,
        modes1=12,
        modes2=12,
        width=32,
        num_layers=4,
        latent_dim=64,
        encoder_channels=[32, 64, 128],
        use_gated_bridge=True
    ).to(device)
    
    # Restore original successful hyperparameters
    loss_fn = LRNLoss(lambda_mse=10000.0, lambda_nce=0.01, use_relative_mse=False)
    
    # V2: 2-stage training (110 + 40 = 150 epochs)
    # Stage 1: NCE + MSE combined (110 epochs)
    # Stage 2: MSE only fine-tuning (40 epochs)
    print("Training LRN-FNO V2 (2-Stage: 110 + 40 = 150 epochs)...")
    trainer = Trainer(
        model=lrn,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        stage1_epochs=110,  # NCE + MSE
        stage2_epochs=40,   # MSE only
        stage1_lr=1e-3,
        stage2_lr=5e-4,     # Slower fine-tuning
        device=str(device),
        checkpoint_dir='checkpoints/burgers2d_v2_checkpoints'
    )
    lrn_history = trainer.train()
    
    # 4. Evaluate
    print("\n--- Final Evaluation ---")
    fno.eval()
    lrn.eval()
    
    l2_fno_list = []
    l2_lrn_list = []
    
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            
            # FNO
            pred_fno = fno(f)
            
            # LRN
            output = lrn(f, return_latents=False)
            pred_lrn = output['prediction']
            
            # Rel L2
            u_flat = u.reshape(u.shape[0], -1)
            fno_flat = pred_fno.reshape(u.shape[0], -1)
            lrn_flat = pred_lrn.reshape(u.shape[0], -1)
            
            diff_fno = torch.norm(fno_flat - u_flat, p=2, dim=1)
            norm_u = torch.norm(u_flat, p=2, dim=1)
            diff_lrn = torch.norm(lrn_flat - u_flat, p=2, dim=1)
            
            l2_fno_list.extend((diff_fno / norm_u).cpu().numpy())
            l2_lrn_list.extend((diff_lrn / norm_u).cpu().numpy())
            
    mean_l2_fno = np.mean(l2_fno_list)
    mean_l2_lrn = np.mean(l2_lrn_list)
    
    print(f"\nFNO Test Rel L2:     {mean_l2_fno:.6f}")
    print(f"LRN-FNO Test Rel L2: {mean_l2_lrn:.6f}")
    improvement = (mean_l2_fno - mean_l2_lrn) / mean_l2_fno * 100
    print(f"Improvement:         {improvement:.2f}%")
    
    # 5. Visualize
    visualize_burgers2d(fno, lrn, test_dataset, device=device, filename='results/plots/burgers2d_comparison_v2.png')
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(fno_losses, label='Vanilla FNO', alpha=0.8)
    valid_mse = [x if x > 0 else np.nan for x in lrn_history['mse_loss']]
    plt.plot(valid_mse, label='LRN-FNO V2', alpha=0.8)
    plt.yscale('log')
    plt.title('Training Loss Comparison (2D Burgers - V2)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/plots/burgers2d_loss_v2.png')
    print("Loss plot saved to results/plots/burgers2d_loss_v2.png")
    
    return {
        'fno_error': mean_l2_fno,
        'lrn_error': mean_l2_lrn,
        'improvement': improvement
    }


if __name__ == '__main__':
    compare_burgers2d_v2()
