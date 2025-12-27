"""
Darcy Flow Demo for LRN-FNO (2D)

Demonstrates the Latent Reciprocity Network on the 2D Darcy Flow equation:
    -∇·(a(x) ∇u(x)) = f(x)

This example shows:
1. Creating an LRN-FNO 2D model
2. Training with 3-stage curriculum
3. Comparing LRN-FNO vs vanilla FNO
4. Visualizing 2D predictions and errors

Usage:
    python examples/darcy_demo.py --mode compare
"""

import os
# Fix for OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LRNFNO2d, FNO2d
from src.losses import LRNLoss
from src.data import DarcyDataset
from src.utils import LRNTrainer, get_device, count_parameters


def visualize_predictions_2d(model_fno, model_lrn, dataset, device='cpu', filename='darcy_comparison.png'):
    """Visualize 2D predictions: Input, Truth, FNO, LRN, and Error maps."""
    model_fno.eval()
    model_lrn.eval()
    
    # Get one sample
    idx = 0
    a, u = dataset[idx]
    a = a.unsqueeze(0).to(device)  # [1, H, W]
    u = u.to(device)               # [H, W]
    
    with torch.no_grad():
        # FNO Prediction
        pred_fno = model_fno(a).squeeze()
        
        # LRN Prediction
        pred_lrn = model_lrn(a, return_latents=False)['prediction'].squeeze()
        
    # Move to CPU for plotting
    a_np = a.squeeze().cpu().numpy()
    u_np = u.cpu().numpy()
    pred_fno_np = pred_fno.cpu().numpy()
    pred_lrn_np = pred_lrn.cpu().numpy()
    
    # Compute Absolute Error
    err_fno = np.abs(u_np - pred_fno_np)
    err_lrn = np.abs(u_np - pred_lrn_np)
    
    # Error Metrics
    l2_fno = np.linalg.norm(u_np - pred_fno_np) / np.linalg.norm(u_np)
    l2_lrn = np.linalg.norm(u_np - pred_lrn_np) / np.linalg.norm(u_np)
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Row 1: Fields
    im0 = axes[0, 0].imshow(a_np, cmap='viridis')
    axes[0, 0].set_title('Input Coefficient a(x)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(u_np, cmap='RdBu_r')
    axes[0, 1].set_title('Ground Truth u(x)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Empty for layout balance or maybe latent visualization later
    axes[0, 2].axis('off')
    
    # Row 2: Predictions and Errors
    im3 = axes[1, 0].imshow(pred_fno_np, cmap='RdBu_r')
    axes[1, 0].set_title(f'Vanilla FNO\nRel L2: {l2_fno:.4f}')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(pred_lrn_np, cmap='RdBu_r')
    axes[1, 1].set_title(f'LRN-FNO\nRel L2: {l2_lrn:.4f}')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Comparison of Error Maps
    vmin, vmax = 0, max(err_fno.max(), err_lrn.max())
    
    # Show difference? Or just errors? Let's show FNO Error
    im5 = axes[0, 2].imshow(err_fno, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[0, 2].set_title('Error: Vanilla FNO')
    axes[0, 2].axis('on')
    plt.colorbar(im5, ax=axes[0, 2])
    
    im6 = axes[1, 2].imshow(err_lrn, cmap='inferno', vmin=vmin, vmax=vmax)
    axes[1, 2].set_title('Error: LRN-FNO')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Comparison plot saved to {filename}")


def compare_darcy():
    print("="*60)
    print("LRN-FNO 2D Darcy Flow Comparison")
    print("="*60)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Generate/Load Data
    print("\nPreparing Darcy Dataset...")
    # Using lower resolution for faster demo training
    RESOLUTION = 32
    N_TRAIN = 400
    N_TEST = 100
    
    train_dataset = DarcyDataset(resolution=RESOLUTION, num_samples=N_TRAIN, train=True)
    test_dataset = DarcyDataset(resolution=RESOLUTION, num_samples=N_TEST, train=False)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 2. Train Vanilla FNO 2D
    print("\n--- Training Vanilla FNO 2D ---")
    fno = FNO2d(
        in_channels=1,
        out_channels=1,
        modes1=12,
        modes2=12,
        width=32,
        num_layers=4
    ).to(device)
    
    optimizer = torch.optim.Adam(fno.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    mse_loss = nn.MSELoss()
    
    print("Training FNO for 50 epochs...")
    fno_losses = []
    
    for epoch in range(50):
        fno.train()
        epoch_loss = 0
        for a, u in train_loader:
            a, u = a.to(device), u.to(device)
            optimizer.zero_grad()
            pred = fno(a)
            # Fix output dims if needed (remove channel if 1)
            if pred.shape[-1] == 1: pred = pred.squeeze(-1)
            
            loss = mse_loss(pred, u)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        fno_losses.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {avg_loss:.6f}")
            
    # 3. Train LRN-FNO 2D
    print("\n--- Training LRN-FNO 2D ---")
    lrn = LRNFNO2d(
        in_channels=1,
        out_channels=1,
        modes1=12,
        modes2=12,
        width=32,
        num_layers=4,
        latent_dim=64,
        encoder_channels=[16, 32, 64]
    ).to(device)
    
    loss_fn = LRNLoss(lambda_mse=1.0)
    
    print("Training LRN-FNO (Curriculum: 10 + 25 + 15 = 50 epochs)...")
    trainer = LRNTrainer(
        model=lrn,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        stage1_epochs=10,
        stage2_epochs=25,
        stage3_epochs=15,
        device=str(device),
        checkpoint_dir='darcy_checkpoints'
    )
    lrn_history = trainer.train()
    
    # 4. Evaluation
    print("\n--- Final Evaluation (Relative L2 Error) ---")
    fno.eval()
    lrn.eval()
    
    l2_fno_list = []
    l2_lrn_list = []
    
    with torch.no_grad():
        for a, u in test_loader:
            a, u = a.to(device), u.to(device)
            
            # FNO
            pred_fno = fno(a)
            if pred_fno.shape[-1] == 1: pred_fno = pred_fno.squeeze(-1)
            
            # LRN
            output = lrn(a, return_latents=False)
            pred_lrn = output['prediction']
            if pred_lrn.shape[-1] == 1: pred_lrn = pred_lrn.squeeze(-1)
            
            # Calculate L2 relative error batch-wise
            # u shape: [B, H, W]
            # Flatten spatial dims for norm: [B, H*W]
            u_flat = u.view(u.shape[0], -1)
            fno_flat = pred_fno.view(u.shape[0], -1)
            lrn_flat = pred_lrn.view(u.shape[0], -1)
            
            diff_fno = torch.norm(fno_flat - u_flat, p=2, dim=1)
            diff_lrn = torch.norm(lrn_flat - u_flat, p=2, dim=1)
            norm_u = torch.norm(u_flat, p=2, dim=1)
            
            l2_fno_list.extend((diff_fno / norm_u).cpu().numpy())
            l2_lrn_list.extend((diff_lrn / norm_u).cpu().numpy())
            
    mean_l2_fno = np.mean(l2_fno_list)
    mean_l2_lrn = np.mean(l2_lrn_list)
    
    print(f"\nFNO Test Rel L2:     {mean_l2_fno:.6f}")
    print(f"LRN-FNO Test Rel L2: {mean_l2_lrn:.6f}")
    print(f"Improvement:         {(mean_l2_fno - mean_l2_lrn) / mean_l2_fno * 100:.2f}%")
    
    # 5. Visualization
    print("\nGenerating comparison plots...")
    visualize_predictions_2d(fno, lrn, test_dataset, device=device)
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(fno_losses, label='Vanilla FNO')
    valid_mse = [x if x > 0 else np.nan for x in lrn_history['mse_loss']]
    plt.plot(valid_mse, label='LRN-FNO')
    plt.yscale('log')
    plt.title('Training Loss Comparison (Darcy)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('darcy_loss.png')
    print("Loss plot saved to darcy_loss.png")


if __name__ == '__main__':
    compare_darcy()
