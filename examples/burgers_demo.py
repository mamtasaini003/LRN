"""
Burgers Equation Demo for LRN-FNO

Demonstrates the Latent Reciprocity Network on the 1D Burgers equation:
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²

This example shows:
1. Creating an LRN-FNO model
2. Training with 3-stage curriculum
3. Comparing LRN-FNO vs vanilla FNO
4. Visualizing predictions and latent space

Usage:
    python examples/burgers_demo.py
"""

import os
# Fix for OpenMP library conflict on Windows/Anaconda
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import LRNFNO1d, FNO1d
from src.losses import LRNLoss, InfoNCELoss
from src.data import BurgersDataset
from src.utils import LRNTrainer, get_device, count_parameters


def visualize_predictions(model, dataset, n_samples=4, device='cpu'):
    """Visualize model predictions vs ground truth."""
    model.eval()
    
    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3*n_samples))
    
    for i in range(n_samples):
        f, u = dataset[i]
        f = f.unsqueeze(0).to(device)
        u = u.numpy()
        
        with torch.no_grad():
            output = model(f, return_latents=False)
            if isinstance(output, dict):
                pred = output['prediction'].cpu().numpy().squeeze()
            else:
                pred = output.cpu().numpy().squeeze()
        
        f_np = f.cpu().numpy().squeeze()
        
        # Input field
        axes[i, 0].plot(f_np, 'b-', linewidth=2, label='Input f(x)')
        axes[i, 0].set_title(f'Sample {i+1}: Initial Condition')
        axes[i, 0].set_xlabel('x')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Prediction vs Ground Truth
        axes[i, 1].plot(u, 'g-', linewidth=2, label='Ground Truth')
        axes[i, 1].plot(pred, 'r--', linewidth=2, label='LRN-FNO Prediction')
        axes[i, 1].set_title(f'Sample {i+1}: Solution')
        axes[i, 1].set_xlabel('x')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('burgers_predictions.png', dpi=150)
    plt.show()


def visualize_latent_space(model, dataset, n_samples=100, device='cpu'):
    """Visualize the learned latent space."""
    model.eval()
    
    z_f_list = []
    z_u_list = []
    
    for i in range(min(n_samples, len(dataset))):
        f, u = dataset[i]
        f = f.unsqueeze(0).to(device)
        u = u.unsqueeze(0).to(device)
        
        with torch.no_grad():
            z_f = model.encoder_f(f)
            if hasattr(model, 'encoder_u') and model.encoder_u is not None:
                z_u = model.encoder_u(u)
                z_u_list.append(z_u.cpu().numpy())
            z_f_list.append(z_f.cpu().numpy())
    
    z_f = np.concatenate(z_f_list, axis=0)
    
    # 2D visualization using first 2 principal components
    from sklearn.decomposition import PCA
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    pca = PCA(n_components=2)
    z_f_2d = pca.fit_transform(z_f)
    
    scatter = ax.scatter(z_f_2d[:, 0], z_f_2d[:, 1], c=range(len(z_f_2d)), 
                        cmap='viridis', alpha=0.7)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_title('Latent Space Visualization (z_f)')
    plt.colorbar(scatter, label='Sample Index')
    
    plt.tight_layout()
    plt.savefig('latent_space.png', dpi=150)
    plt.show()


def quick_demo():
    """Run a quick demonstration of LRN-FNO."""
    print("="*60)
    print("LRN-FNO Burgers Equation Demo")
    print("="*60)
    
    # Configuration
    device = get_device()
    print(f"Device: {device}")
    
    # Generate synthetic data
    print("\nGenerating synthetic Burgers data...")
    train_dataset = BurgersDataset(
        resolution=64,
        num_samples=500,
        train=True,
    )
    test_dataset = BurgersDataset(
        resolution=64,
        num_samples=500,
        train=False,
    )
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create LRN-FNO model
    print("\nCreating LRN-FNO model...")
    model = LRNFNO1d(
        in_channels=1,
        out_channels=1,
        modes=16,
        width=32,
        num_layers=4,
        latent_dim=32,
        encoder_channels=[16, 32, 64],
    ).to(device)
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Quick training demo (reduced epochs)
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    loss_fn = LRNLoss(lambda_mse=1.0, temperature=0.1)
    
    print("\nQuick training demo (5 epochs per stage)...")
    trainer = LRNTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        stage1_epochs=5,
        stage2_epochs=5,
        stage3_epochs=5,
        device=str(device),
        checkpoint_dir='demo_checkpoints',
    )
    
    history = trainer.train()
    
    # Visualize
    print("\nGenerating visualizations...")
    try:
        visualize_predictions(model, test_dataset, n_samples=3, device=device)
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    print("\n" + "="*60)
    print("Demo complete!")
    print(f"Final test MSE: {history['mse_loss'][-1]:.6f}")
    print("="*60)


def compare_fno_vs_lrn():
    """Compare vanilla FNO vs LRN-FNO."""
    print("="*60)
    print("Comparing FNO vs LRN-FNO")
    print("="*60)
    
    device = get_device()
    
    # Data
    train_dataset = BurgersDataset(resolution=64, num_samples=400, train=True)
    test_dataset = BurgersDataset(resolution=64, num_samples=400, train=False)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Train vanilla FNO
    print("\n--- Training Vanilla FNO ---")
    fno = FNO1d(
        in_channels=1,
        out_channels=1,
        modes=16,
        width=32,
        num_layers=4,
    ).to(device)
    
    optimizer_fno = torch.optim.Adam(fno.parameters(), lr=1e-3)
    scheduler_fno = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_fno, T_max=100)
    mse_loss = nn.MSELoss()
    
    fno_losses = []
    print("Training FNO for 100 epochs...")
    for epoch in range(100):
        fno.train()
        epoch_loss = 0
        for f, u in train_loader:
            f, u = f.to(device), u.to(device)
            optimizer_fno.zero_grad()
            pred = fno(f)
            if pred.dim() == 3:
                pred = pred.squeeze(-1)
            loss = mse_loss(pred, u)
            loss.backward()
            optimizer_fno.step()
            epoch_loss += loss.item()
        
        scheduler_fno.step()
        fno_losses.append(epoch_loss / len(train_loader))
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {fno_losses[-1]:.6f}")
    
    # Train LRN-FNO
    print("\n--- Training LRN-FNO ---")
    lrn = LRNFNO1d(
        in_channels=1,
        out_channels=1,
        modes=16,
        width=32,
        num_layers=4,
        latent_dim=32,
        encoder_channels=[16, 32, 64],
    ).to(device)
    
    loss_fn = LRNLoss(lambda_mse=1.0, temperature=0.1)
    
    print("Training LRN-FNO for 100 epochs (20+50+30)...")
    trainer = LRNTrainer(
        model=lrn,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        stage1_epochs=20,
        stage2_epochs=50,
        stage3_epochs=30,
        device=str(device),
        checkpoint_dir='comparison_checkpoints',
    )
    lrn_history = trainer.train()
    
    # Evaluate both with Relative L2 Error
    print("\n--- Final Evaluation (Relative L2 Error) ---")
    fno.eval()
    lrn.eval()
    
    fno_l2_errors = []
    lrn_l2_errors = []
    
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            
            # FNO prediction
            pred_fno = fno(f)
            if pred_fno.dim() == 3:
                pred_fno = pred_fno.squeeze(-1)
            
            # LRN-FNO prediction
            output_lrn = lrn(f, return_latents=False)
            pred_lrn = output_lrn['prediction']
            if pred_lrn.dim() == 3:
                pred_lrn = pred_lrn.squeeze(-1)
            
            # Compute Relative L2 per sample in batch
            # Norm over spatial dim (last dim)
            diff_fno = torch.norm(pred_fno - u, p=2, dim=-1)
            norm_u = torch.norm(u, p=2, dim=-1)
            fno_l2_errors.append((diff_fno / norm_u).cpu().numpy())
            
            diff_lrn = torch.norm(pred_lrn - u, p=2, dim=-1)
            lrn_l2_errors.append((diff_lrn / norm_u).cpu().numpy())
    
    fno_test_l2 = np.concatenate(fno_l2_errors).mean()
    lrn_test_l2 = np.concatenate(lrn_l2_errors).mean()
    
    print(f"\nFNO Test Rel L2:     {fno_test_l2:.6f}")
    print(f"LRN-FNO Test Rel L2: {lrn_test_l2:.6f}")
    print(f"Improvement:         {(fno_test_l2 - lrn_test_l2) / fno_test_l2 * 100:.2f}%")
    
    # --- Visualization ---
    print("\nGenerating comparison plots...")
    
    # 1. Plot Training History
    plt.figure(figsize=(10, 5))
    plt.plot(fno_losses, label='Vanilla FNO', marker='')
    lrn_mse = lrn_history['mse_loss']
    # Filter zeros from Stage 1 (Manifold Alignment)
    # Stage 1 has 20 epochs where MSE is 0
    valid_mse = [x if x > 0 else np.nan for x in lrn_mse]
    
    plt.plot(valid_mse, label='LRN-FNO', alpha=0.9, linewidth=2)
    
    plt.title('Training Loss Comparison (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('comparison_loss.png')
    
    # 2. Plot Prediction Comparison
    model_fno = fno
    model_lrn = lrn
    model_fno.eval()
    model_lrn.eval()
    
    # Get a batch
    f, u = next(iter(test_loader))
    f, u = f.to(device), u.to(device)
    
    with torch.no_grad():
        pred_fno = model_fno(f).squeeze()
        pred_lrn = model_lrn(f, return_latents=False)['prediction'].squeeze()
    
    # Plot first 3 samples with overlapping predictions
    n_samples = 3
    fig, axes = plt.subplots(n_samples, 1, figsize=(10, 5*n_samples))
    if n_samples == 1: axes = [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        
        # Ground Truth
        ax.plot(u[i].cpu().numpy(), 'k-', label='Ground Truth', linewidth=2.5, alpha=0.5)
        
        # Calculate Rel L2 for title
        norm_u_i = torch.norm(u[i], p=2).item()
        
        # FNO
        fno_l2 = torch.norm(pred_fno[i] - u[i], p=2).item() / norm_u_i
        ax.plot(pred_fno[i].cpu().numpy(), 'b--', label=f'Vanilla FNO (L2: {fno_l2:.4f})', linewidth=1.5)
        
        # LRN
        lrn_l2 = torch.norm(pred_lrn[i] - u[i], p=2).item() / norm_u_i
        ax.plot(pred_lrn[i].cpu().numpy(), 'r-.', label=f'LRN-FNO (L2: {lrn_l2:.4f})', linewidth=1.5)
        
        ax.set_title(f'Sample {i+1}: Prediction Comparison')
        ax.set_xlabel('Spatial Coordinate')
        ax.set_ylabel('u(x)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('comparison_predictions.png')
    print("Plots saved: comparison_loss.png, comparison_predictions.png")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='demo',
                        choices=['demo', 'compare'],
                        help='Demo mode or comparison mode')
    args = parser.parse_args()
    
    if args.mode == 'demo':
        quick_demo()
    else:
        compare_fno_vs_lrn()
