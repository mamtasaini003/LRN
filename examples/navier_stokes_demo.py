"""
Navier-Stokes Demo for LRN-FNO

Demonstrates the Latent Reciprocity Network on the 2D Navier-Stokes equation for a viscous, incompressible fluid:
    ∂w/∂t + u·∇w = νΔw + f; ∇·u = 0

Task: Predict the vorticity field w(x,t) at time steps [T, T+10] given [0, T].
This demo simplifies to mapping the initial 10 steps to the next 10 steps.

Usage:
    python examples/navier_stokes_demo.py
"""

import os
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
from src.data import NavierStokesDataset
from src.utils import LRNTrainer, get_device


def visualize_ns_predictions(model_fno, model_lrn, dataset, device='cpu', filename='ns_comparison.png'):
    """Visualize NS predictions at a specific time step."""
    model_fno.eval()
    model_lrn.eval()
    
    idx = 0
    u_in, u_out = dataset[idx]
    
    # u_in: [10, H, W] -> Model expects [1, 10, H, W] -> [1, H, W, 10] permute internally
    # But FNO2d expects [B, C, H, W]. Let's treat time steps as channels.
    
    u_in = u_in.unsqueeze(0).to(device)  # [1, 10, H, W]
    u_out_gt = u_out.to(device)          # [10, H, W]
    
    with torch.no_grad():
        pred_fno = model_fno(u_in).squeeze(0)  # [10, H, W]
        output_lrn = model_lrn(u_in, return_latents=False)
        pred_lrn = output_lrn['prediction'].squeeze(0)
    
    # Select last time step for visualization
    t_idx = -1
    
    gt_np = u_out_gt[t_idx].cpu().numpy()
    fno_np = pred_fno[t_idx].cpu().numpy()
    lrn_np = pred_lrn[t_idx].cpu().numpy()
    
    err_fno = np.abs(gt_np - fno_np)
    err_lrn = np.abs(gt_np - lrn_np)
    
    # Calculate Rel L2 over full sequence
    norm_gt = torch.norm(u_out_gt.reshape(-1)).item()
    l2_fno = torch.norm((pred_fno - u_out_gt).reshape(-1)).item() / norm_gt
    l2_lrn = torch.norm((pred_lrn - u_out_gt).reshape(-1)).item() / norm_gt
    
    # Plotting
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    
    # Input (Last step of input)
    im0 = axes[0, 0].imshow(u_in[0, -1].cpu().numpy(), cmap='jet')
    axes[0, 0].set_title('Input: w(x, t=10)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Truth
    im1 = axes[0, 1].imshow(gt_np, cmap='jet')
    axes[0, 1].set_title('Truth: w(x, t=20)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    axes[0, 2].axis('off')
    
    # Predictions
    im3 = axes[1, 0].imshow(fno_np, cmap='jet')
    axes[1, 0].set_title(f'Vanilla FNO\nRel L2 (Seq): {l2_fno:.4f}')
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(lrn_np, cmap='jet')
    axes[1, 1].set_title(f'LRN-FNO\nRel L2 (Seq): {l2_lrn:.4f}')
    plt.colorbar(im4, ax=axes[1, 1])
    
    # Errors
    vmax_err = max(err_fno.max(), err_lrn.max())
    
    im5 = axes[0, 2].imshow(err_fno, cmap='inferno', vmin=0, vmax=vmax_err)
    axes[0, 2].set_title('Error: Vanilla FNO')
    axes[0, 2].axis('on')
    plt.colorbar(im5, ax=axes[0, 2])
    
    im6 = axes[1, 2].imshow(err_lrn, cmap='inferno', vmin=0, vmax=vmax_err)
    axes[1, 2].set_title('Error: LRN-FNO')
    plt.colorbar(im6, ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")


def compare_navier_stokes():
    print("="*60)
    print("LRN-FNO Navier-Stokes Comparison")
    print("="*60)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Settings
    RES = 64
    T_IN = 10
    T_OUT = 10
    N_TRAIN = 200  # Smaller for demo speed
    N_TEST = 50
    
    print("\nPreparing Navier-Stokes Dataset...")
    train_dataset = NavierStokesDataset(
        resolution=RES, 
        num_samples=N_TRAIN, 
        input_steps=T_IN, 
        output_steps=T_OUT, 
        train=True
    )
    test_dataset = NavierStokesDataset(
        resolution=RES, 
        num_samples=N_TEST, 
        input_steps=T_IN, 
        output_steps=T_OUT, 
        train=False
    )
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Channel dimension is time steps here: FNO maps 10 channels -> 10 channels
    
    # 2. Train FNO
    print("\n--- Training Vanilla FNO ---")
    fno = FNO2d(
        in_channels=T_IN,
        out_channels=T_OUT,
        modes1=12,
        modes2=12,
        width=32,
        num_layers=4
    ).to(device)
    
    optimizer = torch.optim.Adam(fno.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    mse_loss = nn.MSELoss()
    
    print("Training FNO for 100 epochs...")
    fno_losses = []
    
    for epoch in range(100):
        fno.train()
        epoch_loss = 0
        for u_in, u_out in train_loader:
            u_in, u_out = u_in.to(device), u_out.to(device)
            
            optimizer.zero_grad()
            # FNO output: [B, Out, H, W] (already correct shape)
            pred = fno(u_in) 
            
            loss = mse_loss(pred, u_out)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        fno_losses.append(avg_loss)
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {avg_loss:.6f}")
            
    # 3. Train LRN
    print("\n--- Training LRN-FNO ---")
    lrn = LRNFNO2d(
        in_channels=T_IN,
        out_channels=T_OUT,
        modes1=12,
        modes2=12,
        width=32,
        num_layers=4,
        latent_dim=64,
        encoder_channels=[32, 64, 128]
    ).to(device)
    
    loss_fn = LRNLoss(lambda_mse=1.0)
    
    print("Training LRN-FNO (Curriculum: 20 + 50 + 30 = 100 epochs)...")
    trainer = LRNTrainer(
        model=lrn,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        stage1_epochs=20,
        stage2_epochs=50,
        stage3_epochs=30,
        device=str(device),
        checkpoint_dir='ns_checkpoints'
    )
    lrn_history = trainer.train()
    
    # 3. Evaluation
    print("\n--- Final Evaluation (Relative L2) ---")
    fno.eval()
    lrn.eval()
    
    l2_fno_list = []
    l2_lrn_list = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            
            # FNO Prediction
            pred_fno = fno(x)
            
            # LRN Prediction
            out_lrn = lrn(x, return_latents=False)
            pred_lrn = out_lrn['prediction']
            
            # Compute Rel L2 over entire batch & time sequence
            # Reshape to [B, -1] for norm calculation
            y_flat = y.reshape(y.shape[0], -1)
            fno_flat = pred_fno.reshape(y.shape[0], -1)
            lrn_flat = pred_lrn.reshape(y.shape[0], -1)
            
            diff_fno = torch.norm(fno_flat - y_flat, p=2, dim=1)
            diff_lrn = torch.norm(lrn_flat - y_flat, p=2, dim=1)
            norm_y = torch.norm(y_flat, p=2, dim=1)
            
            l2_fno_list.extend((diff_fno / norm_y).cpu().numpy())
            l2_lrn_list.extend((diff_lrn / norm_y).cpu().numpy())
            
    mean_l2_fno = np.mean(l2_fno_list)
    mean_l2_lrn = np.mean(l2_lrn_list)
    
    print(f"\nFNO Test Rel L2:     {mean_l2_fno:.6f}")
    print(f"LRN-FNO Test Rel L2: {mean_l2_lrn:.6f}")
    print(f"Improvement:         {(mean_l2_fno - mean_l2_lrn) / mean_l2_fno * 100:.2f}%")
    
    # 4. Visualization
    print("\nGenerating visualizations...")
    visualize_ns_predictions(fno, lrn, test_dataset, device=device)
    
    # Loss plot
    plt.figure(figsize=(10, 5))
    plt.plot(fno_losses, label='Vanilla FNO')
    valid_mse = [x if x > 0 else np.nan for x in lrn_history['mse_loss']]
    plt.plot(valid_mse, label='LRN-FNO')
    plt.yscale('log')
    plt.title('Training Loss Comparison (Navier-Stokes)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('ns_loss.png')
    print("Loss plot saved to ns_loss.png")


if __name__ == '__main__':
    compare_navier_stokes()
