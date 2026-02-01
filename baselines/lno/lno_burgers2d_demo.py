"""
LNO vs LRN-FNO Comparison Demo - 2D Burgers Equation

This script provides a fair comparison between:
1. Vanilla FNO
2. LRN-FNO (our method)
3. LNO (NeurIPS 2024 baseline)

Following the same experimental setup as the LRN-FNO demos for fair comparison.

Paper Reference:
    "Latent Neural Operator for Solving Forward and Inverse PDE Problems"
    Wang & Wang, NeurIPS 2024
"""

import os
import argparse
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import random
import time

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from lno_model import LNO2d, create_lno_burgers2d


def set_seed(seed=42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def train_lno(model, train_loader, test_loader, device, epochs=150, lr=1e-3):
    """Train LNO model following paper configuration."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    # Using relative L2 loss as in the paper
    def relative_l2_loss(pred, target):
        diff = pred - target
        return torch.norm(diff.reshape(diff.shape[0], -1), dim=1).mean() / \
               torch.norm(target.reshape(target.shape[0], -1), dim=1).mean()
    
    model.train()
    losses = []
    best_test_error = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for f, u in train_loader:
            f, u = f.to(device), u.to(device)
            
            optimizer.zero_grad()
            pred = model(f)
            loss = relative_l2_loss(pred, u)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            # Evaluate
            model.eval()
            test_errors = []
            with torch.no_grad():
                for f, u in test_loader:
                    f, u = f.to(device), u.to(device)
                    pred = model(f)
                    
                    u_flat = u.reshape(u.shape[0], -1)
                    pred_flat = pred.reshape(u.shape[0], -1)
                    
                    errors = torch.norm(pred_flat - u_flat, dim=1) / torch.norm(u_flat, dim=1)
                    test_errors.extend(errors.cpu().numpy())
            
            mean_error = np.mean(test_errors)
            if mean_error < best_test_error:
                best_test_error = mean_error
            
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Test RelL2: {mean_error:.6f}")
            model.train()
    
    return losses, best_test_error


def compare_burgers2d(epochs=150):
    """Compare FNO, LRN-FNO, and LNO on 2D Burgers equation."""
    print("=" * 70)
    print(f"LNO vs LRN-FNO Comparison - 2D Burgers Equation (Epochs: {epochs})")
    print("=" * 70)
    
    # Calculate stage epochs for LRN (approx 75% / 25%)
    stage1_epochs = int(epochs * 0.75)
    stage2_epochs = epochs - stage1_epochs
    if stage1_epochs < 1: stage1_epochs = 1
    if stage2_epochs < 1: stage2_epochs = 0
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Ensure results directory exists
    os.makedirs('../../results/plots', exist_ok=True)
    os.makedirs('lno_checkpoints', exist_ok=True)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Import LRN modules
    from src.models import LRNFNO2d, FNO2d
    from src.losses import LRNLoss
    from src.data import Burgers2dDataset
    from src.utils import LRNTrainerV2
    
    # 1. Prepare Data
    print("\nPreparing 2D Burgers Dataset...")
    train_dataset = Burgers2dDataset(resolution=64, num_samples=300, train=True)
    test_dataset = Burgers2dDataset(resolution=64, num_samples=100, train=False)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    results = {}
    
    # 2. Train Vanilla FNO
    print("\n" + "=" * 40)
    print("Training Vanilla FNO")
    print("=" * 40)
    
    set_seed(42)  # Reset seed for fair comparison
    
    fno = FNO2d(
        in_channels=2,
        out_channels=2,
        modes1=12,
        modes2=12,
        width=32,
        num_layers=4
    ).to(device)
    
    optimizer = torch.optim.Adam(fno.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    
    fno_losses = []
    start_time = time.time()
    
    for epoch in range(epochs):
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
        
        if (epoch + 1) % max(1, epochs//5) == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    fno_time = time.time() - start_time
    print(f"FNO Training Time: {fno_time:.1f}s")
    
    # Evaluate FNO
    fno.eval()
    fno_errors = []
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            pred = fno(f)
            u_flat = u.reshape(u.shape[0], -1)
            pred_flat = pred.reshape(u.shape[0], -1)
            errors = torch.norm(pred_flat - u_flat, dim=1) / torch.norm(u_flat, dim=1)
            fno_errors.extend(errors.cpu().numpy())
    
    results['fno'] = {
        'error': np.mean(fno_errors),
        'std': np.std(fno_errors),
        'time': fno_time,
        'params': sum(p.numel() for p in fno.parameters())
    }
    print(f"FNO Test RelL2: {results['fno']['error']:.6f} (±{results['fno']['std']:.6f})")
    
    # 3. Train LRN-FNO
    print("\n" + "=" * 40)
    print("Training LRN-FNO (2-Stage Protocol)")
    print("=" * 40)
    
    set_seed(42)  # Reset seed
    
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
    
    loss_fn = LRNLoss(lambda_mse=10000.0, lambda_nce=0.01, use_relative_mse=False)
    
    start_time = time.time()
    trainer = LRNTrainerV2(
        model=lrn,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        stage1_lr=1e-3,
        stage2_lr=5e-4,
        device=str(device),
        checkpoint_dir='lno_checkpoints/lrn_burgers2d'
    )
    lrn_history = trainer.train()
    lrn_time = time.time() - start_time
    print(f"LRN-FNO Training Time: {lrn_time:.1f}s")
    
    # Evaluate LRN-FNO
    lrn.eval()
    lrn_errors = []
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            output = lrn(f, return_latents=False)
            pred = output['prediction']
            u_flat = u.reshape(u.shape[0], -1)
            pred_flat = pred.reshape(u.shape[0], -1)
            errors = torch.norm(pred_flat - u_flat, dim=1) / torch.norm(u_flat, dim=1)
            lrn_errors.extend(errors.cpu().numpy())
    
    results['lrn'] = {
        'error': np.mean(lrn_errors),
        'std': np.std(lrn_errors),
        'time': lrn_time,
        'params': sum(p.numel() for p in lrn.parameters())
    }
    print(f"LRN-FNO Test RelL2: {results['lrn']['error']:.6f} (±{results['lrn']['std']:.6f})")
    
    # 4. Train LNO
    print("\n" + "=" * 40)
    print("Training LNO (NeurIPS 2024)")
    print("=" * 40)
    
    set_seed(42)  # Reset seed
    
    lno = LNO2d(
        in_channels=2,
        out_channels=2,
        embed_dim=128,
        latent_size=256,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    ).to(device)
    
    start_time = time.time()
    lno_losses, lno_best = train_lno(lno, train_loader, test_loader, device, epochs=epochs)
    lno_time = time.time() - start_time
    print(f"LNO Training Time: {lno_time:.1f}s")
    
    # Evaluate LNO
    lno.eval()
    lno_errors = []
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            pred = lno(f)
            u_flat = u.reshape(u.shape[0], -1)
            pred_flat = pred.reshape(u.shape[0], -1)
            errors = torch.norm(pred_flat - u_flat, dim=1) / torch.norm(u_flat, dim=1)
            lno_errors.extend(errors.cpu().numpy())
    
    results['lno'] = {
        'error': np.mean(lno_errors),
        'std': np.std(lno_errors),
        'time': lno_time,
        'params': sum(p.numel() for p in lno.parameters())
    }
    print(f"LNO Test RelL2: {results['lno']['error']:.6f} (±{results['lno']['std']:.6f})")
    
    # 5. Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY - 2D Burgers Equation")
    print("=" * 70)
    
    print(f"\n{'Model':<15} {'RelL2 Error':<15} {'Std Dev':<12} {'Time (s)':<12} {'Params':<12}")
    print("-" * 66)
    for name, res in results.items():
        print(f"{name.upper():<15} {res['error']:.6f}       {res['std']:.6f}     {res['time']:.1f}        {res['params']:,}")
    
    # Calculate improvements
    print("\n" + "-" * 66)
    print("Improvements over FNO baseline:")
    fno_err = results['fno']['error']
    for name in ['lrn', 'lno']:
        if name in results:
            improvement = (fno_err - results[name]['error']) / fno_err * 100
            print(f"  {name.upper()}: {improvement:+.2f}%")
    
    print("\nLRN-FNO vs LNO:")
    lrn_err = results['lrn']['error']
    lno_err = results['lno']['error']
    comparison = (lno_err - lrn_err) / lno_err * 100
    if comparison > 0:
        print(f"  LRN-FNO is {comparison:.2f}% better than LNO")
    else:
        print(f"  LNO is {-comparison:.2f}% better than LRN-FNO")
    
    # 6. Visualization
    visualize_comparison(fno, lrn, lno, test_dataset, device, results)
    
    # Save results
    save_results(results, 'burgers2d')
    
    return results


def visualize_comparison(fno, lrn, lno, dataset, device, results):
    """Create comparison visualization."""
    fno.eval()
    lrn.eval()
    lno.eval()
    
    idx = 0
    f, u_gt = dataset[idx]
    f = f.unsqueeze(0).to(device)
    u_gt = u_gt.to(device)
    
    with torch.no_grad():
        pred_fno = fno(f).squeeze(0)
        pred_lrn = lrn(f, return_latents=False)['prediction'].squeeze(0)
        pred_lno = lno(f).squeeze(0)
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    # Row 1: Ground Truth
    im0 = axes[0, 0].imshow(f[0, 0].cpu().numpy(), cmap='jet')
    axes[0, 0].set_title('Input u_0')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(u_gt[0].cpu().numpy(), cmap='jet')
    axes[0, 1].set_title('Ground Truth u_T')
    plt.colorbar(im1, ax=axes[0, 1])
    
    im2 = axes[0, 2].imshow(u_gt[1].cpu().numpy(), cmap='jet')
    axes[0, 2].set_title('Ground Truth v_T')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Row 2: FNO
    im3 = axes[1, 0].imshow(pred_fno[0].cpu().numpy(), cmap='jet')
    axes[1, 0].set_title(f"FNO u_T (RelL2: {results['fno']['error']:.4f})")
    plt.colorbar(im3, ax=axes[1, 0])
    
    im4 = axes[1, 1].imshow(pred_fno[1].cpu().numpy(), cmap='jet')
    axes[1, 1].set_title('FNO v_T')
    plt.colorbar(im4, ax=axes[1, 1])
    
    err_fno = np.abs(u_gt[0].cpu().numpy() - pred_fno[0].cpu().numpy())
    im5 = axes[1, 2].imshow(err_fno, cmap='inferno')
    axes[1, 2].set_title('|Error| FNO')
    plt.colorbar(im5, ax=axes[1, 2])
    
    # Row 3: LRN-FNO
    im6 = axes[2, 0].imshow(pred_lrn[0].cpu().numpy(), cmap='jet')
    axes[2, 0].set_title(f"LRN-FNO u_T (RelL2: {results['lrn']['error']:.4f})")
    plt.colorbar(im6, ax=axes[2, 0])
    
    im7 = axes[2, 1].imshow(pred_lrn[1].cpu().numpy(), cmap='jet')
    axes[2, 1].set_title('LRN-FNO v_T')
    plt.colorbar(im7, ax=axes[2, 1])
    
    err_lrn = np.abs(u_gt[0].cpu().numpy() - pred_lrn[0].cpu().numpy())
    im8 = axes[2, 2].imshow(err_lrn, cmap='inferno')
    axes[2, 2].set_title('|Error| LRN-FNO')
    plt.colorbar(im8, ax=axes[2, 2])
    
    # Row 4: LNO
    im9 = axes[3, 0].imshow(pred_lno[0].cpu().numpy(), cmap='jet')
    axes[3, 0].set_title(f"LNO u_T (RelL2: {results['lno']['error']:.4f})")
    plt.colorbar(im9, ax=axes[3, 0])
    
    im10 = axes[3, 1].imshow(pred_lno[1].cpu().numpy(), cmap='jet')
    axes[3, 1].set_title('LNO v_T')
    plt.colorbar(im10, ax=axes[3, 1])
    
    err_lno = np.abs(u_gt[0].cpu().numpy() - pred_lno[0].cpu().numpy())
    im11 = axes[3, 2].imshow(err_lno, cmap='inferno')
    axes[3, 2].set_title('|Error| LNO')
    plt.colorbar(im11, ax=axes[3, 2])
    
    plt.tight_layout()
    plt.savefig('../../results/plots/lno_vs_lrn_burgers2d.png', dpi=150)
    print("\nVisualization saved to results/plots/lno_vs_lrn_burgers2d.png")
    plt.close()


def save_results(results, task_name):
    """Save results to JSON file."""
    import json
    
    results_serializable = {}
    for name, res in results.items():
        results_serializable[name] = {
            'error': float(res['error']),
            'std': float(res['std']),
            'time': float(res['time']),
            'params': int(res['params'])
        }
    
    with open(f'lno_checkpoints/{task_name}_comparison_results.json', 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"Results saved to lno_checkpoints/{task_name}_comparison_results.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    args = parser.parse_args()
    
    compare_burgers2d(epochs=args.epochs)
