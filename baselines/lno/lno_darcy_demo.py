"""
LNO vs LRN-FNO Comparison Demo - Darcy Flow

This script provides a fair comparison between:
1. Vanilla FNO
2. LRN-FNO (our method)
3. LNO (NeurIPS 2024 baseline)

Following the same experimental setup as the LRN-FNO demos for fair comparison.
"""

import os
import argparse
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

from lno_model import LNO2d


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
    
    def relative_l2_loss(pred, target):
        diff = pred - target
        return torch.norm(diff.reshape(diff.shape[0], -1), dim=1).mean() / \
               torch.norm(target.reshape(target.shape[0], -1), dim=1).mean()
    
    model.train()
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        for f, u in train_loader:
            f, u = f.to(device), u.to(device)
            
            optimizer.zero_grad()
            pred = model(f)
            
            # Handle shape mismatch
            if pred.dim() == 4 and u.dim() == 3:
                pred = pred.squeeze(1)
            
            loss = relative_l2_loss(pred, u)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            test_errors = []
            with torch.no_grad():
                for f, u in test_loader:
                    f, u = f.to(device), u.to(device)
                    pred = model(f)
                    
                    if pred.dim() == 4 and u.dim() == 3:
                        pred = pred.squeeze(1)
                    
                    u_flat = u.reshape(u.shape[0], -1)
                    pred_flat = pred.reshape(u.shape[0], -1)
                    
                    errors = torch.norm(pred_flat - u_flat, dim=1) / torch.norm(u_flat, dim=1)
                    test_errors.extend(errors.cpu().numpy())
            
            mean_error = np.mean(test_errors)
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.6f}, Test RelL2: {mean_error:.6f}")
            model.train()
    
    return losses


def compare_darcy(epochs=150):
    """Compare FNO, LRN-FNO, and LNO on Darcy Flow."""
    print("=" * 70)
    print(f"LNO vs LRN-FNO Comparison - Darcy Flow (Epochs: {epochs})")
    print("=" * 70)
    
    # Calculate stage epochs
    stage1_epochs = int(epochs * 0.75)
    stage2_epochs = epochs - stage1_epochs
    if stage1_epochs < 1: stage1_epochs = 1
    if stage2_epochs < 1: stage2_epochs = 0
    
    set_seed(42)
    
    os.makedirs('../../results/plots', exist_ok=True)
    os.makedirs('lno_checkpoints', exist_ok=True)
    
    device = get_device()
    print(f"Device: {device}")
    
    # Import LRN modules
    from src.models import LRNFNO2d, FNO2d
    from src.losses import LRNLoss
    from src.data import DarcyDataset
    from src.utils import LRNTrainerV2
    
    # 1. Prepare Data
    print("\nPreparing Darcy Flow Dataset...")
    train_dataset = DarcyDataset(resolution=32, num_samples=400, train=True)
    test_dataset = DarcyDataset(resolution=32, num_samples=100, train=False)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    results = {}
    
    # 2. Train Vanilla FNO
    print("\n" + "=" * 40)
    print("Training Vanilla FNO")
    print("=" * 40)
    
    set_seed(42)
    
    fno = FNO2d(
        in_channels=1,
        out_channels=1,
        modes1=12,
        modes2=12,
        width=32,
        num_layers=4
    ).to(device)
    
    optimizer = torch.optim.Adam(fno.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    
    start_time = time.time()
    
    for epoch in range(epochs):
        fno.train()
        for f, u in train_loader:
            f, u = f.to(device), u.to(device)
            optimizer.zero_grad()
            pred = fno(f)
            if pred.dim() == 4 and u.dim() == 3:
                pred = pred.squeeze(1)
            loss = mse_loss(pred, u)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % max(1, epochs//5) == 0:
            print(f"Epoch {epoch+1}/{epochs}")
    
    fno_time = time.time() - start_time
    print(f"FNO Training Time: {fno_time:.1f}s")
    
    fno.eval()
    fno_errors = []
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            pred = fno(f)
            if pred.dim() == 4 and u.dim() == 3:
                pred = pred.squeeze(1)
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
    print(f"FNO Test RelL2: {results['fno']['error']:.6f}")
    
    # 3. Train LRN-FNO
    print("\n" + "=" * 40)
    print("Training LRN-FNO (2-Stage Protocol)")
    print("=" * 40)
    
    set_seed(42)
    
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
    
    loss_fn = LRNLoss(lambda_mse=1.0, lambda_nce=1.0, use_relative_mse=False)
    
    start_time = time.time()
    trainer = LRNTrainerV2(
        model=lrn,
        train_loader=train_loader,
        test_loader=test_loader,
        loss_fn=loss_fn,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        stage1_lr=1e-3,
        stage2_lr=1e-4,
        device=str(device),
        checkpoint_dir='lno_checkpoints/lrn_darcy'
    )
    trainer.train()
    lrn_time = time.time() - start_time
    print(f"LRN-FNO Training Time: {lrn_time:.1f}s")
    
    lrn.eval()
    lrn_errors = []
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            output = lrn(f, return_latents=False)
            pred = output['prediction']
            if pred.dim() == 4 and u.dim() == 3:
                pred = pred.squeeze(1)
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
    print(f"LRN-FNO Test RelL2: {results['lrn']['error']:.6f}")
    
    # 4. Train LNO
    print("\n" + "=" * 40)
    print("Training LNO (NeurIPS 2024)")
    print("=" * 40)
    
    set_seed(42)
    
    # Paper config for Darcy: 4 layers, 128 dim
    lno = LNO2d(
        in_channels=1,
        out_channels=1,
        embed_dim=128,
        latent_size=256,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    ).to(device)
    
    start_time = time.time()
    train_lno(lno, train_loader, test_loader, device, epochs=epochs)
    lno_time = time.time() - start_time
    print(f"LNO Training Time: {lno_time:.1f}s")
    
    lno.eval()
    lno_errors = []
    with torch.no_grad():
        for f, u in test_loader:
            f, u = f.to(device), u.to(device)
            pred = lno(f)
            if pred.dim() == 4 and u.dim() == 3:
                pred = pred.squeeze(1)
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
    print(f"LNO Test RelL2: {results['lno']['error']:.6f}")
    
    # 5. Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY - Darcy Flow")
    print("=" * 70)
    
    print(f"\n{'Model':<15} {'RelL2 Error':<15} {'Std Dev':<12} {'Time (s)':<12} {'Params':<12}")
    print("-" * 66)
    for name, res in results.items():
        print(f"{name.upper():<15} {res['error']:.6f}       {res['std']:.6f}     {res['time']:.1f}        {res['params']:,}")
    
    fno_err = results['fno']['error']
    print("\n" + "-" * 66)
    print("Improvements over FNO baseline:")
    for name in ['lrn', 'lno']:
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
    save_results(results, 'darcy')
    
    return results


def visualize_comparison(fno, lrn, lno, dataset, device, results):
    """Create comparison visualization."""
    fno.eval()
    lrn.eval()
    lno.eval()
    
    idx = 0
    f, u_gt = dataset[idx]
    f_input = f.unsqueeze(0).to(device)
    u_gt = u_gt.to(device)
    
    with torch.no_grad():
        pred_fno = fno(f_input).squeeze()
        pred_lrn = lrn(f_input, return_latents=False)['prediction'].squeeze()
        pred_lno = lno(f_input).squeeze()
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    
    # Row 1: Input and Ground Truth
    im0 = axes[0, 0].imshow(f.squeeze().cpu().numpy(), cmap='viridis')
    axes[0, 0].set_title('Input: Permeability a(x)')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(u_gt.cpu().numpy(), cmap='jet')
    axes[0, 1].set_title('Ground Truth: Pressure u(x)')
    plt.colorbar(im1, ax=axes[0, 1])
    
    axes[0, 2].axis('off')
    axes[0, 2].text(0.5, 0.5, 'Darcy Flow\n-∇·(a(x)∇u) = f', 
                    ha='center', va='center', fontsize=14, transform=axes[0, 2].transAxes)
    
    # Row 2: FNO
    im3 = axes[1, 0].imshow(pred_fno.cpu().numpy(), cmap='jet')
    axes[1, 0].set_title(f"FNO (RelL2: {results['fno']['error']:.4f})")
    plt.colorbar(im3, ax=axes[1, 0])
    
    err_fno = np.abs(u_gt.cpu().numpy() - pred_fno.cpu().numpy())
    im4 = axes[1, 1].imshow(err_fno, cmap='inferno')
    axes[1, 1].set_title('|Error| FNO')
    plt.colorbar(im4, ax=axes[1, 1])
    
    axes[1, 2].axis('off')
    
    # Row 3: LRN-FNO
    im5 = axes[2, 0].imshow(pred_lrn.cpu().numpy(), cmap='jet')
    axes[2, 0].set_title(f"LRN-FNO (RelL2: {results['lrn']['error']:.4f})")
    plt.colorbar(im5, ax=axes[2, 0])
    
    err_lrn = np.abs(u_gt.cpu().numpy() - pred_lrn.cpu().numpy())
    im6 = axes[2, 1].imshow(err_lrn, cmap='inferno')
    axes[2, 1].set_title('|Error| LRN-FNO')
    plt.colorbar(im6, ax=axes[2, 1])
    
    axes[2, 2].axis('off')
    
    # Row 4: LNO
    im7 = axes[3, 0].imshow(pred_lno.cpu().numpy(), cmap='jet')
    axes[3, 0].set_title(f"LNO (RelL2: {results['lno']['error']:.4f})")
    plt.colorbar(im7, ax=axes[3, 0])
    
    err_lno = np.abs(u_gt.cpu().numpy() - pred_lno.cpu().numpy())
    im8 = axes[3, 1].imshow(err_lno, cmap='inferno')
    axes[3, 1].set_title('|Error| LNO')
    plt.colorbar(im8, ax=axes[3, 1])
    
    axes[3, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('../../results/plots/lno_vs_lrn_darcy.png', dpi=150)
    print("\nVisualization saved to results/plots/lno_vs_lrn_darcy.png")
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
    
    compare_darcy(epochs=args.epochs)
