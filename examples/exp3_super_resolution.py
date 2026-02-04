#!/usr/bin/env python3
"""
Experiment 3: Zero-Shot Super-Resolution

Train on 64x64 resolution, test on 128x128 without retraining.
Tests whether LRR anchors representations to a continuous physical manifold.
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
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.components.fno import FNO2d
from models.lrr.model import LRRFNO2d
from losses.infonce import LRNLoss
from data.gaot_datasets import GAOTGridDataset
from torch.utils.data import DataLoader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_relative_l2(pred, target):
    diff = pred - target
    return torch.norm(diff) / torch.norm(target)


def get_dataloaders_multiresolution(nc_path, train_res=64, test_res=128, 
                                     max_samples=200, batch_size=16):
    """Create dataloaders at different resolutions for super-resolution test."""
    # Training data at lower resolution
    train_dataset = GAOTGridDataset(
        nc_path, train=True, resolution=train_res, 
        max_samples=max_samples, normalize=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Test data at both resolutions
    test_low = GAOTGridDataset(
        nc_path, train=False, resolution=train_res,
        max_samples=max_samples, normalize=True
    )
    test_low_loader = DataLoader(test_low, batch_size=batch_size, shuffle=False)
    
    test_high = GAOTGridDataset(
        nc_path, train=False, resolution=test_res,
        max_samples=max_samples, normalize=True
    )
    test_high_loader = DataLoader(test_high, batch_size=batch_size, shuffle=False)
    
    info = {
        'in_channels': train_dataset.c.shape[1],
        'out_channels': train_dataset.u.shape[1],
        'train_res': train_res,
        'test_res': test_res,
        'n_train': len(train_dataset),
        'n_test': len(test_low),
    }
    
    return train_loader, test_low_loader, test_high_loader, info


def train_fno(model, train_loader, device, epochs, lr=1e-3):
    """Train vanilla FNO."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        for c, u in train_loader:
            c, u = c.to(device), u.to(device)
            optimizer.zero_grad()
            pred = model(c)
            loss = mse_loss(pred, u)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"    FNO Epoch {epoch}/{epochs}")
    
    return model


def train_lrr(model, loss_fn, train_loader, device, epochs):
    """Train LRR-FNO with 2-stage protocol."""
    stage1_epochs = int(0.73 * epochs)
    stage2_epochs = epochs - stage1_epochs
    
    # Stage 1: NCE + MSE
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)
    
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        for c, u in train_loader:
            c, u = c.to(device), u.to(device)
            optimizer.zero_grad()
            output = model(c, u, return_latents=True)
            pred = output['prediction']
            z_f = output.get('z_f')
            z_u = output.get('z_u')
            loss_dict = loss_fn(pred, u, z_f, z_u, stage=2)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"    LRR Stage1 Epoch {epoch}/{stage1_epochs}")
    
    # Stage 2: MSE only
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs)
    
    for epoch in range(1, stage2_epochs + 1):
        model.train()
        for c, u in train_loader:
            c, u = c.to(device), u.to(device)
            optimizer.zero_grad()
            output = model(c, u, return_latents=True)
            pred = output['prediction']
            loss_dict = loss_fn(pred, u, None, None, stage=3)
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"    LRR Stage2 Epoch {epoch}/{stage2_epochs}")
    
    return model


def evaluate_at_resolution(model, loader, device, is_lrr=False):
    """Evaluate model at a given resolution."""
    model.eval()
    errors = []
    
    with torch.no_grad():
        for c, u in loader:
            c, u = c.to(device), u.to(device)
            
            if is_lrr:
                pred = model(c)['prediction']
            else:
                pred = model(c)
            
            for i in range(pred.size(0)):
                errors.append(compute_relative_l2(pred[i], u[i]).item())
    
    return np.mean(errors)


def plot_super_resolution_results(results, save_path):
    """Plot super-resolution comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.35
    
    fno_vals = [results['fno_64'], results['fno_128']]
    lrr_vals = [results['lrr_64'], results['lrr_128']]
    
    bars1 = ax.bar(x - width/2, fno_vals, width, label='FNO', color='steelblue')
    bars2 = ax.bar(x + width/2, lrr_vals, width, label='LRR-FNO', color='coral')
    
    ax.set_xlabel('Test Resolution', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Zero-Shot Super-Resolution: FNO vs LRR-FNO\n(Trained on 64×64)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['64×64 (in-dist)', '128×128 (zero-shot)'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    # Add degradation rates
    fno_deg = (results['fno_128'] - results['fno_64']) / results['fno_64'] * 100
    lrr_deg = (results['lrr_128'] - results['lrr_64']) / results['lrr_64'] * 100
    
    ax.text(0.5, 0.95, f'Degradation: FNO {fno_deg:+.1f}% | LRR {lrr_deg:+.1f}%',
           transform=ax.transAxes, ha='center', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def run_experiment(nc_path, epochs=50, max_samples=200, seed=42):
    """Run super-resolution experiment."""
    set_seed(seed)
    device = get_device()
    
    dataset_name = Path(nc_path).stem
    print(f"\n{'='*60}")
    print(f"Experiment 3: Super-Resolution - {dataset_name}")
    print(f"{'='*60}")
    
    output_dir = Path('results/exp3_super_resolution')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_res, test_res = 64, 128
    
    # Load data at multiple resolutions
    print(f"\nLoading data: train@{train_res}x{train_res}, test@{test_res}x{test_res}")
    train_loader, test_low_loader, test_high_loader, info = get_dataloaders_multiresolution(
        nc_path, train_res=train_res, test_res=test_res, max_samples=max_samples
    )
    
    in_channels = info['in_channels']
    out_channels = info['out_channels']
    print(f"Samples: {info['n_train']} train, {info['n_test']} test")
    
    # --- Train FNO ---
    print(f"\n--- Training FNO ({epochs} epochs) ---")
    set_seed(seed)
    
    fno = FNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        modes1=12, modes2=12,
        width=32,
        num_layers=4
    ).to(device)
    
    train_fno(fno, train_loader, device, epochs)
    
    # Evaluate FNO
    fno_64 = evaluate_at_resolution(fno, test_low_loader, device, is_lrr=False)
    fno_128 = evaluate_at_resolution(fno, test_high_loader, device, is_lrr=False)
    print(f"FNO @ 64×64:  {fno_64:.6f}")
    print(f"FNO @ 128×128: {fno_128:.6f}")
    
    # --- Train LRR-FNO ---
    print(f"\n--- Training LRR-FNO ({epochs} epochs) ---")
    set_seed(seed)
    
    lrr_fno = LRRFNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        modes1=12, modes2=12,
        width=32,
        num_layers=4,
        latent_dim=64,
        encoder_channels=[32, 64, 128],
        use_gated_bridge=False
    ).to(device)
    
    lrr_loss_fn = LRNLoss(
        temperature=0.1, lambda_mse=10000.0, lambda_nce=0.01
    )
    
    train_lrr(lrr_fno, lrr_loss_fn, train_loader, device, epochs)
    
    # Evaluate LRR
    lrr_64 = evaluate_at_resolution(lrr_fno, test_low_loader, device, is_lrr=True)
    lrr_128 = evaluate_at_resolution(lrr_fno, test_high_loader, device, is_lrr=True)
    print(f"LRR @ 64×64:  {lrr_64:.6f}")
    print(f"LRR @ 128×128: {lrr_128:.6f}")
    
    # Results
    results = {
        'fno_64': fno_64, 'fno_128': fno_128,
        'lrr_64': lrr_64, 'lrr_128': lrr_128
    }
    
    # Degradation rates
    fno_deg = (fno_128 - fno_64) / fno_64 * 100
    lrr_deg = (lrr_128 - lrr_64) / lrr_64 * 100
    
    print(f"\n--- Super-Resolution Results: {dataset_name} ---")
    print(f"{'Model':<15} {'64×64':<12} {'128×128':<12} {'Degradation':<12}")
    print("-" * 50)
    print(f"{'FNO':<15} {fno_64:<12.6f} {fno_128:<12.6f} {fno_deg:+.2f}%")
    print(f"{'LRR-FNO':<15} {lrr_64:<12.6f} {lrr_128:<12.6f} {lrr_deg:+.2f}%")
    
    # Plot
    plot_super_resolution_results(
        results, output_dir / f'{dataset_name}_super_resolution.png'
    )
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Exp3: Zero-Shot Super-Resolution')
    parser.add_argument('--dataset', type=str, default='dataset/Circle.nc')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Experiment 3: Zero-Shot Super-Resolution")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    run_experiment(
        args.dataset,
        epochs=args.epochs,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
