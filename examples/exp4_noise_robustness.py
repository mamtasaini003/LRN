#!/usr/bin/env python3
"""
Experiment 4: Noise Robustness

Test model degradation with increasing Gaussian noise on input forcing.
Hypothesis: LRR acts as regularizer, filtering high-frequency noise.
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

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.components.fno import FNO2d
from models.lrr.model import LRRFNO2d
from losses.infonce import LRNLoss
from data.gaot_datasets import get_gaot_grid_loaders


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


def add_noise(tensor, noise_level):
    """Add Gaussian noise to tensor."""
    if noise_level == 0:
        return tensor
    noise = torch.randn_like(tensor) * noise_level * tensor.std()
    return tensor + noise


def train_fno(model, train_loader, device, epochs):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
    stage1_epochs = int(0.73 * epochs)
    stage2_epochs = epochs - stage1_epochs
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)
    
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        for c, u in train_loader:
            c, u = c.to(device), u.to(device)
            optimizer.zero_grad()
            output = model(c, u, return_latents=True)
            pred = output['prediction']
            z_f, z_u = output.get('z_f'), output.get('z_u')
            loss_dict = loss_fn(pred, u, z_f, z_u, stage=2)
            loss_dict['total'].backward()
            optimizer.step()
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"    LRR Stage1 {epoch}/{stage1_epochs}")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs)
    
    for epoch in range(1, stage2_epochs + 1):
        model.train()
        for c, u in train_loader:
            c, u = c.to(device), u.to(device)
            optimizer.zero_grad()
            output = model(c, u, return_latents=True)
            loss_dict = loss_fn(output['prediction'], u, None, None, stage=3)
            loss_dict['total'].backward()
            optimizer.step()
        scheduler.step()
        if epoch % 5 == 0 or epoch == 1:
            print(f"    LRR Stage2 {epoch}/{stage2_epochs}")
    
    return model


def evaluate_with_noise(model, loader, device, noise_level, is_lrr=False):
    model.eval()
    errors = []
    
    with torch.no_grad():
        for c, u in loader:
            c, u = c.to(device), u.to(device)
            c_noisy = add_noise(c, noise_level)
            
            if is_lrr:
                pred = model(c_noisy)['prediction']
            else:
                pred = model(c_noisy)
            
            for i in range(pred.size(0)):
                errors.append(compute_relative_l2(pred[i], u[i]).item())
    
    return np.mean(errors)


def plot_noise_robustness(noise_levels, fno_errors, lrr_errors, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(noise_levels, fno_errors, 'o-', linewidth=2, markersize=8, 
            label='FNO', color='steelblue')
    ax.plot(noise_levels, lrr_errors, 's-', linewidth=2, markersize=8,
            label='LRR-FNO', color='coral')
    
    ax.set_xlabel('Noise Level (σ)', fontsize=12)
    ax.set_ylabel('Relative L2 Error', fontsize=12)
    ax.set_title('Noise Robustness: FNO vs LRR-FNO', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Calculate and show average improvement
    improvements = [(f - l) / f * 100 for f, l in zip(fno_errors, lrr_errors)]
    avg_imp = np.mean(improvements)
    ax.text(0.95, 0.05, f'Avg LRR improvement: {avg_imp:+.1f}%',
           transform=ax.transAxes, ha='right', fontsize=11,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def run_experiment(nc_path, epochs=50, max_samples=200, seed=42):
    set_seed(seed)
    device = get_device()
    
    dataset_name = Path(nc_path).stem
    print(f"\n{'='*60}")
    print(f"Experiment 4: Noise Robustness - {dataset_name}")
    print(f"{'='*60}")
    
    output_dir = Path('results/exp4_noise_robustness')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader, test_loader, info = get_gaot_grid_loaders(
        nc_path, batch_size=16, resolution=64, max_samples=max_samples
    )
    
    in_channels = info['in_channels']
    out_channels = info['out_channels']
    print(f"Samples: {info['n_train']} train, {info['n_test']} test")
    
    # Train FNO
    print(f"\n--- Training FNO ({epochs} epochs) ---")
    set_seed(seed)
    fno = FNO2d(
        in_channels=in_channels, out_channels=out_channels,
        modes1=12, modes2=12, width=32, num_layers=4
    ).to(device)
    train_fno(fno, train_loader, device, epochs)
    
    # Train LRR
    print(f"\n--- Training LRR-FNO ({epochs} epochs) ---")
    set_seed(seed)
    lrr_fno = LRRFNO2d(
        in_channels=in_channels, out_channels=out_channels,
        modes1=12, modes2=12, width=32, num_layers=4,
        latent_dim=64, encoder_channels=[32, 64, 128], use_gated_bridge=False
    ).to(device)
    lrr_loss_fn = LRNLoss(temperature=0.1, lambda_mse=10000.0, lambda_nce=0.01)
    train_lrr(lrr_fno, lrr_loss_fn, train_loader, device, epochs)
    
    # Evaluate at different noise levels
    noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    fno_errors = []
    lrr_errors = []
    
    print("\n--- Evaluating with noise ---")
    for noise in noise_levels:
        fno_err = evaluate_with_noise(fno, test_loader, device, noise, is_lrr=False)
        lrr_err = evaluate_with_noise(lrr_fno, test_loader, device, noise, is_lrr=True)
        fno_errors.append(fno_err)
        lrr_errors.append(lrr_err)
        improvement = (fno_err - lrr_err) / fno_err * 100
        print(f"  σ={noise:.2f}: FNO={fno_err:.4f}, LRR={lrr_err:.4f}, Δ={improvement:+.1f}%")
    
    # Results table
    print(f"\n--- Noise Robustness Results: {dataset_name} ---")
    print(f"{'Noise σ':<10} {'FNO':<12} {'LRR-FNO':<12} {'Improvement':<12}")
    print("-" * 50)
    for i, noise in enumerate(noise_levels):
        imp = (fno_errors[i] - lrr_errors[i]) / fno_errors[i] * 100
        print(f"{noise:<10.2f} {fno_errors[i]:<12.6f} {lrr_errors[i]:<12.6f} {imp:+.2f}%")
    
    # Plot
    plot_noise_robustness(
        noise_levels, fno_errors, lrr_errors,
        output_dir / f'{dataset_name}_noise_robustness.png'
    )
    
    return {'noise_levels': noise_levels, 'fno': fno_errors, 'lrr': lrr_errors}


def main():
    parser = argparse.ArgumentParser(description='Exp4: Noise Robustness')
    parser.add_argument('--dataset', type=str, default='dataset/Circle.nc')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Experiment 4: Noise Robustness")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    run_experiment(args.dataset, args.epochs, args.max_samples, args.seed)
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
