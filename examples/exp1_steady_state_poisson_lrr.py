#!/usr/bin/env python3
"""
Steady-State PDE Experiment: FNO vs LRR-FNO (Latent Reciprocity Representation)

Compares vanilla FNO against LRR-FNO on time-independent PDE benchmarks
to demonstrate the effectiveness of Latent Reciprocity Representation.

Datasets:
- Circle: Poisson equation on circular domain
- Ellipse-1/2/3: Poisson on elliptical domains (varying aspect ratios)
- Cone-F: Heat conduction on conical geometry
- Semicircle-F: Forced boundary problem on semicircular domain
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
from utils.training import Trainer
from data.gaot_datasets import get_gaot_grid_loaders

# Human-readable dataset names
DATASET_NAMES = {
    'Circle': 'Poisson Circle Domain',
    'Cone-F': 'Heat Conduction Cone',
    'Ellipse-1': 'Poisson Ellipse (AR=1.5)',
    'Ellipse-2': 'Poisson Ellipse (AR=2.0)',
    'Ellipse-3': 'Poisson Ellipse (AR=2.5)',
    'Semicircle-F': 'Forced Semicircle BVP',
}


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


def train_fno(model, train_loader, test_loader, device, epochs, lr=1e-3):
    """Train vanilla FNO with MSE loss."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    mse_loss = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for c, u in train_loader:
            c, u = c.to(device), u.to(device)
            
            optimizer.zero_grad()
            pred = model(c)
            loss = mse_loss(pred, u)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  FNO Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(train_loader):.6f}")
    
    # Evaluate
    model.eval()
    errors = []
    with torch.no_grad():
        for c, u in test_loader:
            c, u = c.to(device), u.to(device)
            pred = model(c)
            for i in range(pred.size(0)):
                errors.append(compute_relative_l2(pred[i], u[i]).item())
    
    return np.mean(errors)


def train_lrr(model, loss_fn, train_loader, test_loader, device, epochs):
    """Train LRR-FNO using 2-stage training."""
    stage1_epochs = int(0.73 * epochs)
    stage2_epochs = epochs - stage1_epochs
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        stage1_lr=1e-3,
        stage2_lr=1e-4,
        checkpoint_dir='checkpoints_lrr'
    )
    
    trainer.train()
    
    # Evaluate
    model.eval()
    errors = []
    with torch.no_grad():
        for c, u in test_loader:
            c, u = c.to(device), u.to(device)
            output = model(c)
            pred = output['prediction']
            for i in range(pred.size(0)):
                errors.append(compute_relative_l2(pred[i], u[i]).item())
    
    return np.mean(errors)


def run_experiment(nc_path, epochs=50, max_samples=200, seed=42):
    """Run FNO vs LRR comparison on a single dataset."""
    set_seed(seed)
    device = get_device()
    
    dataset_key = Path(nc_path).stem
    dataset_name = DATASET_NAMES.get(dataset_key, dataset_key)
    
    print(f"\n{'='*60}")
    print(f"Experiment: {dataset_name}")
    print(f"File: {Path(nc_path).name}")
    print(f"{'='*60}")
    
    # Load data
    train_loader, test_loader, info = get_gaot_grid_loaders(
        nc_path, 
        batch_size=16, 
        resolution=64,
        max_samples=max_samples
    )
    
    in_channels = info['in_channels']
    out_channels = info['out_channels']
    
    print(f"Samples: {info['n_train']} train, {info['n_test']} test")
    print(f"Channels: {in_channels} -> {out_channels}")
    
    # --- Train Vanilla FNO ---
    print(f"\n--- Training Vanilla FNO ({epochs} epochs) ---")
    set_seed(seed)
    
    fno = FNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        modes1=12, modes2=12,
        width=32,
        num_layers=4
    ).to(device)
    
    fno_error = train_fno(fno, train_loader, test_loader, device, epochs)
    print(f"FNO Test Rel L2: {fno_error:.6f}")
    
    # --- Train LRR-FNO ---
    print(f"\n--- Training LRR-FNO with Latent Reciprocity Representation ({epochs} epochs) ---")
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
        temperature=0.1, 
        lambda_mse=10000.0, 
        lambda_nce=0.01
    )
    
    lrr_error = train_lrr(lrr_fno, lrr_loss_fn, train_loader, test_loader, device, epochs)
    print(f"LRR-FNO Test Rel L2: {lrr_error:.6f}")
    
    # --- Results ---
    improvement = (fno_error - lrr_error) / fno_error * 100
    print(f"\n--- Results: {dataset_name} ---")
    print(f"FNO:     {fno_error:.6f}")
    print(f"LRR-FNO: {lrr_error:.6f}")
    print(f"Improvement: {improvement:+.2f}%")
    
    return {
        'dataset': dataset_name,
        'file': dataset_key,
        'fno_error': fno_error,
        'lrr_error': lrr_error,
        'improvement': improvement
    }


def main():
    parser = argparse.ArgumentParser(description='Steady-State PDE: FNO vs LRR-FNO')
    parser.add_argument('--dataset', type=str, default='dataset/Circle.nc',
                        help='Path to dataset .nc file')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--max_samples', type=int, default=200, help='Max samples to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--all', action='store_true', help='Run on all datasets')
    
    args = parser.parse_args()
    
    print("="*60)
    print("LRR-FNO Latent Reciprocity Representation Experiment")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    if args.all:
        # Run on all available datasets
        dataset_dir = Path('dataset')
        nc_files = sorted(dataset_dir.glob('*.nc'))
        
        results = []
        for nc_path in nc_files:
            try:
                result = run_experiment(
                    str(nc_path), 
                    epochs=args.epochs,
                    max_samples=args.max_samples,
                    seed=args.seed
                )
                results.append(result)
            except Exception as e:
                print(f"Error on {nc_path.name}: {e}")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY: Latent Reciprocity Representation on Steady-State PDEs")
        print("="*70)
        print(f"{'Dataset':<30} {'FNO':<12} {'LRR-FNO':<12} {'Δ':<12}")
        print("-"*70)
        for r in results:
            print(f"{r['dataset']:<30} {r['fno_error']:<12.6f} {r['lrr_error']:<12.6f} {r['improvement']:+.2f}%")
        
        # Save results
        results_path = Path('results') / 'lrr_steady_state_results.txt'
        results_path.parent.mkdir(exist_ok=True)
        with open(results_path, 'w') as f:
            f.write(f"LRR-FNO Latent Reciprocity Representation Experiment\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Epochs: {args.epochs}, Max Samples: {args.max_samples}\n\n")
            f.write(f"{'Dataset':<30} {'FNO':<12} {'LRR-FNO':<12} {'Improvement':<12}\n")
            f.write("-"*70 + "\n")
            for r in results:
                f.write(f"{r['dataset']:<30} {r['fno_error']:<12.6f} {r['lrr_error']:<12.6f} {r['improvement']:+.2f}%\n")
        print(f"\nResults saved to {results_path}")
    else:
        run_experiment(
            args.dataset,
            epochs=args.epochs,
            max_samples=args.max_samples,
            seed=args.seed
        )


if __name__ == '__main__':
    main()
