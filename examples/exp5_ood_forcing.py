#!/usr/bin/env python3
"""
Experiment 5: Out-of-Distribution (OOD) Forcing Generalization

Train on one domain geometry (e.g., Circle) and test on different domains
(e.g., Ellipse, Cone) to evaluate cross-domain generalization.

Hypothesis: LRR's latent anchoring helps models generalize better across
different PDE domains by learning a continuous physical manifold.
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

# Publication-quality plot settings
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['pdf.fonttype'] = 42

# Colorblind-friendly palette
COLORS = {
    'fno': '#0072B2',
    'lrr': '#009E73',
    'improvement': '#E69F00',
    'error': '#D55E00',
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_relative_l2(pred, target):
    diff = pred - target
    return torch.norm(diff) / torch.norm(target)


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
            z_v_k = output.get('z_f')  # Projected backbone latent
            z_u = output.get('z_u')    # Solution encoder latent
            loss_dict = loss_fn(pred, u, z_v_k, z_u, stage=2)
            loss = loss_dict['total']
            loss.backward()
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


def evaluate(model, loader, device, is_lrr=False):
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


def plot_ood_results(train_domain, test_domains, fno_results, lrr_results, save_path):
    """Plot OOD generalization comparison - publication quality, no title."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    x = np.arange(len(test_domains))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fno_results, width, label='FNO', 
                   color=COLORS['fno'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, lrr_results, width, label='LRR-FNO', 
                   color=COLORS['lrr'], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Test Domain')
    ax.set_ylabel('Relative L2 Error')
    ax.set_xticks(x)
    ax.set_xticklabels(test_domains, rotation=25, ha='right', fontsize=9)
    ax.legend(frameon=True, fancybox=False, edgecolor='black', fontsize=9, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add improvement labels
    for i, (fno, lrr) in enumerate(zip(fno_results, lrr_results)):
        improvement = (fno - lrr) / fno * 100
        color = COLORS['improvement'] if improvement > 0 else COLORS['error']
        ax.annotate(f'{improvement:+.1f}%', 
                   xy=(x[i] + width/2, lrr), 
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=8, color=color)
    
    # Mark in-distribution domain
    if train_domain in test_domains:
        id_idx = test_domains.index(train_domain)
        ax.axvspan(id_idx - 0.45, id_idx + 0.45, alpha=0.12, color='gray')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(str(save_path).replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {Path(save_path).stem} (png, pdf)")


def run_experiment(train_dataset, test_datasets, epochs=50, max_samples=200, seed=42):
    """Run OOD generalization experiment."""
    set_seed(seed)
    device = get_device()
    
    train_name = Path(train_dataset).stem
    print(f"\n{'='*60}")
    print(f"Experiment 5: OOD Forcing - Train on {train_name}")
    print(f"{'='*60}")
    
    output_dir = Path('results/exp5_ood_forcing')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    print(f"\nLoading training data: {train_name}")
    train_loader, _, train_info = get_gaot_grid_loaders(
        train_dataset, batch_size=16, resolution=64, max_samples=max_samples
    )
    
    in_channels = train_info['in_channels']
    out_channels = train_info['out_channels']
    print(f"Training samples: {train_info['n_train']}")
    
    # Train FNO
    print(f"\n--- Training FNO ({epochs} epochs) ---")
    set_seed(seed)
    fno = FNO2d(
        in_channels=in_channels, out_channels=out_channels,
        modes1=12, modes2=12, width=32, num_layers=4
    ).to(device)
    train_fno(fno, train_loader, device, epochs)
    
    # Train LRR-FNO
    print(f"\n--- Training LRR-FNO ({epochs} epochs) ---")
    set_seed(seed)
    lrr_fno = LRRFNO2d(
        in_channels=in_channels, out_channels=out_channels,
        modes1=12, modes2=12, width=32, num_layers=4,
        latent_dim=64, encoder_channels=[32, 64, 128], use_gated_bridge=False
    ).to(device)
    lrr_loss_fn = LRNLoss(temperature=0.1, lambda_mse=10000.0, lambda_nce=0.01)
    train_lrr(lrr_fno, lrr_loss_fn, train_loader, device, epochs)
    
    # Evaluate on all test domains
    test_names = []
    fno_results = []
    lrr_results = []
    
    print("\n--- Evaluating on OOD domains ---")
    for test_path in test_datasets:
        test_name = Path(test_path).stem
        test_names.append(test_name)
        
        # Check if in-distribution or OOD
        is_id = (test_path == train_dataset)
        tag = "(In-Dist)" if is_id else "(OOD)"
        
        try:
            _, test_loader, test_info = get_gaot_grid_loaders(
                test_path, batch_size=16, resolution=64, max_samples=max_samples
            )
            
            fno_err = evaluate(fno, test_loader, device, is_lrr=False)
            lrr_err = evaluate(lrr_fno, test_loader, device, is_lrr=True)
            
            fno_results.append(fno_err)
            lrr_results.append(lrr_err)
            
            improvement = (fno_err - lrr_err) / fno_err * 100
            print(f"  {test_name} {tag}: FNO={fno_err:.4f}, LRR={lrr_err:.4f}, Δ={improvement:+.1f}%")
        except Exception as e:
            print(f"  {test_name}: Error - {e}")
            test_names.pop()
    
    # Results table
    print(f"\n--- OOD Generalization Results ---")
    print(f"{'Domain':<25} {'FNO':<12} {'LRR-FNO':<12} {'Improvement':<12}")
    print("-" * 60)
    for i, name in enumerate(test_names):
        is_id = "(ID)" if test_datasets[i] == train_dataset else "(OOD)"
        imp = (fno_results[i] - lrr_results[i]) / fno_results[i] * 100
        print(f"{name+' '+is_id:<25} {fno_results[i]:<12.6f} {lrr_results[i]:<12.6f} {imp:+.2f}%")
    
    # Compute average OOD improvement
    ood_indices = [i for i, p in enumerate(test_datasets) if p != train_dataset]
    if ood_indices:
        ood_fno = [fno_results[i] for i in ood_indices]
        ood_lrr = [lrr_results[i] for i in ood_indices]
        avg_ood_imp = np.mean([(f - l) / f * 100 for f, l in zip(ood_fno, ood_lrr)])
        print(f"\nAverage OOD Improvement: {avg_ood_imp:+.2f}%")
    
    # Plot
    plot_ood_results(
        train_name, test_names, fno_results, lrr_results,
        output_dir / f'{train_name}_ood_generalization.png'
    )
    
    return {
        'train': train_name,
        'test_domains': test_names,
        'fno': fno_results,
        'lrr': lrr_results
    }


def main():
    parser = argparse.ArgumentParser(description='Exp5: OOD Forcing Generalization')
    parser.add_argument('--train', type=str, default='dataset/Circle.nc')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--max_samples', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*60)
    print("Experiment 5: OOD Forcing Generalization")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    # Define test datasets (including train domain for ID reference)
    test_datasets = [
        'dataset/Circle.nc',
        'dataset/Ellipse-1.nc',  # AR 1.5
        'dataset/Ellipse-2.nc',  # AR 2.0
        'dataset/Ellipse-3.nc',  # AR 2.5
        'dataset/Cone-F.nc',
        'dataset/Semicircle-F.nc'
    ]
    
    run_experiment(
        args.train,
        test_datasets,
        epochs=args.epochs,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    print("\n" + "="*60)
    print("Experiment Complete!")
    print("="*60)


if __name__ == '__main__':
    main()
