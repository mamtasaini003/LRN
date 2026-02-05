#!/usr/bin/env python3
"""
Latent Space Analysis with Physical Parameter Coloring.

This script creates t-SNE visualizations of latent representations
colored by a physical parameter (Aspect Ratio for Ellipse datasets).

This is analogous to the RB t-SNE plots colored by Rayleigh/Prandtl numbers
in the latent_reps paper.

Usage:
    python3 examples/latent_analysis_ellipse_parameter.py --epochs 50
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.manifold import TSNE

# Add src to path
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


def extract_latents(model, loader, device, is_lrr=True):
    """Extract latent representations from model."""
    model.eval()
    z_f_list = []
    z_u_list = []
    
    with torch.no_grad():
        for c, u in loader:
            c, u = c.to(device), u.to(device)
            
            if is_lrr:
                output = model(c, u, return_latents=True)
                z_f_list.append(output['z_f'].cpu().numpy())
                if 'z_u' in output:
                    z_u_list.append(output['z_u'].cpu().numpy())
            else:
                pred = model(c)
                z_proxy = pred.mean(dim=(2, 3)).cpu().numpy()
                z_f_list.append(z_proxy)
    
    z_f = np.concatenate(z_f_list, axis=0)
    z_u = np.concatenate(z_u_list, axis=0) if z_u_list else None
    
    return z_f, z_u


def plot_tsne_by_parameter(z, params, param_name, save_path, title="Latent Space"):
    """
    Plot t-SNE visualization colored by a physical parameter.
    
    Creates publication-quality figures similar to rb_tsne_rayleigh.pdf and rb_tsne_prandtl.pdf.
    
    Args:
        z: Latent representations [N, D]
        params: Parameter values [N,]
        param_name: Name of the parameter for labeling
        save_path: Path to save the figure
        title: Plot title
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 12
    
    perplexity = min(30, len(z) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
    embedded = tsne.fit_transform(z)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create colormap based on unique parameter values
    unique_params = np.unique(params)
    if len(unique_params) <= 10:
        # Discrete colormap for few unique values
        cmap = cm.get_cmap('viridis', len(unique_params))
        norm = plt.Normalize(vmin=params.min(), vmax=params.max())
        colors = cmap(norm(params))
        
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], 
                            c=params, cmap='viridis', s=40, alpha=0.7,
                            edgecolors='white', linewidths=0.3)
        
        # Create colorbar with discrete ticks
        cbar = plt.colorbar(scatter, ax=ax, ticks=unique_params)
        cbar.set_label(param_name, fontsize=14)
    else:
        # Continuous colormap for many unique values
        scatter = ax.scatter(embedded[:, 0], embedded[:, 1], 
                            c=params, cmap='viridis', s=40, alpha=0.7,
                            edgecolors='white', linewidths=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(param_name, fontsize=14)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    print(f"Saved: {save_path.replace('.png', '.pdf')}")


def train_lrr_fno(model, loss_fn, train_loader, device, epochs):
    """Train LRR-FNO with 2-stage protocol."""
    stage1_epochs = int(0.73 * epochs)
    stage2_epochs = epochs - stage1_epochs
    
    # Stage 1: NCE + MSE
    print(f"  Stage 1: Combined Optimization [{stage1_epochs} epochs]")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)
    
    for epoch in range(1, stage1_epochs + 1):
        model.train()
        epoch_loss = 0.0
        
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
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch}/{stage1_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Stage 2: MSE only
    print(f"  Stage 2: Autonomous Distillation [{stage2_epochs} epochs]")
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
            print(f"    Epoch {epoch}/{stage2_epochs}")


def run_experiment(epochs=50, max_samples=150, seed=42):
    """Run latent analysis experiment with Ellipse datasets."""
    set_seed(seed)
    device = get_device()
    
    print(f"\n{'='*60}")
    print("Latent Space Analysis with Physical Parameter Coloring")
    print(f"{'='*60}")
    
    output_dir = Path('results/latent_analysis_ellipse')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    latent_reps_dir = Path('latent_reps/figures')
    latent_reps_dir.mkdir(parents=True, exist_ok=True)
    
    # Define datasets with their aspect ratios
    datasets = [
        ('dataset/Ellipse-1.nc', 1.5),
        ('dataset/Ellipse-2.nc', 2.0),
        ('dataset/Ellipse-3.nc', 2.5),
    ]
    
    # Load all datasets
    all_c_train, all_u_train, all_aspect_train = [], [], []
    all_c_test, all_u_test, all_aspect_test = [], [], []
    in_channels, out_channels = None, None
    
    for nc_path, aspect_ratio in datasets:
        if not Path(nc_path).exists():
            print(f"  Warning: {nc_path} not found, skipping.")
            continue
            
        train_loader, test_loader, info = get_gaot_grid_loaders(
            nc_path, batch_size=32, resolution=64, max_samples=max_samples
        )
        
        if in_channels is None:
            in_channels = info['in_channels']
            out_channels = info['out_channels']
        
        for c, u in train_loader:
            all_c_train.append(c)
            all_u_train.append(u)
            all_aspect_train.extend([aspect_ratio] * c.size(0))
        
        for c, u in test_loader:
            all_c_test.append(c)
            all_u_test.append(u)
            all_aspect_test.extend([aspect_ratio] * c.size(0))
        
        print(f"  Loaded {nc_path}: {info['n_train']} train, {info['n_test']} test (AR={aspect_ratio})")
    
    if not all_c_train:
        print("Error: No datasets found!")
        return
    
    # Concatenate data
    all_c_train = torch.cat(all_c_train, dim=0)
    all_u_train = torch.cat(all_u_train, dim=0)
    all_c_test = torch.cat(all_c_test, dim=0)
    all_u_test = torch.cat(all_u_test, dim=0)
    all_aspect_train = np.array(all_aspect_train)
    all_aspect_test = np.array(all_aspect_test)
    
    print(f"\n  Combined Training Set: {len(all_c_train)} samples")
    print(f"  Combined Test Set: {len(all_c_test)} samples")
    
    # Create combined data loaders
    train_dataset = torch.utils.data.TensorDataset(all_c_train, all_u_train)
    test_dataset = torch.utils.data.TensorDataset(all_c_test, all_u_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Train LRR-FNO
    print(f"\n--- Training LRR-FNO ({epochs} epochs) ---")
    
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
    
    train_lrr_fno(lrr_fno, lrr_loss_fn, train_loader, device, epochs)
    
    # Extract latents
    print("\n--- Extracting Latent Representations ---")
    z_f, z_u = extract_latents(lrr_fno, test_loader, device, is_lrr=True)
    print(f"  z_f shape: {z_f.shape}, z_u shape: {z_u.shape if z_u is not None else 'N/A'}")
    
    # Generate plots
    print("\n--- Generating Visualizations ---")
    
    # Plot 1: t-SNE colored by Aspect Ratio (z_f)
    plot_tsne_by_parameter(
        z_f, all_aspect_test, 
        param_name='Aspect Ratio',
        save_path=str(output_dir / 'tsne_aspect_ratio_zf.png'),
        title='Latent Space (z_f) by Aspect Ratio'
    )
    
    # Also save to latent_reps/figures
    plot_tsne_by_parameter(
        z_f, all_aspect_test, 
        param_name='Aspect Ratio',
        save_path=str(latent_reps_dir / 'ellipse_tsne_aspect_ratio.png'),
        title='Latent Space by Aspect Ratio'
    )
    
    # Plot 2: t-SNE colored by Aspect Ratio (z_u)
    if z_u is not None:
        plot_tsne_by_parameter(
            z_u, all_aspect_test, 
            param_name='Aspect Ratio',
            save_path=str(output_dir / 'tsne_aspect_ratio_zu.png'),
            title='Solution Encoding (z_u) by Aspect Ratio'
        )
    
    # Plot 3: z_f vs z_u alignment
    if z_u is not None:
        combined = np.vstack([z_f, z_u])
        perplexity = min(30, len(combined) // 2 - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
        embedded = tsne.fit_transform(combined)
        
        n = len(z_f)
        z_f_embedded = embedded[:n]
        z_u_embedded = embedded[n:]
        
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # Plot connection lines
        for i in range(n):
            ax.plot([z_f_embedded[i, 0], z_u_embedded[i, 0]], 
                    [z_f_embedded[i, 1], z_u_embedded[i, 1]], 
                    'gray', alpha=0.2, linewidth=0.5, zorder=1)
        
        scatter_f = ax.scatter(z_f_embedded[:, 0], z_f_embedded[:, 1], 
                              c='#1f77b4', alpha=0.7, s=60, 
                              label=r'$z_f$ (Backbone)', 
                              edgecolors='white', linewidths=0.5, zorder=2)
        scatter_u = ax.scatter(z_u_embedded[:, 0], z_u_embedded[:, 1], 
                              c='#d62728', alpha=0.7, s=60, 
                              label=r'$z_u$ (Solution)', 
                              marker='s', edgecolors='white', linewidths=0.5, zorder=2)
        
        ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
        ax.set_title('z_f ↔ z_u Alignment', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right', fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'zf_zu_alignment.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'zf_zu_alignment.pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / 'zf_zu_alignment.png'}")
    
    print(f"\n{'='*60}")
    print("Latent Analysis Complete!")
    print(f"Results saved to: {output_dir}/")
    print(f"Also saved to: {latent_reps_dir}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Latent Space Analysis with Parameter Coloring')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--max_samples', type=int, default=150, help='Max samples per dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Latent Space Analysis - Physical Parameter Coloring")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    run_experiment(
        epochs=args.epochs,
        max_samples=args.max_samples,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
