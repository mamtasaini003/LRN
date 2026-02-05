#!/usr/bin/env python3
"""
Latent Space Analysis with Physical Parameter Coloring - Darcy Flow.

This script creates t-SNE visualizations of latent representations
colored by physical parameters, similar to rb_tsne_rayleigh.pdf.

Uses Darcy flow data with varying permeability patterns.

Usage:
    python3 examples/latent_analysis_darcy.py --epochs 50 --num_samples 500
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.manifold import TSNE

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.components.fno import FNO2d
from models.lrr.model import LRRFNO2d
from losses.infonce import LRNLoss


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_darcy_data_with_params(num_samples, resolution=64):
    """
    Generate Darcy flow data with varying permeability characteristics.
    
    Returns data with TWO physical parameters (like Rayleigh and Prandtl):
    - n_blobs: Number of blobs (complexity) - analogous to Rayleigh number
    - avg_permeability: Average permeability value - analogous to Prandtl number
    """
    x = torch.linspace(0, 1, resolution)
    y = torch.linspace(0, 1, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    f_list = []  # Input (coefficient field)
    u_list = []  # Output (solution field)
    n_blobs_list = []  # Parameter 1: Number of blobs (complexity)
    avg_perm_list = []  # Parameter 2: Average permeability
    
    print(f"Generating {num_samples} Darcy samples with varying parameters...")
    
    for i in range(num_samples):
        # Parameter 1: Vary number of blobs (complexity) - like Rayleigh number
        n_blobs = np.random.randint(2, 10)
        
        # Parameter 2: Vary average permeability - like Prandtl number  
        base_permeability = np.random.uniform(0.3, 1.5)
        
        a = torch.ones(resolution, resolution) * base_permeability
        
        for _ in range(n_blobs):
            cx, cy = np.random.rand(2)
            r = np.random.uniform(0.08, 0.25)
            amplitude = np.random.uniform(0.5, 2.5)
            
            mask = ((X - cx)**2 + (Y - cy)**2) < r**2
            a[mask] = amplitude
        
        # Compute average permeability
        avg_perm = a.mean().item()
        
        # Simplified solution approximation using smoothing
        u = a.clone()
        # Multi-scale smoothing to simulate diffusion
        for _ in range(3):
            u = torch.nn.functional.avg_pool2d(
                u.unsqueeze(0).unsqueeze(0), 
                kernel_size=5, stride=1, padding=2
            ).squeeze()
        
        # Add slight variation
        u = u + 0.05 * torch.randn_like(u)
        
        f_list.append(a)
        u_list.append(u)
        n_blobs_list.append(n_blobs)
        avg_perm_list.append(avg_perm)
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
    
    f = torch.stack(f_list).unsqueeze(1)  # [N, 1, H, W]
    u = torch.stack(u_list).unsqueeze(1)  # [N, 1, H, W]
    params = {
        'n_blobs': np.array(n_blobs_list),
        'avg_permeability': np.array(avg_perm_list)
    }
    
    return f, u, params


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


def plot_tsne_publication(z, params, param_name, save_path, title="Latent Space"):
    """
    Plot publication-quality t-SNE visualization with enhanced aesthetics.
    Matches the style of rb_tsne_prandtl.pdf with blue gradients and visible scales.
    """
    import matplotlib
    # Use a modern scientific look
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.serif'] = ['STIXGeneral', 'DejaVu Serif', 'Times New Roman']
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['axes.linewidth'] = 1.0 # Thicker axes
    matplotlib.rcParams['axes.edgecolor'] = '#000000'
    
    perplexity = min(50, len(z) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity), 
                max_iter=1000, learning_rate='auto', init='pca')
    embedded = tsne.fit_transform(z)
    
    # Use a blue gradient colormap as requested
    # 'GnBu' or 'Blues' work well. 'YlGnBu' provides more range.
    cmap_name = 'YlGnBu' 
    
    fig, ax = plt.subplots(figsize=(6.5, 5))
    
    # Add a visible grid BEFORE scatter to keep it in background
    ax.grid(True, linestyle='--', color='#cccccc', linewidth=0.5, zorder=0, alpha=0.6)
    
    # Plot points with high-end styling
    scatter = ax.scatter(
        embedded[:, 0], embedded[:, 1], 
        c=params, 
        cmap=cmap_name,
        s=40, 
        alpha=0.9,
        edgecolors='#333333', # Dark edge for contrast
        linewidths=0.5,
        zorder=2,
        rasterized=True
    )
    
    # Highlight the axes with visible scales (ticks)
    ax.set_xticks(np.linspace(embedded[:,0].min(), embedded[:,0].max(), 5))
    ax.set_yticks(np.linspace(embedded[:,1].min(), embedded[:,1].max(), 5))
    
    # Format tick labels to be minimal but present
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
    
    ax.tick_params(axis='both', which='major', labelsize=9, length=4, width=1.0)
    
    # Label dimensions as requested
    ax.set_xlabel('t-SNE Dimension 1', fontsize=10, fontweight='medium', labelpad=10)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=10, fontweight='medium', labelpad=10)
    
    # Colorbar styling - more integrated and elegant
    # Put colorbar on top or right. Paper usually has it on right.
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, pad=0.03, aspect=20)
    cbar.outline.set_linewidth(0.8)
    cbar.set_label(param_name, fontsize=11, labelpad=12, fontweight='medium')
    cbar.ax.tick_params(labelsize=9)
    
    # Title removed for paper publication
    # ax.set_title(title, fontsize=13, pad=18, fontweight='bold', family='serif')
    
    # Layout adjustment
    plt.tight_layout()
    
    # Save with high resolution
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', transparent=False, facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")
    print(f"Saved: {save_path.replace('.png', '.pdf')}")


def train_lrr_fno(model, loss_fn, train_loader, device, epochs):
    """Train LRR-FNO with 2-stage protocol with improved convergence."""
    stage1_epochs = int(0.73 * epochs)
    stage2_epochs = epochs - stage1_epochs
    
    print(f"  Stage 1: Combined Optimization [{stage1_epochs} epochs]")
    # Higher LR and weight decay to prevent stagnation
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs, eta_min=1e-5)
    
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
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"    Epoch {epoch}/{stage1_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, LR: {lr:.2e}")
    
    print(f"  Stage 2: Autonomous Distillation [{stage2_epochs} epochs]")
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-6)
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        scheduler.step()
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch}/{stage2_epochs}")


def run_experiment(num_samples=500, epochs=50, seed=42):
    """Run latent analysis experiment with Darcy data."""
    set_seed(seed)
    device = get_device()
    
    print(f"\n{'='*60}")
    print("Latent Space Analysis - Darcy Flow")
    print(f"{'='*60}")
    
    output_dir = Path('results/latent_analysis_darcy')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    latent_reps_dir = Path('latent_reps/figures')
    latent_reps_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate Darcy data with TWO varying parameters
    f, u, params = generate_darcy_data_with_params(num_samples, resolution=64)
    
    # Split train/test
    n_train = int(0.8 * num_samples)
    f_train, f_test = f[:n_train], f[n_train:]
    u_train, u_test = u[:n_train], u[n_train:]
    
    # Split parameters
    params_train = {k: v[:n_train] for k, v in params.items()}
    params_test = {k: v[n_train:] for k, v in params.items()}
    
    print(f"\n  Training Set: {len(f_train)} samples")
    print(f"  Test Set: {len(f_test)} samples")
    print(f"  Parameter 1 (n_blobs) range: {params['n_blobs'].min()} - {params['n_blobs'].max()}")
    print(f"  Parameter 2 (avg_permeability) range: {params['avg_permeability'].min():.2f} - {params['avg_permeability'].max():.2f}")
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(f_train, u_train)
    test_dataset = torch.utils.data.TensorDataset(f_test, u_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train LRR-FNO
    print(f"\n--- Training LRR-FNO ({epochs} epochs) ---")
    
    lrr_fno = LRRFNO2d(
        in_channels=1,
        out_channels=1,
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
    
    # Generate TWO publication-quality plots (like Rayleigh and Prandtl)
    print("\n--- Generating Publication-Quality Visualizations ---")
    
    # Plot 1: t-SNE colored by n_blobs (like Rayleigh number)
    plot_tsne_publication(
        z_f, params_test['n_blobs'], 
        param_name='Number of Blobs',
        save_path=str(output_dir / 'darcy_tsne_nblobs.png'),
        title='Latent Space by Number of Blobs'
    )
    
    # Also save to latent_reps/figures
    plot_tsne_publication(
        z_f, params_test['n_blobs'], 
        param_name='Number of Blobs',
        save_path=str(latent_reps_dir / 'darcy_tsne_nblobs.png'),
        title='Latent Space by Number of Blobs'
    )
    
    # Plot 2: t-SNE colored by avg_permeability (like Prandtl number)
    plot_tsne_publication(
        z_f, params_test['avg_permeability'], 
        param_name='Avg. Permeability',
        save_path=str(output_dir / 'darcy_tsne_permeability.png'),
        title='Latent Space by Avg. Permeability'
    )
    
    # Also save to latent_reps/figures
    plot_tsne_publication(
        z_f, params_test['avg_permeability'], 
        param_name='Avg. Permeability',
        save_path=str(latent_reps_dir / 'darcy_tsne_permeability.png'),
        title='Latent Space by Avg. Permeability'
    )
    
    print(f"\n{'='*60}")
    print("Latent Analysis Complete!")
    print(f"Generated TWO t-SNE plots (like Rayleigh and Prandtl):")
    print(f"  - darcy_tsne_nblobs.pdf (analogous to rb_tsne_rayleigh.pdf)")
    print(f"  - darcy_tsne_permeability.pdf (analogous to rb_tsne_prandtl.pdf)")
    print(f"Results saved to: {output_dir}/")
    print(f"Also saved to: {latent_reps_dir}/")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Latent Space Analysis - Darcy Flow')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Latent Space Analysis - Darcy Flow")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    run_experiment(
        num_samples=args.num_samples,
        epochs=args.epochs,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
