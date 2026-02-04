#!/usr/bin/env python3
"""
Experiment 2: Latent Space Analysis and Spectral Error Profiling

This experiment provides:
1. t-SNE visualization of latent representations (z_f vs z_u alignment)
2. Latent space evolution across training epochs
3. Spectral error profiling (fRMSE across frequency bands)
4. Comparison of FNO vs LRR-FNO representation quality
"""

import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
from datetime import datetime
from sklearn.manifold import TSNE
from collections import defaultdict

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


def compute_relative_l2(pred, target):
    diff = pred - target
    return torch.norm(diff) / torch.norm(target)


def compute_spectral_error(pred, target, resolution=64):
    """
    Compute frequency-band RMSE (fRMSE).
    
    Returns errors for low, mid, and high frequency bands.
    """
    # FFT of prediction and target
    pred_fft = torch.fft.fft2(pred)
    target_fft = torch.fft.fft2(target)
    
    # Create frequency masks
    freq_x = torch.fft.fftfreq(resolution, device=pred.device)
    freq_y = torch.fft.fftfreq(resolution, device=pred.device)
    freq_grid = torch.sqrt(freq_x[None, :]**2 + freq_y[:, None]**2)
    
    # Define frequency bands
    low_mask = freq_grid < 0.1
    mid_mask = (freq_grid >= 0.1) & (freq_grid < 0.3)
    high_mask = freq_grid >= 0.3
    
    # Compute errors per band
    def band_error(mask):
        diff = (pred_fft - target_fft) * mask
        return torch.sqrt(torch.mean(torch.abs(diff)**2)).item()
    
    return {
        'low': band_error(low_mask),
        'mid': band_error(mid_mask),
        'high': band_error(high_mask)
    }


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
                # For vanilla FNO, use flattened output features as proxy
                pred = model(c)
                # Global average pool as latent proxy
                z_proxy = pred.mean(dim=(2, 3)).cpu().numpy()
                z_f_list.append(z_proxy)
    
    z_f = np.concatenate(z_f_list, axis=0)
    z_u = np.concatenate(z_u_list, axis=0) if z_u_list else None
    
    return z_f, z_u


def plot_tsne_evolution(latents_history, save_path, title="Latent Space Evolution"):
    """
    Plot t-SNE visualization of latent space evolution across epochs.
    
    latents_history: dict with keys as epoch numbers, values as (z_f, z_u) tuples
    """
    epochs = sorted(latents_history.keys())
    n_epochs = len(epochs)
    
    fig, axes = plt.subplots(1, n_epochs, figsize=(5*n_epochs, 5))
    if n_epochs == 1:
        axes = [axes]
    
    for idx, epoch in enumerate(epochs):
        z_f, z_u = latents_history[epoch]
        
        if z_u is not None:
            # Combine for joint t-SNE
            combined = np.vstack([z_f, z_u])
            labels = ['z_f (backbone)']*len(z_f) + ['z_u (solution)']*len(z_u)
            
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined)-1))
            embedded = tsne.fit_transform(combined)
            
            n_f = len(z_f)
            
            ax = axes[idx]
            ax.scatter(embedded[:n_f, 0], embedded[:n_f, 1], 
                      c='blue', alpha=0.6, label='z_f (backbone)', s=30)
            ax.scatter(embedded[n_f:, 0], embedded[n_f:, 1], 
                      c='red', alpha=0.6, label='z_u (solution)', s=30)
            ax.set_title(f'Epoch {epoch}')
            ax.legend()
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax = axes[idx]
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(z_f)-1))
            embedded = tsne.fit_transform(z_f)
            ax.scatter(embedded[:, 0], embedded[:, 1], c='blue', alpha=0.6, s=30)
            ax.set_title(f'Epoch {epoch}')
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_latent_reps(z_f, z_u, save_path, title="Latent Representation", dataset_name=""):
    """
    Plot publication-quality latent representation visualization.
    
    Creates a single t-SNE plot showing z_f and z_u alignment with:
    - Proper styling for papers
    - Clear legend and labels
    - Connection lines between corresponding pairs
    """
    import matplotlib
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 12
    
    # Combine for joint t-SNE
    combined = np.vstack([z_f, z_u])
    perplexity = min(30, len(combined) // 2 - 1)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
    embedded = tsne.fit_transform(combined)
    
    n = len(z_f)
    z_f_embedded = embedded[:n]
    z_u_embedded = embedded[n:]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot connection lines (faint) between corresponding pairs
    for i in range(n):
        ax.plot([z_f_embedded[i, 0], z_u_embedded[i, 0]], 
                [z_f_embedded[i, 1], z_u_embedded[i, 1]], 
                'gray', alpha=0.2, linewidth=0.5, zorder=1)
    
    # Plot points
    scatter_f = ax.scatter(z_f_embedded[:, 0], z_f_embedded[:, 1], 
                          c='#1f77b4', alpha=0.7, s=60, 
                          label=r'$z_f$ (Backbone Features)', 
                          edgecolors='white', linewidths=0.5, zorder=2)
    scatter_u = ax.scatter(z_u_embedded[:, 0], z_u_embedded[:, 1], 
                          c='#d62728', alpha=0.7, s=60, 
                          label=r'$z_u$ (Solution Encoding)', 
                          marker='s', edgecolors='white', linewidths=0.5, zorder=2)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax.set_title(f'{title}\n{dataset_name}', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Remove axis ticks but keep border
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    # Add text annotation
    cosine_sim = np.mean(np.sum(
        (z_f / (np.linalg.norm(z_f, axis=1, keepdims=True) + 1e-8)) *
        (z_u / (np.linalg.norm(z_u, axis=1, keepdims=True) + 1e-8)), axis=1))
    ax.text(0.02, 0.02, f'Cosine Similarity: {cosine_sim:.3f}', 
            transform=ax.transAxes, fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')  # Also save as PDF
    plt.close()
    print(f"Saved: {save_path}")
    print(f"Saved: {save_path.replace('.png', '.pdf')}")


def plot_spectral_comparison(fno_spectral, lrr_spectral, save_path):
    """Plot spectral error comparison between FNO and LRR-FNO."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bands = ['Low', 'Mid', 'High']
    x = np.arange(len(bands))
    width = 0.35
    
    fno_vals = [fno_spectral['low'], fno_spectral['mid'], fno_spectral['high']]
    lrr_vals = [lrr_spectral['low'], lrr_spectral['mid'], lrr_spectral['high']]
    
    bars1 = ax.bar(x - width/2, fno_vals, width, label='FNO', color='steelblue')
    bars2 = ax.bar(x + width/2, lrr_vals, width, label='LRR-FNO', color='coral')
    
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('RMSE')
    ax.set_title('Spectral Error Profiling (fRMSE)')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend()
    
    # Add improvement percentages
    for i, (f, l) in enumerate(zip(fno_vals, lrr_vals)):
        improvement = (f - l) / f * 100
        ax.annotate(f'{improvement:+.1f}%', 
                   xy=(x[i] + width/2, l), 
                   ha='center', va='bottom', fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_latent_alignment_metrics(alignment_history, save_path):
    """Plot latent alignment metrics over training."""
    epochs = list(alignment_history.keys())
    cosine_sims = [alignment_history[e]['cosine_sim'] for e in epochs]
    l2_distances = [alignment_history[e]['l2_distance'] for e in epochs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(epochs, cosine_sims, 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_title('z_f ↔ z_u Alignment (Cosine Similarity)')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    ax2.plot(epochs, l2_distances, 'r-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L2 Distance')
    ax2.set_title('z_f ↔ z_u Distance (L2)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def compute_alignment_metrics(z_f, z_u):
    """Compute alignment metrics between z_f and z_u."""
    # Normalize
    z_f_norm = z_f / (np.linalg.norm(z_f, axis=1, keepdims=True) + 1e-8)
    z_u_norm = z_u / (np.linalg.norm(z_u, axis=1, keepdims=True) + 1e-8)
    
    # Cosine similarity (average)
    cosine_sim = np.mean(np.sum(z_f_norm * z_u_norm, axis=1))
    
    # L2 distance (average)
    l2_distance = np.mean(np.linalg.norm(z_f - z_u, axis=1))
    
    return {
        'cosine_sim': cosine_sim,
        'l2_distance': l2_distance
    }


def train_with_tracking(model, loss_fn, train_loader, test_loader, device, 
                        epochs, track_epochs=[1, 10, 25, 50]):
    """Train LRR-FNO with 2-stage protocol while tracking latent evolution."""
    latents_history = {}
    alignment_history = {}
    
    # 2-stage split: 73% stage 1, 27% stage 2
    stage1_epochs = int(0.73 * epochs)
    stage2_epochs = epochs - stage1_epochs
    
    # Stage 1: NCE + MSE (combined) 
    print(f"  Stage 1: Combined Optimization [{stage1_epochs} epochs]")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage1_epochs)
    
    current_epoch = 0
    for epoch in range(1, stage1_epochs + 1):
        current_epoch = epoch
        model.train()
        epoch_loss = 0.0
        
        for c, u in train_loader:
            c, u = c.to(device), u.to(device)
            
            optimizer.zero_grad()
            output = model(c, u, return_latents=True)
            pred = output['prediction']
            z_f = output.get('z_f')
            z_u = output.get('z_u')
            loss_dict = loss_fn(pred, u, z_f, z_u, stage=2)  # Stage 2 = NCE + MSE
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Track at specific epochs
        if epoch in track_epochs:
            z_f_np, z_u_np = extract_latents(model, test_loader, device, is_lrr=True)
            latents_history[epoch] = (z_f_np, z_u_np)
            if z_u_np is not None:
                alignment_history[epoch] = compute_alignment_metrics(z_f_np, z_u_np)
        
        if epoch % 10 == 0 or epoch == 1:
            nce = loss_dict.get('nce', 0)
            mse = loss_dict.get('mse', 0)
            print(f"    Epoch {epoch}/{stage1_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, NCE: {nce:.4f}, MSE: {mse:.6f}")
    
    # Stage 2: MSE only (distillation)
    print(f"  Stage 2: Autonomous Distillation [{stage2_epochs} epochs]")
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage2_epochs)
    
    for epoch in range(1, stage2_epochs + 1):
        current_epoch = stage1_epochs + epoch
        model.train()
        epoch_loss = 0.0
        
        for c, u in train_loader:
            c, u = c.to(device), u.to(device)
            
            optimizer.zero_grad()
            output = model(c, u, return_latents=True)
            pred = output['prediction']
            loss_dict = loss_fn(pred, u, None, None, stage=3)  # Stage 3 = MSE only
            loss = loss_dict['total']
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Track at final epoch
        if current_epoch in track_epochs or epoch == stage2_epochs:
            z_f_np, z_u_np = extract_latents(model, test_loader, device, is_lrr=True)
            latents_history[current_epoch] = (z_f_np, z_u_np)
            if z_u_np is not None:
                alignment_history[current_epoch] = compute_alignment_metrics(z_f_np, z_u_np)
        
        if epoch % 5 == 0 or epoch == 1:
            mse = loss_dict.get('mse', 0)
            print(f"    Epoch {epoch}/{stage2_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, MSE: {mse:.6f}")
    
    return latents_history, alignment_history


def run_experiment(nc_path, epochs=50, max_samples=150, seed=42):
    """Run latent analysis experiment."""
    set_seed(seed)
    device = get_device()
    
    dataset_name = Path(nc_path).stem
    print(f"\n{'='*60}")
    print(f"Experiment 2: Latent Analysis - {dataset_name}")
    print(f"{'='*60}")
    
    # Create output directory
    output_dir = Path('results/exp2_latent_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, test_loader, info = get_gaot_grid_loaders(
        nc_path, batch_size=16, resolution=64, max_samples=max_samples
    )
    
    in_channels = info['in_channels']
    out_channels = info['out_channels']
    
    print(f"Dataset: {dataset_name}")
    print(f"Samples: {info['n_train']} train, {info['n_test']} test")
    
    # Track epochs for visualization
    track_epochs = [1, 5, 10, 20, 30, 40, 50][:epochs//10 + 2] if epochs >= 10 else [1, epochs]
    
    # --- Train LRR-FNO with tracking ---
    print(f"\n--- Training LRR-FNO with Latent Tracking ({epochs} epochs) ---")
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
    
    latents_history, alignment_history = train_with_tracking(
        lrr_fno, lrr_loss_fn, train_loader, test_loader, device, 
        epochs, track_epochs
    )
    
    # --- Train Vanilla FNO for comparison ---
    print(f"\n--- Training Vanilla FNO ({epochs} epochs) ---")
    set_seed(seed)
    
    fno = FNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        modes1=12, modes2=12,
        width=32,
        num_layers=4
    ).to(device)
    
    fno_optimizer = optim.Adam(fno.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    
    for epoch in range(1, epochs + 1):
        fno.train()
        for c, u in train_loader:
            c, u = c.to(device), u.to(device)
            fno_optimizer.zero_grad()
            pred = fno(c)
            loss = mse_loss(pred, u)
            loss.backward()
            fno_optimizer.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"  FNO Epoch {epoch}/{epochs}")
    
    # --- Compute spectral errors ---
    print("\n--- Computing Spectral Errors ---")
    
    fno.eval()
    lrr_fno.eval()
    
    fno_spectral = {'low': 0, 'mid': 0, 'high': 0}
    lrr_spectral = {'low': 0, 'mid': 0, 'high': 0}
    n_samples = 0
    
    with torch.no_grad():
        for c, u in test_loader:
            c, u = c.to(device), u.to(device)
            
            fno_pred = fno(c)
            lrr_pred = lrr_fno(c)['prediction']
            
            for i in range(c.size(0)):
                fno_err = compute_spectral_error(fno_pred[i, 0], u[i, 0])
                lrr_err = compute_spectral_error(lrr_pred[i, 0], u[i, 0])
                
                for k in ['low', 'mid', 'high']:
                    fno_spectral[k] += fno_err[k]
                    lrr_spectral[k] += lrr_err[k]
                n_samples += 1
    
    for k in ['low', 'mid', 'high']:
        fno_spectral[k] /= n_samples
        lrr_spectral[k] /= n_samples
    
    print(f"FNO Spectral:  Low={fno_spectral['low']:.4f}, Mid={fno_spectral['mid']:.4f}, High={fno_spectral['high']:.4f}")
    print(f"LRR Spectral:  Low={lrr_spectral['low']:.4f}, Mid={lrr_spectral['mid']:.4f}, High={lrr_spectral['high']:.4f}")
    
    # --- Generate Plots ---
    print("\n--- Generating Visualizations ---")
    
    # 1. t-SNE evolution
    plot_tsne_evolution(
        latents_history,
        output_dir / f'{dataset_name}_tsne_evolution.png',
        title=f'Latent Space Evolution: {dataset_name}'
    )
    
    # 2. Alignment metrics
    if alignment_history:
        plot_latent_alignment_metrics(
            alignment_history,
            output_dir / f'{dataset_name}_alignment_metrics.png'
        )
    
    # 3. Spectral comparison
    plot_spectral_comparison(
        fno_spectral, lrr_spectral,
        output_dir / f'{dataset_name}_spectral_comparison.png'
    )
    
    # 4. Publication-quality latent representation plot
    final_epoch = max(latents_history.keys())
    final_z_f, final_z_u = latents_history[final_epoch]
    if final_z_u is not None:
        # Save to both exp2 results and latent_reps/figures
        latent_reps_dir = Path('latent_reps/figures')
        latent_reps_dir.mkdir(parents=True, exist_ok=True)
        
        plot_latent_reps(
            final_z_f, final_z_u,
            str(output_dir / f'{dataset_name}_latent_reps.png'),
            title='Latent Space Alignment',
            dataset_name=f'{dataset_name} (Epoch {final_epoch})'
        )
        # Also save to latent_reps folder
        plot_latent_reps(
            final_z_f, final_z_u,
            str(latent_reps_dir / f'{dataset_name}_latent_reps.png'),
            title='LRR-FNO Latent Alignment',
            dataset_name=dataset_name
        )
    
    # --- Final Evaluation ---
    fno_errors = []
    lrr_errors = []
    
    with torch.no_grad():
        for c, u in test_loader:
            c, u = c.to(device), u.to(device)
            
            fno_pred = fno(c)
            lrr_pred = lrr_fno(c)['prediction']
            
            for i in range(c.size(0)):
                fno_errors.append(compute_relative_l2(fno_pred[i], u[i]).item())
                lrr_errors.append(compute_relative_l2(lrr_pred[i], u[i]).item())
    
    fno_error = np.mean(fno_errors)
    lrr_error = np.mean(lrr_errors)
    improvement = (fno_error - lrr_error) / fno_error * 100
    
    print(f"\n--- Results: {dataset_name} ---")
    print(f"FNO Rel L2:     {fno_error:.6f}")
    print(f"LRR-FNO Rel L2: {lrr_error:.6f}")
    print(f"Improvement:    {improvement:+.2f}%")
    
    return {
        'dataset': dataset_name,
        'fno_error': fno_error,
        'lrr_error': lrr_error,
        'improvement': improvement,
        'fno_spectral': fno_spectral,
        'lrr_spectral': lrr_spectral,
        'alignment_history': alignment_history
    }


def main():
    parser = argparse.ArgumentParser(description='Exp2: Latent Analysis and Spectral Profiling')
    parser.add_argument('--dataset', type=str, default='dataset/Circle.nc')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_samples', type=int, default=150)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    print("="*60)
    print("LRR-FNO Experiment 2: Latent Analysis & Spectral Profiling")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*60)
    
    result = run_experiment(
        args.dataset,
        epochs=args.epochs,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    print("\n" + "="*60)
    print("Experiment 2 Complete!")
    print(f"Visualizations saved to: results/exp2_latent_analysis/")
    print("="*60)


if __name__ == '__main__':
    main()
