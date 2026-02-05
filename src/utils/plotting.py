#!/usr/bin/env python3
"""
Publication-Quality Plotting Utilities for LRR Experiments

This module provides consistent, research paper quality visualizations
without titles (suitable for latex captions), high DPI, and PDF export.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# Set publication-quality defaults
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 12
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['axes.linewidth'] = 0.8
matplotlib.rcParams['grid.linewidth'] = 0.5
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['pdf.fonttype'] = 42  # Ensure fonts are editable in PDF
matplotlib.rcParams['ps.fonttype'] = 42

# Consistent color palette (colorblind-friendly)
COLORS = {
    'fno': '#0072B2',        # Blue
    'lrr': '#009E73',        # Green  
    'improvement': '#E69F00', # Orange
    'error': '#D55E00',       # Red-Orange
    'neutral': '#56B4E9',     # Light blue
    'accent': '#CC79A7',      # Pink
}

# Marker styles
MARKERS = {
    'fno': 'o',
    'lrr': 's',
}


def setup_figure(figsize=(6, 4.5), constrained_layout=True):
    """Create a publication-quality figure."""
    fig = plt.figure(figsize=figsize, constrained_layout=constrained_layout)
    return fig


def save_figure(fig, save_path, formats=['png', 'pdf']):
    """Save figure in multiple formats for publication."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        output_path = save_path.with_suffix(f'.{fmt}')
        fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  Saved: {save_path.stem} ({', '.join(formats)})")


def plot_training_curves(fno_losses, lrr_losses, save_path):
    """
    Plot training loss curves for FNO and LRR-FNO.
    Research paper quality without title.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    epochs = range(1, len(fno_losses) + 1)
    
    ax.semilogy(epochs, fno_losses, '-', color=COLORS['fno'], 
                linewidth=1.8, label='FNO', marker='o', markersize=3, markevery=max(1, len(epochs)//10))
    ax.semilogy(epochs, lrr_losses[:len(fno_losses)], '-', color=COLORS['lrr'], 
                linewidth=1.8, label='LRR-FNO', marker='s', markersize=3, markevery=max(1, len(epochs)//10))
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss (MSE)')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_figure(fig, save_path)


def plot_error_comparison_bar(datasets, fno_errors, lrr_errors, save_path, short_labels=None):
    """
    Plot grouped bar chart comparing FNO vs LRR-FNO errors.
    Research paper quality without title.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fno_errors, width, label='FNO', 
                   color=COLORS['fno'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, lrr_errors, width, label='LRR-FNO', 
                   color=COLORS['lrr'], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Relative L2 Error')
    ax.set_xticks(x)
    labels = short_labels if short_labels else datasets
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 2), textcoords='offset points', ha='center', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 2), textcoords='offset points', ha='center', fontsize=8)
    
    save_figure(fig, save_path)


def plot_improvement_bar(datasets, improvements, save_path, short_labels=None):
    """
    Plot improvement percentage bar chart.
    Research paper quality without title.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    
    x = np.arange(len(datasets))
    colors = [COLORS['improvement'] if imp > 0 else COLORS['error'] for imp in improvements]
    
    bars = ax.bar(x, improvements, color=colors, edgecolor='black', linewidth=0.5, width=0.6)
    
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Improvement (%)')
    ax.set_xticks(x)
    labels = short_labels if short_labels else datasets
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add value labels
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height >= 0 else 'top'
        offset = 2 if height >= 0 else -2
        ax.annotate(f'{imp:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, offset), textcoords='offset points', ha='center', va=va, 
                   fontsize=9, fontweight='bold')
    
    # Average line
    avg = np.mean(improvements)
    ax.axhline(y=avg, color=COLORS['accent'], linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(len(datasets) - 0.3, avg + 1, f'Avg: {avg:.1f}%', fontsize=9, color=COLORS['accent'])
    
    save_figure(fig, save_path)


def plot_predictions_grid(inputs, ground_truths, fno_preds, lrr_preds, save_path, num_samples=3):
    """
    Plot prediction comparison grid: Input, GT, FNO, LRR-FNO, FNO Error, LRR Error.
    Research paper quality without titles on individual panels.
    """
    num_samples = min(num_samples, len(inputs))
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(14, 2.5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    col_labels = ['Input $f$', 'Ground Truth $u$', 'FNO Prediction', 
                  'LRR-FNO Prediction', 'FNO Error', 'LRR-FNO Error']
    
    for i in range(num_samples):
        inp = inputs[i].squeeze()
        gt = ground_truths[i].squeeze()
        fno = fno_preds[i].squeeze()
        lrr = lrr_preds[i].squeeze()
        
        fno_err = np.abs(fno - gt)
        lrr_err = np.abs(lrr - gt)
        
        vmin, vmax = gt.min(), gt.max()
        err_max = max(fno_err.max(), lrr_err.max())
        
        # Input
        im0 = axes[i, 0].imshow(inp, cmap='viridis', aspect='equal')
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)
        
        # Ground Truth
        im1 = axes[i, 1].imshow(gt, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)
        
        # FNO Prediction
        im2 = axes[i, 2].imshow(fno, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046, pad=0.04)
        
        # LRR-FNO Prediction
        im3 = axes[i, 3].imshow(lrr, cmap='RdBu_r', vmin=vmin, vmax=vmax, aspect='equal')
        plt.colorbar(im3, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        # FNO Error
        im4 = axes[i, 4].imshow(fno_err, cmap='hot', vmin=0, vmax=err_max, aspect='equal')
        plt.colorbar(im4, ax=axes[i, 4], fraction=0.046, pad=0.04)
        
        # LRR Error
        im5 = axes[i, 5].imshow(lrr_err, cmap='hot', vmin=0, vmax=err_max, aspect='equal')
        plt.colorbar(im5, ax=axes[i, 5], fraction=0.046, pad=0.04)
        
        for j in range(6):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(col_labels[j], fontsize=10, pad=5)
    
    save_figure(fig, save_path)


def plot_error_distribution(fno_errors, lrr_errors, save_path):
    """
    Plot error distribution histogram.
    Research paper quality without title.
    """
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    all_errors = np.concatenate([fno_errors, lrr_errors])
    bins = np.linspace(all_errors.min(), all_errors.max(), 25)
    
    ax.hist(fno_errors, bins=bins, alpha=0.6, label='FNO', color=COLORS['fno'], edgecolor='black', linewidth=0.5)
    ax.hist(lrr_errors, bins=bins, alpha=0.6, label='LRR-FNO', color=COLORS['lrr'], edgecolor='black', linewidth=0.5)
    
    # Mean lines
    ax.axvline(np.mean(fno_errors), color=COLORS['fno'], linestyle='--', linewidth=2, alpha=0.9)
    ax.axvline(np.mean(lrr_errors), color=COLORS['lrr'], linestyle='--', linewidth=2, alpha=0.9)
    
    ax.set_xlabel('Relative L2 Error')
    ax.set_ylabel('Frequency')
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_figure(fig, save_path)


def plot_noise_robustness(noise_levels, fno_errors, lrr_errors, save_path):
    """
    Plot noise robustness curve.
    Research paper quality without title.
    """
    fig, ax = plt.subplots(figsize=(5.5, 4))
    
    ax.plot(noise_levels, fno_errors, '-o', color=COLORS['fno'], linewidth=2, 
            markersize=7, label='FNO', markeredgecolor='black', markeredgewidth=0.5)
    ax.plot(noise_levels, lrr_errors, '-s', color=COLORS['lrr'], linewidth=2, 
            markersize=7, label='LRR-FNO', markeredgecolor='black', markeredgewidth=0.5)
    
    ax.set_xlabel('Noise Level ($\\sigma$)')
    ax.set_ylabel('Relative L2 Error')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Fill between to show improvement region
    ax.fill_between(noise_levels, lrr_errors, fno_errors, alpha=0.2, color=COLORS['improvement'],
                    where=[l < f for l, f in zip(lrr_errors, fno_errors)])
    
    save_figure(fig, save_path)


def plot_super_resolution(train_res, test_res, fno_train, fno_test, lrr_train, lrr_test, save_path):
    """
    Plot super-resolution results.
    Research paper quality without title.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    
    x = np.arange(2)
    width = 0.35
    
    fno_vals = [fno_train, fno_test]
    lrr_vals = [lrr_train, lrr_test]
    
    bars1 = ax.bar(x - width/2, fno_vals, width, label='FNO', 
                   color=COLORS['fno'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, lrr_vals, width, label='LRR-FNO', 
                   color=COLORS['lrr'], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Test Resolution')
    ax.set_ylabel('Relative L2 Error')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{train_res}×{train_res}\n(In-Dist)', f'{test_res}×{test_res}\n(Zero-Shot)'])
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 2), textcoords='offset points', ha='center', fontsize=9)
    
    save_figure(fig, save_path)


def plot_ood_generalization(train_domain, test_domains, fno_errors, lrr_errors, save_path):
    """
    Plot OOD generalization results.
    Research paper quality without title.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    x = np.arange(len(test_domains))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, fno_errors, width, label='FNO', 
                   color=COLORS['fno'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, lrr_errors, width, label='LRR-FNO', 
                   color=COLORS['lrr'], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Test Domain')
    ax.set_ylabel('Relative L2 Error')
    ax.set_xticks(x)
    ax.set_xticklabels(test_domains, rotation=25, ha='right')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Improvement annotations
    for i, (f, l) in enumerate(zip(fno_errors, lrr_errors)):
        imp = (f - l) / f * 100
        color = COLORS['improvement'] if imp > 0 else COLORS['error']
        ax.annotate(f'{imp:+.1f}%', xy=(x[i] + width/2, l),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, color=color)
    
    # Mark in-distribution domain
    id_idx = test_domains.index(train_domain) if train_domain in test_domains else None
    if id_idx is not None:
        ax.axvspan(id_idx - 0.45, id_idx + 0.45, alpha=0.15, color='gray')
        ax.text(id_idx, ax.get_ylim()[1] * 0.95, 'ID', ha='center', fontsize=9, style='italic')
    
    save_figure(fig, save_path)


def plot_spectral_error(fno_spectral, lrr_spectral, save_path):
    """
    Plot frequency-band RMSE (fRMSE) comparison.
    Research paper quality without title.
    """
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    
    bands = ['Low', 'Mid', 'High']
    x = np.arange(len(bands))
    width = 0.35
    
    fno_vals = [fno_spectral['low'], fno_spectral['mid'], fno_spectral['high']]
    lrr_vals = [lrr_spectral['low'], lrr_spectral['mid'], lrr_spectral['high']]
    
    bars1 = ax.bar(x - width/2, fno_vals, width, label='FNO', 
                   color=COLORS['fno'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, lrr_vals, width, label='LRR-FNO', 
                   color=COLORS['lrr'], edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Frequency Band')
    ax.set_ylabel('fRMSE')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend(frameon=True, fancybox=False, edgecolor='black')
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    save_figure(fig, save_path)


def plot_tsne_latent(z_f, z_u, save_path, show_connections=True):
    """
    Plot t-SNE visualization of latent space alignment.
    Research paper quality without title.
    """
    from sklearn.manifold import TSNE
    
    fig, ax = plt.subplots(figsize=(5, 4.5))
    
    # Combine for joint t-SNE
    combined = np.vstack([z_f, z_u])
    perplexity = min(30, len(combined) // 2 - 1)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=max(5, perplexity))
    embedded = tsne.fit_transform(combined)
    
    n = len(z_f)
    z_f_emb = embedded[:n]
    z_u_emb = embedded[n:]
    
    # Connection lines
    if show_connections:
        for i in range(n):
            ax.plot([z_f_emb[i, 0], z_u_emb[i, 0]], [z_f_emb[i, 1], z_u_emb[i, 1]], 
                   color='gray', alpha=0.15, linewidth=0.5, zorder=1)
    
    # Points
    ax.scatter(z_f_emb[:, 0], z_f_emb[:, 1], c=COLORS['fno'], alpha=0.7, s=40, 
              label=r'$z_f$ (Backbone)', edgecolors='white', linewidths=0.5, zorder=2)
    ax.scatter(z_u_emb[:, 0], z_u_emb[:, 1], c=COLORS['lrr'], alpha=0.7, s=40, 
              marker='s', label=r'$z_u$ (Solution)', edgecolors='white', linewidths=0.5, zorder=2)
    
    ax.set_xlabel('t-SNE Dimension 1')
    ax.set_ylabel('t-SNE Dimension 2')
    ax.legend(frameon=True, fancybox=False, edgecolor='black', loc='upper right')
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Cosine similarity annotation
    z_f_norm = z_f / (np.linalg.norm(z_f, axis=1, keepdims=True) + 1e-8)
    z_u_norm = z_u / (np.linalg.norm(z_u, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.mean(np.sum(z_f_norm * z_u_norm, axis=1))
    ax.text(0.02, 0.02, f'Cosine Sim: {cos_sim:.3f}', transform=ax.transAxes, fontsize=9,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    
    save_figure(fig, save_path)


def plot_alignment_evolution(alignment_history, save_path):
    """
    Plot latent alignment metrics over training epochs.
    Research paper quality without title.
    """
    epochs = sorted(alignment_history.keys())
    cosine_sims = [alignment_history[e]['cosine_sim'] for e in epochs]
    l2_dists = [alignment_history[e]['l2_distance'] for e in epochs]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    
    # Cosine similarity
    ax1.plot(epochs, cosine_sims, '-o', color=COLORS['fno'], linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # L2 distance
    ax2.plot(epochs, l2_dists, '-s', color=COLORS['lrr'], linewidth=2, markersize=6)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('L2 Distance')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    save_figure(fig, save_path)


def create_comprehensive_summary(results, save_path):
    """
    Create multi-panel summary figure for experiment results.
    Research paper quality.
    """
    if len(results) < 2:
        return
    
    fig = plt.figure(figsize=(12, 8))
    
    datasets = [r['file'] for r in results]
    fno_errors = [r['fno_error'] for r in results]
    lrr_errors = [r['lrr_error'] for r in results]
    improvements = [r['improvement'] for r in results]
    
    # Panel 1: Error comparison
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(datasets))
    width = 0.35
    ax1.bar(x - width/2, fno_errors, width, label='FNO', color=COLORS['fno'], edgecolor='black', linewidth=0.5)
    ax1.bar(x + width/2, lrr_errors, width, label='LRR-FNO', color=COLORS['lrr'], edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Relative L2 Error')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Panel 2: Improvement
    ax2 = fig.add_subplot(2, 2, 2)
    colors = [COLORS['improvement'] if imp > 0 else COLORS['error'] for imp in improvements]
    ax2.bar(x, improvements, color=colors, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.axhline(y=np.mean(improvements), color=COLORS['accent'], linestyle='--', linewidth=1.5)
    ax2.set_ylabel('Improvement (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Panel 3: Scatter
    ax3 = fig.add_subplot(2, 2, 3)
    scatter = ax3.scatter(fno_errors, lrr_errors, c=improvements, cmap='RdYlGn', 
                          s=100, edgecolors='black', linewidth=0.8)
    max_val = max(max(fno_errors), max(lrr_errors)) * 1.1
    ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1)
    ax3.fill_between([0, max_val], [0, 0], [0, max_val], alpha=0.1, color=COLORS['lrr'])
    ax3.set_xlabel('FNO Error')
    ax3.set_ylabel('LRR-FNO Error')
    ax3.set_xlim(0, max_val)
    ax3.set_ylim(0, max_val)
    ax3.set_aspect('equal')
    plt.colorbar(scatter, ax=ax3, label='Improvement %')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Panel 4: Summary stats
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    
    avg_fno = np.mean(fno_errors)
    avg_lrr = np.mean(lrr_errors)
    avg_imp = np.mean(improvements)
    max_imp = max(improvements)
    
    summary = (
        f"Datasets Evaluated: {len(results)}\n\n"
        f"Average FNO Error: {avg_fno:.4f}\n"
        f"Average LRR-FNO Error: {avg_lrr:.4f}\n\n"
        f"Average Improvement: {avg_imp:+.2f}%\n"
        f"Maximum Improvement: {max_imp:+.2f}%\n\n"
        f"LRR-FNO wins: {sum(1 for i in improvements if i > 0)}/{len(improvements)}"
    )
    ax4.text(0.1, 0.5, summary, transform=ax4.transAxes, fontsize=11,
            verticalalignment='center', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='whitesmoke', edgecolor='gray', alpha=0.9))
    
    save_figure(fig, save_path)
