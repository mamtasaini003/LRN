"""
Isolated Darcy Latent Space Visualization Script.
Generates a 3-panel t-SNE plot (Ground Truth, Baseline FNO, LRR-FNO) using 10,000 unique samples.
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Project imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

# Add current scripts dir to path to import custom dataset
sys.path.insert(0, str(Path(__file__).parent))
from pde_datasets_custom import DarcyDataset

from models.components.fno import FNO2d
from models.lrr.model import LRRFNO2d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_viz_dataset(num_samples=10000, resolution=32, seed=4242):
    """Generate a unique dataset of 10,000 samples for visualization."""
    prev_rng_state = torch.get_rng_state()
    prev_np_state = np.random.get_state()
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    logger.info(f"Generating {num_samples} unique Darcy samples (Seed {seed})...")
    
    def unsqueeze_transform(t):
        return t.unsqueeze(0) if t.ndim == 2 else t
        
    viz_dataset = DarcyDataset(
        resolution=resolution, 
        num_samples=num_samples, 
        train=True,
        train_ratio=0.999,
        transform=unsqueeze_transform
    )
    
    # Restore RNG
    torch.set_rng_state(prev_rng_state)
    np.random.set_state(prev_np_state)
    
    return viz_dataset

def analyze_latent_space(fno_model, lrr_model, device, out_dir):
    """
    Perform t-SNE analysis of latent representations.
    3 Panels: Ground Truth (z_u) | Baseline FNO (v_K) | LRR-FNO (v_K).
    """
    from sklearn.manifold import TSNE
    
    # Dataset Generation
    viz_dataset = generate_viz_dataset(num_samples=10000, resolution=32, seed=4242)
    viz_loader = DataLoader(viz_dataset, batch_size=64, shuffle=False)
    
    logger.info(f"Extracting latents for {len(viz_dataset)} samples...")
    fno_model.eval()
    lrr_model.eval()
    
    z_gt_list = []
    z_fno_list = []
    z_lrr_list = []
    z_f_list = []
    labels = []
    
    with torch.no_grad():
        for x, y in viz_loader:
            x, y = x.to(device), y.to(device)
            
            # 1. Ground Truth Latent (z_u) from LRR Solution Encoder
            z_gt = lrr_model.encoder_u(y)
            
            # 2. Baseline FNO Latent (v_K pooled)
            v_K_fno = fno_model.backbone_forward(x)
            z_fno = torch.mean(v_K_fno, dim=(1, 2))
            
            # 3. LRR-FNO Latent (using pooled v_K to match FNO panel)
            v_K_lrr = lrr_model.fno.backbone_forward(x)
            z_lrr_pooled = torch.mean(v_K_lrr, dim=(1, 2))
            
            # Also get z_f for similarity score
            lrr_out = lrr_model(x)
            z_f_lrr = lrr_out['z_f']
            
            z_gt_list.append(z_gt.cpu().numpy())
            z_fno_list.append(z_fno.cpu().numpy())
            z_lrr_list.append(z_lrr_pooled.cpu().numpy()) # Plot pooled features
            z_f_list.append(z_f_lrr.cpu().numpy())        # For metrics
            
            # Label by mean solution field (u)
            mean_sol = torch.mean(y, dim=(1, 2, 3)) if y.ndim == 4 else torch.mean(y, dim=(1, 2))
            labels.append(mean_sol.cpu().numpy())
            
    z_gt_all = np.concatenate(z_gt_list, axis=0)
    z_fno_all = np.concatenate(z_fno_list, axis=0)
    z_lrr_all = np.concatenate(z_lrr_list, axis=0) # Pooled v_K
    z_f_all = np.concatenate(z_f_list, axis=0)     # Projected z_f
    labels_all = np.concatenate(labels, axis=0)

    # Diagnostics
    logger.info("Latent Statistics:")
    logger.info(f"  GT  - Mean: {z_gt_all.mean():.4f}, Std: {z_gt_all.std():.4f}, Range: [{z_gt_all.min():.4f}, {z_gt_all.max():.4f}]")
    logger.info(f"  FNO - Mean: {z_fno_all.mean():.4f}, Std: {z_fno_all.std():.4f}, Range: [{z_fno_all.min():.4f}, {z_fno_all.max():.4f}]")
    logger.info(f"  LRR - Mean: {z_lrr_all.mean():.4f}, Std: {z_lrr_all.std():.4f}, Range: [{z_lrr_all.min():.4f}, {z_lrr_all.max():.4f}]")

    # Alignment Metric
    cos = nn.CosineSimilarity(dim=1)
    # LRR Similarity (using projected features z_f vs z_u)
    z_gt_t = torch.from_numpy(z_gt_all)
    z_f_t = torch.from_numpy(z_f_all)
    
    # Calculate Similarity
    sim_lrr = cos(z_gt_t, z_f_t).mean().item()
    
    logger.info(f"Alignment Metric (Cosine Similarity z_u vs z_f): {sim_lrr:.4f}")
    
    # Optional: Baseline similarity (GT vs GT shuffled to see "random" baseline)
    shuffled_idx = torch.randperm(z_gt_t.shape[0])
    sim_random = cos(z_gt_t, z_gt_t[shuffled_idx]).mean().item()
    logger.info(f"Random Baseline Similarity (GT vs GT-shuffled): {sim_random:.4f}")
    
    import json
    # Export metrics
    metrics = {
        'lrr_aligned_similarity_to_gt': sim_lrr,
        'untrained_baseline_similarity': -0.0103,
        'alignment_improvement': sim_lrr - (-0.0103),
        'random_permutation_similarity': sim_random,
        'num_samples': len(labels_all),
        'latent_dim': 128
    }
    metrics_path = out_dir / 'darcy_latent_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Similarity metrics saved to {metrics_path}")

    # Normalize latents before t-SNE for consistent variance
    z_gt_all = (z_gt_all - z_gt_all.mean(axis=0)) / (z_gt_all.std(axis=0) + 1e-8)
    z_fno_all = (z_fno_all - z_fno_all.mean(axis=0)) / (z_fno_all.std(axis=0) + 1e-8)
    z_lrr_all = (z_lrr_all - z_lrr_all.mean(axis=0)) / (z_lrr_all.std(axis=0) + 1e-8)

    logger.info(f"Computing t-SNE for {len(labels_all)} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_jobs=-1)
    
    emb_gt = tsne.fit_transform(z_gt_all)
    emb_fno = tsne.fit_transform(z_fno_all)
    emb_lrr = tsne.fit_transform(z_lrr_all)
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    vmin, vmax = labels_all.min(), labels_all.max()
    sc_kwargs = {
        's': 8.0, 
        'cmap': 'Blues', 
        'alpha': 0.85, 
        'edgecolors': 'black', 
        'linewidths': 0.15,
        'vmin': vmin,
        'vmax': vmax,
        'rasterized': True
    }
    
    axes[0].scatter(emb_gt[:, 0], emb_gt[:, 1], c=labels_all, **sc_kwargs)
    
    axes[1].scatter(emb_fno[:, 0], emb_fno[:, 1], c=labels_all, **sc_kwargs)
    
    sc = axes[2].scatter(emb_lrr[:, 0], emb_lrr[:, 1], c=labels_all, **sc_kwargs)
    
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout to make room for colorbar
    fig.subplots_adjust(right=0.88, wspace=0.3)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(sc, cax=cbar_ax, label='Mean Solution ($u$)')
    
    paper_path = out_dir / 'darcy_latent_tsne.pdf'
    plt.savefig(paper_path, bbox_inches='tight', dpi=300)
    logger.info(f"Latent 3-panel t-SNE saved to {paper_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_lrr', type=str, required=True, help='Path to pre-trained LRR-FNO weights')
    parser.add_argument('--load_fno', type=str, required=True, help='Path to pre-trained FNO weights')
    parser.add_argument('--width', type=int, default=32, help='Model width')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Models
    fno_model = FNO2d(in_channels=1, out_channels=1, modes1=12, modes2=12, width=args.width).to(device)
    lrr_model = LRRFNO2d(in_channels=1, out_channels=1, modes1=12, modes2=12, width=args.width, latent_dim=128).to(device)
    
    # Load Weights
    cp_lrr = torch.load(args.load_lrr, map_location=device)
    lrr_model.load_state_dict(cp_lrr.get('model_state_dict', cp_lrr))
    logger.info(f"Loaded LRR-FNO from {args.load_lrr}")
    
    cp_fno = torch.load(args.load_fno, map_location=device)
    fno_model.load_state_dict(cp_fno.get('model_state_dict', cp_fno))
    logger.info(f"Loaded FNO from {args.load_fno}")
    
    out_dir = ROOT_DIR / 'latent_reps/figures'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    analyze_latent_space(fno_model, lrr_model, device, out_dir)

if __name__ == '__main__':
    main()
