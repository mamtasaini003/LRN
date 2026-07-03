"""
Script to demonstrate how LRR training improves latent alignment.
3 Panels: Ground Truth | Untrained LRR | Aligned LRR
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
sys.path.insert(0, str(Path(__file__).parent))

from pde_datasets_custom import DarcyDataset
from models.lrr.model import LRRFNO2d

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_viz_dataset(num_samples=5000, resolution=32, seed=4242):
    def unsqueeze_transform(t):
        return t.unsqueeze(0) if t.ndim == 2 else t
    return DarcyDataset(resolution=resolution, num_samples=num_samples, train=True, train_ratio=0.999, transform=unsqueeze_transform)

def get_latents(model, loader, device):
    model.eval()
    z_gt_list, z_lrr_list, labels = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            z_gt = model.encoder_u(y)
            out = model(x)
            z_f = out['z_f']
            z_gt_list.append(z_gt.cpu().numpy())
            z_lrr_list.append(z_f.cpu().numpy())
            labels.append(torch.mean(x, dim=(1,2,3)).cpu().numpy())
    return np.concatenate(z_gt_list), np.concatenate(z_lrr_list), np.concatenate(labels)

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Measuring alignment on {device}...")
    
    dataset = generate_viz_dataset(num_samples=4000)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # 1. Untrained Model
    model_untrained = LRRFNO2d(1, 1, 12, 12, 32, latent_dim=128).to(device)
    z_gt_un, z_lrr_un, l_un = get_latents(model_untrained, loader, device)
    
    # 2. Aligned Model
    model_aligned = LRRFNO2d(1, 1, 12, 12, 32, latent_dim=128).to(device)
    cp = torch.load('latent_reps/scripts/lrr_aligned_weights.pt', map_location=device)
    model_aligned.load_state_dict(cp.get('model_state_dict', cp))
    z_gt_al, z_lrr_al, l_al = get_latents(model_aligned, loader, device)
    
    # Metrics
    cos = nn.CosineSimilarity(dim=1)
    sim_un = cos(torch.from_numpy(z_gt_un), torch.from_numpy(z_lrr_un)).mean().item()
    sim_al = cos(torch.from_numpy(z_gt_al), torch.from_numpy(z_lrr_al)).mean().item()
    
    logger.info(f"Untrained Similarity: {sim_un:.4f}")
    logger.info(f"Aligned Similarity:   {sim_al:.4f}")
    
    # t-SNE
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_jobs=-1)
    
    logger.info("Computing t-SNE for Ground Truth...")
    emb_gt = tsne.fit_transform(z_gt_al)
    logger.info("Computing t-SNE for Untrained LRR...")
    emb_un = tsne.fit_transform(z_lrr_un)
    logger.info("Computing t-SNE for Aligned LRR...")
    emb_al = tsne.fit_transform(z_lrr_al)
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    kwargs = {'s': 1.5, 'cmap': 'viridis', 'alpha': 0.6, 'edgecolors': 'none', 'rasterized': True}
    
    axes[0].scatter(emb_gt[:, 0], emb_gt[:, 1], c=l_al, **kwargs)
    axes[0].set_title("Ground Truth ($z_u$)", fontsize=14)
    
    axes[1].scatter(emb_un[:, 0], emb_un[:, 1], c=l_un, **kwargs)
    axes[1].set_title(f"Untrained LRR ($z_f$)\nSimilarity: {sim_un:.4f}", fontsize=14)
    
    sc = axes[2].scatter(emb_al[:, 0], emb_al[:, 1], c=l_al, **kwargs)
    axes[2].set_title(f"Aligned LRR ($z_f$)\nSimilarity: {sim_al:.4f}", fontsize=14)
    
    for ax in axes: ax.axis('off')
    plt.colorbar(sc, ax=axes, shrink=0.7, pad=0.02, label='Mean Permeability')
    
    out_path = Path('latent_reps/figures/lrr_alignment_improvement.pdf')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    logger.info(f"Result saved to {out_path}")

if __name__ == '__main__':
    run()
