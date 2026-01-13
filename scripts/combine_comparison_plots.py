#!/usr/bin/env python3
"""
Script to combine the three comparison plots (Burgers 2D, Darcy Flow, Navier-Stokes)
into a single combined figure for the final performance report.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

def combine_comparison_plots():
    """Combine the three v2 comparison plots into a single figure."""
    
    # Define paths
    plots_dir = Path(__file__).parent.parent / "results" / "plots"
    output_path = plots_dir / "combined_comparison_v2.png"
    
    # Load the three comparison plots
    burgers_path = plots_dir / "burgers2d_comparison_v2.png"
    darcy_path = plots_dir / "darcy_comparison_v2.png"
    ns_path = plots_dir / "ns_comparison_v2.png"
    
    print(f"Loading plots from: {plots_dir}")
    print(f"  - Burgers 2D: {burgers_path.exists()}")
    print(f"  - Darcy Flow: {darcy_path.exists()}")
    print(f"  - Navier-Stokes: {ns_path.exists()}")
    
    # Load images
    burgers_img = mpimg.imread(burgers_path)
    darcy_img = mpimg.imread(darcy_path)
    ns_img = mpimg.imread(ns_path)
    
    # Create figure with 3 rows (one for each plot)
    # Set aspect ratio based on image dimensions
    fig, axes = plt.subplots(3, 1, figsize=(15, 30))
    
    # Set up the main title
    fig.suptitle('Truth | Vanilla FNO | LRN-FNO', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # Plot each comparison - simple labels only
    titles = [
        'Burgers 2D',
        'Darcy Flow', 
        'Navier-Stokes'
    ]
    images = [burgers_img, darcy_img, ns_img]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, aspect='equal')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=10)
        ax.axis('off')
    
    # Adjust layout to prevent overlap and maintain aspect ratio
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    
    # Save the combined figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nCombined plot saved to: {output_path}")
    
    # Also print image dimensions for verification
    print(f"\nImage dimensions:")
    print(f"  - Burgers 2D: {burgers_img.shape}")
    print(f"  - Darcy Flow: {darcy_img.shape}")
    print(f"  - Navier-Stokes: {ns_img.shape}")
    
    plt.close()
    return output_path

if __name__ == "__main__":
    output = combine_comparison_plots()
    print(f"\n✓ Successfully created combined comparison plot!")
