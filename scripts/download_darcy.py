#!/usr/bin/env python3
"""
Download FNO Benchmark Darcy Dataset

Uses the neuraloperator library to download and prepare the Darcy flow dataset.
"""

import os
from pathlib import Path
import numpy as np
import torch

DATASET_DIR = Path("dataset/darcy")
DATASET_DIR.mkdir(parents=True, exist_ok=True)


def download_via_neuralop():
    """Download Darcy dataset using neuralop library."""
    try:
        from neuralop.data.datasets import load_darcy_flow_small
        
        print("Downloading Darcy dataset via neuralop...")
        train_loader, test_loaders, _ = load_darcy_flow_small(
            n_train=1000, 
            n_tests=[200],
            batch_size=1,
            test_batch_sizes=[1]
        )

        
        x_train, y_train = [], []
        for batch in train_loader:
            if isinstance(batch, dict):
                x = batch['x']
                y = batch['y']
            else:
                x, y = batch
            x_train.append(x.numpy())
            y_train.append(y.numpy())
        
        x_test, y_test = [], []
        # test_loaders is a dictionary {batch_size: loader} or just a loader depending on version
        if isinstance(test_loaders, dict):
            # Try to get the loader for batch size 1
            loader = test_loaders.get(1) or list(test_loaders.values())[0]
        else:
            loader = test_loaders

        for batch in loader:
            if isinstance(batch, dict):
                x = batch['x']
                y = batch['y']
            else:
                x, y = batch
            x_test.append(x.numpy())
            y_test.append(y.numpy())

        
        x_train = np.concatenate(x_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        x_test = np.concatenate(x_test, axis=0)
        y_test = np.concatenate(y_test, axis=0)
        
        # Save
        np.save(DATASET_DIR / 'Darcy_x_train.npy', x_train)
        np.save(DATASET_DIR / 'Darcy_y_train.npy', y_train)
        np.save(DATASET_DIR / 'Darcy_x_test.npy', x_test)
        np.save(DATASET_DIR / 'Darcy_y_test.npy', y_test)
        
        print(f"Saved: x_train {x_train.shape}, y_train {y_train.shape}")
        print(f"Saved: x_test {x_test.shape}, y_test {y_test.shape}")
        return True
        
    except ImportError:
        print("neuralop not installed. Install with: pip install neuraloperator")
        return False


def main():
    print("="*60)
    print("FNO Darcy Dataset Downloader")
    print("="*60)
    
    success = download_via_neuralop()
    
    if success:
        print(f"\nDone! Dataset saved to: {DATASET_DIR}")
        print("\nTo use in ablation studies:")
        print("  python examples/exp_ablation_darcy.py --epochs 100")
    else:
        print("\nFailed to download. Please install neuraloperator:")
        print("  pip install neuraloperator")


if __name__ == "__main__":
    main()

