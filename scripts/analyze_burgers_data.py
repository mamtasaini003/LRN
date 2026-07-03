
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.pde_datasets import BurgersDataset

def analyze_synthetic():
    print("--- Analyzing Synthetic BurgersDataset ---")
    dataset = BurgersDataset(resolution=128, num_samples=10, train=True)
    f, u = dataset[0]
    
    print(f"Dataset Length: {len(dataset)}")
    print(f"Sample 0 Input (f) Shape: {f.shape}")
    print(f"Sample 0 Output (u) Shape: {u.shape}")
    
    print("\nData Semantics:")
    print("Input 'f': Initial condition u(x, t=0)")
    print("Output 'u': Evolved solution u(x, t=T)")
    print("Time Information: Implicit mapping from t=0 to t=T (Fixed time horizon)")
    print("Temporal Resolution: None (1-step mapping)")
    
    # Check values
    print(f"\nSample stats:")
    print(f"f range: [{f.min():.4f}, {f.max():.4f}]")
    print(f"u range: [{u.min():.4f}, {u.max():.4f}]")

def check_neuralop_files():
    print("\n--- Checking NeuralOperator Data Files ---")
    data_root = Path("datasets/burgers")
    if data_root.exists():
        files = list(data_root.glob("*.pt"))
        if files:
            print(f"Found files: {[f.name for f in files]}")
            # Load basic info if possible
            try:
                data = torch.load(files[0])
                if isinstance(data, dict):
                    print(f"Keys: {data.keys()}")
                    if 'x' in data: print(f"x shape: {data['x'].shape}")
            except Exception as e:
                print(f"Error loading file: {e}")
        else:
            print("No .pt files found in datasets/burgers.")
    else:
        print("datasets/burgers directory does not exist.")

if __name__ == "__main__":
    analyze_synthetic()
    check_neuralop_files()
