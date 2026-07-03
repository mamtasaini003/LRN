
import torch
import sys
from pathlib import Path

def analyze_ns_file(path_str):
    path = Path(path_str)
    print(f"--- Analyzing {path.name} ---")
    
    if not path.exists():
        print(f"File not found: {path}")
        return

    try:
        data = torch.load(path)
        print(f"Type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in data.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}: {v.shape} (dtype: {v.dtype})")
                else:
                    print(f"  {k}: {type(v)}")
        else:
            print("Data is not a dict.")
            # It might be a tensor directly?
            if hasattr(data, 'shape'):
                 print(f"Shape: {data.shape}")

        # Deep dive into 'x' and 'y' if present
        if isinstance(data, dict):
            if 'x' in data and 'y' in data:
                x = data['x']
                y = data['y']
                analyze_tensors(x, y)
            elif 'input' in data and 'output' in data:
                x = data['input']
                y = data['output']
                analyze_tensors(x, y)
                
    except Exception as e:
        print(f"Error loading: {e}")

def analyze_tensors(x, y):
    print("\nTensor Analysis:")
    print(f"Input (x) shape: {x.shape}")
    print(f"Output (y) shape: {y.shape}")
    
    # Check for time dimension
    # NS usually [B, T, H, W] or [B, H, W, T] or [B, X, Y, T]
    
    print("\nHypothesis check:")
    if x.ndim == 4:
        print(f"x might be [Batch, Time/Channel, H, W] or [Batch, H, W, Time/Channel]")
    if x.ndim == 2:
        print("x is likely flattened or 1D.")

    # Check values
    print(f"x range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"y range: [{y.min():.4f}, {y.max():.4f}]")

if __name__ == "__main__":
    analyze_ns_file("datasets/navier_stokes/nsforcing_train_128.pt")
