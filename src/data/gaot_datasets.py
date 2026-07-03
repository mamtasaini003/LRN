"""
GAOT Dataset Loader for Time-Independent PDE Cases.

Handles unstructured point cloud data by interpolating to regular grids.
Uses h5py for HDF5-based NetCDF4 files.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from typing import Tuple, Optional
from scipy.interpolate import griddata


class GAOTGridDataset(Dataset):
    """
    Dataset loader for GAOT NetCDF/HDF5 files.
    Interpolates unstructured point cloud data to regular grids for FNO.
    
    Maps: c (spatial parameters/forcing) -> u (solution)
    """
    
    def __init__(
        self,
        nc_path: str,
        train: bool = True,
        train_ratio: float = 0.8,
        resolution: int = 64,
        normalize: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            nc_path: Path to the .nc file
            train: If True, use training split
            train_ratio: Fraction of data for training
            resolution: Grid resolution for interpolation
            normalize: If True, normalize inputs and outputs
            max_samples: Maximum number of samples to use
        """
        self.nc_path = Path(nc_path)
        self.train = train
        self.normalize = normalize
        self.resolution = resolution
        
        # Load data using h5py
        with h5py.File(nc_path, 'r') as f:
            u = f['u'][:]  # [batch, 1, points, 1]
            c = f['c'][:]  # [batch, 1, points, features]
            x = f['x'][:]  # [batch, 1, points, 2] or [1, 1, points, 2] (shared coords)
        
        # Squeeze singleton time dimension
        u = u[:, 0, :, :]  # [batch, points, 1]
        c = c[:, 0, :, :]  # [batch, points, features]
        x = x[:, 0, :, :]  # [batch, points, 2] or [1, points, 2]
        
        # Handle shared coordinates (x has batch dim of 1)
        shared_coords = (x.shape[0] == 1 and u.shape[0] > 1)
        if shared_coords:
            x = x[0]  # [points, 2] - single set of coordinates for all samples
        
        # Limit samples
        if max_samples is not None:
            u = u[:max_samples]
            c = c[:max_samples]
            if not shared_coords:
                x = x[:max_samples]
        
        # Split train/test
        n_samples = u.shape[0]
        n_train = int(n_samples * train_ratio)
        
        if train:
            u = u[:n_train]
            c = c[:n_train]
            if not shared_coords:
                x = x[:n_train]
        else:
            u = u[n_train:]
            c = c[n_train:]
            if not shared_coords:
                x = x[n_train:]
        
        self.shared_coords = shared_coords

        
        # Interpolate to regular grid
        print(f"Interpolating {len(u)} samples to {resolution}x{resolution} grid...")
        
        # Create target grid based on data bounds
        if self.shared_coords:
            # x is [points, 2]
            x_min, x_max = x[:, 0].min(), x[:, 0].max()
            y_min, y_max = x[:, 1].min(), x[:, 1].max()
        else:
            # x is [batch, points, 2]
            x_min, x_max = x[:, :, 0].min(), x[:, :, 0].max()
            y_min, y_max = x[:, :, 1].min(), x[:, :, 1].max()

        
        # Enforce square aspect ratio to preserve geometry
        w = x_max - x_min
        h = y_max - y_min
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Use the larger dimension for the square box
        L = max(w, h)
        
        # Add small padding (10% of the size)
        pad = 0.1 * L
        L_padded = L + 2 * pad
        
        # Define square bounds
        x_min = center_x - L_padded / 2
        x_max = center_x + L_padded / 2
        y_min = center_y - L_padded / 2
        y_max = center_y + L_padded / 2
        
        grid_x = np.linspace(x_min, x_max, resolution)
        grid_y = np.linspace(y_min, y_max, resolution)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        
        # Interpolate each sample
        u_grid = []
        c_grid = []
        
        for i in range(len(u)):
            # Use shared coordinates or per-sample coordinates
            points = x if self.shared_coords else x[i]  # [points, 2]
            
            # Interpolate u (solution)
            u_interp = griddata(
                points, u[i, :, 0], (grid_xx, grid_yy), 
                method='linear', fill_value=0.0
            )
            u_grid.append(u_interp)
            
            # Interpolate c (forcing) - each channel
            c_channels = []
            for ch in range(c.shape[-1]):
                c_interp = griddata(
                    points, c[i, :, ch], (grid_xx, grid_yy),
                    method='linear', fill_value=0.0
                )
                c_channels.append(c_interp)
            c_grid.append(np.stack(c_channels, axis=0))

        
        self.u = np.stack(u_grid, axis=0)  # [batch, H, W]
        self.c = np.stack(c_grid, axis=0)  # [batch, channels, H, W]
        
        # Normalize
        if normalize:
            self.c_mean = self.c.mean()
            self.c_std = self.c.std() + 1e-8
            self.u_mean = self.u.mean()
            self.u_std = self.u.std() + 1e-8
            
            self.c = (self.c - self.c_mean) / self.c_std
            self.u = (self.u - self.u_mean) / self.u_std
        
        # Convert to torch
        self.c = torch.from_numpy(self.c).float()
        self.u = torch.from_numpy(self.u).float().unsqueeze(1)  # Add channel dim
        
        print(f"Loaded {self.nc_path.name}: {len(self)} samples")
        print(f"  c shape: {self.c.shape}, u shape: {self.u.shape}")
    
    def __len__(self) -> int:
        return self.c.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            c: Forcing [channels, H, W]
            u: Solution [1, H, W]
        """
        return self.c[idx], self.u[idx]


def get_gaot_grid_loaders(
    nc_path: str,
    batch_size: int = 16,
    resolution: int = 64,
    train_ratio: float = 0.8,
    max_samples: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, dict]:
    """
    Create train and test data loaders for a GAOT dataset.
    
    Returns:
        train_loader, test_loader, info_dict
    """
    train_dataset = GAOTGridDataset(
        nc_path, train=True, train_ratio=train_ratio, 
        resolution=resolution, max_samples=max_samples
    )
    test_dataset = GAOTGridDataset(
        nc_path, train=False, train_ratio=train_ratio,
        resolution=resolution, max_samples=max_samples
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    info = {
        'in_channels': train_dataset.c.shape[1],
        'out_channels': train_dataset.u.shape[1],
        'resolution': resolution,
        'n_train': len(train_dataset),
        'n_test': len(test_dataset),
    }
    
    return train_loader, test_loader, info


if __name__ == "__main__":
    import sys
    nc_path = sys.argv[1] if len(sys.argv) > 1 else "dataset/Circle.nc"
    
    train_loader, test_loader, info = get_gaot_grid_loaders(
        nc_path, max_samples=50, resolution=64
    )
    
    print(f"\nDataset info: {info}")
    
    for c, u in train_loader:
        print(f"Batch c shape: {c.shape}, u shape: {u.shape}")
        break
