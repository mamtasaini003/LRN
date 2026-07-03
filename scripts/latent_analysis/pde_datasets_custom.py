"""
PDE Datasets for LRN Training (Custom for Latent Analysis)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable
import h5py


class DarcyDataset(Dataset):
    """
    2D Darcy Flow Dataset.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        resolution: int = 85,
        num_samples: int = 1000,
        sub_sampling: int = 1,
        train: bool = True,
        train_ratio: float = 0.8,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.sub_sampling = sub_sampling
        self.train = train
        self.train_ratio = train_ratio
        self.transform = transform
        
        if data_path is not None and Path(data_path).exists():
            self._load_from_file(data_path, train, train_ratio)
        else:
            self._generate_synthetic(num_samples, resolution)
    
    def _load_from_file(self, data_path: str, train: bool, train_ratio: float):
        """Load Darcy data from file."""
        with h5py.File(data_path, 'r') as f:
            if 'coeff' in f:
                inputs = torch.tensor(f['coeff'][:], dtype=torch.float32)
                outputs = torch.tensor(f['sol'][:], dtype=torch.float32)
            elif 'input' in f:
                inputs = torch.tensor(f['input'][:], dtype=torch.float32)
                outputs = torch.tensor(f['output'][:], dtype=torch.float32)
            else:
                inputs = torch.tensor(f['x'][:], dtype=torch.float32)
                outputs = torch.tensor(f['y'][:], dtype=torch.float32)
        
        # Subsampling
        if self.sub_sampling > 1:
            inputs = inputs[:, ::self.sub_sampling, ::self.sub_sampling]
            outputs = outputs[:, ::self.sub_sampling, ::self.sub_sampling]
            
        # Split
        n_train = int(inputs.shape[0] * train_ratio)
        if train:
            self.f = inputs[:n_train]
            self.u = outputs[:n_train]
        else:
            self.f = inputs[n_train:]
            self.u = outputs[n_train:]

    def _generate_synthetic(self, num_samples, resolution):
        """Synthetic Darcy flow data generation for fallback."""
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        f_list = []
        u_list = []
        
        for _ in range(num_samples):
            n_blobs = np.random.randint(3, 8)
            a = torch.ones(resolution, resolution) * 0.5
            
            for _ in range(n_blobs):
                cx, cy = np.random.rand(2)
                r = np.random.uniform(0.1, 0.3)
                amplitude = np.random.uniform(0.5, 2.0)
                mask = ((X - cx)**2 + (Y - cy)**2) < r**2
                a[mask] = amplitude
            
            from torch.nn.functional import avg_pool2d
            u = a.unsqueeze(0).unsqueeze(0)
            u = avg_pool2d(u, kernel_size=5, stride=1, padding=2)
            u = u.squeeze()
            u = u + 0.1 * torch.randn_like(u)
            
            f_list.append(a)
            u_list.append(u)
        
        self.f = torch.stack(f_list)
        self.u = torch.stack(u_list)
        
        n_train = int(num_samples * self.train_ratio)
        if self.train:
            self.f = self.f[:n_train]
            self.u = self.u[:n_train]
        else:
            self.f = self.f[n_train:]
            self.u = self.u[n_train:]
    
    def __len__(self) -> int:
        return self.f.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.f[idx]
        u = self.u[idx]
        if self.transform:
            f = self.transform(f)
            u = self.transform(u)
        return f, u
