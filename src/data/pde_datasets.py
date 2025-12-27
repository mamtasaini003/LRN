"""
PDE Datasets for LRN Training

Implements dataset classes for common PDE benchmarks:
- Burgers equation (1D)
- Darcy flow (2D)
- Navier-Stokes (2D)

Each dataset provides (f, u) pairs where f is the input/forcing field
and u is the corresponding PDE solution.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Callable
import h5py


class BurgersDataset(Dataset):
    """
    1D Burgers Equation Dataset.
    
    The viscous Burgers equation:
        ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    
    with periodic boundary conditions. The input f is the initial condition
    at t=0, and the output u is the solution at some future time T.
    
    Supports loading from HDF5 files or generating synthetic data.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        resolution: int = 128,
        num_samples: int = 1000,
        sub_sampling: int = 1,
        train: bool = True,
        train_ratio: float = 0.8,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_path: Path to HDF5 data file (if None, generates synthetic data)
            resolution: Spatial resolution
            num_samples: Number of samples if generating synthetic data
            sub_sampling: Subsample factor for resolution
            train: Whether to load training or test split
            train_ratio: Ratio of data for training
            transform: Optional transform to apply
        """
        super().__init__()
        self.resolution = resolution
        self.sub_sampling = sub_sampling
        self.train = train
        self.transform = transform
        
        if data_path is not None and Path(data_path).exists():
            self._load_from_file(data_path, train, train_ratio)
        else:
            self._generate_synthetic(num_samples, resolution)
    
    def _load_from_file(self, data_path: str, train: bool, train_ratio: float):
        """Load data from HDF5 file."""
        with h5py.File(data_path, 'r') as f:
            # Standard format: 'input' and 'output' or 'f' and 'u'
            if 'input' in f:
                inputs = torch.tensor(f['input'][:], dtype=torch.float32)
                outputs = torch.tensor(f['output'][:], dtype=torch.float32)
            else:
                inputs = torch.tensor(f['f'][:], dtype=torch.float32)
                outputs = torch.tensor(f['u'][:], dtype=torch.float32)
        
        # Split data
        n_total = inputs.shape[0]
        n_train = int(n_total * train_ratio)
        
        if train:
            self.f = inputs[:n_train]
            self.u = outputs[:n_train]
        else:
            self.f = inputs[n_train:]
            self.u = outputs[n_train:]
        
        # Subsample if needed
        if self.sub_sampling > 1:
            self.f = self.f[:, ::self.sub_sampling]
            self.u = self.u[:, ::self.sub_sampling]
    
    def _generate_synthetic(self, num_samples: int, resolution: int):
        """Generate synthetic Burgers-like data for testing."""
        print(f"Generating {num_samples} synthetic Burgers samples...")
        
        x = torch.linspace(0, 2 * np.pi, resolution)
        
        # Generate random initial conditions (superposition of harmonics)
        f_list = []
        u_list = []
        
        for _ in range(num_samples):
            # Random Fourier coefficients
            n_modes = 8
            coeffs = torch.randn(n_modes) * 0.5
            phases = torch.rand(n_modes) * 2 * np.pi
            
            # Initial condition
            f = torch.zeros(resolution)
            for k in range(n_modes):
                f += coeffs[k] * torch.sin((k + 1) * x + phases[k])
            
            # Simplified forward map (not actual Burgers solution)
            # For real training, replace with proper data
            viscosity = 0.01
            decay = torch.exp(-viscosity * torch.arange(1, n_modes + 1).float() ** 2)
            
            u = torch.zeros(resolution)
            for k in range(n_modes):
                u += decay[k] * coeffs[k] * torch.sin((k + 1) * x + phases[k])
            
            f_list.append(f)
            u_list.append(u)
        
        self.f = torch.stack(f_list)
        self.u = torch.stack(u_list)
        
        # Split for train/test
        n_train = int(num_samples * 0.8)
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
        
        if self.transform is not None:
            f = self.transform(f)
            u = self.transform(u)
        
        return f, u


class DarcyDataset(Dataset):
    """
    2D Darcy Flow Dataset.
    
    The Darcy flow equation:
        -∇·(a(x)∇u(x)) = f(x)
    
    where a(x) is the permeability field (input) and u(x) is the pressure (output).
    
    Common benchmark: given coefficient field a, predict solution u.
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
        self.transform = transform
        
        if data_path is not None and Path(data_path).exists():
            self._load_from_file(data_path, train, train_ratio)
        else:
            self._generate_synthetic(num_samples, resolution)
    
    def _load_from_file(self, data_path: str, train: bool, train_ratio: float):
        """Load Darcy data from file."""
        with h5py.File(data_path, 'r') as f:
            if 'coeff' in f:
                # Common format: coeff (a) and sol (u)
                inputs = torch.tensor(f['coeff'][:], dtype=torch.float32)
                outputs = torch.tensor(f['sol'][:], dtype=torch.float32)
            elif 'input' in f:
                inputs = torch.tensor(f['input'][:], dtype=torch.float32)
                outputs = torch.tensor(f['output'][:], dtype=torch.float32)
            else:
                inputs = torch.tensor(f['a'][:], dtype=torch.float32)
                outputs = torch.tensor(f['u'][:], dtype=torch.float32)
        
        n_total = inputs.shape[0]
        n_train = int(n_total * train_ratio)
        
        if train:
            self.f = inputs[:n_train]
            self.u = outputs[:n_train]
        else:
            self.f = inputs[n_train:]
            self.u = outputs[n_train:]
        
        if self.sub_sampling > 1:
            s = self.sub_sampling
            self.f = self.f[:, ::s, ::s]
            self.u = self.u[:, ::s, ::s]
    
    def _generate_synthetic(self, num_samples: int, resolution: int):
        """Generate synthetic Darcy-like data."""
        print(f"Generating {num_samples} synthetic Darcy samples...")
        
        # 2D grid
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        f_list = []
        u_list = []
        
        for _ in range(num_samples):
            # Random coefficient field (piecewise constant approximation)
            n_blobs = np.random.randint(3, 8)
            a = torch.ones(resolution, resolution) * 0.5
            
            for _ in range(n_blobs):
                cx, cy = np.random.rand(2)
                r = np.random.uniform(0.1, 0.3)
                amplitude = np.random.uniform(0.5, 2.0)
                
                mask = ((X - cx)**2 + (Y - cy)**2) < r**2
                a[mask] = amplitude
            
            # Simplified "solution" (not actual Darcy solver)
            # Smooth the coefficient field as proxy for solution
            from torch.nn.functional import avg_pool2d
            u = a.unsqueeze(0).unsqueeze(0)
            u = avg_pool2d(u, kernel_size=5, stride=1, padding=2)
            u = u.squeeze()
            
            # Add some variation
            u = u + 0.1 * torch.randn_like(u)
            
            f_list.append(a)
            u_list.append(u)
        
        self.f = torch.stack(f_list)
        self.u = torch.stack(u_list)
        
        n_train = int(num_samples * 0.8)
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
        
        if self.transform is not None:
            f = self.transform(f)
            u = self.transform(u)
        
        return f, u


class NavierStokesDataset(Dataset):
    """
    2D Navier-Stokes Dataset.
    
    Incompressible Navier-Stokes equations:
        ∂w/∂t + u·∇w = ν∇²w + f
        ∇·u = 0
    
    where w is the vorticity and u is velocity. Typically predicts future
    vorticity states given initial conditions.
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        resolution: int = 64,
        num_samples: int = 1000,
        sub_sampling: int = 1,
        input_steps: int = 10,
        output_steps: int = 10,
        train: bool = True,
        train_ratio: float = 0.8,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            data_path: Path to HDF5 data file
            resolution: Spatial resolution
            num_samples: Number of samples for synthetic data
            sub_sampling: Spatial subsampling factor
            input_steps: Number of input time steps
            output_steps: Number of output time steps used for prediction
            train: Training or test split
            train_ratio: Ratio for training split
            transform: Optional transform
        """
        super().__init__()
        self.resolution = resolution
        self.sub_sampling = sub_sampling
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.train = train
        self.transform = transform
        
        if data_path is not None and Path(data_path).exists():
            self._load_from_file(data_path, train, train_ratio)
        else:
            self._generate_synthetic(num_samples, resolution)
    
    def _load_from_file(self, data_path: str, train: bool, train_ratio: float):
        """Load NS data from file."""
        with h5py.File(data_path, 'r') as f:
            if 'vorticity' in f:
                data = torch.tensor(f['vorticity'][:], dtype=torch.float32)
            elif 'w' in f:
                data = torch.tensor(f['w'][:], dtype=torch.float32)
            else:
                data = torch.tensor(f['data'][:], dtype=torch.float32)
        
        # Data format: [N, T, H, W]
        # We assume T >= input_steps + output_steps
        n_total = data.shape[0]
        n_train = int(n_total * train_ratio)
        
        if train:
            self.data = data[:n_train]
        else:
            self.data = data[n_train:]
            
        if self.sub_sampling > 1:
            s = self.sub_sampling
            self.data = self.data[:, :, ::s, ::s]
    
    def _generate_synthetic(self, num_samples: int, resolution: int):
        """Generate synthetic advection data (rotating Gaussian blobs)."""
        print(f"Generating {num_samples} synthetic Navier-Stokes sequences...")
        
        total_steps = self.input_steps + self.output_steps
        x = torch.linspace(0, 1, resolution)
        y = torch.linspace(0, 1, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Center of rotation
        cx, cy = 0.5, 0.5
        
        data_list = []
        
        for _ in range(num_samples):
            # T steps of rotating field
            seq = []
            
            # Random initial vortices
            n_blobs = np.random.randint(2, 5)
            blob_params = []
            for _ in range(n_blobs):
                bx, by = np.random.rand(2) * 0.6 + 0.2
                sigma = np.random.uniform(0.05, 0.15)
                amp = np.random.uniform(0.5, 2.0)
                # Random rotation speed
                omega = np.random.uniform(-0.1, 0.1)
                blob_params.append([bx, by, sigma, amp, omega])
            
            for t in range(total_steps):
                field = torch.zeros(resolution, resolution)
                for (bx, by, sigma, amp, omega) in blob_params:
                    # Rotate center
                    angle = omega * t
                    dx = bx - cx
                    dy = by - cy
                    rbx = cx + dx * np.cos(angle) - dy * np.sin(angle)
                    rby = cy + dx * np.sin(angle) + dy * np.cos(angle)
                    
                    dist_sq = (X - rbx)**2 + (Y - rby)**2
                    field += amp * torch.exp(-dist_sq / (2 * sigma**2))
                
                seq.append(field)
            
            data_list.append(torch.stack(seq))
            
        self.data = torch.stack(data_list)
        
        # Split
        n_train = int(num_samples * 0.8)
        if self.train:
            self.data = self.data[:n_train]
        else:
            self.data = self.data[n_train:]
            
    def __len__(self) -> int:
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # f: [Input_Steps, H, W]
        # u: [Output_Steps, H, W]
        # Typically we predict the NEXT output_steps
        traj = self.data[idx]
        
        f = traj[:self.input_steps]
        u = traj[self.input_steps : self.input_steps + self.output_steps]
        
        if self.transform is not None:
            f = self.transform(f)
            u = self.transform(u)
        
        return f, u


class Burgers2dDataset(Dataset):
    """
    2D Coupled Burgers Equation Dataset.
    
    ∂u/∂t + u·∇u = νΔu
    ∂v/∂t + v·∇v = νΔv
    
    Variables: u(x,y), v(x,y)
    Input: Initial conditions (u0, v0)
    Output: Evolved state (uT, vT)
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        resolution: int = 64,
        num_samples: int = 1000,
        sub_sampling: int = 1,
        time_step: float = 1.0,  # Time to evolve
        train: bool = True,
        train_ratio: float = 0.8,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.resolution = resolution
        self.sub_sampling = sub_sampling
        self.time_step = time_step
        self.train = train
        self.transform = transform
        
        if data_path is not None and Path(data_path).exists():
            self._load_from_file(data_path, train, train_ratio)
        else:
            self._generate_synthetic(num_samples, resolution)
            
    def _load_from_file(self, data_path: str, train: bool, train_ratio: float):
        # Placeholder for loading actual 2D Burgers data
        # Assume format [N, 2, H, W] for both input and output or [N, T, 2, H, W]
        # For now, just generate synthetic
        print("Loading from file not fully implemented for Burgers2d, generating synthetic.")
        self._generate_synthetic(100, self.resolution)
            
    def _generate_synthetic(self, num_samples: int, resolution: int):
        """Generate synthetic 2D Fields using Finite Difference Time Stepping."""
        print(f"Generating {num_samples} synthetic 2D Burgers samples (Non-Linear Physics)...")
        
        x = torch.linspace(0, 2*np.pi, resolution)
        y = torch.linspace(0, 2*np.pi, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        f_list = [] # Input (t=0)
        u_list = [] # Output (t=T)
        
        # Physics parameters
        nu = 0.01
        dt = 0.01
        steps = int(self.time_step / dt)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        for i in range(num_samples):
            if (i+1) % 10 == 0:
                print(f"Simulating sample {i+1}/{num_samples}...")
                
            # 1. Generate Random Initial Conditions (Spectral)
            u = torch.zeros(resolution, resolution)
            v = torch.zeros(resolution, resolution)
            
            K = 4 # Fewer modes for cleaner starting field
            for kx in range(K):
                for ky in range(K):
                    if kx==0 and ky==0: continue
                    # Random phase and amplitude
                    phase_u = torch.rand(1) * 2 * np.pi
                    phase_v = torch.rand(1) * 2 * np.pi
                    amp_u = torch.randn(1) * 0.5
                    amp_v = torch.randn(1) * 0.5
                    
                    basis = torch.sin(kx*X + ky*Y + phase_u)
                    u += amp_u * basis
                    basis_v = torch.sin(kx*X + ky*Y + phase_v)
                    v += amp_v * basis_v
            
            # Normalize to avoid instability
            u = 0.5 * u / u.std()
            v = 0.5 * v / v.std()
            
            # Save Initial Condition
            f_sample = torch.stack([u.clone(), v.clone()])
            f_list.append(f_sample)
            
            # 2. Time Stepping (Finite Difference)
            # du/dt = -u du/dx - v du/dy + nu del^2 u
            
            for _ in range(steps):
                # Gradients (Periodic BCs via roll)
                u_ip = torch.roll(u, -1, 0)
                u_im = torch.roll(u, 1, 0)
                u_jp = torch.roll(u, -1, 1)
                u_jm = torch.roll(u, 1, 1)
                
                v_ip = torch.roll(v, -1, 0)
                v_im = torch.roll(v, 1, 0)
                v_jp = torch.roll(v, -1, 1)
                v_jm = torch.roll(v, 1, 1)
                
                # First derivatives (Central Difference)
                dudx = (u_ip - u_im) / (2*dx)
                dudy = (u_jp - u_jm) / (2*dy)
                dvdx = (v_ip - v_im) / (2*dx)
                dvdy = (v_jp - v_jm) / (2*dy)
                
                # Laplacian (Central Difference)
                lap_u = (u_ip - 2*u + u_im)/(dx**2) + (u_jp - 2*u + u_jm)/(dy**2)
                lap_v = (v_ip - 2*v + v_im)/(dx**2) + (v_jp - 2*v + v_jm)/(dy**2)
                
                # Update
                du_dt = -(u*dudx + v*dudy) + nu*lap_u
                dv_dt = -(u*dvdx + v*dvdy) + nu*lap_v
                
                u = u + dt * du_dt
                v = v + dt * dv_dt
                
            # Save Final State
            u_sample = torch.stack([u, v])
            u_list.append(u_sample)
            
        self.f = torch.stack(f_list)
        self.u = torch.stack(u_list)
        
        n_train = int(num_samples * 0.8)
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


def create_dataloaders(
    dataset_name: str,
    data_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
):
    """
    Factory function to create train and test dataloaders.
    
    Args:
        dataset_name: 'burgers', 'burgers2d', 'darcy', or 'navier_stokes'
        data_path: Path to data file
        batch_size: Batch size
        num_workers: Number of data loading workers
        **kwargs: Additional dataset arguments
        
    Returns:
        train_loader, test_loader
    """
    from torch.utils.data import DataLoader
    
    dataset_classes = {
        'burgers': BurgersDataset,
        'burgers2d': Burgers2dDataset,
        'darcy': DarcyDataset,
        'navier_stokes': NavierStokesDataset,
        'ns': NavierStokesDataset,
    }
    
    DatasetClass = dataset_classes[dataset_name.lower()]
    
    train_dataset = DatasetClass(data_path=data_path, train=True, **kwargs)
    test_dataset = DatasetClass(data_path=data_path, train=False, **kwargs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, test_loader
