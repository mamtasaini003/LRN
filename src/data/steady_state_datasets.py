"""
Steady-State PDE Datasets for LRN-FNO Experiments
Date: 2026-02-01
Purpose: Time-independent versions of Burgers 2D and Navier-Stokes for comparison with transient baselines.
"""

import torch
import numpy as np
from torch.utils.data import Dataset


class Burgers2dSteadyDataset(Dataset):
    """
    Steady-state 2D Burgers dataset.
    
    Mapping: Forcing field f(x,y) -> Steady velocity (u, v)
    
    The steady Burgers equation (viscous limit):
        ν∇²u = f_u(x,y)
        ν∇²v = f_v(x,y)
    
    We generate random forcing functions and compute the steady-state solution
    using a spectral (Fourier) approach.
    """
    
    def __init__(self, resolution: int = 64, num_samples: int = 300, 
                 train: bool = True, nu: float = 0.1):
        self.resolution = resolution
        self.train = train
        self.nu = nu
        self._generate_synthetic(num_samples, resolution)
    
    def _generate_synthetic(self, num_samples: int, resolution: int):
        """Generate steady-state Burgers data using spectral Poisson solve."""
        print(f"Generating {num_samples} synthetic steady-state Burgers 2D samples...")
        
        # 2D grid
        x = torch.linspace(0, 2*np.pi, resolution)
        y = torch.linspace(0, 2*np.pi, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Wavenumbers for spectral solve
        k = torch.fft.fftfreq(resolution, d=1.0/resolution) * 2 * np.pi / (2*np.pi)
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        lap = -(kx**2 + ky**2)
        lap[0, 0] = 1.0  # Avoid division by zero
        
        f_list = []  # Forcing (input)
        u_list = []  # Velocity (output)
        
        for _ in range(num_samples):
            # Random forcing via Fourier modes
            K = 6
            fu = torch.zeros(resolution, resolution)
            fv = torch.zeros(resolution, resolution)
            
            for kx_i in range(1, K):
                for ky_i in range(1, K):
                    amp_u = torch.randn(1).item() * 0.5
                    amp_v = torch.randn(1).item() * 0.5
                    phase_u = torch.rand(1).item() * 2 * np.pi
                    phase_v = torch.rand(1).item() * 2 * np.pi
                    
                    fu += amp_u * torch.sin(kx_i * X + ky_i * Y + phase_u)
                    fv += amp_v * torch.sin(kx_i * X + ky_i * Y + phase_v)
            
            # Solve Poisson: ν∇²u = f => u = F^{-1}[F[f] / (ν * lap)]
            fu_hat = torch.fft.fft2(fu)
            fv_hat = torch.fft.fft2(fv)
            
            u_hat = fu_hat / (self.nu * lap)
            v_hat = fv_hat / (self.nu * lap)
            
            u = torch.fft.ifft2(u_hat).real
            v = torch.fft.ifft2(v_hat).real
            
            # Stack channels: [2, H, W]
            f = torch.stack([fu, fv])
            sol = torch.stack([u, v])
            
            f_list.append(f)
            u_list.append(sol)
        
        self.f = torch.stack(f_list)
        self.u = torch.stack(u_list)
        
        # Train/test split
        n_train = int(num_samples * 0.8)
        if self.train:
            self.f = self.f[:n_train]
            self.u = self.u[:n_train]
        else:
            self.f = self.f[n_train:]
            self.u = self.u[n_train:]
    
    def __len__(self):
        return len(self.f)
    
    def __getitem__(self, idx):
        return self.f[idx], self.u[idx]


class NavierStokesSteadyDataset(Dataset):
    """
    Steady-state 2D Navier-Stokes (vorticity formulation).
    
    Mapping: Forcing field f(x,y) -> Steady vorticity ω(x,y)
    
    The steady vorticity equation:
        ν∇²ω = f(x,y)
    
    This is a Poisson problem solved spectrally.
    """
    
    def __init__(self, resolution: int = 64, num_samples: int = 200,
                 train: bool = True, nu: float = 0.1):
        self.resolution = resolution
        self.train = train
        self.nu = nu
        self._generate_synthetic(num_samples, resolution)
    
    def _generate_synthetic(self, num_samples: int, resolution: int):
        """Generate steady-state NS data using spectral Poisson solve."""
        print(f"Generating {num_samples} synthetic steady-state Navier-Stokes samples...")
        
        # 2D grid
        x = torch.linspace(0, 2*np.pi, resolution)
        y = torch.linspace(0, 2*np.pi, resolution)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Wavenumbers
        k = torch.fft.fftfreq(resolution, d=1.0/resolution) * 2 * np.pi / (2*np.pi)
        kx, ky = torch.meshgrid(k, k, indexing='ij')
        lap = -(kx**2 + ky**2)
        lap[0, 0] = 1.0
        
        f_list = []
        u_list = []
        
        for _ in range(num_samples):
            # Random forcing
            K = 6
            forcing = torch.zeros(resolution, resolution)
            
            for kx_i in range(1, K):
                for ky_i in range(1, K):
                    amp = torch.randn(1).item() * 0.5
                    phase = torch.rand(1).item() * 2 * np.pi
                    forcing += amp * torch.sin(kx_i * X + ky_i * Y + phase)
            
            # Solve Poisson: ν∇²ω = f => ω = F^{-1}[F[f] / (ν * lap)]
            f_hat = torch.fft.fft2(forcing)
            omega_hat = f_hat / (self.nu * lap)
            # Negate to make output positively correlated with forcing
            omega = -torch.fft.ifft2(omega_hat).real
            
            f_list.append(forcing)
            u_list.append(omega)
        
        self.f = torch.stack(f_list)
        self.u = torch.stack(u_list)
        
        # Train/test split
        n_train = int(num_samples * 0.8)
        if self.train:
            self.f = self.f[:n_train]
            self.u = self.u[:n_train]
        else:
            self.f = self.f[n_train:]
            self.u = self.u[n_train:]
    
    def __len__(self):
        return len(self.f)
    
    def __getitem__(self, idx):
        return self.f[idx], self.u[idx]
