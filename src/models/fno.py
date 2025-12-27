"""
Fourier Neural Operator (FNO) Implementation

Implements spectral convolution layers and FNO architectures for 1D and 2D PDEs.
The FNO learns in Fourier space: v_{k+1}(x) = σ(W_k * F^{-1}(R_k · F(v_k)) + b_k)

Reference: Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class SpectralConv1d(nn.Module):
    """
    1D Spectral Convolution Layer
    
    Performs convolution in Fourier space by learning complex weights R_k(ξ)
    for the first `modes` Fourier modes.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels  
            modes: Number of Fourier modes to keep (low frequency)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Scale for initialization
        scale = 1 / (in_channels * out_channels)
        
        # Complex weights for Fourier modes: R_k(ξ)
        self.weights = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )
    
    def complex_mul1d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication in Fourier space.
        
        Args:
            x: Input tensor [batch, in_channels, modes]
            weights: Weight tensor [in_channels, out_channels, modes]
            
        Returns:
            Output tensor [batch, out_channels, modes]
        """
        return torch.einsum("bim,iom->bom", x, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through spectral convolution.
        
        Args:
            x: Input tensor [batch, channels, spatial_dim]
            
        Returns:
            Output tensor [batch, channels, spatial_dim]
        """
        batch_size = x.shape[0]
        spatial_size = x.shape[-1]
        
        # Transform to Fourier space: F(v_k)
        x_ft = torch.fft.rfft(x)
        
        # Multiply relevant Fourier modes: R_k · F(v_k)
        out_ft = torch.zeros(
            batch_size, self.out_channels, spatial_size // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :self.modes] = self.complex_mul1d(
            x_ft[:, :, :self.modes], self.weights
        )
        
        # Transform back to physical space: F^{-1}(...)
        x = torch.fft.irfft(out_ft, n=spatial_size)
        
        return x


class SpectralConv2d(nn.Module):
    """
    2D Spectral Convolution Layer
    
    Performs convolution in 2D Fourier space by learning complex weights
    for the first `modes1 x modes2` Fourier modes.
    """
    
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes1: Number of Fourier modes in first dimension
            modes2: Number of Fourier modes in second dimension
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        scale = 1 / (in_channels * out_channels)
        
        # Complex weights for positive and negative frequency modes
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def complex_mul2d(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Complex multiplication for 2D Fourier modes.
        
        Args:
            x: Input tensor [batch, in_channels, modes1, modes2]
            weights: Weight tensor [in_channels, out_channels, modes1, modes2]
            
        Returns:
            Output tensor [batch, out_channels, modes1, modes2]
        """
        return torch.einsum("bixy,ioxy->boxy", x, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through 2D spectral convolution.
        
        Args:
            x: Input tensor [batch, channels, height, width]
            
        Returns:
            Output tensor [batch, channels, height, width]
        """
        batch_size = x.shape[0]
        size1, size2 = x.shape[-2], x.shape[-1]
        
        # 2D FFT
        x_ft = torch.fft.rfft2(x)
        
        # Initialize output in Fourier space
        out_ft = torch.zeros(
            batch_size, self.out_channels, size1, size2 // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        # Multiply relevant modes (both positive and negative frequencies in first dim)
        out_ft[:, :, :self.modes1, :self.modes2] = self.complex_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.complex_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # Inverse 2D FFT
        x = torch.fft.irfft2(out_ft, s=(size1, size2))
        
        return x


class FNO1d(nn.Module):
    """
    1D Fourier Neural Operator
    
    Architecture:
        1. Input lifting: ℓ_0: f → v_0 (pointwise MLP)
        2. K Fourier layers: v_{k+1} = σ(SpectralConv(v_k) + W·v_k)
        3. Output projection: Π: v_K → ũ (pointwise MLP)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 4,
        padding: int = 8,
    ):
        """
        Args:
            in_channels: Input field channels
            out_channels: Output field channels
            modes: Number of Fourier modes
            width: Hidden channel width
            num_layers: Number of Fourier layers (K)
            padding: Padding for non-periodic domains
        """
        super().__init__()
        self.modes = modes
        self.width = width
        self.num_layers = num_layers
        self.padding = padding
        
        # Input lifting: ℓ_0
        self.lifting = nn.Linear(in_channels + 1, width)  # +1 for spatial coordinate
        
        # Fourier layers
        self.spectral_convs = nn.ModuleList([
            SpectralConv1d(width, width, modes) for _ in range(num_layers)
        ])
        self.linear_convs = nn.ModuleList([
            nn.Conv1d(width, width, 1) for _ in range(num_layers)
        ])
        
        # Output projection: Π
        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, out_channels)
        )
    
    def get_grid(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate normalized spatial grid coordinates."""
        batch_size, size = shape[0], shape[1]  # shape is [B, S, C]
        grid = torch.linspace(0, 1, size, device=device)
        grid = grid.reshape(1, size, 1).repeat(batch_size, 1, 1)
        return grid
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input field [batch, spatial_dim] or [batch, channels, spatial_dim]
            return_features: If True, return features before projection (v_K)
            
        Returns:
            output: Predicted solution [batch, spatial_dim, out_channels]
            features: (optional) Features after K layers [batch, spatial_dim, width]
        """
        # Ensure 3D input in format [B, S, C]
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, S] -> [B, S, 1]
        elif x.dim() == 3:
            # Check if input is [B, C, S] (channels first) and convert to [B, S, C]
            # Heuristic: if channel dim (dim 1) is small and last dim is large, permute
            if x.shape[1] <= 10 and x.shape[-1] > x.shape[1]:
                x = x.permute(0, 2, 1)  # [B, C, S] -> [B, S, C]
        
        # Add spatial grid
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat([x, grid], dim=-1)
        
        # Lifting: [B, S, C+1] -> [B, S, W]
        x = self.lifting(x)
        x = x.permute(0, 2, 1)  # [B, W, S] for conv
        
        # Pad for non-periodic boundaries
        if self.padding > 0:
            x = F.pad(x, [0, self.padding])
        
        # Fourier layers
        for i in range(self.num_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.linear_convs[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        
        # Remove padding
        if self.padding > 0:
            x = x[..., :-self.padding]
        
        x = x.permute(0, 2, 1)  # [B, S, W]
        features = x if return_features else None
        
        # Projection
        x = self.projection(x)
        
        return (x, features) if return_features else x
    
    def backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through backbone only (for LRN injection).
        Returns features v_K before projection.
        """
        output, features = self.forward(x, return_features=True)
        return features
    
    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Apply output projection to features."""
        return self.projection(features)


class FNO2d(nn.Module):
    """
    2D Fourier Neural Operator
    
    For 2D PDEs like Darcy flow and Navier-Stokes.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        num_layers: int = 4,
        padding: int = 9,
    ):
        """
        Args:
            in_channels: Input field channels
            out_channels: Output field channels
            modes1: Fourier modes in first spatial dimension
            modes2: Fourier modes in second spatial dimension
            width: Hidden channel width
            num_layers: Number of Fourier layers (K)
            padding: Padding for non-periodic domains
        """
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_layers = num_layers
        self.padding = padding
        
        # Input lifting: +2 for spatial coordinates (x, y)
        self.lifting = nn.Linear(in_channels + 2, width)
        
        # Fourier layers
        self.spectral_convs = nn.ModuleList([
            SpectralConv2d(width, width, modes1, modes2) for _ in range(num_layers)
        ])
        self.linear_convs = nn.ModuleList([
            nn.Conv2d(width, width, 1) for _ in range(num_layers)
        ])
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(width, 128),
            nn.GELU(),
            nn.Linear(128, out_channels)
        )
    
    def get_grid(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Generate 2D normalized spatial grid coordinates."""
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]
        
        grid_x = torch.linspace(0, 1, size_x, device=device)
        grid_y = torch.linspace(0, 1, size_y, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing='ij')
        
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        
        return grid
    
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input field [batch, height, width] or [batch, channels, height, width]
            return_features: If True, return features before projection
            
        Returns:
            output: Predicted solution [batch, height, width, out_channels]
            features: (optional) Features after K layers
        """
        # Ensure 4D input: [B, H, W, C]
        if x.dim() == 3:
            x = x.unsqueeze(-1)
        elif x.dim() == 4 and x.shape[1] < x.shape[-1]:
            x = x.permute(0, 2, 3, 1)
        
        batch_size, size_x, size_y, _ = x.shape
        
        # Add spatial grid
        grid = self.get_grid((batch_size, size_x, size_y), x.device)
        x = torch.cat([x, grid], dim=-1)
        
        # Lifting
        x = self.lifting(x)
        x = x.permute(0, 3, 1, 2)  # [B, W, H, W] for conv
        
        # Pad
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # Fourier layers
        for i in range(self.num_layers):
            x1 = self.spectral_convs[i](x)
            x2 = self.linear_convs[i](x)
            x = x1 + x2
            if i < self.num_layers - 1:
                x = F.gelu(x)
        
        # Remove padding
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
        features = x if return_features else None
        
        # Projection
        x = self.projection(x)
        
        # x is [B, H, W, C]
        # Permute back to [B, C, H, W] for consistency with input
        x = x.permute(0, 3, 1, 2)
        
        return (x, features) if return_features else x
    
    def backbone_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through backbone only, returns v_K."""
        _, features = self.forward(x, return_features=True)
        return features
    
    def project(self, features: torch.Tensor) -> torch.Tensor:
        """Apply output projection."""
        return self.projection(features)
