"""
Dual Encoders for Latent Reciprocity Network

Implements the forward encoder E_f and reverse encoder E_u that map
input fields and solutions to a shared latent space z ∈ ℝ^{d_z}.

The encoders use CNN architectures with global average pooling to produce
fixed-dimensional latent codes regardless of input resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FieldEncoder(nn.Module):
    """
    Base encoder architecture for mapping spatial fields to latent codes.
    
    Architecture:
        - CNN backbone with progressively increasing channels
        - Global average pooling for resolution invariance
        - MLP head to produce latent code z ∈ ℝ^{d_z}
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        hidden_channels: list = [32, 64, 128],
        spatial_dim: int = 1,
    ):
        """
        Args:
            in_channels: Number of input channels
            latent_dim: Dimensionality of latent space (d_z)
            hidden_channels: Channel widths for CNN layers
            spatial_dim: 1 for 1D fields, 2 for 2D fields
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.spatial_dim = spatial_dim
        
        # Select appropriate conv and pool layers based on spatial dimension
        if spatial_dim == 1:
            Conv = nn.Conv1d
            BatchNorm = nn.BatchNorm1d
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            Conv = nn.Conv2d
            BatchNorm = nn.BatchNorm2d
            self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Build CNN backbone
        layers = []
        channels = [in_channels] + hidden_channels
        
        for i in range(len(channels) - 1):
            layers.extend([
                Conv(channels[i], channels[i + 1], kernel_size=3, padding=1, stride=2),
                BatchNorm(channels[i + 1]),
                nn.GELU(),
            ])
        
        self.backbone = nn.Sequential(*layers)
        
        # MLP head for latent projection
        self.head = nn.Sequential(
            nn.Linear(hidden_channels[-1], hidden_channels[-1]),
            nn.GELU(),
            nn.Linear(hidden_channels[-1], latent_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode field to latent code.
        
        Args:
            x: Input field
               - 1D: [batch, spatial] or [batch, channels, spatial]
               - 2D: [batch, height, width] or [batch, channels, height, width]
               
        Returns:
            z: Latent code [batch, latent_dim]
        """
        # Ensure proper input shape
        if self.spatial_dim == 1:
            if x.dim() == 2:
                x = x.unsqueeze(1)  # [B, S] -> [B, 1, S]
        else:
            if x.dim() == 3:
                x = x.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        # CNN backbone
        x = self.backbone(x)
        
        # Global average pooling
        x = self.pool(x)
        x = x.flatten(1)  # [B, C]
        
        # Project to latent space
        z = self.head(x)
        
        return z


class ForwardEncoder(FieldEncoder):
    """
    Forward Encoder E_f: Maps input source field f to latent code z_f.
    
    z_f = E_f(f; φ_f)
    
    This encoder remains active during both training and inference,
    providing the latent prior for conditioning the operator.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        hidden_channels: list = [32, 64, 128],
        spatial_dim: int = 1,
    ):
        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            spatial_dim=spatial_dim,
        )


class ReverseEncoder(FieldEncoder):
    """
    Reverse Encoder E_u: Maps solution field u to latent code z_u.
    
    z_u = E_u(u; φ_u)
    
    This encoder is only used during training (Stages I and II) to establish
    bidirectional latent reciprocity. It is discarded during inference
    (Stage III and deployment).
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        latent_dim: int = 64,
        hidden_channels: list = [32, 64, 128],
        spatial_dim: int = 1,
    ):
        super().__init__(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=hidden_channels,
            spatial_dim=spatial_dim,
        )


class ProjectionHead(nn.Module):
    """
    Optional projection head for contrastive learning.
    
    Projects latent codes to a normalized embedding space for computing
    similarity in the InfoNCE loss. This can improve representation quality
    by decoupling the contrastive objective from the reconstruction objective.
    """
    
    def __init__(self, latent_dim: int, projection_dim: int = 128):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, projection_dim),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Project and L2-normalize latent codes."""
        z_proj = self.projector(z)
        z_proj = F.normalize(z_proj, dim=-1)
        return z_proj
