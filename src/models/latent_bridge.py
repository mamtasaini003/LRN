"""
Latent Bridge Module for LRN

Implements the latent injection mechanism that conditions the FNO backbone
on the latent code z_f after K Fourier layers.

The bridge broadcasts z_f spatially and fuses it with backbone features:
    v^latent = σ(MLP(v_K ⊕ Proj(z_f)))

This regularizes the operator to produce physically plausible outputs
constrained by the learned reciprocal manifold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LatentBridge(nn.Module):
    """
    Latent Bridge for injecting latent codes into FNO features.
    
    After K backbone layers produce v_K, the latent bridge:
    1. Projects z_f to feature-compatible dimension
    2. Broadcasts z_f spatially to match v_K dimensions
    3. Concatenates v_K and projected z_f
    4. Fuses through MLP: v^latent = σ(MLP(v_K ⊕ Proj(z_f)))
    """
    
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        projection_dim: Optional[int] = None,
        spatial_dim: int = 1,
    ):
        """
        Args:
            feature_dim: Dimension of backbone features (FNO width)
            latent_dim: Dimension of latent codes (d_z)
            projection_dim: Dimension for projecting latent (default: feature_dim)
            spatial_dim: 1 for 1D, 2 for 2D spatial data
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim or feature_dim
        self.spatial_dim = spatial_dim
        
        # Project latent to feature-compatible dimension: Proj(z_f)
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, self.projection_dim),
            nn.GELU(),
            nn.Linear(self.projection_dim, self.projection_dim),
        )
        
        # Fusion MLP: MLP(v_K ⊕ Proj(z_f))
        fused_dim = feature_dim + self.projection_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        z_f: torch.Tensor
    ) -> torch.Tensor:
        """
        Inject latent code into backbone features.
        
        Args:
            features: Backbone features v_K
                - 1D: [batch, spatial, feature_dim]
                - 2D: [batch, height, width, feature_dim]
            z_f: Latent code from forward encoder [batch, latent_dim]
            
        Returns:
            v_latent: Fused features [batch, ..., feature_dim]
        """
        # Project latent: [B, latent_dim] -> [B, projection_dim]
        z_proj = self.latent_projection(z_f)
        
        # Broadcast spatially to match feature dimensions
        if self.spatial_dim == 1:
            # [B, proj_dim] -> [B, S, proj_dim]
            spatial_size = features.shape[1]
            z_broadcast = z_proj.unsqueeze(1).expand(-1, spatial_size, -1)
        else:
            # [B, proj_dim] -> [B, H, W, proj_dim]
            height, width = features.shape[1], features.shape[2]
            z_broadcast = z_proj.unsqueeze(1).unsqueeze(2).expand(-1, height, width, -1)
        
        # Concatenate: v_K ⊕ Proj(z_f)
        fused = torch.cat([features, z_broadcast], dim=-1)
        
        # Apply fusion MLP: v^latent = σ(MLP(...))
        v_latent = self.fusion_mlp(fused)
        
        return v_latent


class GatedLatentBridge(nn.Module):
    """
    Gated variant of the Latent Bridge.
    
    Uses a gating mechanism to control how much latent information
    is injected into the features, allowing the model to learn
    adaptive modulation.
    
    v^latent = v_K + gate * transform(v_K, z_f)
    """
    
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        projection_dim: Optional[int] = None,
        spatial_dim: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.projection_dim = projection_dim or feature_dim
        self.spatial_dim = spatial_dim
        
        # Latent projection
        self.latent_projection = nn.Sequential(
            nn.Linear(latent_dim, self.projection_dim),
            nn.GELU(),
        )
        
        # Transform branch
        fused_dim = feature_dim + self.projection_dim
        self.transform = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        
        # Gating branch
        self.gate = nn.Sequential(
            nn.Linear(fused_dim, feature_dim),
            nn.Sigmoid(),
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        z_f: torch.Tensor
    ) -> torch.Tensor:
        """
        Gated latent injection.
        
        Args:
            features: Backbone features v_K
            z_f: Latent code from forward encoder
            
        Returns:
            v_latent: Gated fused features
        """
        z_proj = self.latent_projection(z_f)
        
        # Broadcast
        if self.spatial_dim == 1:
            spatial_size = features.shape[1]
            z_broadcast = z_proj.unsqueeze(1).expand(-1, spatial_size, -1)
        else:
            height, width = features.shape[1], features.shape[2]
            z_broadcast = z_proj.unsqueeze(1).unsqueeze(2).expand(-1, height, width, -1)
        
        # Concatenate
        fused = torch.cat([features, z_broadcast], dim=-1)
        
        # Compute gated modulation
        gate = self.gate(fused)
        modulation = self.transform(fused)
        
        # Apply residual gated connection
        v_latent = features + gate * modulation
        
        return v_latent


class SpectralLatentBridge(nn.Module):
    """
    Spectral variant of the Latent Bridge.
    
    Modulates features in the Fourier domain, which is particularly
    suited for FNO architectures. The latent code controls spectral
    mode amplitudes to reduce aliasing in high modes.
    """
    
    def __init__(
        self,
        feature_dim: int,
        latent_dim: int,
        num_modes: int = 16,
        spatial_dim: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.spatial_dim = spatial_dim
        
        # Generate spectral modulation weights from latent
        self.mode_generator = nn.Sequential(
            nn.Linear(latent_dim, feature_dim * num_modes * 2),  # *2 for complex
        )
        
        # Fallback spatial bridge for non-spectral components
        self.spatial_bridge = LatentBridge(
            feature_dim=feature_dim,
            latent_dim=latent_dim,
            spatial_dim=spatial_dim,
        )
    
    def forward(
        self, 
        features: torch.Tensor, 
        z_f: torch.Tensor
    ) -> torch.Tensor:
        """
        Spectral latent modulation.
        
        For 1D: Modulates Fourier modes of feature channels.
        Falls back to spatial bridge for simplicity in current implementation.
        """
        # Current implementation uses spatial bridge
        # Full spectral modulation can be added for advanced use
        return self.spatial_bridge(features, z_f)
