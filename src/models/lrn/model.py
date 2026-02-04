"""
LRN-FNO Model Definitions.

Standard Latent Reciprocity Network implementation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from ..components.fno import FNO1d, FNO2d
from ..components.encoders import ForwardEncoder, ReverseEncoder
from ..components.latent_bridge import LatentBridge, GatedLatentBridge


class LRNFNO1d(nn.Module):
    """
    LRN-FNO for 1D PDEs (e.g., Burgers equation).
    
    Integrates bidirectional latent alignment with FNO1d backbone.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: int = 16,
        width: int = 64,
        num_layers: int = 4,
        latent_dim: int = 64,
        encoder_channels: list = [32, 64, 128],
        padding: int = 8,
        use_gated_bridge: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.width = width
        
        # Forward encoder E_f: f → z_f
        self.encoder_f = ForwardEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=encoder_channels,
            spatial_dim=1,
        )
        
        # Reverse encoder E_u: u → z_u (discarded at inference)
        self.encoder_u = ReverseEncoder(
            in_channels=out_channels,
            latent_dim=latent_dim,
            hidden_channels=encoder_channels,
            spatial_dim=1,
        )
        
        # FNO backbone G_θ
        self.fno = FNO1d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            width=width,
            num_layers=num_layers,
            padding=padding,
        )
        
        # Latent bridge for injection
        BridgeClass = GatedLatentBridge if use_gated_bridge else LatentBridge
        self.latent_bridge = BridgeClass(
            feature_dim=width,
            latent_dim=latent_dim,
            spatial_dim=1,
        )
        
        # Flag for inference mode (E_u discarded)
        self._inference_mode = False
    
    def set_inference_mode(self, mode: bool = True):
        """Enable/disable inference mode (discards E_u usage)."""
        self._inference_mode = mode
    
    def forward(
        self,
        f: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        return_latents: bool = True,
    ) -> Dict[str, torch.Tensor]:
        output = {}
        
        # Encode input: z_f = E_f(f)
        z_f = self.encoder_f(f)
        
        # Encode solution if provided and not in inference mode
        z_u = None
        if u is not None and not self._inference_mode:
            z_u = self.encoder_u(u)
        
        # FNO backbone: f → v_K
        v_K = self.fno.backbone_forward(f)
        
        # Latent bridge injection
        v_latent = self.latent_bridge(v_K, z_f)
        
        # Output projection
        u_pred = self.fno.project(v_latent)
        
        if u_pred.shape[-1] == 1:
            u_pred = u_pred.squeeze(-1)
        
        output['prediction'] = u_pred
        
        if return_latents:
            output['z_f'] = z_f
            if z_u is not None:
                output['z_u'] = z_u
        
        return output
    
    def forward_simple(self, f: torch.Tensor) -> torch.Tensor:
        """Simplified forward for inference - returns prediction only."""
        output = self.forward(f, return_latents=False)
        return output['prediction']


class LRNFNO2d(nn.Module):
    """
    LRN-FNO for 2D PDEs (e.g., Darcy flow, Navier-Stokes).
    
    Integrates bidirectional latent alignment with FNO2d backbone.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes1: int = 12,
        modes2: int = 12,
        width: int = 32,
        num_layers: int = 4,
        latent_dim: int = 64,
        encoder_channels: list = [32, 64, 128],
        padding: int = 9,
        use_gated_bridge: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.width = width
        
        # Forward encoder E_f
        self.encoder_f = ForwardEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=encoder_channels,
            spatial_dim=2,
        )
        
        # Reverse encoder E_u
        self.encoder_u = ReverseEncoder(
            in_channels=out_channels,
            latent_dim=latent_dim,
            hidden_channels=encoder_channels,
            spatial_dim=2,
        )
        
        # FNO2d backbone
        self.fno = FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            num_layers=num_layers,
            padding=padding,
        )
        
        # Latent bridge
        BridgeClass = GatedLatentBridge if use_gated_bridge else LatentBridge
        self.latent_bridge = BridgeClass(
            feature_dim=width,
            latent_dim=latent_dim,
            spatial_dim=2,
        )
        
        self._inference_mode = False
    
    def set_inference_mode(self, mode: bool = True):
        """Enable/disable inference mode."""
        self._inference_mode = mode
    
    def forward(
        self,
        f: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        return_latents: bool = True,
    ) -> Dict[str, torch.Tensor]:
        output = {}
        
        # Encode input
        z_f = self.encoder_f(f)
        
        # Encode solution if available
        z_u = None
        if u is not None and not self._inference_mode:
            z_u = self.encoder_u(u)
        
        # FNO backbone
        v_K = self.fno.backbone_forward(f)
        
        # Latent bridge
        v_latent = self.latent_bridge(v_K, z_f)
        
        # Projection
        u_pred = self.fno.project(v_latent)
        
        # u_pred is [B, H, W, C]. Permute to [B, C, H, W]
        u_pred = u_pred.permute(0, 3, 1, 2)
        
        if u_pred.shape[1] == 1:
            u_pred = u_pred.squeeze(1)
        
        output['prediction'] = u_pred
        
        if return_latents:
            output['z_f'] = z_f
            if z_u is not None:
                output['z_u'] = z_u
        
        return output
    
    def forward_simple(self, f: torch.Tensor) -> torch.Tensor:
        """Simplified forward for inference."""
        output = self.forward(f, return_latents=False)
        return output['prediction']


def create_lrn_fno(
    spatial_dim: int = 1,
    **kwargs
) -> nn.Module:
    """
    Factory function to create LRN-FNO model.
    """
    if spatial_dim == 1:
        return LRNFNO1d(**kwargs)
    elif spatial_dim == 2:
        return LRNFNO2d(**kwargs)
    else:
        raise ValueError(f"Unsupported spatial_dim: {spatial_dim}")
