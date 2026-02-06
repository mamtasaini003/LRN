"""
LRN-FNO Base Model Definitions.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Dict

from ..components.fno import FNO1d, FNO2d
from ..components.encoders import ForwardEncoder, ReverseEncoder
from ..components.latent_bridge import LatentBridge


class LRNFNO1d(nn.Module):
    """
    Latent Rescaling Network (LRN) wrapping FNO1d backbone.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        width: int,
        num_layers: int = 4,
        latent_dim: int = 128,
        encoder_channels: List[int] = [16, 32, 64],
        use_gated_bridge: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.width = width
        self.latent_dim = latent_dim
        
        # Backbone
        self.fno = FNO1d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            width=width,
            num_layers=num_layers
        )
        
        # Encoders
        self.encoder_f = ForwardEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=encoder_channels,
            spatial_dim=1
        )
        
        self.encoder_u = ReverseEncoder(
            in_channels=out_channels,
            latent_dim=latent_dim,
            hidden_channels=encoder_channels,
            spatial_dim=1
        )
        
        # Bridge
        self.latent_bridge = LatentBridge(
            feature_dim=width,
            latent_dim=latent_dim,
            spatial_dim=1
        )
        
        self._inference_mode = False

    def set_inference_mode(self, mode: bool = True):
        self._inference_mode = mode

    def forward(
        self, 
        f: torch.Tensor, 
        u: Optional[torch.Tensor] = None,
        return_latents: bool = True
    ) -> Dict[str, torch.Tensor]:
        output = {}
        
        # Encode input
        z_f = self.encoder_f(f)
        output['z_f_input'] = z_f # Store raw encoder output 
        
        # Encode solution if available (and not inference)
        z_u = None
        if u is not None and not self._inference_mode:
            z_u = self.encoder_u(u)
            if return_latents:
                output['z_u'] = z_u

        # Backbone features
        v_K = self.fno.backbone_forward(f)
        
        # Bridge
        v_latent = self.latent_bridge(v_K, z_f)
        
        # Projection
        pred = self.fno.project(v_latent)
        
        # Permute like FNO1d typically needs if output is [B, L, C] -> [B, C, L]
        # But FNO1d.project likely outputs [B, L, out_channels].
        # In demos, we permute manually. Let's return as is or standardize?
        # Standard FNO1d returns [B, L, C]. FNO2d returns [B, H, W, C].
        # We'll leave permutations to the user or subclass.
        
        output['prediction'] = pred
        if return_latents:
             # Default z_f is just the input encoder latent
            output['z_f'] = z_f
            
        return output


class LRNFNO2d(nn.Module):
    """
    Latent Rescaling Network (LRN) wrapping FNO2d backbone.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        width: int,
        num_layers: int = 4,
        latent_dim: int = 128,
        encoder_channels: List[int] = [16, 32, 64],
        use_gated_bridge: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.latent_dim = latent_dim
        
        # Backbone
        self.fno = FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            num_layers=num_layers
        )
        
        # Encoders
        self.encoder_f = ForwardEncoder(
            in_channels=in_channels,
            latent_dim=latent_dim,
            hidden_channels=encoder_channels,
            spatial_dim=2
        )
        
        self.encoder_u = ReverseEncoder(
            in_channels=out_channels,
            latent_dim=latent_dim,
            hidden_channels=encoder_channels,
            spatial_dim=2
        )
        
        # Bridge
        self.latent_bridge = LatentBridge(
            feature_dim=width,
            latent_dim=latent_dim,
            spatial_dim=2
        )
        
        self._inference_mode = False

    def set_inference_mode(self, mode: bool = True):
        self._inference_mode = mode

    def forward(
        self, 
        f: torch.Tensor, 
        u: Optional[torch.Tensor] = None,
        return_latents: bool = True
    ) -> Dict[str, torch.Tensor]:
        output = {}
        
        # Encode input
        z_f = self.encoder_f(f)
        output['z_f_input'] = z_f
        
        # Encode solution
        if u is not None and not self._inference_mode:
            z_u = self.encoder_u(u)
            if return_latents:
                output['z_u'] = z_u

        # Backbone
        v_K = self.fno.backbone_forward(f)
        
        # Bridge
        v_latent = self.latent_bridge(v_K, z_f)
        
        # Projection
        pred = self.fno.project(v_latent)
        
        output['prediction'] = pred
        if return_latents:
            output['z_f'] = z_f
            
        return output
