"""
LRR-FNO Model Definitions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from ..lrn.model import LRNFNO1d, LRNFNO2d


class LRRFNO1d(LRNFNO1d):
    """
    LRR-FNO 1D.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        hidden_dim = 128
        self.vk_projection = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(self.width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.latent_dim)
        )
    
    def forward(
        self,
        f: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        return_latents: bool = True,
    ) -> Dict[str, torch.Tensor]:
        output = {}
        
        # Encode solution if available (for NCE loss target)
        z_u = None
        if u is not None and not self._inference_mode:
            z_u = self.encoder_u(u)
            
        # FNO Features v_K
        v_K = self.fno.backbone_forward(f) # [B, S, W]
        
        # Latent Supervision: Project v_K to match z_u space
        v_K_perm = v_K.permute(0, 2, 1)
        z_v_K = self.vk_projection(v_K_perm) # [B, latent_dim]
        
        # Latent bridge with ZERO context (no encoder_f dependency)
        # This isolates the effect of latent supervision
        batch_size = f.shape[0]
        z_zero = torch.zeros(batch_size, self.latent_dim, device=f.device)
        v_latent = self.latent_bridge(v_K, z_zero)
        
        # Output projection
        u_pred = self.fno.project(v_latent)
        
        # Permute to [B, C, L]
        u_pred = u_pred.permute(0, 2, 1)
            
        output['prediction'] = u_pred
        
        if return_latents:
            # z_v_k: Projected backbone features for latent supervision
            # This aligns with z_u (solution latent) via InfoNCE loss
            output['z_v_k'] = z_v_K
            output['z_f'] = z_v_K  # Backward compatibility alias
            if z_u is not None:
                output['z_u'] = z_u
            
        return output


class LRRFNO2d(LRNFNO2d):
    """
    LRR-FNO 2D.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Projection Head: GAP + MLP
        # v_K from FNO2d is [B, H, W, W_chan]
        # Match FieldEncoder head capacity: Linear -> GELU -> Linear
        hidden_dim = 128
        self.vk_projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.width, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.latent_dim)
        )

    def forward(
        self,
        f: torch.Tensor,
        u: Optional[torch.Tensor] = None,
        return_latents: bool = True,
    ) -> Dict[str, torch.Tensor]:
        output = {}
        
        # Encode solution if available (for NCE loss target)
        z_u = None
        if u is not None and not self._inference_mode:
            z_u = self.encoder_u(u)

        # FNO Features v_K
        v_K = self.fno.backbone_forward(f) # [B, H, W, C]
        
        # Latent Supervision: Project v_K to match z_u space
        v_K_perm = v_K.permute(0, 3, 1, 2)
        z_v_K = self.vk_projection(v_K_perm)
        
        # Latent bridge with ZERO context (no encoder_f dependency)
        # This isolates the effect of latent supervision
        batch_size = f.shape[0]
        z_zero = torch.zeros(batch_size, self.latent_dim, device=f.device)
        v_latent = self.latent_bridge(v_K, z_zero)
        
        # Projection
        u_pred = self.fno.project(v_latent)
        
        # Permute back to [B, C, H, W]
        u_pred = u_pred.permute(0, 3, 1, 2) 
        
        output['prediction'] = u_pred
        
        if return_latents:
            # z_v_k: Projected backbone features for latent supervision
            # This aligns with z_u (solution latent) via InfoNCE loss
            output['z_v_k'] = z_v_K
            output['z_f'] = z_v_K  # Backward compatibility alias
            if z_u is not None:
                output['z_u'] = z_u
            
        return output
