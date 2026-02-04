"""
LRR-FNO Model Definitions.

Latent Reciprocity Representation (LRR) implementation where the backbone
features v_K are projected to align with the solution latent z_u.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

from ..lrn.model import LRNFNO1d, LRNFNO2d


class LRRFNO1d(LRNFNO1d):
    """
    Latent Reciprocity Representation FNO (LRR-FNO) for 1D.
    
    Implements Latent Space Supervision where internal backbone features v_K
    are projected and aligned with the solution latent z_u.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Projection Head: GAP + MLP
        # v_K from FNO1d is [B, S, W]
        # Match FieldEncoder head capacity: Linear -> GELU -> Linear
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
        
        # Encode input z_f (still used for context injection)
        z_f_input = self.encoder_f(f)
        
        # Encode solution if available
        z_u = None
        if u is not None and not self._inference_mode:
            z_u = self.encoder_u(u)
            
        # FNO Features v_K
        v_K = self.fno.backbone_forward(f) # [B, S, W]
        
        # Latent Supervision: Project v_K to match z_u space
        v_K_perm = v_K.permute(0, 2, 1)
        z_v_K = self.vk_projection(v_K_perm) # [B, latent_dim]
        
        # Latent bridge uses z_f_input (input context)
        v_latent = self.latent_bridge(v_K, z_f_input)
        
        # Output projection
        u_pred = self.fno.project(v_latent)
            
        output['prediction'] = u_pred
        
        if return_latents:
            # RETURN TRICK: return projected backbone features as z_f
            # so the loss function aligns v_K with z_u.
            output['z_f'] = z_v_K 
            if z_u is not None:
                output['z_u'] = z_u
            
            # For logging or debugging, return real input latent
            output['z_f_real'] = z_f_input 
            
        return output


class LRRFNO2d(LRNFNO2d):
    """
    Latent Reciprocity Representation FNO (LRR-FNO) for 2D.
    
    Implements Latent Space Supervision where internal backbone features v_K
    are projected and aligned with the solution latent z_u.
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
        
        # Encode input z_f (for bridge context)
        z_f_input = self.encoder_f(f)
        
        # Encode solution if available
        z_u = None
        if u is not None and not self._inference_mode:
            z_u = self.encoder_u(u)

        # FNO Features v_K
        v_K = self.fno.backbone_forward(f) # [B, H, W, C]
        
        # Latent Supervision: Project v_K to match z_u space
        v_K_perm = v_K.permute(0, 3, 1, 2)
        z_v_K = self.vk_projection(v_K_perm)
        
        # Latent bridge
        # Normalize z_f_input before injection to ensure unit norm (consistent with NCE)
        z_f_norm = F.normalize(z_f_input, dim=-1)
        v_latent = self.latent_bridge(v_K, z_f_norm)
        
        # Projection
        u_pred = self.fno.project(v_latent)
        
        # Permute back to [B, C, H, W]
        u_pred = u_pred.permute(0, 3, 1, 2) 
        
        output['prediction'] = u_pred
        
        if return_latents:
            # RETURN TRICK: return projected backbone features as z_f
            # so the loss function aligns v_K with z_u.
            output['z_f'] = z_v_K
            if z_u is not None:
                output['z_u'] = z_u
            
            output['z_f_real'] = z_f_input
            
        return output
