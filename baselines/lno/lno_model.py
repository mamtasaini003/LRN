"""
Latent Neural Operator (LNO) - PyTorch Implementation

Paper: "Latent Neural Operator for Solving Forward and Inverse PDE Problems"
       Wang & Wang, NeurIPS 2024
       
Reference: https://github.com/L-I-M-I-T/LatentNeuralOperator

This implementation follows the paper architecture:
1. Embedding Layer: trunk-projector (position) and branch-projector (position + value)
2. PhCA Encoder: Transform from geometric space to latent space
3. Transformer Blocks: Learn operator in latent space
4. PhCA Decoder: Transform from latent space back to geometric space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron."""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        num_layers: int = 2,
        activation: nn.Module = nn.GELU
    ):
        super().__init__()
        layers = []
        
        if num_layers == 1:
            layers.append(nn.Linear(in_features, out_features))
        else:
            layers.append(nn.Linear(in_features, hidden_features))
            layers.append(activation())
            
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_features, hidden_features))
                layers.append(activation())
            
            layers.append(nn.Linear(hidden_features, out_features))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionProjector(nn.Module):
    """
    Attention Projector for Physics-Cross-Attention.
    
    This is the generalized linear projection W1, W2 in the paper,
    implemented as an MLP for improved performance.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsCrossAttention(nn.Module):
    """
    Physics-Cross-Attention (PhCA) Module.
    
    The core module in LNO for transforming between geometric and latent spaces.
    
    For encoding (geometric -> latent):
        Z = softmax(W1 @ X^T) @ Y @ Wv
        where X is position embeddings and Y is position+value embeddings
    
    For decoding (latent -> geometric):
        U = softmax(P @ W2^T) @ Z @ Wv'
        where P is query position embeddings
    
    Key insight: W1 = W2 (shared weights between encoder and decoder)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        share_projector: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.share_projector = share_projector
        
        # Attention projector (shared for encoder and decoder)
        self.attn_projector = AttentionProjector(embed_dim)
        
        # Value projection
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def encode(
        self,
        query_latent: torch.Tensor,  # [B, M, D] - learnable latent positions
        key_pos: torch.Tensor,       # [B, N, D] - geometric position embeddings
        value: torch.Tensor          # [B, N, D] - position+value embeddings
    ) -> torch.Tensor:
        """
        Encode from geometric space to latent space.
        
        Args:
            query_latent: Learnable latent position embeddings [B, M, D]
            key_pos: Position embeddings from geometric space [B, N, D]  
            value: Position+value embeddings [B, N, D]
        
        Returns:
            Latent representation [B, M, D]
        """
        B, M, D = query_latent.shape
        N = key_pos.shape[1]
        
        # Apply attention projector
        # In the simplified equation: softmax(W1 @ X^T) @ Y @ Wv
        # where W1 absorbs the learnable query H
        attn_key = self.attn_projector(key_pos)  # [B, N, D]
        
        # Compute attention scores
        # query_latent acts as the learnable latent positions
        attn_scores = torch.bmm(query_latent, attn_key.transpose(1, 2))  # [B, M, N]
        attn_scores = attn_scores * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply value projection and compute output
        v = self.v_proj(value)  # [B, N, D]
        output = torch.bmm(attn_probs, v)  # [B, M, D]
        output = self.out_proj(output)
        
        return output
    
    def decode(
        self,
        query_pos: torch.Tensor,     # [B, N_out, D] - output position embeddings
        key_latent: torch.Tensor,    # [B, M, D] - learnable latent positions
        value_latent: torch.Tensor   # [B, M, D] - latent representations
    ) -> torch.Tensor:
        """
        Decode from latent space to geometric space.
        
        Args:
            query_pos: Output position embeddings [B, N_out, D]
            key_latent: Learnable latent position embeddings [B, M, D]
            value_latent: Latent representations [B, M, D]
        
        Returns:
            Output in geometric space [B, N_out, D]
        """
        B, N_out, D = query_pos.shape
        M = key_latent.shape[1]
        
        # Apply shared attention projector (W2 = W1)
        attn_query = self.attn_projector(query_pos)  # [B, N_out, D]
        
        # Compute attention scores
        attn_scores = torch.bmm(attn_query, key_latent.transpose(1, 2))  # [B, N_out, M]
        attn_scores = attn_scores * self.scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply value projection and compute output
        v = self.v_proj(value_latent)  # [B, M, D]
        output = torch.bmm(attn_probs, v)  # [B, N_out, D]
        output = self.out_proj(output)
        
        return output


class TransformerBlock(nn.Module):
    """
    Standard Transformer Block with Pre-LayerNorm.
    
    Uses scaled dot-product attention as per the paper's ablation study,
    which shows it performs better than linear attention variants.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LayerNorm Transformer
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        
        return x


class LNO2d(nn.Module):
    """
    2D Latent Neural Operator.
    
    Architecture:
    1. Trunk-projector: embeds positions to D-dimensional space
    2. Branch-projector: embeds position+value pairs to D-dimensional space
    3. PhCA Encoder: transforms to latent space (M tokens)
    4. Transformer Blocks: processes in latent space
    5. PhCA Decoder: transforms back to geometric space
    6. Output MLP: maps to output values
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        embed_dim: Embedding dimension (D in paper)
        latent_size: Number of latent tokens (M in paper)
        num_layers: Number of transformer layers (L in paper)
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dimension ratio
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int = 128,
        latent_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        resolution: int = None  # Optional: for fixed grid
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.resolution = resolution
        
        # Trunk-projector: embeds positions only (2D coordinates)
        self.trunk_projector = MLP(
            in_features=2,  # (x, y) coordinates
            hidden_features=embed_dim,
            out_features=embed_dim,
            num_layers=3
        )
        
        # Branch-projector: embeds position + value
        self.branch_projector = MLP(
            in_features=2 + in_channels,  # (x, y) + value
            hidden_features=embed_dim,
            out_features=embed_dim,
            num_layers=3
        )
        
        # Learnable latent positions (H in the paper)
        # These are the hypothetical sampling positions in latent space
        self.latent_positions = nn.Parameter(torch.randn(1, latent_size, embed_dim))
        
        # PhCA for encoding and decoding (shared weights)
        self.phca = PhysicsCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            share_projector=True
        )
        
        # Transformer blocks for processing in latent space
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = MLP(
            in_features=embed_dim,
            hidden_features=embed_dim,
            out_features=out_channels,
            num_layers=2
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Initialize latent positions
        nn.init.trunc_normal_(self.latent_positions, std=0.02)
    
    def _create_coordinate_grid(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device
    ) -> torch.Tensor:
        """Create normalized coordinate grid."""
        # Create grid coordinates normalized to [0, 1]
        y = torch.linspace(0, 1, height, device=device)
        x = torch.linspace(0, 1, width, device=device)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        
        # Stack and reshape to [H*W, 2]
        coords = torch.stack([grid_x, grid_y], dim=-1)  # [H, W, 2]
        coords = coords.reshape(-1, 2)  # [H*W, 2]
        
        # Expand for batch
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)  # [B, H*W, 2]
        
        return coords
    
    def forward(
        self,
        x: torch.Tensor,
        query_coords: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C_in, H, W]
            query_coords: Optional query coordinates for decoupled prediction
                          [B, N_out, 2]. If None, uses same grid as input.
        
        Returns:
            Output tensor [B, C_out, H, W] or [B, N_out, C_out] if query_coords given
        """
        B, C_in, H, W = x.shape
        device = x.device
        
        # 1. Create coordinate grid
        coords = self._create_coordinate_grid(B, H, W, device)  # [B, H*W, 2]
        N = coords.shape[1]
        
        # 2. Flatten input values
        values = x.reshape(B, C_in, -1).permute(0, 2, 1)  # [B, H*W, C_in]
        
        # 3. Apply projectors
        # Trunk-projector: position only
        pos_embed = self.trunk_projector(coords)  # [B, N, D]
        
        # Branch-projector: position + value
        pos_val = torch.cat([coords, values], dim=-1)  # [B, N, 2 + C_in]
        val_embed = self.branch_projector(pos_val)  # [B, N, D]
        
        # 4. Get latent positions
        latent_pos = self.latent_positions.expand(B, -1, -1)  # [B, M, D]
        
        # 5. PhCA Encoding: geometric -> latent
        z = self.phca.encode(latent_pos, pos_embed, val_embed)  # [B, M, D]
        
        # 6. Process in latent space with Transformer blocks
        for block in self.transformer_blocks:
            z = block(z)
        
        # 7. PhCA Decoding: latent -> geometric
        if query_coords is not None:
            # Use provided query coordinates
            query_pos_embed = self.trunk_projector(query_coords)  # [B, N_out, D]
            output = self.phca.decode(query_pos_embed, latent_pos, z)  # [B, N_out, D]
            output = self.output_proj(output)  # [B, N_out, C_out]
        else:
            # Use same coordinates as input
            output = self.phca.decode(pos_embed, latent_pos, z)  # [B, N, D]
            output = self.output_proj(output)  # [B, N, C_out]
            
            # Reshape back to image format
            output = output.permute(0, 2, 1).reshape(B, self.out_channels, H, W)
        
        return output


class LNO1d(nn.Module):
    """
    1D Latent Neural Operator (e.g., for Burgers equation).
    
    Similar to LNO2d but for 1D spatial domains with optional time dimension.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embed_dim: int = 96,
        latent_size: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        include_time: bool = True  # Whether input includes time dimension
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.latent_size = latent_size
        self.include_time = include_time
        
        coord_dim = 2 if include_time else 1  # (x, t) or just x
        
        # Projectors
        self.trunk_projector = MLP(
            in_features=coord_dim,
            hidden_features=embed_dim,
            out_features=embed_dim,
            num_layers=3
        )
        
        self.branch_projector = MLP(
            in_features=coord_dim + in_channels,
            hidden_features=embed_dim,
            out_features=embed_dim,
            num_layers=3
        )
        
        # Learnable latent positions
        self.latent_positions = nn.Parameter(torch.randn(1, latent_size, embed_dim))
        
        # PhCA
        self.phca = PhysicsCrossAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = MLP(
            in_features=embed_dim,
            hidden_features=embed_dim,
            out_features=out_channels,
            num_layers=2
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.trunc_normal_(self.latent_positions, std=0.02)
    
    def forward(
        self,
        x: torch.Tensor,
        input_coords: torch.Tensor = None,
        query_coords: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input values [B, N_in, C_in] or [B, C_in, H, W] for 2D grid
            input_coords: Input coordinates [B, N_in, coord_dim]
            query_coords: Query coordinates [B, N_out, coord_dim]
        
        Returns:
            Output tensor
        """
        # Handle different input formats
        if x.dim() == 4:
            # [B, C, H, W] format - flatten to sequence
            B, C, H, W = x.shape
            x = x.reshape(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
            
            # Create coordinates
            device = x.device
            t = torch.linspace(0, 1, H, device=device)
            s = torch.linspace(0, 1, W, device=device)
            grid_t, grid_s = torch.meshgrid(t, s, indexing='ij')
            coords = torch.stack([grid_s, grid_t], dim=-1).reshape(-1, 2)
            input_coords = coords.unsqueeze(0).expand(B, -1, -1)
            
            if query_coords is None:
                query_coords = input_coords
            
            output = self._forward_sequence(x, input_coords, query_coords)
            output = output.permute(0, 2, 1).reshape(B, self.out_channels, H, W)
            return output
        else:
            return self._forward_sequence(x, input_coords, query_coords)
    
    def _forward_sequence(
        self,
        values: torch.Tensor,
        input_coords: torch.Tensor,
        query_coords: torch.Tensor
    ) -> torch.Tensor:
        """Process sequence data."""
        B = values.shape[0]
        device = values.device
        
        # Embed positions and values
        pos_embed = self.trunk_projector(input_coords)
        pos_val = torch.cat([input_coords, values], dim=-1)
        val_embed = self.branch_projector(pos_val)
        
        # Latent positions
        latent_pos = self.latent_positions.expand(B, -1, -1)
        
        # Encode
        z = self.phca.encode(latent_pos, pos_embed, val_embed)
        
        # Process
        for block in self.transformer_blocks:
            z = block(z)
        
        # Decode
        if query_coords is not None:
            query_embed = self.trunk_projector(query_coords)
            output = self.phca.decode(query_embed, latent_pos, z)
        else:
            output = self.phca.decode(pos_embed, latent_pos, z)
        
        output = self.output_proj(output)
        
        return output


# Convenience function for creating LNO with recommended hyperparameters
def create_lno_darcy(in_channels: int = 1, out_channels: int = 1) -> LNO2d:
    """Create LNO for Darcy flow problem (paper configuration)."""
    return LNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=128,
        latent_size=256,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    )


def create_lno_ns2d(in_channels: int = 10, out_channels: int = 10) -> LNO2d:
    """Create LNO for 2D Navier-Stokes problem (paper configuration)."""
    return LNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=256,
        latent_size=256,
        num_layers=8,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    )


def create_lno_burgers2d(in_channels: int = 2, out_channels: int = 2) -> LNO2d:
    """Create LNO for 2D coupled Burgers equation."""
    return LNO2d(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=128,
        latent_size=256,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0
    )


if __name__ == "__main__":
    # Test the model
    print("Testing LNO2d...")
    
    model = LNO2d(
        in_channels=2,
        out_channels=2,
        embed_dim=128,
        latent_size=64,
        num_layers=4,
        num_heads=8
    )
    
    x = torch.randn(4, 2, 32, 32)
    y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nTesting LNO1d...")
    
    model1d = LNO1d(
        in_channels=1,
        out_channels=1,
        embed_dim=96,
        latent_size=64,
        num_layers=4
    )
    
    # Test with 2D grid input
    x1d = torch.randn(4, 1, 64, 64)
    y1d = model1d(x1d)
    
    print(f"Input shape: {x1d.shape}")
    print(f"Output shape: {y1d.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model1d.parameters()):,}")
    
    print("\nAll tests passed!")
