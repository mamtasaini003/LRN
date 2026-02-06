"""
InfoNCE Loss for Latent Reciprocity

Implements the InfoNCE contrastive loss for enforcing bidirectional
latent alignment between source field embeddings z_f and solution embeddings z_u.

L_NCE = -Σ_i log( exp(sim(z_f,i, z_u,i)/τ) / Σ_j exp(sim(z_f,i, z_u,j)/τ) )

This maximizes mutual information between matched (f, u) pairs while
distinguishing mismatches, shaping a PDE-consistent latent manifold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class InfoNCELoss(nn.Module):
    """
    InfoNCE Contrastive Loss for Latent Reciprocity.
    
    Enforces that matched (f_i, u_i) pairs have similar latent embeddings
    while unmatched pairs are pushed apart in the latent space.
    
    Uses cosine similarity and cross-entropy formulation for numerical stability.
    """
    
    def __init__(self, temperature: float = 0.1):
        """
        Args:
            temperature: Temperature parameter τ for scaling similarities.
                        Lower values create sharper distributions.
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self, 
        z_f: torch.Tensor, 
        z_u: torch.Tensor,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            z_f: Forward encoder embeddings [batch, latent_dim]
            z_u: Reverse encoder embeddings [batch, latent_dim]
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            loss: Scalar InfoNCE loss value
        """
        batch_size = z_f.shape[0]
        device = z_f.device
        
        # L2 normalize embeddings for cosine similarity
        if normalize:
            z_f = F.normalize(z_f, dim=-1)
            z_u = F.normalize(z_u, dim=-1)
        
        # Compute similarity matrix: sim(z_f_i, z_u_j) for all pairs
        # Shape: [batch, batch]
        similarity_matrix = torch.mm(z_f, z_u.t()) / self.temperature
        
        # Labels: positive pairs are on the diagonal (i == j)
        labels = torch.arange(batch_size, device=device)
        
        # Cross-entropy loss treats each row as a classification problem
        # where the correct class is the diagonal element
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
    
    def compute_accuracy(
        self, 
        z_f: torch.Tensor, 
        z_u: torch.Tensor
    ) -> float:
        """
        Compute retrieval accuracy for monitoring.
        
        Returns the fraction of samples where the correct match
        has the highest similarity score.
        """
        with torch.no_grad():
            z_f = F.normalize(z_f, dim=-1)
            z_u = F.normalize(z_u, dim=-1)
            
            similarity = torch.mm(z_f, z_u.t())
            predictions = similarity.argmax(dim=1)
            labels = torch.arange(z_f.shape[0], device=z_f.device)
            
            accuracy = (predictions == labels).float().mean().item()
            
        return accuracy


class SymmetricInfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE Loss.
    
    Computes bidirectional contrastive loss:
    L = (L_NCE(z_f → z_u) + L_NCE(z_u → z_f)) / 2
    
    This ensures both directions are aligned, providing stronger
    reciprocity constraints.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.infonce = InfoNCELoss(temperature)
    
    def forward(
        self, 
        z_f: torch.Tensor, 
        z_u: torch.Tensor
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss."""
        # Forward direction: z_f → z_u
        loss_forward = self.infonce(z_f, z_u)
        
        # Backward direction: z_u → z_f
        loss_backward = self.infonce(z_u, z_f)
        
        return (loss_forward + loss_backward) / 2
    
    def compute_accuracy(self, z_f: torch.Tensor, z_u: torch.Tensor) -> float:
        """Delegate accuracy computation to base InfoNCELoss."""
        return self.infonce.compute_accuracy(z_f, z_u)


class LRNLoss(nn.Module):
    """
    Combined Loss for LRN Training.
    
    Implements the total loss function:
    L_total = L_NCE + λ · L_MSE
    
    Supports all three training stages:
    - Stage I: L_NCE only (λ = 0 or skip MSE)
    - Stage II: L_NCE + λ · L_MSE
    - Stage III: L_MSE only (skip NCE)
    """
    
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_nce: float = 1.0,
        temperature: float = 0.1,
        symmetric_nce: bool = False,
        use_relative_mse: bool = False,
    ):
        """
        Args:
            lambda_mse: Weight for MSE loss (λ_mse)
            lambda_nce: Weight for InfoNCE loss (λ_nce)
            temperature: Temperature for InfoNCE (τ)
            symmetric_nce: Use symmetric InfoNCE loss
            use_relative_mse: Use Relative MSE instead of standard MSE
        """
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_nce = lambda_nce
        
        if symmetric_nce:
            self.nce_loss = SymmetricInfoNCELoss(temperature)
        else:
            self.nce_loss = InfoNCELoss(temperature)
        
        if use_relative_mse:
            self.mse_loss = RelativeMSELoss()
        else:
            self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        z_f: Optional[torch.Tensor] = None,
        z_u: Optional[torch.Tensor] = None,
        stage: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined loss based on training stage.
        
        Args:
            prediction: Predicted solution ũ
            target: Ground truth solution u
            z_f: Forward encoder latent code
            z_u: Reverse encoder latent code
            stage: Training stage (1, 2, or 3)
            
        Returns:
            Dictionary containing:
                - 'total': Total combined loss
                - 'nce': InfoNCE loss (if computed)
                - 'mse': MSE loss (if computed)
                - 'accuracy': Retrieval accuracy (if NCE computed)
        """
        losses = {}
        total = torch.tensor(0.0, device=prediction.device)
        
        # Stage I: NCE only
        # Stage II: NCE + MSE
        # Stage III: MSE only
        
        compute_nce = stage in [1, 2] and z_f is not None and z_u is not None
        compute_mse = stage in [2, 3]
        
        if compute_nce:
            nce = self.nce_loss(z_f, z_u)
            losses['nce'] = nce
            total = total + self.lambda_nce * nce
            
            # Compute accuracy for monitoring
            accuracy = self.nce_loss.compute_accuracy(z_f, z_u)
            losses['accuracy'] = accuracy
        
        if compute_mse:
            mse = self.mse_loss(prediction, target)
            losses['mse'] = mse
            total = total + self.lambda_mse * mse
        
        losses['total'] = total
        
        return losses


class RelativeMSELoss(nn.Module):
    """
    Relative MSE Loss for better scaling across different PDE magnitudes.
    
    L = ||ũ - u||² / ||u||²
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(
        self, 
        prediction: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """Compute relative MSE loss."""
        diff = prediction - target
        
        # Compute norms
        diff_norm = torch.norm(diff.flatten(1), dim=1)
        target_norm = torch.norm(target.flatten(1), dim=1)
        
        # Relative error
        relative_error = diff_norm / (target_norm + self.eps)
        
        return relative_error.pow(2).mean()
