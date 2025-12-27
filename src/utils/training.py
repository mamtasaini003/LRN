"""
Training Utilities for LRN

Implements the 3-stage curriculum training protocol:
    - Stage I: Manifold Alignment (NCE only)
    - Stage II: Hybrid Optimization (NCE + MSE)
    - Stage III: Autonomous Distillation (MSE only, discard E_u)

Also includes standard training utilities and learning rate scheduling.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path
import time
from tqdm import tqdm


class Trainer:
    """
    Base trainer class for neural operators.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.optimizer = optimizer or Adam(model.parameters(), lr=1e-3)
        self.scheduler = scheduler
        
        self.history: Dict[str, List[float]] = {
            'train_loss': [],
            'test_loss': [],
        }
    
    def train_epoch(self) -> float:
        """Train for one epoch. Override in subclass."""
        raise NotImplementedError
    
    def evaluate(self) -> float:
        """Evaluate on test set. Override in subclass."""
        raise NotImplementedError
    
    def save_checkpoint(self, path: str, **kwargs):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            **kwargs
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'history' in checkpoint:
            self.history = checkpoint['history']


class LRNTrainer(Trainer):
    """
    Trainer for LRN-FNO with 3-stage curriculum learning.
    
    Stage I: Manifold Alignment
        - Train only encoders E_f, E_u
        - Loss: L_NCE (contrastive only)
        - Purpose: Establish bidirectional latent reciprocity
    
    Stage II: Hybrid Optimization
        - Train all components: E_f, E_u, G_θ
        - Loss: L_NCE + λ·L_MSE
        - Purpose: Joint optimization of latent space and reconstruction
    
    Stage III: Autonomous Distillation
        - Discard E_u, train E_f, G_θ
        - Loss: L_MSE only
        - Purpose: Prepare deployable forward model
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: nn.Module,
        stage1_epochs: int = 50,
        stage2_epochs: int = 100,
        stage3_epochs: int = 50,
        stage1_lr: float = 1e-3,
        stage2_lr: float = 1e-3,
        stage3_lr: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        
        self.loss_fn = loss_fn
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.stage3_lr = stage3_lr
        self.log_interval = log_interval
        
        # Extended history for LRN
        self.history.update({
            'nce_loss': [],
            'mse_loss': [],
            'accuracy': [],
            'stage': [],
        })
    
    def _get_encoder_params(self):
        """Get parameters for encoders only."""
        params = []
        if hasattr(self.model, 'encoder_f'):
            params.extend(self.model.encoder_f.parameters())
        if hasattr(self.model, 'encoder_u') and self.model.encoder_u is not None:
            params.extend(self.model.encoder_u.parameters())
        return params
    
    def _get_all_params(self):
        """Get all model parameters."""
        return self.model.parameters()
    
    def _get_stage3_params(self):
        """Get parameters for Stage III (excluding E_u)."""
        params = []
        for name, param in self.model.named_parameters():
            if 'encoder_u' not in name:
                params.append(param)
        return params
    
    def _freeze_backbone(self, freeze: bool = True):
        """Freeze/unfreeze FNO backbone."""
        if hasattr(self.model, 'fno'):
            for param in self.model.fno.parameters():
                param.requires_grad = not freeze
        if hasattr(self.model, 'latent_bridge'):
            for param in self.model.latent_bridge.parameters():
                param.requires_grad = not freeze
    
    def train_epoch(self, stage: int) -> Dict[str, float]:
        """
        Train for one epoch at specified stage.
        
        Args:
            stage: Training stage (1, 2, or 3)
            
        Returns:
            Dictionary of average losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_nce = 0.0
        total_mse = 0.0
        total_acc = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f'Stage {stage} Training')
        
        for batch_idx, (f, u) in enumerate(pbar):
            f = f.to(self.device)
            u = u.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(f, u, return_latents=True)
            prediction = output['prediction']
            z_f = output.get('z_f')
            z_u = output.get('z_u')
            
            # Compute loss based on stage
            losses = self.loss_fn(
                prediction=prediction,
                target=u,
                z_f=z_f,
                z_u=z_u,
                stage=stage,
            )
            
            loss = losses['total']
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            if 'nce' in losses:
                total_nce += losses['nce'].item()
            if 'mse' in losses:
                total_mse += losses['mse'].item()
            if 'accuracy' in losses:
                total_acc += losses['accuracy']
            num_batches += 1
            
            # Update progress bar
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'nce': f"{losses.get('nce', 0):.4f}" if 'nce' in losses else 'N/A',
                    'mse': f"{losses.get('mse', 0):.4f}" if 'mse' in losses else 'N/A',
                })
        
        return {
            'total': total_loss / num_batches,
            'nce': total_nce / num_batches if total_nce > 0 else 0,
            'mse': total_mse / num_batches if total_mse > 0 else 0,
            'accuracy': total_acc / num_batches if total_acc > 0 else 0,
        }
    
    @torch.no_grad()
    def evaluate(self, stage: int = 3) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            stage: Training stage (affects loss computation)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0
        
        for f, u in self.test_loader:
            f = f.to(self.device)
            u = u.to(self.device)
            
            output = self.model(f, u, return_latents=True)
            prediction = output['prediction']
            z_f = output.get('z_f')
            z_u = output.get('z_u')
            
            losses = self.loss_fn(
                prediction=prediction,
                target=u,
                z_f=z_f,
                z_u=z_u,
                stage=stage,
            )
            
            total_loss += losses['total'].item()
            if 'mse' in losses:
                total_mse += losses['mse'].item()
            num_batches += 1
        
        return {
            'total': total_loss / num_batches,
            'mse': total_mse / num_batches if total_mse > 0 else 0,
        }
    
    def train_stage(
        self, 
        stage: int, 
        epochs: int, 
        lr: float
    ) -> Dict[str, List[float]]:
        """
        Train for a complete stage.
        
        Args:
            stage: Training stage (1, 2, or 3)
            epochs: Number of epochs for this stage
            lr: Learning rate for this stage
            
        Returns:
            Training history for this stage
        """
        print(f"\n{'='*60}")
        print(f"STAGE {stage}: ", end="")
        if stage == 1:
            print("Manifold Alignment (NCE only)")
            self._freeze_backbone(True)
            params = self._get_encoder_params()
        elif stage == 2:
            print("Hybrid Optimization (NCE + MSE)")
            self._freeze_backbone(False)
            params = self._get_all_params()
        else:
            print("Autonomous Distillation (MSE only)")
            self.model.set_inference_mode(True)
            params = self._get_stage3_params()
        print(f"{'='*60}")
        
        self.optimizer = Adam(params, lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs)
        
        stage_history = {
            'train_loss': [],
            'test_loss': [],
            'nce_loss': [],
            'mse_loss': [],
        }
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            train_metrics = self.train_epoch(stage)
            
            # Evaluate
            test_metrics = self.evaluate(stage)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Record history
            stage_history['train_loss'].append(train_metrics['total'])
            stage_history['test_loss'].append(test_metrics['total'])
            stage_history['nce_loss'].append(train_metrics['nce'])
            stage_history['mse_loss'].append(train_metrics['mse'])
            
            self.history['train_loss'].append(train_metrics['total'])
            self.history['test_loss'].append(test_metrics['total'])
            self.history['nce_loss'].append(train_metrics['nce'])
            self.history['mse_loss'].append(train_metrics['mse'])
            self.history['stage'].append(stage)
            
            # Logging
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_metrics['total']:.6f} | "
                  f"Test: {test_metrics['total']:.6f} | "
                  f"NCE: {train_metrics['nce']:.6f} | "
                  f"MSE: {train_metrics['mse']:.6f}")
            
            # Save best model
            if test_metrics['total'] < best_loss:
                best_loss = test_metrics['total']
                self.save_checkpoint(
                    self.checkpoint_dir / f'best_stage{stage}.pt',
                    stage=stage,
                    epoch=epoch,
                    best_loss=best_loss,
                )
        
        # Save final stage checkpoint
        self.save_checkpoint(
            self.checkpoint_dir / f'final_stage{stage}.pt',
            stage=stage,
        )
        
        return stage_history
    
    def train(self) -> Dict[str, List[float]]:
        """
        Execute full 3-stage curriculum training.
        
        Returns:
            Complete training history
        """
        print("\n" + "="*60)
        print("LATENT RECIPROCITY NETWORK - 3-STAGE CURRICULUM TRAINING")
        print("="*60)
        
        start_time = time.time()
        
        # Stage I: Manifold Alignment
        if self.stage1_epochs > 0:
            self.train_stage(stage=1, epochs=self.stage1_epochs, lr=self.stage1_lr)
        
        # Stage II: Hybrid Optimization
        if self.stage2_epochs > 0:
            self.train_stage(stage=2, epochs=self.stage2_epochs, lr=self.stage2_lr)
        
        # Stage III: Autonomous Distillation
        if self.stage3_epochs > 0:
            self.train_stage(stage=3, epochs=self.stage3_epochs, lr=self.stage3_lr)
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
        
        # Save final model
        self.save_checkpoint(
            self.checkpoint_dir / 'final_model.pt',
        )
        
        return self.history


class LRNTrainerV2(LRNTrainer):
    """
    Version 2 Trainer for LRN-FNO with a simplified 2-stage curriculum.
    
    Stage 1: Combined Optimization (InfoNCE + MSE)
        - Train all components: E_f, E_u, G_θ
        - Loss: L_NCE + λ·L_MSE
        - Purpose: Jointly learn the latent manifold and the solution mapping.
    
    Stage 2: Autonomous Distillation (MSE only)
        - Discard E_u, train E_f, G_θ
        - Loss: L_MSE only
        - Purpose: Refine the forward model for inference.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        loss_fn: nn.Module,
        stage1_epochs: int = 100,
        stage2_epochs: int = 50,
        stage1_lr: float = 1e-3,
        stage2_lr: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints_v2',
        log_interval: int = 10,
    ):
        # We reuse the base LRNTrainer logic but only implement 2 stages.
        # We map V2's Stage 1 -> Original Stage 2
        # We map V2's Stage 2 -> Original Stage 3
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            stage1_epochs=0, # Skip manifold-only alignment
            stage2_epochs=stage1_epochs,
            stage3_epochs=stage2_epochs,
            stage2_lr=stage1_lr,
            stage3_lr=stage2_lr,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_interval=log_interval,
        )

    def train(self) -> Dict[str, List[float]]:
        """
        Execute simplified 2-stage training.
        """
        print("\n" + "="*60)
        print("LATENT RECIPROCITY NETWORK - VERSION 2 (2-STAGE TRAINING)")
        print("="*60)
        
        start_time = time.time()
        
        # Stage 1: Joint NCE + MSE
        if self.stage2_epochs > 0:
            print("Starting Stage 1: Combined Optimization (Joint Training)...")
            self.train_stage(stage=2, epochs=self.stage2_epochs, lr=self.stage2_lr)
        
        # Stage 2: MSE Only (Distillation)
        if self.stage3_epochs > 0:
            print("Starting Stage 2: Autonomous Distillation (Fine-tuning)...")
            self.train_stage(stage=3, epochs=self.stage3_epochs, lr=self.stage3_lr)
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal training time (V2): {elapsed_time/60:.2f} minutes")
        
        self.save_checkpoint(self.checkpoint_dir / 'final_model_v2.pt')
        return self.history


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
