"""
Training Utilities for LRN
Supports multi-stage curriculum training.
"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from typing import Optional, Dict, List, Callable, Any
from pathlib import Path
import time
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable
        def __iter__(self):
            return iter(self.iterable)
        def update(self, *args, **kwargs):
            pass
        def set_postfix(self, *args, **kwargs):
            pass
        def set_description(self, *args, **kwargs):
            pass
        def close(self):
            pass


class BaseTrainer:
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


class CurriculumTrainer(BaseTrainer):
    """
    Trainer for curriculum-based learning.
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
        weight_decay: float = 1e-4, # Default L2 regularization
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints',
        log_interval: int = 10,
        scheduler_type: str = 'cosine', # 'cosine' or 'step'
        scheduler_kwargs: Optional[Dict] = None,
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        # ... existing init code ...
        self.loss_fn = loss_fn
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.stage3_lr = stage3_lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.scheduler_type = scheduler_type
        self.scheduler_kwargs = scheduler_kwargs or {}
        
        # Extended history
        self.history.update({
            'nce_loss': [],
            'mse_loss': [],
            'accuracy': [],
            'stage': [],
        })

    def _get_encoder_params(self):
        """Get parameters for encoders only (and projection heads for LRR)."""
        params = []
        if hasattr(self.model, 'encoder_f'):
            params.extend(self.model.encoder_f.parameters())
        if hasattr(self.model, 'encoder_u') and self.model.encoder_u is not None:
            params.extend(self.model.encoder_u.parameters())
        # For LRR models, include the projection head
        if hasattr(self.model, 'vk_projection'):
            params.extend(self.model.vk_projection.parameters())
        if hasattr(self.model, 'fno') and hasattr(self.model, 'vk_projection'):
             # Also include the backbone if we are effectively unfreezing it
             params.extend(self.model.fno.parameters())
        if hasattr(self.model, 'latent_bridge') and hasattr(self.model, 'vk_projection'):
             params.extend(self.model.latent_bridge.parameters())
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
        # For LRR models, we generally WANT the backbone to learn during Stage 1 
        # (Latent Supervision), or at least the interaction affects it.
        # However, if we strictly follow the 'Manifold Alignment' (Stage 1), 
        # it usually implies freezing the generator.
        # BUT for LRR, 'v_K' comes from the backbone. If backbone is random, v_K is random.
        # So we should probably NOT freeze backbone for LRR models in Stage 1?
        # Let's assume if it has 'vk_projection', it's an LRR model and we should unfreeze.
        is_lrr = hasattr(self.model, 'vk_projection')
        
        should_freeze = freeze
        if is_lrr and freeze:
            # We must NOT freeze backbone for LRR if we want v_K to align with z_u 
            # effectively (assuming we want to shape v_K, not just map z_u to random v_K).
            should_freeze = False

        if hasattr(self.model, 'fno'):
            for param in self.model.fno.parameters():
                param.requires_grad = not should_freeze
        if hasattr(self.model, 'latent_bridge'):
            for param in self.model.latent_bridge.parameters():
                param.requires_grad = not should_freeze
    
    def train_epoch(self, stage: int, display_stage: Optional[int] = None) -> Dict[str, float]:
        """
        Train for one epoch at specified stage.
        """
        self.model.train()
        
        total_loss = 0.0
        total_nce = 0.0
        total_mse = 0.0
        total_acc = 0.0
        num_batches = 0
        
        d_stage = display_stage or stage
        pbar = tqdm(self.train_loader, desc=f'Stage {d_stage} Training')
        
        for batch_idx, (f, u) in enumerate(pbar):
            f = f.to(self.device)
            u = u.to(self.device)
            
            # Handle 1D data: ensure channel dimension exists [B, L] -> [B, 1, L]
            if f.dim() == 2:
                f = f.unsqueeze(1)
            if u.dim() == 2:
                u = u.unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(f, u, return_latents=True)
            prediction = output['prediction']
            z_v_k = output.get('z_f')  # Projected backbone latent (for NCE alignment)
            z_u = output.get('z_u')    # Solution encoder latent
            
            # Compute loss based on stage
            losses = self.loss_fn(
                prediction=prediction,
                target=u,
                z_f=z_v_k,   # Pass projected backbone features
                z_u=z_u,      # Pass solution latent
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
        """
        self.model.eval()
        
        total_loss = 0.0
        total_mse = 0.0
        num_batches = 0
        
        for f, u in self.test_loader:
            f = f.to(self.device)
            u = u.to(self.device)
            
            # Handle 1D data: ensure channel dimension exists
            if f.dim() == 2:
                f = f.unsqueeze(1)
            if u.dim() == 2:
                u = u.unsqueeze(1)
            
            output = self.model(f, u, return_latents=True)
            prediction = output['prediction']
            z_v_k = output.get('z_f')  # Projected backbone latent
            z_u = output.get('z_u')    # Solution encoder latent
            
            losses = self.loss_fn(
                prediction=prediction,
                target=u,
                z_f=z_v_k,
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
        lr: float,
        custom_header: Optional[str] = None,
        display_stage: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        Train for a complete stage.
        """
        if stage == 1:
            print(f"STAGE 1: Alignment")
        elif stage == 2:
            print(f"STAGE 2: Hybrid")
        else:
            print(f"STAGE 3: Distillation")
        
        if stage == 1:
            if hasattr(self.model, 'set_inference_mode'):
                self.model.set_inference_mode(False)
            self._freeze_backbone(True)
            params = self._get_encoder_params()
        elif stage == 2:
            if hasattr(self.model, 'set_inference_mode'):
                self.model.set_inference_mode(False)
            self._freeze_backbone(False)
            params = self._get_all_params()
        else:
            if hasattr(self.model, 'set_inference_mode'):
                self.model.set_inference_mode(True)
            params = self._get_stage3_params()
        
        self.optimizer = Adam(params, lr=lr, weight_decay=self.weight_decay)
        
        if self.scheduler_type == 'step':
            step_size = self.scheduler_kwargs.get('step_size', 100)
            gamma = self.scheduler_kwargs.get('gamma', 0.5)
            self.scheduler = StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        else:
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
            train_metrics = self.train_epoch(stage, display_stage=display_stage)
            
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

class Trainer(CurriculumTrainer):
    """
    Standard Trainer for Neural Operators.
    
    Implements a simplified 2-stage curriculum which is the recommended default.
    
    Stage 1: Combined Optimization (InfoNCE + MSE)
        - Train all components components.
        - Purpose: Jointly learn the latent manifold and the solution mapping.
    
    Stage 2: Autonomous Distillation (MSE only)
        - Train forward model components only.
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
        weight_decay: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'checkpoints_v2',
        log_interval: int = 10,
        scheduler_type: str = 'cosine',
        scheduler_kwargs: Optional[Dict] = None,
    ):
        # We reuse the base CurriculumTrainer logic but only implement 2 stages.
        # Map V2/Standard Stage 1 -> Original Stage 2
        # Map V2/Standard Stage 2 -> Original Stage 3
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
            weight_decay=weight_decay,
            device=device,
            checkpoint_dir=checkpoint_dir,
            log_interval=log_interval,
            scheduler_type=scheduler_type,
            scheduler_kwargs=scheduler_kwargs,
        )

    def train(self) -> Dict[str, List[float]]:
        """
        Execute simplified 2-stage training.
        """
        print("\n" + "="*60)
        print("TRAINER - 2-STAGE CURRICULUM")
        print("="*60)
        
        start_time = time.time()
        
        # Stage 1: Joint NCE + MSE
        if self.stage2_epochs > 0:
            header = f"STAGE 1: Combined Optimization (NCE + MSE) [{self.stage2_epochs} epochs]"
            self.train_stage(stage=2, epochs=self.stage2_epochs, lr=self.stage2_lr, custom_header=header, display_stage=1)
        
        # Stage 2: MSE Only (Distillation)
        if self.stage3_epochs > 0:
            header = f"STAGE 2: Autonomous Distillation (MSE only) [{self.stage3_epochs} epochs]"
            self.train_stage(stage=3, epochs=self.stage3_epochs, lr=self.stage3_lr, custom_header=header, display_stage=2)
        
        elapsed_time = time.time() - start_time
        print(f"\nTotal training time: {elapsed_time/60:.2f} minutes")
        
        self.save_checkpoint(self.checkpoint_dir / 'final_model.pt')
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
