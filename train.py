"""
LRN-FNO Training Script

Main training script for the Latent Reciprocity Network with FNO backbone.
Implements the 3-stage curriculum training protocol:
    - Stage I: Manifold Alignment (NCE only)
    - Stage II: Hybrid Optimization (NCE + λ·MSE)
    - Stage III: Autonomous Distillation (MSE only)

Usage:
    python train.py --config configs/default.yaml
    python train.py --dataset burgers --epochs 100
"""

import os
# Fix for OpenMP library conflict on Windows/Anaconda
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

from src.models import LRNFNO1d, LRNFNO2d
from src.losses import LRNLoss
from src.data import create_dataloaders
from src.utils import LRNTrainer, LRNTrainerV2, get_device, count_parameters


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, device: torch.device):
    """Create LRN-FNO model from configuration."""
    model_config = config['model']
    spatial_dim = model_config.get('spatial_dim', 1)
    
    fno_config = model_config.get('fno', {})
    latent_config = model_config.get('latent', {})
    
    common_args = {
        'in_channels': model_config.get('in_channels', 1),
        'out_channels': model_config.get('out_channels', 1),
        'width': fno_config.get('width', 64),
        'num_layers': fno_config.get('num_layers', 4),
        'latent_dim': latent_config.get('dim', 64),
        'encoder_channels': latent_config.get('encoder_channels', [32, 64, 128]),
        'use_gated_bridge': latent_config.get('use_gated_bridge', False),
    }
    
    if spatial_dim == 1:
        model = LRNFNO1d(
            modes=fno_config.get('modes', 16),
            padding=fno_config.get('padding', 8),
            **common_args
        )
    else:
        model = LRNFNO2d(
            modes1=fno_config.get('modes1', 12),
            modes2=fno_config.get('modes2', 12),
            padding=fno_config.get('padding', 9),
            **common_args
        )
    
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description='Train LRN-FNO Model')
    
    # Configuration
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    
    # Override config options
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name: burgers, darcy, navier_stokes')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset file')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Spatial resolution')
    
    # Training overrides
    parser.add_argument('--stage1_epochs', type=int, default=None,
                        help='Stage I epochs')
    parser.add_argument('--stage2_epochs', type=int, default=None,
                        help='Stage II epochs')
    parser.add_argument('--stage3_epochs', type=int, default=None,
                        help='Stage III epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    
    # Model overrides
    parser.add_argument('--modes', type=int, default=None,
                        help='Number of Fourier modes')
    parser.add_argument('--width', type=int, default=None,
                        help='FNO hidden width')
    parser.add_argument('--latent_dim', type=int, default=None,
                        help='Latent space dimension')
    
    # Training control
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto, cuda, cpu, mps')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--v2', action='store_true',
                        help='Use Version 2 training (2 stages)')
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = {
            'model': {'spatial_dim': 1, 'fno': {}, 'latent': {}},
            'training': {'stage1': {}, 'stage2': {}, 'stage3': {}},
            'loss': {},
            'data': {},
        }
    
    # Apply command-line overrides
    if args.dataset:
        config['data']['dataset'] = args.dataset
    if args.data_path:
        config['data']['data_path'] = args.data_path
    if args.resolution:
        config['data']['resolution'] = args.resolution
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.stage1_epochs:
        config['training']['stage1']['epochs'] = args.stage1_epochs
    if args.stage2_epochs:
        config['training']['stage2']['epochs'] = args.stage2_epochs
    if args.stage3_epochs:
        config['training']['stage3']['epochs'] = args.stage3_epochs
    if args.modes:
        config['model']['fno']['modes'] = args.modes
    if args.width:
        config['model']['fno']['width'] = args.width
    if args.latent_dim:
        config['model']['latent']['dim'] = args.latent_dim
    if args.lr:
        for stage in ['stage1', 'stage2', 'stage3']:
            config['training'][stage]['lr'] = args.lr
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    if args.device == 'auto':
        device = get_device()
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    data_config = config.get('data', {})
    train_loader, test_loader = create_dataloaders(
        dataset_name=data_config.get('dataset', 'burgers'),
        data_path=data_config.get('data_path'),
        batch_size=config['training'].get('batch_size', 32),
        num_workers=config['training'].get('num_workers', 4),
        resolution=data_config.get('resolution', 128),
        num_samples=data_config.get('num_samples', 1000),
    )
    print(f"Dataset: {data_config.get('dataset', 'burgers')}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Set spatial dimension based on dataset
    dataset_name = data_config.get('dataset', 'burgers').lower()
    if dataset_name == 'burgers':
        config['model']['spatial_dim'] = 1
    else:
        config['model']['spatial_dim'] = 2
    
    # Create model
    model = create_model(config, device)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Create loss function
    loss_config = config.get('loss', {})
    loss_fn = LRNLoss(
        lambda_mse=loss_config.get('lambda_mse', 1.0),
        temperature=loss_config.get('temperature', 0.1),
        symmetric_nce=loss_config.get('symmetric_nce', False),
    )
    
    # Create trainer
    training_config = config.get('training', {})
    
    if args.v2:
        print("Using LRNTrainerV2 (2-stage protocol)")
        trainer = LRNTrainerV2(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            stage1_epochs=training_config.get('stage2', {}).get('epochs', 100),
            stage2_epochs=training_config.get('stage3', {}).get('epochs', 50),
            stage1_lr=training_config.get('stage2', {}).get('lr', 1e-3),
            stage2_lr=training_config.get('stage3', {}).get('lr', 1e-4),
            device=str(device),
            checkpoint_dir=args.checkpoint_dir,
        )
    else:
        trainer = LRNTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            stage1_epochs=training_config.get('stage1', {}).get('epochs', 50),
            stage2_epochs=training_config.get('stage2', {}).get('epochs', 100),
            stage3_epochs=training_config.get('stage3', {}).get('epochs', 50),
            stage1_lr=training_config.get('stage1', {}).get('lr', 1e-3),
            stage2_lr=training_config.get('stage2', {}).get('lr', 1e-3),
            stage3_lr=training_config.get('stage3', {}).get('lr', 1e-4),
            device=str(device),
            checkpoint_dir=args.checkpoint_dir,
        )
    
    # Train!
    print("\nStarting training...")
    history = trainer.train()
    
    # Print final results
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final test loss: {history['test_loss'][-1]:.6f}")
    print(f"Best test loss: {min(history['test_loss']):.6f}")
    print(f"Checkpoints saved to: {args.checkpoint_dir}")
    
    return history


if __name__ == '__main__':
    main()
