"""
Retrain LRR-FNO with meaningful NCE weight to demonstrate alignment.
"""

import sys
import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

# Project imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from models.lrr.model import LRRFNO2d
from losses.infonce import LRNLoss
from utils.training import Trainer
from data.pde_datasets import DarcyDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_aligned(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training Aligned LRR-FNO on {device} with lambda_nce={args.lambda_nce}...")
    
    # Dataset
    def unsqueeze_transform(t):
        return t.unsqueeze(0) if t.ndim == 2 else t
        
    train_dataset = DarcyDataset(resolution=32, num_samples=1000, train=True, transform=unsqueeze_transform)
    test_dataset = DarcyDataset(resolution=32, num_samples=200, train=False, transform=unsqueeze_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    model = LRRFNO2d(
        in_channels=1, out_channels=1, modes1=12, modes2=12, width=32, 
        latent_dim=128
    ).to(device)
    
    # LOSS with HIGHER NCE WEIGHT
    loss_fn = LRNLoss(
        temperature=0.1,
        lambda_mse=1.0,
        lambda_nce=args.lambda_nce, # Usually 0.1 - 1.0
        symmetric_nce=True
    )
    
    # TRAINER
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        stage1_epochs=args.epochs, # Combined stage
        stage2_epochs=0,          # Skip distillation for alignment focus
        stage1_lr=1e-3,
        weight_decay=0.0,
        checkpoint_dir='checkpoints_darcy_aligned',
        scheduler_type='cosine'
    )
    
    trainer.train()
    
    save_path = Path("latent_reps/scripts/lrr_aligned_weights.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Aligned LRR-FNO weights saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lambda_nce', type=float, default=1.0)
    args = parser.parse_args()
    train_aligned(args)
