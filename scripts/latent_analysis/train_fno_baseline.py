"""
Standalone script to train a baseline FNO model for Darcy Flow.
Used to generate weights for latent space comparison.
"""

import sys
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader

# Project imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / 'src'))

from models.components.fno import FNO2d
from data.pde_datasets import DarcyDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_fno(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training Baseline FNO on {device}...")
    
    # Dataset
    def unsqueeze_transform(t):
        return t.unsqueeze(0) if t.ndim == 2 else t
        
    train_dataset = DarcyDataset(resolution=32, num_samples=args.n_train, train=True, transform=unsqueeze_transform)
    loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Model
    model = FNO2d(in_channels=1, out_channels=1, modes1=12, modes2=12, width=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if out.ndim != y.ndim: out = out.squeeze(1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{args.epochs} | Loss: {total_loss/len(loader):.6f}")
            
    save_path = Path("latent_reps/scripts/fno_baseline_weights.pt")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Baseline FNO weights saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_train', type=int, default=1000)
    args = parser.parse_args()
    train_fno(args)
