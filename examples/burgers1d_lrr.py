"""
Latent Rescaling Network (LRN) - 1D Burgers Equation Experiment.
Compares standard FNO training with LRR-FNO (Latent Reciprocity).

Usage:
    python burgers1d_lrr.py --epochs 200 --batch_size 32
"""

import sys
import argparse
import random
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models.components.fno import FNO1d
from models.lrr.model import LRRFNO1d
from losses.infonce import LRNLoss
from utils.training import Trainer
from data.neuralop_loaders import create_neuralop_dataloaders
from data.pde_datasets import BurgersDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True 


def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Relative L2 error norm."""
    diff_norm = torch.norm(pred.flatten(1) - target.flatten(1), p=2, dim=1)
    target_norm = torch.norm(target.flatten(1), p=2, dim=1)
    return (diff_norm / (target_norm + 1e-8)).mean().item()


def load_data(args):
    """Load Burgers dataset using NeuralOperator or synthetic fallback."""
    logger.info("Loading Burgers 1D dataset...")
    resolution = 128
    
    try:
        return create_neuralop_dataloaders(
            dataset_name='burgers',
            n_train=args.n_train,
            n_test=args.n_test,
            batch_size=args.batch_size,
            test_batch_size=args.batch_size,
            resolution=resolution,
            return_tuple_format=True,
            encode_input=True
        )
    except Exception as e:
        logger.warning(f"NeuralOperator loader failed ({e}). using synthetic fallback.")
        # Helper to ensure [C, L] dims
        def unsqueeze_transform(x, y):
            if x.ndim == 1: x = x.unsqueeze(0)
            if y.ndim == 1: y = y.unsqueeze(0)
            return x, y

        train_dataset = BurgersDataset(
            resolution=resolution, num_samples=args.n_train, train=True,
            transform=unsqueeze_transform
        )
        test_dataset = BurgersDataset(
            resolution=resolution, num_samples=args.n_test, train=False,
            transform=unsqueeze_transform
        )
        return (
            torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
            torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False),
            None
        )


def train_fno(args, device, train_loader, test_loader):
    """Train baseline FNO model."""
    logger.info("Training Baseline FNO...")
    
    model = FNO1d(
        in_channels=1, out_channels=1,
        modes=args.modes, width=args.width, num_layers=4
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if x.ndim == 2: x = x.unsqueeze(1)
            if y.ndim == 2: y = y.unsqueeze(1)
            
            optimizer.zero_grad()
            out = model(x)
            
            # FNO1d outputs [B, L, C], permute to [B, C, L] if needed
            if out.ndim == 3 and out.shape[-1] == 1:
                out = out.permute(0, 2, 1)
                
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}/{args.epochs} | Loss: {train_loss/len(train_loader):.6f}")

    return evaluate_model(model, test_loader, device), model


def train_lrr(args, device, train_loader, test_loader):
    """Train LRR-FNO model with Latent Reciprocity."""
    logger.info("Training LRR-FNO (Latent Supervision)...")
    
    model = LRRFNO1d(
        in_channels=1, out_channels=1,
        modes=args.modes, width=args.width, num_layers=4,
        latent_dim=128,
        encoder_channels=[16, 32, 64],
        use_gated_bridge=False
    ).to(device)
    
    loss_fn = LRNLoss(
        temperature=0.07,
        lambda_mse=5.0,
        lambda_nce=0.001,
        symmetric_nce=True
    )
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        stage1_epochs=args.epochs,
        stage2_epochs=0,
        stage1_lr=1e-3,
        weight_decay=1e-4,
        checkpoint_dir='checkpoints_burgers'
    )
    
    trainer.train()
    model.eval()
    return evaluate_model(model, test_loader, device, is_lrr=True), model


def evaluate_model(model, loader, device, is_lrr=False):
    """Evaluate model using Relative L2 Error."""
    model.eval()
    errors = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if x.ndim == 2: x = x.unsqueeze(1)
            if y.ndim == 2: y = y.unsqueeze(1)
            
            if is_lrr:
                out = model(x)['prediction']
            else:
                out = model(x)
                if out.ndim == 3 and out.shape[-1] == 1:
                    out = out.permute(0, 2, 1)
            
            errors.append(compute_relative_l2(out, y))
            
    return np.mean(errors)


def plot_comparison(fno_model, lrr_model, loader, device, metrics):
    """Visualize qualitative comparison."""
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    if x.ndim == 2: x = x.unsqueeze(1)
    if y.ndim == 2: y = y.unsqueeze(1)
    
    with torch.no_grad():
        fno_pred = fno_model(x)
        if fno_pred.ndim == 3 and fno_pred.shape[-1] == 1:
            fno_pred = fno_pred.permute(0, 2, 1)
            
        lrr_pred = lrr_model(x)['prediction']
        
    # Plot first sample
    y_plot = y[0, 0].cpu().numpy()
    fno_plot = fno_pred[0, 0].cpu().numpy()
    lrr_plot = lrr_pred[0, 0].cpu().numpy()

    fno_err, lrr_err = metrics
    imp = (fno_err - lrr_err) / fno_err * 100
    
    plt.figure(figsize=(10, 5))
    plt.plot(y_plot, 'k-', label='Ground Truth', linewidth=2)
    plt.plot(fno_plot, 'b--', label='Baseline FNO', alpha=0.8)
    plt.plot(lrr_plot, 'g--', label='LRR-FNO', alpha=0.8)
    plt.legend()
    plt.title(f"Burgers 1D | FNO: {fno_err:.4f} | LRR: {lrr_err:.4f} | Δ: {imp:.1f}%")
    plt.grid(True, alpha=0.3)
    
    out_path = Path('results/plots/burgers_comparison.png')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    logger.info(f"Comparison plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--width', type=int, default=64, help='Model width')
    parser.add_argument('--modes', type=int, default=16, help='Fourier modes')
    parser.add_argument('--n_train', type=int, default=1000, help='Training samples')
    parser.add_argument('--n_test', type=int, default=200, help='Test samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()
    
    logger.info(f"Device: {device}")
    
    train_loader, test_loader, _ = load_data(args)
    
    fno_err, fno_model = train_fno(args, device, train_loader, test_loader)
    logger.info(f"Baseline FNO Error: {fno_err:.4f}")
    
    lrr_err, lrr_model = train_lrr(args, device, train_loader, test_loader)
    logger.info(f"LRR-FNO Error:      {lrr_err:.4f}")
    
    improvement = (fno_err - lrr_err) / fno_err * 100
    logger.info(f"Final Improvement:  {improvement:.2f}%")
    
    plot_comparison(fno_model, lrr_model, test_loader, device, (fno_err, lrr_err))


if __name__ == '__main__':
    main()
