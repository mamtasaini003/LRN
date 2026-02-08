"""
Latent Rescaling Network (LRN) - Navier-Stokes Experiment.
Compares standard FNO training with LRR-FNO (Latent Reciprocity).

Usage:
    python navier_stokes_lrr.py --epochs 200 --batch_size 16
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

from models.components.fno import FNO2d
from models.lrr.model import LRRFNO2d
from losses.infonce import LRNLoss
from utils.training import Trainer
from data.neuralop_loaders import create_neuralop_dataloaders
from data.pde_datasets import NavierStokesDataset

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
    """Load Navier-Stokes dataset using NeuralOperator or synthetic fallback."""
    logger.info("Loading Navier-Stokes dataset...")
    resolution = 128
    
    try:
        try:
            return create_neuralop_dataloaders(
                dataset_name='navier_stokes',
                n_train=args.n_train,
                n_test=args.n_test,
                batch_size=args.batch_size,
                test_batch_size=args.batch_size,
                resolution=resolution,
                return_tuple_format=True,
                encode_input=True
            )
        except Exception as e:
            if "neuraloperator" in str(e) and "not installed" in str(e):
                raise e
                
            if args.n_train is not None:
                logger.warning(f"Failed to load {args.n_train} samples ({e}). Retrying with all available data...")
                return create_neuralop_dataloaders(
                    dataset_name='navier_stokes',
                    n_train=None,
                    n_test=args.n_test,
                    batch_size=args.batch_size,
                    test_batch_size=args.batch_size,
                    resolution=resolution,
                    return_tuple_format=True,
                    encode_input=True
                )
            raise e
    except Exception as e:
        logger.warning(f"NeuralOperator loader failed ({e}). using synthetic fallback.")
        # Helper to ensure [C, H, W] dims (applied per-tensor)
        def unsqueeze_transform(t):
            return t.unsqueeze(0) if t.ndim == 2 else t

        # Ensure input/output steps align with 1-channel expectation for this demo
        train_dataset = NavierStokesDataset(
            resolution=64, num_samples=args.n_train, train=True,
            input_steps=1, output_steps=1, transform=unsqueeze_transform
        )
        test_dataset = NavierStokesDataset(
            resolution=64, num_samples=args.n_test, train=False,
            input_steps=1, output_steps=1, transform=unsqueeze_transform
        )
        return (
            torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True),
            torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False),
            None
        )


def train_fno(args, device, train_loader, test_loader):
    """Train baseline FNO model."""
    logger.info("Training Baseline FNO...")
    
    model = FNO2d(
        in_channels=1, out_channels=1,
        modes1=12, modes2=12,
        width=args.width, num_layers=4
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if x.ndim == 3: x = x.unsqueeze(1)
            if y.ndim == 3: y = y.unsqueeze(1)
            
            optimizer.zero_grad()
            out = model(x)
            
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        avg_loss = train_loss / len(train_loader)
        test_err = evaluate_model(model, test_loader, device)
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}/{args.epochs} | Loss: {avg_loss:.6f}")

    # For legacy script, we can just return a simple history dict
    # But since we want to plot, let's track it properly or use the fact that Trainer tracks it.
    return test_err, model


def train_lrr(args, device, train_loader, test_loader):
    """Train LRR-FNO model with Latent Reciprocity."""
    logger.info("Training LRR-FNO (Latent Supervision)...")
    
    model = LRRFNO2d(
        in_channels=1, out_channels=1,
        modes1=12, modes2=12,
        width=args.width, num_layers=4,
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
    
    # Split epochs: majority for Hybrid (Stage 2), last 100 for Distillation (Stage 3)
    distill_epochs = 100
    hybrid_epochs = args.epochs - distill_epochs
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        stage1_epochs=hybrid_epochs,
        stage2_epochs=distill_epochs,
        stage1_lr=1e-3,
        stage2_lr=1e-4,
        weight_decay=0.0,
        checkpoint_dir='checkpoints_ns_lrr',
        scheduler_type='step',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5}
    )
    
    trainer.train()
    model.eval()
    return evaluate_model(model, test_loader, device, is_lrr=True), model, trainer.history


def evaluate_model(model, loader, device, is_lrr=False):
    """Evaluate model using Relative L2 Error."""
    model.eval()
    errors = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if x.ndim == 3: x = x.unsqueeze(1)
            if y.ndim == 3: y = y.unsqueeze(1)
            
            if is_lrr:
                out = model(x)['prediction']
            else:
                out = model(x)
            
            errors.append(compute_relative_l2(out, y))
            
    return np.mean(errors)


def plot_comparison(fno_model, lrr_model, loader, device, metrics):
    """Visualize qualitative comparison."""
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    if x.ndim == 3: x = x.unsqueeze(1)
    if y.ndim == 3: y = y.unsqueeze(1)
    
    with torch.no_grad():
        fno_pred = fno_model(x)
        lrr_pred = lrr_model(x)['prediction']
        
    # Plot first sample
    y_plot = y[0, 0].cpu()
    fno_plot = fno_pred[0, 0].cpu()
    lrr_plot = lrr_pred[0, 0].cpu()

    fno_err, lrr_err = metrics
    imp = (fno_err - lrr_err) / fno_err * 100
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Ground Truth', 'Baseline FNO', 'LRR-FNO']
    data = [y_plot, fno_plot, lrr_plot]
    
    for ax, title, img in zip(axes, titles, data):
        im = ax.imshow(img, cmap='viridis')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    out_dir = Path('results/plots/navier_stokes')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'comparison.png'
    
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Comparison plot saved to {out_path}")


def plot_loss_history(lrr_history, out_dir):
    """Plot convergence curves for LRR."""
    plt.figure(figsize=(10, 6))
    plt.plot(lrr_history['train_loss'], 'g--', label='LRR Train', alpha=0.5)
    plt.plot(lrr_history['test_loss'], 'g-', label='LRR Test')
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    out_path = out_dir / 'loss_curves.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Loss curves saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--width', type=int, default=32, help='Model width (d_v)')
    parser.add_argument('--n_train', type=int, default=1000, help='Training samples')
    parser.add_argument('--n_test', type=int, default=200, help='Test samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()
    
    logger.info(f"Device: {device}")
    
    train_loader, test_loader, _ = load_data(args)
    
    # Results Dir
    out_dir = Path('results/plots/navier_stokes/root')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fno_err, fno_model = train_fno(args, device, train_loader, test_loader)
    logger.info(f"Baseline FNO Error: {fno_err:.4f}")
    
    lrr_err, lrr_model, lrr_hist = train_lrr(args, device, train_loader, test_loader)
    logger.info(f"LRR-FNO Error:      {lrr_err:.4f}")
    
    # Summary
    improvement = (fno_err - lrr_err) / fno_err * 100
    
    plot_comparison(fno_model, lrr_model, test_loader, device, (fno_err, lrr_err))
    plot_loss_history(lrr_hist, out_dir)


if __name__ == '__main__':
    main()
