"""
Latent Rescaling Network (LRN) - Darcy Flow Experiment (Two-Stage).
Training Strategy: 400 Epochs Hybrid + 100 Epochs Distillation
"""

import sys
import argparse
import random
import logging
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from models.components.fno import FNO2d
from models.lrr.model import LRRFNO2d
from losses.infonce import LRNLoss
from utils.training import Trainer
from data.neuralop_loaders import create_neuralop_dataloaders
from data.pde_datasets import DarcyDataset

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
    """Load Darcy dataset using NeuralOperator or fallback."""
    logger.info("Loading Darcy Flow dataset...")
    resolution = 32
    
    try:
        try:
            return create_neuralop_dataloaders(
                dataset_name='darcy',
                n_train=args.n_train,
                n_test=args.n_test,
                batch_size=args.batch_size,
                test_batch_size=args.batch_size,
                resolution=resolution,
                encode_output=True,
                return_tuple_format=True,
                encode_input=True
            )
        except Exception as e:
            if "neuraloperator" in str(e) and "not installed" in str(e):
                raise e
                
            if args.n_train is not None:
                logger.warning(f"Failed to load {args.n_train} samples ({e}). Retrying with all available data...")
                return create_neuralop_dataloaders(
                    dataset_name='darcy',
                    n_train=None,
                    n_test=args.n_test,
                    batch_size=args.batch_size,
                    test_batch_size=args.batch_size,
                    resolution=resolution,
                    encode_output=True,
                    return_tuple_format=True,
                    encode_input=True
                )
            raise e
    except Exception as e:
        logger.warning(f"NeuralOperator loader failed ({e}). using synthetic fallback.")
        def unsqueeze_transform(t):
            return t.unsqueeze(0) if t.ndim == 2 else t

        train_dataset = DarcyDataset(
            resolution=resolution, num_samples=args.n_train, train=True,
            transform=unsqueeze_transform
        )
        test_dataset = DarcyDataset(
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
    
    model = FNO2d(
        in_channels=1, out_channels=1,
        modes1=12, modes2=12,
        width=args.width, num_layers=4
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()
    
    history = {'train': [], 'test': []}
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            out = model(x)
            
            if out.ndim != y.ndim:
                out = out.squeeze(1) if out.ndim > y.ndim else out
            
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        test_err = evaluate_model(model, test_loader, device)
        
        history['train'].append(avg_train_loss)
        history['test'].append(test_err)
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}/{args.epochs} | Loss: {avg_train_loss:.6f}")

    return test_err, model, history


def train_lrr(args, device, train_loader, test_loader):
    """Train LRR-FNO model (Two-Stage: Hybrid + Distill)."""
    logger.info("Training LRR-FNO (Two-Stage: Hybrid + Distill)...")
    
    model = LRRFNO2d(
        in_channels=1, out_channels=1,
        modes1=12, modes2=12,
        width=args.width, num_layers=4,
        latent_dim=128,
        encoder_channels=[16, 32, 64],
        use_gated_bridge=False
    ).to(device)
    
    loss_fn = LRNLoss(
        temperature=0.1,
        lambda_mse=5.0,
        lambda_nce=0.001,
        symmetric_nce=True
    )
    
    # TWO STAGE
    distill_epochs = 250
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
        checkpoint_dir='checkpoints_darcy_twostage',
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
            
            if is_lrr:
                out = model(x)['prediction']
            else:
                out = model(x)
                if out.ndim != y.ndim:
                    out = out.squeeze(1) if out.ndim > y.ndim else out
            
            errors.append(compute_relative_l2(out, y))
            
    return np.mean(errors)


def plot_comparison(fno_model, lrr_model, loader, device, metrics, out_dir):
    """Visualize qualitative comparison using subplots."""
    fno_model.eval()
    lrr_model.eval()
    
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        fno_pred = fno_model(x)
        lrr_pred = lrr_model(x)['prediction']
        
    # Prepare for plotting (first sample)
    if fno_pred.ndim == 4: fno_pred = fno_pred.squeeze(1)
    if lrr_pred.ndim == 4: lrr_pred = lrr_pred.squeeze(1)
    if y.ndim == 4: y = y.squeeze(1)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['Ground Truth', 'Baseline FNO', 'LRR-FNO']
    data = [y[0], fno_pred[0], lrr_pred[0]]
    
    for ax, title, img in zip(axes, titles, data):
        im = ax.imshow(img.cpu(), cmap='viridis', aspect='equal')
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    fno_err, lrr_err = metrics
    imp = (fno_err - lrr_err) / fno_err * 100
    
    out_path = out_dir / 'comparison.png'

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison plot saved to {out_path}")


def plot_loss_history(fno_history, lrr_history, out_dir):
    """Plot convergence curves."""
    plt.figure(figsize=(10, 6))
    
    # FNO
    plt.plot(fno_history['train'], 'b--', label='FNO Train', alpha=0.5)
    plt.plot(fno_history['test'], 'b-', label='FNO Test')
    
    # LRR
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
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--width', type=int, default=32, help='Model width (d_v)')
    parser.add_argument('--n_train', type=int, default=1000, help='Training samples')
    parser.add_argument('--n_test', type=int, default=200, help='Test samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    set_seed(args.seed)
    device = get_device()
    
    logger.info(f"Device: {device}")
    
    train_loader, test_loader, _ = load_data(args)
    
    # Baseline Training
    fno_err, fno_model, fno_hist = train_fno(args, device, train_loader, test_loader)
    logger.info(f"Baseline FNO Error: {fno_err:.4f}")
    
    # LRR Training
    lrr_err, lrr_model, lrr_hist = train_lrr(args, device, train_loader, test_loader)
    logger.info(f"LRR-FNO Error:      {lrr_err:.4f}")
    
    improvement = (fno_err - lrr_err) / fno_err * 100
    logger.info(f"Final Improvement:  {improvement:.2f}%")
    
    # Save Results
    out_dir = Path('results/plots/darcy/two_stage')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(fno_model, lrr_model, test_loader, device, (fno_err, lrr_err), out_dir)
    plot_loss_history(fno_hist, lrr_hist, out_dir)
    
    metrics = {
        'fno_error': fno_err,
        'lrr_error': lrr_err,
        'improvement_percent': improvement,
        'config': vars(args)
    }
    with open(out_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Metrics saved to {out_dir / 'metrics.json'}")


if __name__ == '__main__':
    main()
