"""
Latent Rescaling Network (LRN) - 1D Burgers Experiment (One-Stage).
Training Strategy: 500 Epochs Hybrid (MSE + NCE)
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
            if "neuraloperator" in str(e) and "not installed" in str(e):
                raise e 
            
            if args.n_train is not None:
                logger.warning(f"Failed to load {args.n_train} samples ({e}). Retrying with all available data...")
                return create_neuralop_dataloaders(
                    dataset_name='burgers',
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
        def unsqueeze_transform(t):
            return t.unsqueeze(0) if t.ndim == 1 else t

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
        modes=args.modes, width=args.width, num_layers=4,
        padding=0  # Periodic boundary
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()
    
    history = {'train': [], 'test': []}
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if x.ndim == 2: x = x.unsqueeze(1)
            if y.ndim == 2: y = y.unsqueeze(1)
            
            optimizer.zero_grad()
            out = model(x)
            
            if out.ndim == 3 and out.shape[-1] == 1:
                out = out.permute(0, 2, 1)
                
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
    """Train LRR-FNO model (One-Stage Hybrid)."""
    logger.info("Training LRR-FNO (One-Stage Hybrid)...")
    
    model = LRRFNO1d(
        in_channels=1, out_channels=1,
        modes=args.modes, width=args.width, num_layers=4,
        latent_dim=128,
        encoder_channels=[16, 32, 64],
        use_gated_bridge=False,
        padding=0  # Periodic boundary
    ).to(device)
    
    loss_fn = LRNLoss(
        temperature=0.07,
        lambda_mse=5.0,
        lambda_nce=0.001,
        symmetric_nce=True
    )
    
    # ONE STAGE: 100% Hybrid
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        stage1_epochs=args.epochs, # All epochs for Stage 2 (Hybrid)
        stage2_epochs=0,           # No Stage 3 (Distill)
        stage1_lr=1e-3,
        weight_decay=0.0,
        checkpoint_dir='checkpoints_burgers_onestage',
        scheduler_type='step',
        scheduler_kwargs={'step_size': 100, 'gamma': 0.5},
        weight_decay=1e-4
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


def plot_comparison(fno_model, lrr_model, loader, device, metrics, out_dir):
    """Visualize qualitative comparison."""
    fno_model.eval()
    lrr_model.eval()
    
    batch_errors_lrr = []
    
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    if x.ndim == 2: x = x.unsqueeze(1)
    if y.ndim == 2: y = y.unsqueeze(1)
    
    with torch.no_grad():
        fno_out = fno_model(x)
        if fno_out.ndim == 3 and fno_out.shape[-1] == 1:
            fno_out = fno_out.permute(0, 2, 1)
        
        lrr_out = lrr_model(x)['prediction']
    
    for i in range(x.shape[0]):
        err_lrr = compute_relative_l2(lrr_out[i:i+1], y[i:i+1])
        batch_errors_lrr.append(err_lrr)
    
    batch_errors_lrr = np.array(batch_errors_lrr)
    
    indices = [
        np.argmin(batch_errors_lrr),              # Best
        np.argsort(batch_errors_lrr)[len(batch_errors_lrr)//2], # Median
        np.argmax(batch_errors_lrr)               # Worst
    ]
    titles = ['Best Case', 'Median Case', 'Worst Case']
    filenames = ['best_case.png', 'median_case.png', 'worst_case.png']
    
    fno_total, lrr_total = metrics
    imp = (fno_total - lrr_total) / fno_total * 100
    
    for idx, title, fname in zip(indices, titles, filenames):
        y_true = y[idx, 0].cpu().numpy()
        y_fno = fno_out[idx, 0].cpu().numpy()
        y_lrr = lrr_out[idx, 0].cpu().numpy()
        
        plt.figure(figsize=(6, 4))
        plt.plot(y_true, 'k-', label='Ground Truth', linewidth=2)
        plt.plot(y_fno, 'b--', label='FNO', alpha=0.7)
        plt.plot(y_lrr, 'g-.', label='LRR', alpha=0.9, linewidth=2)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        out_path = out_dir / fname
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved {title} plot to {out_path}")


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
    
    fno_err, fno_model, fno_hist = train_fno(args, device, train_loader, test_loader)
    logger.info(f"Baseline FNO Error: {fno_err:.4f}")
    
    lrr_err, lrr_model, lrr_hist = train_lrr(args, device, train_loader, test_loader)
    logger.info(f"LRR-FNO Error:      {lrr_err:.4f}")
    
    improvement = (fno_err - lrr_err) / fno_err * 100
    logger.info(f"Final Improvement:  {improvement:.2f}%")
    
    # Save Results
    out_dir = Path('results/plots/burgers/one_stage')
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
