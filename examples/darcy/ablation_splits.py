"""
Ablation Study: Optimal Training Split for LRR-FNO (Darcy Flow).
This script compares different splits of Stage 1 (Hybrid) and Stage 2 (Distillation)
within a fixed epoch budget (default: 500).
"""

import sys
import argparse
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

def compute_relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    diff_norm = torch.norm(pred.flatten(1) - target.flatten(1), p=2, dim=1)
    target_norm = torch.norm(target.flatten(1), p=2, dim=1)
    return (diff_norm / (target_norm + 1e-8)).mean().item()

def evaluate_model(model, loader, device, is_lrr=False):
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

def train_fno(args, device, train_loader, test_loader):
    logger.info("Training Baseline FNO...")
    model = FNO2d(
        in_channels=1, out_channels=1,
        modes1=args.modes, modes2=args.modes, 
        width=args.width, num_layers=4
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    criterion = nn.MSELoss()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            if out.ndim != y.ndim:
                out = out.squeeze(1) if out.ndim > y.ndim else out
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return evaluate_model(model, test_loader, device)

def load_data(args):
    """Load Darcy dataset with synthetic fallback."""
    logger.info("Loading Darcy dataset...")
    resolution = 32
    try:
        train_loader, test_loader, _ = create_neuralop_dataloaders(
            dataset_name='darcy',
            n_train=args.n_train,
            n_test=args.n_test,
            batch_size=args.batch_size,
            resolution=resolution,
            encode_output=True,
            return_tuple_format=True,
            encode_input=True
        )
        return train_loader, test_loader
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
            torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        )

def run_ablation(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_data(args)
    
    fno_err = train_fno(args, device, train_loader, test_loader)
    logger.info(f"Baseline FNO Error: {fno_err:.6f}")
    
    # Test different splits (Stage1_Epochs, Stage2_Epochs)
    splits = [
        (args.epochs, 0),         # 100/0
        (int(args.epochs*0.9), int(args.epochs*0.1)), # 90/10
        (int(args.epochs*0.8), int(args.epochs*0.2)), # 80/20
        (int(args.epochs*0.7), int(args.epochs*0.3)), # 70/30
        (int(args.epochs*0.5), int(args.epochs*0.5)), # 50/50
        (int(args.epochs*0.4), int(args.epochs*0.6)), # 40/60
        (int(args.epochs*0.3), int(args.epochs*0.7)), # 30/70
        (int(args.epochs*0.2), int(args.epochs*0.8)), # 20/80
        (int(args.epochs*0.1), int(args.epochs*0.9)), # 10/90
    ]
    
    results = []
    
    for stage1, stage2 in splits:
        logger.info(f"Testing split: Hybrid={stage1} | Distill={stage2}")
        model = LRRFNO2d(
            in_channels=1, out_channels=1,
            modes1=args.modes, modes2=args.modes,
            width=args.width, num_layers=4,
            latent_dim=128, encoder_channels=[16, 32, 64]
        ).to(device)
        
        loss_fn = LRNLoss(temperature=0.1, lambda_mse=5.0, lambda_nce=0.001, symmetric_nce=True)
        
        trainer = Trainer(
            model=model, loss_fn=loss_fn, train_loader=train_loader, test_loader=test_loader,
            device=device, stage1_epochs=stage1, stage2_epochs=stage2,
            stage1_lr=1e-3, stage2_lr=1e-4, checkpoint_dir=f'checkpoints_ablation_darcy_{stage1}_{stage2}'
        )
        trainer.train()
        
        lrr_err = evaluate_model(model, test_loader, device, is_lrr=True)
        results.append({
            'hybrid_epochs': stage1,
            'distill_epochs': stage2,
            'lrr_error': lrr_err,
            'split_ratio': f"{stage1}/{stage2}"
        })
        logger.info(f"Result for {stage1}/{stage2}: {lrr_err:.6f}")

    # Save results
    out_dir = Path('results/ablation/darcy')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    final_data = {
        'fno_error': fno_err,
        'ablation_results': results
    }
    with open(out_dir / 'splits_data.json', 'w') as f:
        json.dump(final_data, f, indent=4)
        
    # Plotting
    ratios = [r['split_ratio'] for r in results]
    errors = [r['lrr_error'] for r in results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(ratios, errors, 'o-', label='LRR-FNO (Ours)', linewidth=2)
    plt.axhline(y=fno_err, color='r', linestyle='--', label='Baseline FNO')
    plt.xlabel('Hybrid / Distillation Epoch Split')
    plt.ylabel('Relative L2 Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(out_dir / 'splits_trend.png', bbox_inches='tight')
    plt.close()
    
    logger.info(f"Ablation study complete. Results saved to {out_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--modes', type=int, default=12)
    args = parser.parse_args()
    run_ablation(args)
