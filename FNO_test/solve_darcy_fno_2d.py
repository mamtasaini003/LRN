import sys
import torch
import torch.nn as nn
import torch.optim as optim
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import json
import argparse

# Add src to python path to allow imports
# Assuming script is in FNO_test/ and src is in ../src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.components.fno import FNO2d
from src.data.neuralop_loaders import create_neuralop_dataloaders

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def compute_relative_mse(pred, y):
    """Compute relative MSE: ||pred - y||^2 / ||y||^2"""
    diff_norm = torch.norm(pred.flatten(1) - y.flatten(1), p=2, dim=1) ** 2
    y_norm = torch.norm(y.flatten(1), p=2, dim=1) ** 2
    return (diff_norm / y_norm).mean().item()

def save_animation(history_data, out_path, test_rels=None):
    """Create and save 3-panel animation: Exact, Prediction, Error."""
    logger.info("Generating refined animation...")
    
    plt.style.use('default') 
    # Use constrained_layout=False to avoid engine conflicts with tight_layout
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=False)
    
    # Subplot titles
    titles = ["Ground Truth", "FNO Prediction", "Piecewise Abs. Error"]
    cmaps = ['viridis', 'viridis', 'magma']
    
    # Get common range from first frame ground truth
    _, _, first_true = history_data[0]
    vmin, vmax = first_true.min(), first_true.max()
    
    # Initialize images and colorbars
    images = []
    cbs = []
    
    for i, (ax, title, cmap) in enumerate(zip(axes, titles, cmaps)):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_aspect('equal')
        
        # Initial display
        init_data = np.zeros_like(first_true)
        im = ax.imshow(init_data, cmap=cmap, origin='lower')
        images.append(im)
        
        # Specific color limits
        if i < 2: # GT and Prediction
            im.set_clim(vmin, vmax)
        
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cb.ax.tick_params(labelsize=10)
        cbs.append(cb)
        
        ax.axis('off')

    def update(frame_idx):
        epoch, pred, true_val = history_data[frame_idx]
        rel_err = test_rels[frame_idx] if test_rels else 0.0
        error = np.abs(true_val - pred)
        
        # Update images
        images[0].set_data(true_val)
        images[1].set_data(pred)
        images[2].set_data(error)
        images[2].set_clim(0, max(error.max(), 1e-4)) 
        
        fig.suptitle(f"Epoch {epoch} | Test Rel. L2 Error: {rel_err:.4f}", 
                     fontsize=20, fontweight='bold', y=0.98)
        
        return images

    anim = animation.FuncAnimation(
        fig, update, frames=len(history_data), interval=150, blit=False
    )
    
    # Final layout adjustment
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    anim.save(out_path, writer='pillow', fps=7, dpi=100)
    logger.info(f"Refined animation saved to {out_path}")
    plt.close(fig)

def save_static_plot(pred, true_val, epoch, out_path, rel_err=0.0):
    """Save final 3-panel comparison with improved aesthetics."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=False)
    
    error = np.abs(true_val - pred)
    vmin, vmax = true_val.min(), true_val.max()
    
    titles = ["Ground Truth", f"Prediction (Epoch {epoch})", "Piecewise Abs. Error"]
    data = [true_val, pred, error]
    cmaps = ['viridis', 'viridis', 'magma']
    
    for i, (ax, title, d, cmap) in enumerate(zip(axes, titles, data, cmaps)):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_aspect('equal')
        
        curr_vmax = vmax if i < 2 else d.max()
        curr_vmin = vmin if i < 2 else 0
        
        im = ax.imshow(d, cmap=cmap, origin='lower', vmin=curr_vmin, vmax=curr_vmax)
        cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cb.ax.tick_params(labelsize=10)
        ax.axis('off')
        
    fig.suptitle(f"Final Results | Test Rel. L2 Error: {rel_err:.4f}", 
                 fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Refined static result saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description='FNO 2D Darcy Solver')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--modes', type=int, default=12, help='Number of Fourier modes')
    parser.add_argument('--width', type=int, default=32, help='Width of the network')
    parser.add_argument('--train_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--test_samples', type=int, default=200, help='Number of test samples')
    parser.add_argument('--res', type=int, default=32, help='Resolution')
    args = parser.parse_args()

    # Hyperparameters
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    MODES = args.modes
    WIDTH = args.width
    TRAIN_SAMPLES = args.train_samples
    TEST_SAMPLES = args.test_samples
    RESOLUTION = args.res
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # 1. Data Loading
    logger.info("Loading Darcy Flow dataset...")
    try:
        train_loader, test_loader, _ = create_neuralop_dataloaders(
            dataset_name='darcy',
            n_train=TRAIN_SAMPLES,
            n_test=TEST_SAMPLES,
            batch_size=BATCH_SIZE,
            test_batch_size=BATCH_SIZE,
            resolution=RESOLUTION,
            encode_input=True,
            encode_output=True,
            return_tuple_format=True
        )
    except Exception as e:
        logger.warning(f"Failed to load data via neuraloperator: {e}")
        logger.info("Attempting fallback to local synthetic/pre-generated dataset...")
        
        from src.data.pde_datasets import DarcyDataset
        
        # Helper to ensure [C, H, W] dims
        def unsqueeze_transform(t):
            return t.unsqueeze(0) if t.ndim == 2 else t

        # Load synthetic/local data
        train_dataset = DarcyDataset(
            resolution=RESOLUTION, 
            num_samples=TRAIN_SAMPLES, 
            train=True,
            transform=unsqueeze_transform
        )
        test_dataset = DarcyDataset(
            resolution=RESOLUTION, 
            num_samples=TEST_SAMPLES, 
            train=False,
            transform=unsqueeze_transform
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

    # 2. Model Initialization
    logger.info("Initializing FNO2d model...")
    model = FNO2d(
        in_channels=1,
        out_channels=1,
        modes1=MODES,
        modes2=MODES,
        width=WIDTH,
        num_layers=4,
        padding=9
    ).to(device)
    
    # 3. Optimization Setup
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    criterion = nn.MSELoss()
    
    # Fixed sample for visualization
    fixed_x, fixed_y = next(iter(test_loader))
    fixed_x = fixed_x[:1].to(device) # Keep batch dim [1, C, H, W]
    fixed_y = fixed_y[:1].to(device) # [1, H, W] or [1, 1, H, W]
    
    # Ensure y is proper shape for plotting [H, W]
    if fixed_y.ndim == 4: fixed_y_plot = fixed_y.squeeze(0).squeeze(0).cpu().numpy()
    elif fixed_y.ndim == 3: fixed_y_plot = fixed_y.squeeze(0).cpu().numpy()
    
    # 4. Training Loop
    logger.info("Starting training...")
    train_losses = []
    test_rels = []
    
    # Store frames for animation
    history_frames = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_mse = 0.0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            out = model(x)
            
            if out.ndim != y.ndim:
                 out = out.squeeze(1) if out.ndim > y.ndim else out

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_mse += loss.item()
            
        avg_train_mse = train_mse / len(train_loader)
        scheduler.step()
        
        # Evaluation loop for metrics
        model.eval()
        total_rel_mse = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                if out.ndim != y.ndim:
                    out = out.squeeze(1) if out.ndim > y.ndim else out
                
                total_rel_mse += compute_relative_mse(out, y)
        
        avg_rel_mse = total_rel_mse / len(test_loader)
        test_rels.append(avg_rel_mse)
        train_losses.append(avg_train_mse)
        
        # Visualization Snapshot
        with torch.no_grad():
            pred_vis = model(fixed_x)
            if pred_vis.ndim == 4: pred_vis = pred_vis.squeeze(1)
            pred_frame = pred_vis.squeeze(0).cpu().numpy()
            
        history_frames.append((epoch, pred_frame, fixed_y_plot))
        
        # Print metrics
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{EPOCHS} | Train MSE: {avg_train_mse:.6f} | Test Rel L2: {avg_rel_mse:.6f}")
        
    logger.info("Training completed.")
    
    # 5. Save Results
    # Animation
    anim_path = Path('FNO_test/evolution_animation.gif')
    save_animation(history_frames, anim_path, test_rels=test_rels)
    
    # Final Static Plot
    final_path = Path('FNO_test/final_result.png')
    save_static_plot(history_frames[-1][1], history_frames[-1][2], EPOCHS, final_path, rel_err=test_rels[-1])
    
    # Final metrics calculation
    final_metrics = {
        "configuration": {
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "modes": MODES,
            "width": WIDTH,
            "train_samples": TRAIN_SAMPLES,
            "test_samples": TEST_SAMPLES,
            "resolution": RESOLUTION,
            "device": str(device)
        },
        "results": {
            "final_train_mse": train_losses[-1],
            "final_test_rel_l2": test_rels[-1],
            "min_test_rel_l2": min(test_rels),
            "train_losses": train_losses,
            "test_rel_l2_history": test_rels
        }
    }
    
    with open('FNO_test/results.json', 'w') as f:
        json.dump(final_metrics, f, indent=4)
    logger.info("Saved configuration and results to FNO_test/results.json")

if __name__ == "__main__":
    main()
