"""
Generate Burgers 1D Dataset following FNO paper specifications (Appendix A.3).

Initial condition: u_0 ~ N(0, 625(-Δ + 25I)^{-2}) with periodic BC
Viscosity: ν = 0.1
Solver: Split-step method (Fourier space heat + forward Euler advection)
Resolution: 8192 (subsampled to 128 for training)
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_gaussian_field(resolution: int, n_samples: int, alpha: float = 2.0, 
                            tau: float = 5.0, sigma: float = 625.0, device='cpu'):
    """
    Generate Gaussian random field samples from N(0, sigma*(-Δ + tau^2*I)^{-alpha})
    
    For FNO Burgers: sigma=625, tau=5 (so tau^2=25), alpha=2
    """
    # Wavenumbers for 1D periodic domain
    k = torch.fft.fftfreq(resolution, d=1.0/resolution).to(device)
    
    # (-Δ + tau^2*I) in Fourier space = (4π²k² + tau²)
    # Eigenvalues of inverse: 1 / (4π²k² + tau²)^alpha
    coeff = (4.0 * np.pi**2 * k**2 + tau**2) ** (-alpha / 2.0)
    coeff = coeff * np.sqrt(sigma)
    
    # Generate random coefficients in Fourier space
    # Real and imaginary parts are independent Gaussians
    xi = torch.randn(n_samples, resolution, dtype=torch.cfloat, device=device)
    
    # Apply covariance structure
    u_hat = xi * coeff.unsqueeze(0)
    
    # Transform back to physical space
    # Normalize: IFFT divides by N, so multiply by N to get correct variance
    u = torch.fft.ifft(u_hat).real * resolution
    
    return u

def solve_burgers_spectral(u0: torch.Tensor, nu: float = 0.1, T: float = 1.0, 
                           dt: float = 1e-4, device='cpu'):
    """
    Solve 1D viscous Burgers equation using split-step spectral method.
    
    ∂u/∂t + u·∂u/∂x = ν·∂²u/∂x²
    
    Uses operator splitting:
    1. Solve heat equation exactly in Fourier space
    2. Advance nonlinear term with forward Euler
    """
    resolution = u0.shape[-1]
    n_steps = int(T / dt)
    
    u = u0.clone().to(device)
    
    # Wavenumbers
    k = torch.fft.fftfreq(resolution, d=1.0/resolution).to(device) * 2 * np.pi
    
    for _ in range(n_steps):
        # Step 1: Solve heat equation ∂u/∂t = ν·∂²u/∂x² exactly in Fourier space
        u_hat = torch.fft.fft(u)
        diffusion_factor = torch.exp(-nu * k**2 * dt)
        u_hat = u_hat * diffusion_factor.unsqueeze(0)
        u = torch.fft.ifft(u_hat).real
        
        # Step 2: Advance nonlinear term ∂u/∂t + u·∂u/∂x = 0
        # Compute ∂u/∂x in Fourier space then transform back
        u_hat = torch.fft.fft(u)
        du_dx_hat = 1j * k.unsqueeze(0) * u_hat
        du_dx = torch.fft.ifft(du_dx_hat).real
        
        # Forward Euler in physical space: u^{n+1} = u^n - dt * u * ∂u/∂x
        u = u - dt * u * du_dx
    
    return u

def generate_burgers_dataset(n_train: int = 1000, n_test: int = 200, 
                              resolution: int = 8192, output_resolution: int = 128,
                              nu: float = 0.1, T: float = 1.0, 
                              output_dir: str = 'datasets/burgers',
                              device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate Burgers dataset following FNO paper specifications.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating Burgers dataset on {device}...")
    print(f"  High-fidelity resolution: {resolution}")
    print(f"  Output resolution: {output_resolution}")
    print(f"  Viscosity: {nu}")
    print(f"  Time horizon: {T}")
    
    # Time step for stability (CFL-like condition)
    # N=8192 => dx ~ 1.2e-4. Max u ~ 3. dt=1e-4 is empirically stable.
    dt = 1e-4
    
    # Subsampling factor
    sub = resolution // output_resolution
    
    for split, n_samples in [('train', n_train), ('test', n_test)]:
        print(f"\nGenerating {split} set ({n_samples} samples)...")
        
        # Generate initial conditions
        u0_full = generate_gaussian_field(
            resolution=resolution, n_samples=n_samples,
            alpha=2.0, tau=5.0, sigma=625.0, device=device
        )
        
        # Solve Burgers equation
        uT_full = torch.zeros_like(u0_full)
        batch_size = 50  # Process in batches for memory efficiency
        
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Solving {split}"):
            end_idx = min(i + batch_size, n_samples)
            uT_full[i:end_idx] = solve_burgers_spectral(
                u0_full[i:end_idx], nu=nu, T=T, dt=dt, device=device
            )
        
        # Subsample to output resolution
        u0 = u0_full[:, ::sub].cpu()
        uT = uT_full[:, ::sub].cpu()
        
        # Save dataset
        data = {
            'x': u0,  # Initial condition (input)
            'y': uT,  # Solution at T (output)
        }
        
        save_path = output_path / f'burgers_{split}_{output_resolution}.pt'
        torch.save(data, save_path)
        print(f"Saved {save_path} ({u0.shape})")
    
    print("\nDataset generation complete!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--resolution', type=int, default=8192)
    parser.add_argument('--output_resolution', type=int, default=128)
    parser.add_argument('--nu', type=float, default=0.1)
    parser.add_argument('--T', type=float, default=1.0)
    parser.add_argument('--output_dir', type=str, default='datasets/burgers')
    args = parser.parse_args()
    
    generate_burgers_dataset(
        n_train=args.n_train,
        n_test=args.n_test,
        resolution=args.resolution,
        output_resolution=args.output_resolution,
        nu=args.nu,
        T=args.T,
        output_dir=args.output_dir
    )
