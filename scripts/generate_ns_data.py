"""
Generate Navier-Stokes 2D Dataset following FNO paper specifications (Appendix A.3).

Initial vorticity: w_0 ~ N(0, 7^{3/2}(-Δ + 49I)^{-2.5}) with periodic BC
Forcing: f(x) = 0.1(sin(2π(x₁+x₂)) + cos(2π(x₁+x₂)))
Solver: Pseudospectral with stream-function formulation, Crank-Nicolson time stepping
Resolution: 256x256, subsampled to 64x64
Time step: 1e-4, recorded every t=1
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

def generate_vorticity_field(resolution: int, n_samples: int, device='cpu'):
    """
    Generate initial vorticity from N(0, 7^{3/2}(-Δ + 49I)^{-2.5})
    """
    kx = torch.fft.fftfreq(resolution, d=1.0/resolution).to(device)
    ky = torch.fft.fftfreq(resolution, d=1.0/resolution).to(device)
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    
    k_sq = kx**2 + ky**2
    sigma = 7.0 ** 1.5  # 7^{3/2}
    tau_sq = 49.0  # tau^2 = 49
    alpha = 2.5
    
    coeff = sigma * (4.0 * np.pi**2 * k_sq + tau_sq) ** (-alpha / 2.0)
    
    xi = torch.randn(n_samples, resolution, resolution, dtype=torch.cfloat, device=device)
    w_hat = xi * coeff.unsqueeze(0)
    w = torch.fft.ifft2(w_hat).real * (resolution**2)
    
    return w

def solve_ns_spectral(w0: torch.Tensor, nu: float = 1e-3, T: float = 10.0, 
                      dt: float = 1e-4, record_interval: float = 1.0, device='cpu'):
    """
    Solve 2D Navier-Stokes in vorticity form using pseudospectral method.
    
    ∂w/∂t + u·∇w = ν∇²w + f
    ∇·u = 0
    
    Uses stream-function formulation and Crank-Nicolson time stepping.
    """
    batch_size, resolution, _ = w0.shape
    n_steps = int(T / dt)
    record_steps = int(record_interval / dt)
    n_records = int(T / record_interval) + 1
    
    # Wavenumbers
    kx = torch.fft.fftfreq(resolution, d=1.0/resolution).to(device) * 2 * np.pi
    ky = torch.fft.fftfreq(resolution, d=1.0/resolution).to(device) * 2 * np.pi
    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing='ij')
    k_sq = kx_grid**2 + ky_grid**2
    k_sq_safe = k_sq.clone()
    k_sq_safe[0, 0] = 1.0  # Avoid division by zero
    
    # Forcing: f(x) = 0.1(sin(2π(x₁+x₂)) + cos(2π(x₁+x₂)))
    x = torch.linspace(0, 1, resolution, device=device)
    y = torch.linspace(0, 1, resolution, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    forcing = 0.1 * (torch.sin(2*np.pi*(xx + yy)) + torch.cos(2*np.pi*(xx + yy)))
    f_hat = torch.fft.fft2(forcing)
    
    # Dealiasing mask (2/3 rule)
    dealias = torch.ones_like(k_sq)
    kmax = resolution // 3
    dealias[torch.abs(kx_grid) > kmax * 2 * np.pi / resolution] = 0
    dealias[torch.abs(ky_grid) > kmax * 2 * np.pi / resolution] = 0
    
    # Initialize
    w = w0.clone().to(device)
    trajectory = torch.zeros(batch_size, n_records, resolution, resolution, device=device)
    trajectory[:, 0] = w
    
    record_idx = 1
    
    for step in range(1, n_steps + 1):
        w_hat = torch.fft.fft2(w)
        
        # Stream function: ψ = -w / k²
        psi_hat = -w_hat / k_sq_safe.unsqueeze(0)
        psi_hat[:, 0, 0] = 0
        
        # Velocity: u = ∂ψ/∂y, v = -∂ψ/∂x
        u_hat = 1j * ky_grid.unsqueeze(0) * psi_hat
        v_hat = -1j * kx_grid.unsqueeze(0) * psi_hat
        
        u = torch.fft.ifft2(u_hat).real
        v = torch.fft.ifft2(v_hat).real
        
        # Vorticity gradients
        dw_dx_hat = 1j * kx_grid.unsqueeze(0) * w_hat
        dw_dy_hat = 1j * ky_grid.unsqueeze(0) * w_hat
        dw_dx = torch.fft.ifft2(dw_dx_hat).real
        dw_dy = torch.fft.ifft2(dw_dy_hat).real
        
        # Nonlinear term: u·∇w
        nonlin = u * dw_dx + v * dw_dy
        nonlin_hat = torch.fft.fft2(nonlin) * dealias.unsqueeze(0)
        
        # Crank-Nicolson: (1 + νΔtk²/2)w^{n+1} = (1 - νΔtk²/2)w^n - Δt*NL + Δt*f
        denom = 1.0 + nu * dt * k_sq.unsqueeze(0) / 2
        numer = (1.0 - nu * dt * k_sq.unsqueeze(0) / 2) * w_hat - dt * nonlin_hat + dt * f_hat.unsqueeze(0)
        w_hat = numer / denom
        
        w = torch.fft.ifft2(w_hat).real
        
        # Record
        if step % record_steps == 0 and record_idx < n_records:
            trajectory[:, record_idx] = w
            record_idx += 1
    
    return trajectory

def generate_ns_dataset(n_train: int = 1000, n_test: int = 200,
                        resolution: int = 64, T: float = 10.0, nu: float = 1e-3,
                        output_dir: str = 'datasets/navier_stokes',
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate Navier-Stokes dataset following FNO paper specifications.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating Navier-Stokes dataset on {device}...")
    print(f"  Resolution: {resolution}")
    print(f"  Viscosity: {nu}")
    print(f"  Time horizon: {T}")
    
    dt = 1e-3 if resolution <= 64 else 5e-4
    
    for split, n_samples in [('train', n_train), ('test', n_test)]:
        print(f"\nGenerating {split} set ({n_samples} samples)...")
        
        batch_size = 20
        all_x, all_y = [], []
        
        for i in tqdm(range(0, n_samples, batch_size), desc=f"Solving {split}"):
            batch_n = min(batch_size, n_samples - i)
            
            w0 = generate_vorticity_field(resolution, batch_n, device=device)
            trajectory = solve_ns_spectral(w0, nu=nu, T=T, dt=dt, device=device)
            
            # Input: first time step, Output: last time step
            all_x.append(trajectory[:, 0].cpu())
            all_y.append(trajectory[:, -1].cpu())
        
        x = torch.cat(all_x, dim=0)
        y = torch.cat(all_y, dim=0)
        
        data = {
            'x': x,  # Initial vorticity
            'y': y,  # Final vorticity
        }
        
        save_path = output_path / f'ns_{split}_{resolution}.pt'
        torch.save(data, save_path)
        print(f"Saved {save_path} ({x.shape})")
    
    print("\nNavier-Stokes dataset generation complete!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--T', type=float, default=10.0)
    parser.add_argument('--nu', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='datasets/navier_stokes')
    args = parser.parse_args()
    
    generate_ns_dataset(
        n_train=args.n_train,
        n_test=args.n_test,
        resolution=args.resolution,
        T=args.T,
        nu=args.nu,
        output_dir=args.output_dir
    )
