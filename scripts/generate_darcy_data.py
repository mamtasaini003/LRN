"""
Generate Darcy Flow 2D Dataset following FNO paper specifications (Appendix A.3).

Coefficient: a ~ ψ_# N(0, (-Δ + 9I)^{-2}) with zero Neumann BC
Mapping ψ: Takes value 12 for positive, 3 for negative (piecewise constant)
Forcing: f(x) = 1 (constant)
Solver: Second-order finite difference on 421x421 grid, subsampled
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from scipy import sparse
from scipy.sparse.linalg import spsolve

def generate_gaussian_field_2d(resolution: int, n_samples: int, alpha: float = 2.0, 
                                tau: float = 3.0, device='cpu'):
    """
    Generate 2D Gaussian random field from N(0, (-Δ + tau^2*I)^{-alpha})
    with zero Neumann boundary conditions.
    
    For FNO Darcy: tau=3 (so tau^2=9), alpha=2
    """
    # Use spectral method with periodic BC as approximation
    # then the piecewise mapping handles the boundary effects
    kx = torch.fft.fftfreq(resolution, d=1.0/resolution).to(device)
    ky = torch.fft.fftfreq(resolution, d=1.0/resolution).to(device)
    kx, ky = torch.meshgrid(kx, ky, indexing='ij')
    
    # (-Δ + tau^2) in Fourier space = (4π²(kx² + ky²) + tau²)
    k_sq = kx**2 + ky**2
    coeff = (4.0 * np.pi**2 * k_sq + tau**2) ** (-alpha / 2.0)
    
    # Generate random coefficients
    xi = torch.randn(n_samples, resolution, resolution, dtype=torch.cfloat, device=device)
    
    # Apply covariance structure
    u_hat = xi * coeff.unsqueeze(0)
    
    # Transform back to physical space
    # Normalize: IFFT2 divides by N^2, so multiply by N^2 to get correct variance
    u = torch.fft.ifft2(u_hat).real * (resolution**2)
    
    return u

def psi_mapping(field: torch.Tensor, high_val: float = 12.0, low_val: float = 3.0):
    """
    Piecewise constant mapping: 12 for positive, 3 for negative.
    """
    return torch.where(field > 0, high_val, low_val)

def build_darcy_matrix(a: np.ndarray, h: float):
    """
    Build sparse matrix for -∇·(a∇u) = f using second-order FD.
    Dirichlet BC: u = 0 on boundary.
    """
    ny, nx = a.shape
    N = nx * ny
    
    # Interior stencil coefficients
    rows, cols, vals = [], [], []
    
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                # Dirichlet boundary: u = 0
                rows.append(idx)
                cols.append(idx)
                vals.append(1.0)
            else:
                # Interior point: 5-point stencil
                # -( a_{i+1/2}(u_{i+1} - u_i) - a_{i-1/2}(u_i - u_{i-1}) ) / h^2
                
                a_im = 0.5 * (a[j, i] + a[j, i-1])  # a_{i-1/2}
                a_ip = 0.5 * (a[j, i] + a[j, i+1])  # a_{i+1/2}
                a_jm = 0.5 * (a[j, i] + a[j-1, i])  # a_{j-1/2}
                a_jp = 0.5 * (a[j, i] + a[j+1, i])  # a_{j+1/2}
                
                diag = (a_im + a_ip + a_jm + a_jp) / h**2
                
                # Diagonal
                rows.append(idx)
                cols.append(idx)
                vals.append(diag)
                
                # Off-diagonals
                rows.append(idx); cols.append(idx - 1); vals.append(-a_im / h**2)
                rows.append(idx); cols.append(idx + 1); vals.append(-a_ip / h**2)
                rows.append(idx); cols.append(idx - nx); vals.append(-a_jm / h**2)
                rows.append(idx); cols.append(idx + nx); vals.append(-a_jp / h**2)
    
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(N, N)).tocsr()
    return A

def solve_darcy(a: np.ndarray, f: float = 1.0):
    """
    Solve -∇·(a∇u) = f with Dirichlet BC u=0.
    """
    ny, nx = a.shape
    h = 1.0 / (nx - 1)
    
    A = build_darcy_matrix(a, h)
    
    # RHS: f=1 in interior, 0 on boundary
    b = np.ones(nx * ny) * f
    for j in range(ny):
        for i in range(nx):
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                b[j * nx + i] = 0.0
    
    u = spsolve(A, b)
    return u.reshape(ny, nx)

def generate_darcy_dataset(n_train: int = 1000, n_test: int = 200,
                           solver_resolution: int = 421, output_resolution: int = 85,
                           output_dir: str = 'datasets/darcy',
                           device: str = 'cpu'):
    """
    Generate Darcy dataset following FNO paper specifications.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating Darcy dataset...")
    print(f"  Solver resolution: {solver_resolution}")
    print(f"  Output resolution: {output_resolution}")
    
    # Subsampling
    sub = solver_resolution // output_resolution
    
    for split, n_samples in [('train', n_train), ('test', n_test)]:
        print(f"\nGenerating {split} set ({n_samples} samples)...")
        
        # Generate random Gaussian fields at solver resolution
        fields = generate_gaussian_field_2d(
            resolution=solver_resolution, n_samples=n_samples,
            alpha=2.0, tau=3.0, device=device
        )
        
        # Apply piecewise mapping
        a_full = psi_mapping(fields, high_val=12.0, low_val=3.0).cpu().numpy()
        
        # Solve for each sample
        u_full = np.zeros_like(a_full)
        for i in tqdm(range(n_samples), desc=f"Solving {split}"):
            u_full[i] = solve_darcy(a_full[i], f=1.0)
        
        # Subsample
        a_out = torch.from_numpy(a_full[:, ::sub, ::sub]).float()
        u_out = torch.from_numpy(u_full[:, ::sub, ::sub]).float()
        
        # Save
        data = {
            'x': a_out,  # Coefficient field (input)
            'y': u_out,  # Solution field (output)
        }
        
        save_path = output_path / f'darcy_{split}_{output_resolution}.pt'
        torch.save(data, save_path)
        print(f"Saved {save_path} ({a_out.shape})")
    
    print("\nDarcy dataset generation complete!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_train', type=int, default=1000)
    parser.add_argument('--n_test', type=int, default=200)
    parser.add_argument('--solver_resolution', type=int, default=421)
    parser.add_argument('--output_resolution', type=int, default=85)
    parser.add_argument('--output_dir', type=str, default='datasets/darcy')
    args = parser.parse_args()
    
    generate_darcy_dataset(
        n_train=args.n_train,
        n_test=args.n_test,
        solver_resolution=args.solver_resolution,
        output_resolution=args.output_resolution,
        output_dir=args.output_dir
    )
