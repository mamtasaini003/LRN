# 2D Darcy Flow Problem Description

This document outlines the problem statement, dataset specifications, and model configuration used in the FNO Darcy example.

## Problem Statement

The goal is to solve the **2D Darcy Flow** equation, which describes the flow of a fluid through a porous medium. The governing steady-state equation is:

$$-\nabla \cdot (a(x) \nabla u(x)) = f(x), \quad x \in (0,1)^2$$

Where:
- $a(x)$ is the **permeability field** (input).
- $u(x)$ is the **fluid pressure** (output/solution).
- $f(x)$ is the **forcing function** (source term).

### Boundary Conditions
The solution $u(x)$ is subject to **Dirichlet boundary conditions**:
- $u(x) = 0$ on the boundary $\partial [0,1]^2$.

## Dataset Specifications (FNO Paper Specs)

Following the specifications in the original FNO paper (*Li et al., 2020, Appendix A.3*):

### 1. Coefficient Generation ($a(x)$)
The permeability field $a(x)$ is generated as a piecewise constant field derived from a Gaussian Random Field (GRF):
- **Base GRF**: $a \sim N(0, (-\Delta + 9I)^{-2})$ with zero Neumann boundary conditions.
- **Mapping**: A non-linear thresholding function $\psi$ is applied to make the field piecewise constant:
  - $a(x) = 12$ where the GRF is positive.
  - $a(x) = 3$ where the GRF is negative.

### 2. Forcing Function ($f(x)$)
- The forcing is constant across the entire domain: $f(x) = 1$.

### 3. Numerical Solver
- **Method**: Second-order finite difference scheme.
- **Internal Resolution**: The ground truth is computed on a fine $421 \times 421$ grid to ensure high accuracy.
- **Subsampling**: The data is then subsampled to the target resolution for training/testing.

## Experiment Configuration

| Parameter | Value |
| :--- | :--- |
| **Model** | FNO2d (Fourier Neural Operator) |
| **Training Resolution** | $32 \times 32$ |
| **Input Channels** | 1 (Permeability field $a$) |
| **Output Channels** | 1 (Pressure field $u$) |
| **Training Epochs** | 100 (Default) |
| **Modes** | 12 (Fourier modes in each dimension) |
| **Width** | 32 (Hidden dimension / channel width) |
| **Batch Size** | 32 |
| **Optimizer** | Adam |
| **Learning Rate** | $1 \times 10^{-3}$ |

## Understanding the Goal
The FNO model learns the **operator mapping** from the permeability coefficient $a(x)$ to the pressure solution $u(x)$. Unlike traditional solvers that solve the PDE for a specific $a(x)$, the FNO is trained to generalize across different permeability distributions, allowing for near-instantaneous inference once trained.
