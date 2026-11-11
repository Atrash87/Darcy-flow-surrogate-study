"""Cell-centred finite-volume solver for steady, isotropic Darcy flow.

Problem
-------
    -div(k(x, y) grad(p)) = 0 in (0, 1) x (0, 1)

Boundary conditions
-------------------
    p = 1 on x = 0 (left)
    p = 0 on x = 1 (right)
    n . k grad(p) = 0 on y = 0 and y = 1 (top/bottom)

Parameterization
----------------
    params = (logk_mean, logk_amp, pattern_angle_deg)

The angle rotates the spatial scalar heterogeneity pattern. It does not
represent an anisotropic permeability tensor.

The implementation preserves the return signature:
    pressure, X, Y, q_left = solve_darcy(...)
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


def make_logk_field(
    X: np.ndarray,
    Y: np.ndarray,
    params: tuple[float, float, float],
) -> np.ndarray:
    """Evaluate scalar log-permeability at cell centres."""
    if len(params) != 3:
        raise ValueError("params must be (logk_mean, logk_amp, pattern_angle_deg)")

    logk_mean, logk_amp, angle_deg = map(float, params)
    theta = np.deg2rad(angle_deg)

    xr = np.cos(theta) * X + np.sin(theta) * Y
    yr = -np.sin(theta) * X + np.cos(theta) * Y

    basis = (
        np.sin(2.0 * np.pi * xr) * np.cos(2.0 * np.pi * yr)
        + 0.5 * np.sin(4.0 * np.pi * xr) * np.sin(2.0 * np.pi * yr)
        + 0.3 * np.cos(6.0 * np.pi * xr) * np.cos(2.0 * np.pi * yr)
    )
    return logk_mean + logk_amp * basis


def solve_darcy(
    nx: int = 64,
    ny: int = 64,
    params: tuple[float, float, float] = (0.0, 1.0, 0.0),
    *,
    p_left: float = 1.0,
    p_right: float = 0.0,
):
    """Solve steady scalar Darcy flow; return pressure, coordinates and inflow."""
    if not isinstance(nx, (int, np.integer)) or not isinstance(ny, (int, np.integer)):
        raise TypeError("nx and ny must be integers")
    if nx < 2 or ny < 2:
        raise ValueError("nx and ny must both be at least 2")
    if not np.isfinite(p_left) or not np.isfinite(p_right):
        raise ValueError("Boundary pressures must be finite")

    hx, hy = 1.0 / nx, 1.0 / ny
    x = (np.arange(nx, dtype=float) + 0.5) * hx
    y = (np.arange(ny, dtype=float) + 0.5) * hy
    X, Y = np.meshgrid(x, y)

    logk = make_logk_field(X, Y, params)
    with np.errstate(over="raise", invalid="raise"):
        k = np.exp(logk)
    if not np.all(np.isfinite(k)) or np.any(k <= 0.0):
        raise ValueError("Permeability must be finite and strictly positive")

    kx = 2.0 * k[:, :-1] * k[:, 1:] / (k[:, :-1] + k[:, 1:])
    ky = 2.0 * k[:-1, :] * k[1:, :] / (k[:-1, :] + k[1:, :])

    n_cells = nx * ny
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs = np.zeros(n_cells, dtype=float)

    def idx(i: int, j: int) -> int:
        return j * nx + i

    for j in range(ny):
        for i in range(nx):
            row = idx(i, j)
            diagonal = 0.0

            if i > 0:
                g = kx[j, i - 1] / hx**2
                diagonal += g
                rows.append(row); cols.append(idx(i - 1, j)); values.append(-g)
            else:
                g = 2.0 * k[j, 0] / hx**2
                diagonal += g
                rhs[row] += g * p_left

            if i < nx - 1:
                g = kx[j, i] / hx**2
                diagonal += g
                rows.append(row); cols.append(idx(i + 1, j)); values.append(-g)
            else:
                g = 2.0 * k[j, -1] / hx**2
                diagonal += g
                rhs[row] += g * p_right

            if j > 0:
                g = ky[j - 1, i] / hy**2
                diagonal += g
                rows.append(row); cols.append(idx(i, j - 1)); values.append(-g)
            if j < ny - 1:
                g = ky[j, i] / hy**2
                diagonal += g
                rows.append(row); cols.append(idx(i, j + 1)); values.append(-g)

            rows.append(row); cols.append(row); values.append(diagonal)

    A = coo_matrix((values, (rows, cols)), shape=(n_cells, n_cells)).tocsr()
    pressure = spsolve(A, rhs).reshape(ny, nx)
    if not np.all(np.isfinite(pressure)):
        raise RuntimeError("Darcy linear solve returned non-finite pressure values")

    q_left = float(np.sum(k[:, 0] * (p_left - pressure[:, 0]) / (0.5 * hx)) * hy)
    return pressure, X, Y, q_left


def boundary_fluxes(
    p: np.ndarray,
    params: tuple[float, float, float],
    nx: int | None = None,
    ny: int | None = None,
    *,
    p_left: float = 1.0,
    p_right: float = 0.0,
) -> tuple[float, float]:
    """Return positive left inflow and positive right outflow."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 2:
        raise ValueError("p must be a two-dimensional pressure array")
    inferred_ny, inferred_nx = p.shape
    nx = inferred_nx if nx is None else nx
    ny = inferred_ny if ny is None else ny
    if (ny, nx) != p.shape:
        raise ValueError("nx and ny must match the shape of p")

    hx, hy = 1.0 / nx, 1.0 / ny
    x = (np.arange(nx, dtype=float) + 0.5) * hx
    y = (np.arange(ny, dtype=float) + 0.5) * hy
    X, Y = np.meshgrid(x, y)
    k = np.exp(make_logk_field(X, Y, params))

    q_left = float(np.sum(k[:, 0] * (p_left - p[:, 0]) / (0.5 * hx)) * hy)
    q_right = float(np.sum(k[:, -1] * (p[:, -1] - p_right) / (0.5 * hx)) * hy)
    return q_left, q_right


if __name__ == "__main__":
    p, X, Y, q = solve_darcy(nx=32, ny=24, params=(0.0, 0.0, 0.0))
    p_exact = 1.0 - X
    q_left, q_right = boundary_fluxes(p, (0.0, 0.0, 0.0))
    print("Constant-k verification")
    print(f"max pressure error : {np.max(np.abs(p - p_exact)):.3e}")
    print(f"left/right flux    : {q_left:.12f} / {q_right:.12f}")
