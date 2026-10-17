"""Pure-Neumann depth experiment; distinct from the mixed-boundary solver."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from models import exact_solution


def assemble_pure_neumann(N: int, closure: str):
    """All vertices unknown; FO derivative or centred ghost boundary rows."""
    if N < 2 or closure not in ("first", "ghost"):
        raise ValueError("N >= 2 and closure in {'first','ghost'} are required")
    h = 1.0 / N
    size = (N + 1) ** 2
    ii = np.broadcast_to(np.arange(N + 1), (N + 1, N + 1)).ravel()
    jj = np.broadcast_to(np.arange(N + 1)[:, None], (N + 1, N + 1)).ravel()
    kk = np.arange(size)
    left, right, bottom, top = ii == 0, ii == N, jj == 0, jj == N
    edge_count = left.astype(int) + right.astype(int) + bottom.astype(int) + top.astype(int)
    boundary = edge_count > 0
    x, y = ii * h, jj * h
    f = -2.0 * exact_solution(x, y)
    flux = (
        np.where(left, -np.exp(y), 0.0)
        + np.where(right, np.exp(1.0 + y), 0.0)
        + np.where(bottom, -np.exp(x), 0.0)
        + np.where(top, np.exp(x + 1.0), 0.0)
    )
    if closure == "first":
        b = np.where(boundary, flux, f)
        diagonal = np.where(boundary, edge_count / h, 4.0 / h**2)
    else:
        b = f + 2.0 * flux / h
        diagonal = np.full(size, 4.0 / h**2)
    rows, cols, data = [kk], [kk], [diagonal]

    def append(mask, offset, coefficient):
        selected = mask & (coefficient != 0.0)
        rows.append(kk[selected])
        cols.append(kk[selected] + offset)
        data.append(coefficient[selected])

    for mask, offset, normal in (
        (ii > 0, -1, right), (ii < N, 1, left),
        (jj > 0, -(N + 1), top), (jj < N, N + 1, bottom),
    ):
        if closure == "first":
            coefficient = np.where(boundary, np.where(normal, -1.0 / h, 0.0), -1.0 / h**2)
        else:
            coefficient = np.where(normal, -2.0 / h**2, -1.0 / h**2)
        append(mask, offset, coefficient)

    A = sparse.coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(size, size)).tocsr()
    return A, b


def continuum_compatibility_quadrature(N: int) -> float:
    """Tensor trapezoid volume integral plus four edge-flux integrals."""
    h = 1.0 / N
    grid = np.linspace(0.0, 1.0, N + 1)
    xx, yy = np.meshgrid(grid, grid)
    weights = np.ones(N + 1); weights[[0, -1]] = 0.5
    volume = h**2 * np.sum(np.outer(weights, weights) * (-2.0 * exact_solution(xx, yy)))
    flux = h * np.sum(weights * (
        -np.exp(grid) + np.exp(1.0 + grid)
        - np.exp(grid) + np.exp(grid + 1.0)
    ))
    return float(volume + flux)


def gauge_solve(A, b):
    """Compute left-null compatibility and a mean-zero bordered solve."""
    n = A.shape[0]
    ones = np.ones(n)
    column = sparse.csr_matrix(ones[:, None])
    row = sparse.csr_matrix(ones[None, :])
    zero = sparse.csr_matrix((1, 1))
    left_border = sparse.bmat([[A.T, column], [row, zero]], format="csc")
    left_null = spsolve(left_border, np.r_[np.zeros(n), 1.0])[:n]
    defect = float(left_null @ b)
    right_border = sparse.bmat([[A, column], [row, zero]], format="csc")
    solved = spsolve(right_border, np.r_[b, 0.0])
    u, lagrange = solved[:n], float(solved[-1])
    return u, defect, lagrange, float(np.linalg.norm(A.T @ left_null, ord=np.inf))


def run_pure_neumann(output_dir: Path, grids=(8, 16, 32, 64)) -> list[dict]:
    records = []
    for closure in ("first", "ghost"):
        for N in grids:
            A, b = assemble_pure_neumann(N, closure)
            null_defect = float(np.linalg.norm(A @ np.ones(A.shape[0]), ord=np.inf))
            u, compatibility, lagrange, left_defect = gauge_solve(A, b)
            xx, yy = np.meshgrid(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1))
            exact = exact_solution(xx, yy).ravel()
            exact -= np.mean(exact)
            records.append({
                "closure": closure, "N": N, "ndof": A.shape[0],
                "null_vector_defect_inf": null_defect,
                "left_null_defect_inf": left_defect,
                "discrete_compatibility_defect": compatibility,
                "compatibility_correction": lagrange,
                "continuum_quadrature_defect": continuum_compatibility_quadrature(N),
                "gauge_mean": float(np.mean(u)),
                "gauge_solution_linf_error": float(np.max(np.abs(u - exact))),
                "bordered_residual_inf": float(np.linalg.norm(A @ u + lagrange - b, ord=np.inf)),
            })
    path = output_dir / "pure_neumann_depth.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    return records
