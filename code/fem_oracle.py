"""Independent P1 finite-element oracle for the report's mixed weak form."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from skfem import Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm, MeshTri, asm, condense, solve
from skfem.helpers import dot, grad

from models import exact_solution, run_case


@BilinearForm
def diffusion(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def boundary_mass(u, v, w):
    return u * v


@LinearForm
def source(v, w):
    return -2.0 * np.exp(w.x[0] + w.x[1]) * v


@LinearForm
def boundary_flux(v, w):
    return (1.0 + w.beta) * np.exp(1.0 + w.x[1]) * v


def solve_mixed_fem(N: int, boundary: str) -> np.ndarray:
    """Return P1 nodal values as U[j, i] on the shared Cartesian vertices."""
    points = np.linspace(0.0, 1.0, N + 1)
    mesh = MeshTri.init_tensor(points, points)
    basis = Basis(mesh, ElementTriP1())
    facets = mesh.facets_satisfying(lambda x: np.isclose(x[0], 1.0))
    right = FacetBasis(mesh, ElementTriP1(), facets=facets)
    beta = 0.0 if boundary == "neumann" else 2.0
    A = asm(diffusion, basis) + beta * asm(boundary_mass, right)
    b = asm(source, basis) + asm(boundary_flux, right, beta=beta)
    xy = mesh.p
    dirichlet = np.flatnonzero(
        np.isclose(xy[0], 0.0) | np.isclose(xy[1], 0.0) | np.isclose(xy[1], 1.0)
    )
    exact = exact_solution(xy[0], xy[1])
    values = solve(*condense(A, b, x=exact.copy(), D=dirichlet))
    ii = np.rint(xy[0] * N).astype(int)
    jj = np.rint(xy[1] * N).astype(int)
    if len(np.unique(jj * (N + 1) + ii)) != (N + 1) ** 2:
        raise AssertionError("FEM vertices do not form the expected common grid")
    full = np.empty((N + 1, N + 1), dtype=float)
    full[jj, ii] = values
    return full


def run_fem_comparison(output_dir: Path, grids=(16, 32, 64, 128)) -> list[dict]:
    rows = []
    for boundary in ("neumann", "robin"):
        for N in grids:
            fem = solve_mixed_fem(N, boundary)
            points = np.linspace(0.0, 1.0, N + 1)
            xx, yy = np.meshgrid(points, points)
            fem_error = float(np.max(np.abs(fem - exact_solution(xx, yy))))
            for closure in ("first", "ghost"):
                fdm = run_case(N, boundary, closure).full_solution
                difference = float(np.max(np.abs(fdm - fem)))
                rows.append({
                    "boundary": boundary, "closure": closure, "N": N,
                    "fem_linf_error": fem_error,
                    "fdm_linf_error": float(np.max(np.abs(fdm - exact_solution(xx, yy)))),
                    "fdm_fem_linf_difference": difference,
                })
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            group = [r for r in rows if r["boundary"] == boundary and r["closure"] == closure]
            for previous, current in zip(group, group[1:]):
                current["fdm_fem_pairwise_order"] = float(np.log2(
                    previous["fdm_fem_linf_difference"] / current["fdm_fem_linf_difference"]
                ))
    path = output_dir / "fem_oracle_comparison.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[-1]))
        writer.writeheader(); writer.writerows(rows)
    return rows
