"""Sparse FDM for -Delta u=f on (0,1)^2 with a mixed right boundary.

Unknowns are (i,j), i=1,...,N and j=1,...,N-1.  All four corners and
the other three edges are Dirichlet nodes.  The right open edge is either
Neumann or Robin, using a first-order derivative row (``first``) or a
second-order centred ghost condition eliminated into a boundary PDE row
(``ghost``).
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

BoundaryKind = Literal["neumann", "robin"]
ClosureKind = Literal["first", "ghost"]
ScalarField = Callable[[np.ndarray, np.ndarray], np.ndarray]


def exact_solution(x, y):
    return np.exp(np.asarray(x) + np.asarray(y))


def forcing(x, y):
    return -2.0 * exact_solution(x, y)


def boundary_data(y, boundary: BoundaryKind):
    beta = 0.0 if boundary == "neumann" else 2.0
    return (1.0 + beta) * np.exp(1.0 + np.asarray(y))


def index(i: int, j: int, N: int) -> int:
    """Zero-based vector index for the unknown grid node (i,j)."""
    if not (1 <= i <= N and 1 <= j <= N - 1):
        raise IndexError(f"({i},{j}) is not an unknown for N={N}")
    return (j - 1) * N + (i - 1)


def is_unknown(i: int, j: int, N: int) -> bool:
    return 1 <= i <= N and 1 <= j <= N - 1


@dataclass(frozen=True)
class Assembly:
    A: sparse.csr_matrix
    b: np.ndarray
    N: int
    h: float
    boundary: BoundaryKind
    closure: ClosureKind
    beta: float
    assembly_seconds: float


@dataclass(frozen=True)
class Diagnostics:
    l2: float
    linf: float
    boundary_l2: float
    boundary_linf: float
    interior_l2: float
    interior_linf: float
    residual_componentwise: float
    residual_relative_inf: float
    asymmetry_frobenius: float


@dataclass(frozen=True)
class ExperimentResult:
    assembly: Assembly
    solution: np.ndarray
    full_solution: np.ndarray
    diagnostics: Diagnostics
    solve_seconds: float


def _validate_options(N: int, boundary: str, closure: str) -> None:
    if not isinstance(N, (int, np.integer)) or N < 2:
        raise ValueError("N must be an integer >= 2")
    if boundary not in {"neumann", "robin"}:
        raise ValueError("boundary must be 'neumann' or 'robin'")
    if closure not in {"first", "ghost"}:
        raise ValueError("closure must be 'first' or 'ghost'")


def assemble_system(
    N: int,
    boundary: BoundaryKind,
    closure: ClosureKind,
    *,
    u_exact: ScalarField = exact_solution,
    f_source: ScalarField = forcing,
    s_boundary: Callable[[np.ndarray, BoundaryKind], np.ndarray] = boundary_data,
) -> Assembly:
    """Assemble the reduced CSR system after exact Dirichlet elimination."""
    _validate_options(N, boundary, closure)
    start = perf_counter()
    h = 1.0 / N
    beta = 0.0 if boundary == "neumann" else 2.0
    ndof = N * (N - 1)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    b = np.empty(ndof, dtype=float)

    def add_or_eliminate(k: int, i: int, j: int, coefficient: float, rhs: float) -> float:
        if is_unknown(i, j, N):
            rows.append(k)
            cols.append(index(i, j, N))
            vals.append(coefficient)
        else:
            if not (0 <= i <= N and 0 <= j <= N):
                raise AssertionError("a retained stencil referenced an outside-grid node")
            rhs -= coefficient * float(u_exact(i * h, j * h))
        return rhs

    for j in range(1, N):
        y = j * h
        for i in range(1, N + 1):
            k = index(i, j, N)
            if i < N:
                rhs = float(f_source(i * h, y))
                stencil = ((i, j, 4.0 / h**2), (i - 1, j, -1.0 / h**2),
                           (i + 1, j, -1.0 / h**2), (i, j - 1, -1.0 / h**2),
                           (i, j + 1, -1.0 / h**2))
            elif closure == "first":
                rhs = float(s_boundary(y, boundary))
                stencil = ((i, j, 1.0 / h + beta), (i - 1, j, -1.0 / h))
            else:
                rhs = float(f_source(1.0, y) + 2.0 * s_boundary(y, boundary) / h)
                stencil = ((i, j, 4.0 / h**2 + 2.0 * beta / h),
                           (i - 1, j, -2.0 / h**2),
                           (i, j - 1, -1.0 / h**2),
                           (i, j + 1, -1.0 / h**2))
            for ni, nj, a in stencil:
                rhs = add_or_eliminate(k, ni, nj, a, rhs)
            b[k] = rhs

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    A.sum_duplicates()
    if A.shape != (ndof, ndof) or b.shape != (ndof,):
        raise AssertionError("assembled system has the wrong size")
    if not (np.isfinite(A.data).all() and np.isfinite(b).all()):
        raise FloatingPointError("assembled system contains NaN or infinity")
    return Assembly(A, b, N, h, boundary, closure, beta, perf_counter() - start)


def solve_system(system: Assembly) -> tuple[np.ndarray, float]:
    """Solve the generally nonsymmetric sparse system with SuperLU via spsolve."""
    start = perf_counter()
    solution = np.asarray(
        spsolve(system.A.tocsc(), system.b, use_umfpack=False), dtype=float
    )
    elapsed = perf_counter() - start
    if solution.shape != system.b.shape or not np.isfinite(solution).all():
        raise FloatingPointError("sparse direct solve returned an invalid solution")
    return solution, elapsed


def reconstruct(solution: np.ndarray, N: int, u_exact: ScalarField = exact_solution) -> np.ndarray:
    h = 1.0 / N
    x = np.linspace(0.0, 1.0, N + 1)
    y = np.linspace(0.0, 1.0, N + 1)
    U = np.empty((N + 1, N + 1), dtype=float)  # U[j,i]
    U[0, :] = u_exact(x, 0.0)
    U[N, :] = u_exact(x, 1.0)
    U[:, 0] = u_exact(0.0, y)
    U[:, N] = u_exact(1.0, y)  # corners stay exact; open edge overwritten next
    for j in range(1, N):
        for i in range(1, N + 1):
            U[j, i] = solution[index(i, j, N)]
    return U


def exact_vector(N: int, u_exact: ScalarField = exact_solution) -> np.ndarray:
    h = 1.0 / N
    return np.array([float(u_exact(i * h, j * h)) for j in range(1, N) for i in range(1, N + 1)])


def componentwise_backward_error(A: sparse.spmatrix, x: np.ndarray, b: np.ndarray) -> float:
    residual = b - A @ x
    denominator = abs(A) @ np.abs(x) + np.abs(b)
    ratios = np.zeros_like(residual)
    np.divide(np.abs(residual), denominator, out=ratios, where=denominator > 0.0)
    return float(np.max(ratios, initial=0.0))


def compute_diagnostics(system: Assembly, solution: np.ndarray, full: np.ndarray) -> Diagnostics:
    N, h = system.N, system.h
    grid = np.linspace(0.0, 1.0, N + 1)
    X, Y = np.meshgrid(grid, grid)
    error = full - exact_solution(X, Y)
    weights = np.ones(N + 1)
    weights[[0, -1]] = 0.5
    l2 = float(np.sqrt(h**2 * np.sum(error**2 * np.outer(weights, weights))))
    interior = error[1:N, 1:N]
    right = error[1:N, N]
    residual = system.b - system.A @ solution
    matrix_inf = float(np.max(np.asarray(abs(system.A).sum(axis=1)).ravel()))
    residual_den = matrix_inf * np.linalg.norm(solution, ord=np.inf) + np.linalg.norm(system.b, ord=np.inf)
    asym = system.A - system.A.T
    A_frob = float(sparse.linalg.norm(system.A, ord="fro"))
    return Diagnostics(
        l2=l2,
        linf=float(np.max(np.abs(error))),
        boundary_l2=float(np.sqrt(h * np.sum(right**2))),
        boundary_linf=float(np.max(np.abs(right), initial=0.0)),
        interior_l2=float(np.sqrt(h**2 * np.sum(interior**2))),
        interior_linf=float(np.max(np.abs(interior), initial=0.0)),
        residual_componentwise=componentwise_backward_error(system.A, solution, system.b),
        residual_relative_inf=float(np.linalg.norm(residual, ord=np.inf) / residual_den),
        asymmetry_frobenius=float(sparse.linalg.norm(asym, ord="fro") / A_frob),
    )


def run_case(N: int, boundary: BoundaryKind, closure: ClosureKind) -> ExperimentResult:
    system = assemble_system(N, boundary, closure)
    solution, solve_seconds = solve_system(system)
    full = reconstruct(solution, N)
    diagnostics = compute_diagnostics(system, solution, full)
    return ExperimentResult(system, solution, full, diagnostics, solve_seconds)


def consistency_residual(system: Assembly) -> np.ndarray:
    """Raw manufactured-solution row defect A*u_exact-b."""
    return np.asarray(system.A @ exact_vector(system.N) - system.b)


def expected_relative_defect(h: float, closure: ClosureKind) -> float:
    """Analytic exponential defect divided by u* on a representative row."""
    if closure == "first":
        return (1.0 - np.exp(-h)) / h - 1.0
    return (4.0 - 3.0 * np.exp(-h) - np.exp(h)) / h**2 - 2.0 / h + 2.0


def ghost_recovery(U_boundary: float, U_west: float, h: float, s: float, beta: float) -> float:
    return U_west + 2.0 * h * (s - beta * U_boundary)
