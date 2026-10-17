"""Refinement study, strict sanity checks, tables, and figures."""

from __future__ import annotations

import csv
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                     "axes.titlesize": 14, "xtick.labelsize": 11,
                     "ytick.labelsize": 11, "legend.fontsize": 11})
import numpy as np
import scipy
from scipy.sparse.linalg import LinearOperator, cg, eigsh, gmres, spilu

from models import (
    assemble_system,
    boundary_data,
    consistency_residual,
    exact_solution,
    exact_vector,
    ghost_recovery,
    index,
    run_case,
    solve_system,
    componentwise_backward_error,
)


def observed_orders(values: list[float]) -> list[float]:
    return [float("nan")] + [float(np.log(values[k - 1] / values[k]) / np.log(2.0)) for k in range(1, len(values))]


def fit_loglog_order(h_values, error_values) -> tuple[float, float]:
    """Least-squares slope and R^2 for log(error) versus log(h)."""
    h = np.asarray(h_values, dtype=float)
    error = np.asarray(error_values, dtype=float)
    if h.ndim != 1 or error.ndim != 1 or h.size != error.size or h.size < 3:
        raise ValueError("fitted order needs at least three paired one-dimensional values")
    if np.any(h <= 0.0) or np.any(error <= 0.0):
        raise ValueError("mesh sizes and errors must be positive")
    log_h = np.log(h)
    log_error = np.log(error)
    slope, intercept = np.polyfit(log_h, log_error, 1)
    fitted = slope * log_h + intercept
    ss_res = float(np.sum((log_error - fitted) ** 2))
    ss_tot = float(np.sum((log_error - np.mean(log_error)) ** 2))
    r_squared = 1.0 if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return float(slope), r_squared


def _assert_close(actual, expected, *, rtol=2e-12, atol=2e-12, message=""):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise AssertionError(message or f"{actual!r} != {expected!r}")


def run_sanity_checks() -> dict[str, str]:
    """Checks independent formulas, ownership, signs, rows, and convergence."""
    # N=4 hand row from the report/scaffold.
    N, h, j = 4, 0.25, 1
    for boundary, beta in (("neumann", 0.0), ("robin", 2.0)):
        sys = assemble_system(N, boundary, "ghost")
        row = sys.A.getrow(index(N, j, N)).toarray().ravel()
        _assert_close(row[index(N, j, N)], 64.0 + 8.0 * beta)
        _assert_close(row[index(N - 1, j, N)], -32.0)
        _assert_close(row[index(N, j + 1, N)], -16.0)
        expected_rhs = ((6.0 if boundary == "neumann" else 22.0) * np.exp(1.25) + 16.0 * np.e)
        _assert_close(sys.b[index(N, j, N)], expected_rhs, message="ghost boundary RHS sign/factor failure")

    # Exact exponential row-defect identities, including the boundary-row scaling.
    for closure in ("first", "ghost"):
        sys = assemble_system(16, "robin", closure)
        defects = consistency_residual(sys)
        boundary_defects = defects[[index(16, jj, 16) for jj in range(1, 16)]]
        relative = boundary_defects / exact_vector(16)[[index(16, jj, 16) for jj in range(1, 16)]]
        if closure == "first":
            expected = (1.0 - np.exp(-sys.h)) / sys.h - 1.0
        else:
            expected = (4.0 - 3.0 * np.exp(-sys.h) - np.exp(sys.h)) / sys.h**2 - 2.0 / sys.h + 2.0
        _assert_close(relative, expected, rtol=3e-11, atol=3e-11, message="manufactured boundary defect mismatch")

    # Outward normal is +x: q=ux=e^(1+y)>0, and centred ghost derivative is exact after recovery.
    y = 0.375
    for boundary, beta in (("neumann", 0.0), ("robin", 2.0)):
        ub, uw = np.exp(1.0 + y), np.exp(1.0 - h + y)
        s = float(boundary_data(y, boundary))
        ug = ghost_recovery(ub, uw, h, s, beta)
        _assert_close((ug - uw) / (2.0 * h) + beta * ub, s)
        if not s > 0.0:
            raise AssertionError("outward-normal boundary data must be positive")

    # Corners are not unknowns and retain exact Dirichlet values after reconstruction.
    sample = run_case(8, "robin", "ghost")
    _assert_close(sample.full_solution[[0, 0, -1, -1], [0, -1, 0, -1]], [1.0, np.e, np.e, np.e**2])
    if sample.assembly.A.shape != (56, 56) or not np.isfinite(sample.full_solution).all():
        raise AssertionError("index/size/finite-value check failed")
    if sample.diagnostics.residual_componentwise > 5e-14:
        raise AssertionError("componentwise algebraic residual is too large")

    # Independent data: affine is exact for both closures; quadratic has ghost defect zero
    # and first-order boundary defect -h.
    affine = lambda x, y: np.asarray(x) * 0 + np.asarray(y) * 0 + 1.0 + np.asarray(x) + np.asarray(y)
    zero_f = lambda x, y: np.asarray(x) * 0 + np.asarray(y) * 0
    affine_s = lambda y, boundary: 1.0 + (0.0 if boundary == "neumann" else 2.0) * affine(1.0, y)
    quadratic = lambda x, y: 1.0 + np.asarray(x) ** 2 + np.asarray(y) ** 2
    quad_f = lambda x, y: np.asarray(x) * 0 + np.asarray(y) * 0 - 4.0
    quad_s = lambda y, boundary: 2.0 + (0.0 if boundary == "neumann" else 2.0) * quadratic(1.0, y)
    for closure in ("first", "ghost"):
        a = assemble_system(8, "robin", closure, u_exact=affine, f_source=zero_f, s_boundary=affine_s)
        _assert_close(a.A @ np.array([affine(i / 8, j / 8) for j in range(1, 8) for i in range(1, 9)]) - a.b, 0.0)
        q = assemble_system(8, "robin", closure, u_exact=quadratic, f_source=quad_f, s_boundary=quad_s)
        qdef = q.A @ np.array([quadratic(i / 8, j / 8) for j in range(1, 8) for i in range(1, 9)]) - q.b
        qb = qdef[[index(8, jj, 8) for jj in range(1, 8)]]
        _assert_close(qb, -q.h if closure == "first" else 0.0)

    # Dense solve is independent of SuperLU on a small mesh.
    sys = assemble_system(6, "neumann", "ghost")
    sparse_x, _ = solve_system(sys)
    dense_x = np.linalg.solve(sys.A.toarray(), sys.b)
    _assert_close(sparse_x, dense_x, rtol=2e-12, atol=2e-12)

    # Asymptotic trend over four meshes, not a single mesh pair.
    grids = [16, 32, 64, 128]
    for boundary in ("neumann", "robin"):
        first = [run_case(n, boundary, "first").diagnostics.boundary_linf for n in grids]
        ghost_l2 = [run_case(n, boundary, "ghost").diagnostics.l2 for n in grids]
        ghost_inf = [run_case(n, boundary, "ghost").diagnostics.linf for n in grids]
        p_first = observed_orders(first)[-2:]
        p_g_l2 = observed_orders(ghost_l2)[-2:]
        p_g_inf = observed_orders(ghost_inf)[-2:]
        if not all(0.85 < p < 1.15 for p in p_first):
            raise AssertionError(f"{boundary} first-order boundary trend failed: {p_first}")
        if not all(1.80 < p < 2.20 for p in p_g_l2 + p_g_inf):
            raise AssertionError(f"{boundary} ghost global trend failed: {p_g_l2}, {p_g_inf}")
    return {
        "hand_derived_rows": "passed",
        "manufactured_consistency": "passed",
        "normal_and_ghost_elimination": "passed",
        "corner_ownership_and_dimensions": "passed",
        "affine_and_quadratic_independent_MMS": "passed",
        "independent_dense_solve": "passed",
        "four_mesh_convergence_trends": "passed",
    }


def _timed_medians(N: int, boundary: str, closure: str, repeats: int = 5):
    run_case(N, boundary, closure)  # warm-up excluded
    assembly_times, solve_times, total_times = [], [], []
    representative = None
    for _ in range(repeats):
        start = perf_counter()
        system = assemble_system(N, boundary, closure)
        assembly_times.append(perf_counter() - start)
        _, solve_time = solve_system(system)
        total_times.append(perf_counter() - start)
        solve_times.append(solve_time)
        representative = system
    return {
        "assembly_seconds_median": median(assembly_times),
        "solve_seconds_median": median(solve_times),
        "total_seconds_median": median(total_times),
        "total_seconds_min": min(total_times),
        "total_seconds_max": max(total_times),
        "timing_repeats": repeats,
    }, representative


def run_refinement(output_dir: Path, grids=(16, 32, 64, 128)) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            cases = [run_case(N, boundary, closure) for N in grids]
            metrics = ["l2", "linf", "boundary_linf", "interior_linf"]
            orders = {m: observed_orders([getattr(c.diagnostics, m) for c in cases]) for m in metrics}
            for level, case in enumerate(cases):
                timing, _ = _timed_medians(case.assembly.N, boundary, closure)
                d = asdict(case.diagnostics)
                rows.append({
                    "boundary": boundary, "closure": closure, "N": case.assembly.N,
                    "h": case.assembly.h, "ndof": case.assembly.A.shape[0],
                    "nnz": case.assembly.A.nnz, **d,
                    "order_l2": orders["l2"][level], "order_linf": orders["linf"][level],
                    "order_boundary_linf": orders["boundary_linf"][level],
                    "order_interior_linf": orders["interior_linf"][level],
                    **timing,
                })
    with (output_dir / "refinement_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return rows


def choose_accuracy_case(rows: list[dict], target: float):
    """Coarsest tested grid satisfying a fixed nodal-max accuracy target."""
    if not np.isfinite(target) or target <= 0.0:
        raise ValueError("accuracy target must be finite and positive")
    feasible = [r for r in rows if float(r["linf"]) <= target]
    return min(feasible, key=lambda r: int(r["N"])) if feasible else None


def run_matched_accuracy(output_dir: Path, refinement_rows: list[dict],
                         targets=(1e-2, 5e-3)) -> list[dict]:
    """Fixed-target study; never extrapolate beyond the tested candidates."""
    candidates = [dict(r) for r in refinement_rows]
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            for N in (4, 8):
                case = run_case(N, boundary, closure)
                timing, _ = _timed_medians(N, boundary, closure)
                candidates.append({
                    "boundary": boundary, "closure": closure, "N": N,
                    "h": 1.0 / N, "ndof": case.assembly.A.shape[0],
                    "nnz": case.assembly.A.nnz,
                    **asdict(case.diagnostics), **timing,
                })
    candidate_fields = ["boundary", "closure", "N", "h", "ndof", "nnz",
                        "linf", "assembly_seconds_median", "solve_seconds_median",
                        "total_seconds_median", "total_seconds_min", "total_seconds_max"]
    with (output_dir / "cost_accuracy_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=candidate_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(candidates, key=lambda r: (r["boundary"], r["closure"], r["N"])))
    selected = []
    for boundary in ("neumann", "robin"):
        for target in targets:
            for closure in ("first", "ghost"):
                group = [r for r in candidates if r["boundary"] == boundary and r["closure"] == closure]
                row = choose_accuracy_case(group, target)
                record = {
                    "boundary": boundary, "closure": closure, "target_linf": target,
                    "status": "attained" if row is not None else "not attained",
                    "selected_N": row["N"] if row is not None else "",
                    "achieved_linf": row["linf"] if row is not None else "",
                    "ndof": row["ndof"] if row is not None else "",
                    "total_seconds_median": row["total_seconds_median"] if row is not None else "",
                    "total_seconds_min": row["total_seconds_min"] if row is not None else "",
                    "total_seconds_max": row["total_seconds_max"] if row is not None else "",
                    "assembly_seconds_median": row["assembly_seconds_median"] if row is not None else "",
                    "solve_seconds_median": row["solve_seconds_median"] if row is not None else "",
                }
                selected.append(record)
    with (output_dir / "matched_accuracy.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)
    return selected


def write_fitted_orders(rows: list[dict], output_dir: Path, fit_levels: int = 3) -> list[dict]:
    """Fit asymptotic orders on the finest consecutive mesh levels."""
    fitted: list[dict] = []
    metrics = ("l2", "linf", "boundary_linf", "interior_linf")
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            subset = sorted(
                (r for r in rows if (r["boundary"], r["closure"]) == (boundary, closure)),
                key=lambda r: r["N"],
            )
            selected = subset[-fit_levels:]
            record = {
                "boundary": boundary,
                "closure": closure,
                "fit_N": ",".join(str(r["N"]) for r in selected),
                "fit_h_max": max(r["h"] for r in selected),
                "fit_h_min": min(r["h"] for r in selected),
            }
            for metric in metrics:
                slope, r_squared = fit_loglog_order(
                    [r["h"] for r in selected], [r[metric] for r in selected]
                )
                record[f"fitted_order_{metric}"] = slope
                record[f"r_squared_{metric}"] = r_squared
            fitted.append(record)
    path = output_dir / "fitted_orders.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fitted[0]))
        writer.writeheader()
        writer.writerows(fitted)
    return fitted


def symmetrising_weights(system) -> np.ndarray:
    """Return the report's exact positive row weights W_h^c."""
    weights = np.ones(system.A.shape[0], dtype=float)
    boundary_weight = 1.0 / system.h if system.closure == "first" else 0.5
    for j in range(1, system.N):
        weights[index(system.N, j, system.N)] = boundary_weight
    return weights


def weighted_system(system):
    """Construct K_h^c = W_h^c A_h^c and W_h^c b_h^c."""
    weights = symmetrising_weights(system)
    K = (scipy.sparse.diags(weights) @ system.A).tocsr()
    return K, weights * system.b, weights


def _zero_extended_vector(vector: np.ndarray, N: int) -> np.ndarray:
    full = np.zeros((N + 1, N + 1), dtype=float)
    for j in range(1, N):
        for i in range(1, N + 1):
            full[j, i] = vector[index(i, j, N)]
    return full


def energy_identity_terms(system, vector: np.ndarray) -> tuple[float, float, float]:
    """Return quadratic form, edge-sum form, and their relative discrepancy."""
    K, _, _ = weighted_system(system)
    full = _zero_extended_vector(vector, system.N)
    horizontal = float(np.sum(np.diff(full[1:system.N, :], axis=1) ** 2))
    vertical_interior = float(np.sum(np.diff(full[:, 1:system.N], axis=0) ** 2))
    vertical_boundary = float(np.sum(np.diff(full[:, system.N], axis=0) ** 2))
    theta = 0.0 if system.closure == "first" else 0.5
    robin = float(
        system.beta * system.h * np.sum(full[1:system.N, system.N] ** 2)
    )
    quadratic = float(system.h**2 * vector @ (K @ vector))
    edge_sum = horizontal + vertical_interior + theta * vertical_boundary + robin
    scale = max(abs(quadratic), abs(edge_sum), np.finfo(float).tiny)
    return quadratic, edge_sum, abs(quadratic - edge_sum) / scale


def run_weighted_matrix_diagnostics(output_dir: Path, N: int = 32) -> list[dict]:
    """Numerically verify the weighted symmetry, SPD, and energy identity."""
    records: list[dict] = []
    rng = np.random.default_rng(321)
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            case = run_case(N, boundary, closure)
            system = case.assembly
            K, _, weights = weighted_system(system)
            K_frob = float(scipy.sparse.linalg.norm(K, ord="fro"))
            weighted_asymmetry = float(
                scipy.sparse.linalg.norm(K - K.T, ord="fro") / K_frob
            )
            smallest_eigenvalue = float(
                eigsh(K, k=1, which="SA", return_eigenvectors=False, tol=1e-10,
                      v0=rng.standard_normal(K.shape[0]))[0]
            )
            trial = rng.standard_normal(system.A.shape[0])
            quadratic, edge_sum, identity_defect = energy_identity_terms(system, trial)
            records.append({
                "boundary": boundary,
                "closure": closure,
                "N": N,
                "ndof": system.A.shape[0],
                "raw_asymmetry_frobenius": case.diagnostics.asymmetry_frobenius,
                "weighted_asymmetry_frobenius": weighted_asymmetry,
                "smallest_eigenvalue": smallest_eigenvalue,
                "minimum_diagonal": float(np.min(K.diagonal())),
                "minimum_weight": float(np.min(weights)),
                "maximum_weight": float(np.max(weights)),
                "energy_quadratic_form": quadratic,
                "energy_edge_sum": edge_sum,
                "energy_identity_relative_defect": identity_defect,
            })
    with (output_dir / "weighted_matrix_diagnostics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if max(r["weighted_asymmetry_frobenius"] for r in records) > 1e-13:
        raise AssertionError("the report's weighted matrix was not numerically symmetric")
    if min(r["smallest_eigenvalue"] for r in records) <= 0.0:
        raise AssertionError("the report's weighted matrix was not numerically positive definite")
    if max(r["energy_identity_relative_defect"] for r in records) > 1e-12:
        raise AssertionError("the report's weighted energy identity failed")
    return records


def run_energy_convergence(
    output_dir: Path, grids=(16, 32, 64, 128)
) -> list[dict]:
    """Measure the report-defined weighted energy norm on four meshes."""
    records: list[dict] = []
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            group: list[dict] = []
            for N in grids:
                case = run_case(N, boundary, closure)
                error = case.solution - exact_vector(N)
                quadratic, edge_sum, identity_defect = energy_identity_terms(
                    case.assembly, error
                )
                if quadratic < -1e-12:
                    raise AssertionError("negative weighted energy quadratic form")
                group.append({
                    "boundary": boundary,
                    "closure": closure,
                    "N": N,
                    "h": case.assembly.h,
                    "ndof": case.assembly.A.shape[0],
                    "energy_error": float(np.sqrt(max(quadratic, 0.0))),
                    "energy_edge_sum": edge_sum,
                    "energy_identity_relative_defect": identity_defect,
                    "l2_error": case.diagnostics.l2,
                    "linf_error": case.diagnostics.linf,
                })
            pairwise = observed_orders([r["energy_error"] for r in group])
            fitted, r_squared = fit_loglog_order(
                [r["h"] for r in group[-3:]],
                [r["energy_error"] for r in group[-3:]],
            )
            for row, order in zip(group, pairwise):
                row["pairwise_energy_order"] = order
                row["fitted_energy_order_finest_three"] = fitted
                row["r_squared_energy_finest_three"] = r_squared
            records.extend(group)
    with (output_dir / "energy_convergence.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def run_cg_comparison(output_dir: Path, N: int = 64, rtol: float = 1e-10) -> list[dict]:
    """Solve the report's SPD weighted system with Jacobi-preconditioned CG."""
    records: list[dict] = []
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            case = run_case(N, boundary, closure)
            system = case.assembly
            K, weighted_rhs, _ = weighted_system(system)
            diagonal = K.diagonal()
            if np.any(diagonal <= 0.0):
                raise AssertionError("weighted matrix has a nonpositive diagonal")
            preconditioner = LinearOperator(K.shape, matvec=lambda x, d=diagonal: x / d)
            iterations = 0

            def count_iteration(_):
                nonlocal iterations
                iterations += 1

            start = perf_counter()
            cg_solution, info = cg(
                K,
                weighted_rhs,
                x0=np.zeros_like(weighted_rhs),
                rtol=rtol,
                atol=0.0,
                maxiter=10000,
                M=preconditioner,
                callback=count_iteration,
            )
            elapsed = perf_counter() - start
            if info != 0 or not np.isfinite(cg_solution).all():
                raise RuntimeError(f"CG failed for {boundary}/{closure}: info={info}")
            difference = float(np.linalg.norm(cg_solution - case.solution, ord=np.inf))
            discretisation = case.diagnostics.linf
            weighted_residual = weighted_rhs - K @ cg_solution
            weighted_residual_relative = float(
                np.linalg.norm(weighted_residual, ord=np.inf)
                / (
                    float(np.max(np.asarray(abs(K).sum(axis=1)).ravel()))
                    * np.linalg.norm(cg_solution, ord=np.inf)
                    + np.linalg.norm(weighted_rhs, ord=np.inf)
                )
            )
            records.append({
                "boundary": boundary,
                "closure": closure,
                "N": N,
                "ndof": K.shape[0],
                "solver": "CG on K=W*A with Jacobi preconditioner",
                "rtol": rtol,
                "atol": 0.0,
                "exit_status": int(info),
                "iterations": iterations,
                "solve_seconds": elapsed,
                "weighted_relative_inf_residual": weighted_residual_relative,
                "original_componentwise_backward_error": componentwise_backward_error(
                    system.A, cg_solution, system.b
                ),
                "cg_direct_difference_linf": difference,
                "discretisation_error_linf": discretisation,
                "algebraic_to_discretisation_ratio": difference / discretisation,
            })
    with (output_dir / "cg_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if max(r["algebraic_to_discretisation_ratio"] for r in records) >= 0.01:
        raise AssertionError("CG algebraic error was not below one percent of discretisation error")
    return records


def run_algebraic_error_separation(
    output_dir: Path,
    N: int = 64,
    tolerances=(1e-8, 1e-10, 1e-12),
) -> list[dict]:
    """Cross-check SuperLU with preconditioned GMRES on a representative mesh."""
    records: list[dict] = []
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            case = run_case(N, boundary, closure)
            system = case.assembly
            direct = case.solution
            exact = exact_vector(N)
            discretisation_linf = float(np.linalg.norm(direct - exact, ord=np.inf))
            right_indices = np.array([index(N, j, N) for j in range(1, N)])
            direct_qoi = float(np.mean(direct[right_indices]))

            row_norms = np.asarray(abs(system.A).sum(axis=1)).ravel()
            if np.any(row_norms <= 0.0):
                raise AssertionError("assembled matrix contains a zero row")
            row_scale = 1.0 / row_norms
            scaled_A = scipy.sparse.diags(row_scale) @ system.A
            scaled_b = row_scale * system.b
            ilu = spilu(scaled_A.tocsc(), drop_tol=1e-4, fill_factor=20.0)
            preconditioner = LinearOperator(system.A.shape, matvec=ilu.solve)
            for rtol in tolerances:
                iteration_residuals: list[float] = []
                start = perf_counter()
                iterative, info = gmres(
                    scaled_A,
                    scaled_b,
                    x0=np.zeros_like(system.b),
                    rtol=rtol,
                    atol=0.0,
                    restart=50,
                    maxiter=1000,
                    M=preconditioner,
                    callback=iteration_residuals.append,
                    callback_type="pr_norm",
                )
                elapsed = perf_counter() - start
                if info != 0 or not np.isfinite(iterative).all():
                    raise RuntimeError(
                        f"GMRES failed for {boundary}/{closure}, rtol={rtol}: info={info}"
                    )
                algebraic_difference = float(np.linalg.norm(iterative - direct, ord=np.inf))
                qoi = float(np.mean(iterative[right_indices]))
                records.append({
                    "boundary": boundary,
                    "closure": closure,
                    "N": N,
                    "ndof": system.A.shape[0],
                    "reference_solver": "spsolve/SuperLU",
                    "iterative_solver": "GMRES+ILU",
                    "row_scaling": "inverse absolute row sum",
                    "rtol": rtol,
                    "atol": 0.0,
                    "restart": 50,
                    "maxiter": 1000,
                    "exit_status": int(info),
                    "iterations": len(iteration_residuals),
                    "solve_seconds": elapsed,
                    "componentwise_backward_error": componentwise_backward_error(
                        system.A, iterative, system.b
                    ),
                    "algebraic_difference_linf": algebraic_difference,
                    "discretisation_error_linf": discretisation_linf,
                    "algebraic_to_discretisation_ratio": (
                        algebraic_difference / discretisation_linf
                    ),
                    "right_boundary_mean": qoi,
                    "right_boundary_mean_difference": abs(qoi - direct_qoi),
                })
    path = output_dir / "algebraic_error_separation.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    if max(r["algebraic_to_discretisation_ratio"] for r in records) >= 0.01:
        raise AssertionError("algebraic error was not below one percent of discretisation error")
    return records


def write_report_summary(rows: list[dict], fitted: list[dict], output_dir: Path) -> None:
    """Write compact CSV and LaTeX tables that can be included in the report."""
    fitted_by_key = {(r["boundary"], r["closure"]): r for r in fitted}
    finest = []
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            candidates = [
                r for r in rows if (r["boundary"], r["closure"]) == (boundary, closure)
            ]
            row = max(candidates, key=lambda r: r["N"])
            fit = fitted_by_key[(boundary, closure)]
            finest.append({
                "boundary": boundary,
                "closure": closure,
                "N": row["N"],
                "ndof": row["ndof"],
                "nnz": row["nnz"],
                "l2": row["l2"],
                "linf": row["linf"],
                "boundary_linf": row["boundary_linf"],
                "interior_linf": row["interior_linf"],
                "fitted_order_l2": fit["fitted_order_l2"],
                "fitted_order_linf": fit["fitted_order_linf"],
                "residual_componentwise": row["residual_componentwise"],
                "asymmetry_frobenius": row["asymmetry_frobenius"],
                "total_seconds_median": row["total_seconds_median"],
                "solver": "spsolve/SuperLU",
                "solver_tolerance": "N/A (direct solver)",
            })
    with (output_dir / "report_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(finest[0]))
        writer.writeheader()
        writer.writerows(finest)

    latex_lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\hline",
        r"Boundary & Closure & $N$ & $E_{2,h}$ & $p_2$ & $E_{\infty,h}$ & $p_\infty$ & $\eta$ \\",
        r"\hline",
    ]
    for row in finest:
        latex_lines.append(
            f"{row['boundary'].title()} & {row['closure'].title()} & {row['N']} & "
            f"{row['l2']:.3e} & {row['fitted_order_l2']:.3f} & "
            f"{row['linf']:.3e} & {row['fitted_order_linf']:.3f} & "
            f"{row['residual_componentwise']:.2e} \\\\" 
        )
    latex_lines.extend([r"\hline", r"\end{tabular}"])
    (output_dir / "report_summary.tex").write_text(
        "\n".join(latex_lines) + "\n", encoding="utf-8"
    )


def write_section5_summary(
    weighted_rows: list[dict], energy_rows: list[dict], cg_rows: list[dict], output_dir: Path
) -> None:
    """Write report-ready LaTeX tables for the weighted-energy section."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Boundary & Closure & $\|K-K^T\|_F/\|K\|_F$ & $\lambda_{\min}(K)$ & Energy identity defect \\",
        r"\hline",
    ]
    for row in weighted_rows:
        lines.append(
            f"{row['boundary'].title()} & {row['closure'].title()} & "
            f"{row['weighted_asymmetry_frobenius']:.2e} & "
            f"{row['smallest_eigenvalue']:.4f} & "
            f"{row['energy_identity_relative_defect']:.2e} \\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Numerical checks of weighted symmetry, positive definiteness and the energy identity on $N=32$.}",
        r"\label{tab:weighted-diagnostics}",
        r"\end{table}",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Boundary & Closure & $E_{E,h}$ at $N=128$ & Fitted order & $R^2$ \\",
        r"\hline",
    ])
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            subset = [
                r for r in energy_rows
                if (r["boundary"], r["closure"]) == (boundary, closure)
            ]
            finest = max(subset, key=lambda r: r["N"])
            lines.append(
                f"{boundary.title()} & {closure.title()} & "
                f"{finest['energy_error']:.3e} & "
                f"{finest['fitted_energy_order_finest_three']:.3f} & "
                f"{finest['r_squared_energy_finest_three']:.6f} \\\\"
            )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Weighted-energy errors and fitted orders on the finest three levels, $N=32,64,128$.}",
        r"\label{tab:energy-convergence}",
        r"\end{table}",
        "",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\hline",
        r"Boundary & Closure & CG iterations & Weighted residual & Algebraic/discretisation \\",
        r"\hline",
    ])
    for row in cg_rows:
        lines.append(
            f"{row['boundary'].title()} & {row['closure'].title()} & "
            f"{row['iterations']} & {row['weighted_relative_inf_residual']:.2e} & "
            f"{row['algebraic_to_discretisation_ratio']:.2e} \\\\"
        )
    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\caption{Jacobi-preconditioned CG results for $K_h^cU=W_h^cb$ on $N=64$ with relative tolerance $10^{-10}$.}",
        r"\label{tab:weighted-cg}",
        r"\end{table}",
    ])
    (output_dir / "section5_summary.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_all(output_dir: Path) -> None:
    checks = run_sanity_checks()
    rows = run_refinement(output_dir)
    fitted = write_fitted_orders(rows, output_dir)
    algebraic = run_algebraic_error_separation(output_dir)
    weighted_diagnostics = run_weighted_matrix_diagnostics(output_dir)
    energy_rows = run_energy_convergence(output_dir)
    cg_rows = run_cg_comparison(output_dir)
    matched = run_matched_accuracy(output_dir, rows)
    write_report_summary(rows, fitted, output_dir)
    write_section5_summary(weighted_diagnostics, energy_rows, cg_rows, output_dir)
    from make_figures import make_plots, make_tolerance_plot, make_weighted_energy_plots
    make_plots(rows, output_dir, fitted)
    make_tolerance_plot(algebraic, output_dir)
    make_weighted_energy_plots(energy_rows, cg_rows, output_dir)
    metadata = {
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "machine": platform.machine(),
        "thread_environment": {name: os.environ.get(name, "unset") for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        "solver": "scipy.sparse.linalg.spsolve (SuperLU), CSC input, use_umfpack=False",
        "solver_tolerance": "N/A (sparse direct solver)",
        "algebraic_cross_check": "GMRES+ILU at rtol 1e-8, 1e-10, 1e-12; atol 0; N=64",
        "weighted_matrix_check": (
            "K=W*A with report weights; symmetry, smallest eigenvalue, and energy identity; N=32"
        ),
        "weighted_cg_check": "Jacobi-preconditioned CG on K=W*A; rtol 1e-10; atol 0; N=64",
        "maximum_weighted_symmetry_defect": max(
            r["weighted_asymmetry_frobenius"] for r in weighted_diagnostics
        ),
        "minimum_weighted_eigenvalue": min(
            r["smallest_eigenvalue"] for r in weighted_diagnostics
        ),
        "fitted_order_range": "finest three levels: N=32,64,128",
        "timing": "perf_counter; median of 5 individually measured assembly-plus-solve totals after one warm-up; sample min/max retained",
        "cost_candidates": [4, 8, 16, 32, 64, 128],
        "accuracy_targets_linf": [1e-2, 5e-3],
        "accuracy_selection": "coarsest tested grid meeting the target; not attained if none",
        "timing_includes": "assembly, COO/CSR/CSC conversion, ordering, LU factorisation and triangular solves",
        "timing_excludes": "imports, warm-up, diagnostics, independent tests, extra Krylov/eigenvalue runs, plotting and file I/O",
        "sanity_checks": checks,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    from reporting import write_report_material
    write_report_material(rows, fitted, weighted_diagnostics, energy_rows,
                          cg_rows, algebraic, matched, metadata, output_dir)
