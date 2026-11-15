"""Export the ten numerical tables in Sections 8--9, without report prose."""

import csv
from pathlib import Path

from reporting import decimal, sci, table


def read_csv(root, name):
    with (root / "data" / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def order(value):
    return "--" if value == "" else decimal(float(value), 2)


def milliseconds(value):
    return decimal(1000 * float(value), 2)


def write_supplementary_tables(root: Path):
    """Read freshly computed CSVs and generate only body Tables 14--23."""
    out = root / "generated"
    solver = read_csv(root, "depth_extension/direct_jacobi.csv")
    table(out / "depth_direct_jacobi.tex", "rrrrrrr",
          r"$N$ & free DOF & $E_{\rm disc,\infty}$ & Jacobi steps & final $\|r\|_\infty$ & $\|u_J-u_D\|_\infty/E_{\rm disc,\infty}$ & direct/Jacobi solve (ms)",
          [[r["N"], r["free_dof"], sci(r["discretisation_error_inf"], 2),
            r["iterations"], sci(r["jacobi_residual_final"], 2),
            sci(r["algebraic_over_discretisation"], 2),
            milliseconds(r["direct_s"]) + "/" + milliseconds(r["jacobi_s"])] for r in solver],
          r"Matched-accuracy sparse direct and Jacobi comparison. Full residuals and errors are in "
          r"\path{data/depth_extension/direct_jacobi.csv}. Times are single local measurements.",
          "tab:depth-direct-jacobi", fit_width=True)

    square = read_csv(root, "depth_extension/square_fdm_fem.csv")
    table(out / "depth_square_fdm_fem.tex", "lrrrrrrr",
          r"Method & $N$ & free DOF & nnz & $L^2$ error & order & nodal $L^\infty$ & assembly/solve (ms)",
          [[r["method"], r["N"], r["dof_free"], r["nnz"], sci(r["L2"], 2),
            order(r["L2_order"]), sci(r["nodal_Linf"], 2),
            milliseconds(r["assembly_s"]) + "/" + milliseconds(r["solve_s"])] for r in square],
          r"Square all-Dirichlet benchmark using a common P1 reconstruction for the $L^2$ comparison. "
          r"Orders are adjacent-grid log ratios. Full values are in \path{data/depth_extension/square_fdm_fem.csv}.",
          "tab:depth-square-fdm-fem", fit_width=True)

    lshape = read_csv(root, "depth_extension/lshape_smooth_singular.csv")
    table(out / "depth_lshape_errors.tex", "lrrrrrrr",
          r"Solution & level & $h_{\max}$ & free DOF & $L^2$ error & $L^2$ order & energy error & energy order",
          [[r["kind"], r["level"], f'{float(r["h"]):.3g}', r["dof_free"],
            sci(r["L2"], 2), order(r["L2_order"]), sci(r["energy"], 2),
            order(r["energy_order"])] for r in lshape],
          r"Five L-shaped uniform refinements. The energy error is $\|\nabla(u_*-u_h)\|_{L^2(\Omega)}$. "
          r"Additional diagnostics are in \path{data/depth_extension/lshape_smooth_singular.csv}.",
          "tab:depth-lshape-errors", fit_width=True)
    table(out / "depth_lshape_corner.tex", "lrrrrr",
          r"Solution & level & free DOF & nnz & solve (ms) & corner error fraction",
          [[r["kind"], r["level"], r["dof_free"], r["nnz"],
            milliseconds(r["solve_s"]), decimal(float(r["corner_energy_fraction"]), 2)] for r in lshape],
          r"L-shaped solve costs and fraction of the computed squared energy error in whole triangles "
          r"whose centroids are within $2h_{\max}$ of the re-entrant corner. Fractions are rounded to two decimal places.",
          "tab:depth-lshape-corner")

    sine = read_csv(root, "variable_diffusion/sine_required.csv")
    table(out / "variable_sine.tex", "lrrrrr",
          r"Method & $N$ & $L^2$ error & $L^2$ order & nodal $L^\infty$ & $L^\infty$ order",
          [[r["method"], r["N"], sci(r["error_l2"], 2), order(r["order_l2"]),
            sci(r["error_inf"], 2), order(r["order_inf"])] for r in sine],
          r"Prescribed sine benchmark on four matched meshes. FDM uses a discrete nodal $L^2$ norm; "
          r"FEM integrates the continuous P1 error. Full diagnostics are in \path{data/variable_diffusion/sine_required.csv}.",
          "tab:variable-sine")
    smooth = read_csv(root, "variable_diffusion/smooth.csv")
    table(out / "variable_smooth.tex", "lrrrr",
          r"Method & $N$ & nodal $L^\infty$ error & order & CG steps",
          [[r["method"], r["N"], sci(r["error_inf"], 2), order(r["order"]),
            r["cg_iterations"]] for r in smooth],
          r"Additional exponential test with $a=1+x$. Full residuals, timings, nonzeros and condition estimates "
          r"are in \path{data/variable_diffusion/smooth.csv}.", "tab:variable-smooth")

    layered = read_csv(root, "variable_diffusion/layered.csv")
    table(out / "variable_layered.tex", "rlrr",
          r"$N$ & face mean & maximum error & sampled flux error",
          [[r["N"], r["mean"], sci(r["error_inf"], 2), sci(r["flux_error"], 2)] for r in layered],
          r"Layered benchmark. The last column is the absolute error in positive $a u_x$ at one sampled face; "
          r"it is not a whole-interface maximum. Full values are in \path{data/variable_diffusion/layered.csv}.",
          "tab:variable-layered")
    interface = read_csv(root, "variable_diffusion/interface_diagnostics.csv")
    table(out / "variable_interface.tex", "rlrrr",
          r"$N$ & mean & $E_{J,\infty}$ & $D_h$ & $\mathcal J_h$",
          [[r["N"], r["mean"], sci(r["interface_flux_error_max"], 2),
            sci(r["adjacent_flux_jump_max"], 2), sci(r["reconstructed_flux_jump_max"], 2)] for r in interface],
          r"Whole-interface physical flux error, adjacent-face difference and material-side reconstructed jump. "
          r"All fluxes use the common $+x$ orientation. Full values and traces are in "
          r"\path{data/variable_diffusion/interface_diagnostics.csv} and \path{data/variable_diffusion/interface_flux_profiles.csv}.",
          "tab:variable-interface")
    contrast = read_csv(root, "variable_diffusion/contrast_sweep.csv")
    table(out / "variable_contrast.tex", "rlrr",
          r"$a_R/a_L$ & mean & Jacobi-CG steps & direct nodal error",
          [[f'{float(r["contrast"]):g}', r["mean"], r["cg_iterations"],
            sci(r["nodal_error_inf"], 2)] for r in contrast if r["N"] == "63"],
          r"Material-contrast sweep at $N=63$. The error belongs to the direct solve; CG contributes the iteration count. "
          r"CG errors, residuals and direct/CG differences on all grids are retained separately in "
          r"\path{data/variable_diffusion/contrast_sweep.csv}.", "tab:variable-contrast")
    fits = read_csv(root, "variable_diffusion/sine_fitted_orders.csv")
    table(out / "variable_sine_fits.tex", "lrrr",
          r"Method & fitted $p_{L^2}$ & fitted $p_\infty$ & max $\delta_A$",
          [[r["method"], decimal(float(r["fitted_order_l2"]), 4),
            decimal(float(r["fitted_order_inf"]), 4),
            sci(max(float(v["symmetry_defect_fro"]) for v in sine if v["method"] == r["method"]), 1)] for r in fits],
          r"Sine fits on $N=16,32,64$ and maximum reduced-matrix Frobenius symmetry defect over the four sine grids. "
          r"Fits and $R^2$ values are in \path{data/variable_diffusion/sine_fitted_orders.csv}.",
          "tab:variable-sine-fits")
