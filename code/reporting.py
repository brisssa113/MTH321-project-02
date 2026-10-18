"""Generate report tables and fixed-target evidence from measured values only."""

from pathlib import Path
import csv
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from models import run_case


def sci(value, digits=3):
    value = float(value)
    if not np.isfinite(value):
        return "--"
    if value == 0.0:
        return "$0$"
    mantissa, exponent = f"{value:.{digits}e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def decimal(value, digits=3):
    return f"{value:.{digits}f}" if np.isfinite(float(value)) else "--"


def sig2(value):
    """Two significant figures in plain decimal notation (no '1.7e+02')."""
    return f"{float(f'{float(value):.2g}'):g}"


def ratio_text(value):
    value = float(value)
    return sci(value, 1) if value >= 1000 else sig2(value)


def table(path, columns, header, rows, caption, label, size="small"):
    text = [r"\begin{table}[htbp]", r"\centering", "\\" + size,
            r"\setlength{\tabcolsep}{4pt}", f"\\begin{{tabular}}{{{columns}}}",
            r"\toprule", header + " " + chr(92)*2, r"\midrule"]
    text += [" & ".join(str(v) for v in row) + r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", r"\end{table}"]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def write_report_material(rows, fitted, weighted, energy, cg_rows, algebraic,
                          matched, metadata, destination: Path, weak_gmres=None,
                          fem_rows=None, pure_rows=None, condition_rows=None):
    labels = {"first": "FO", "ghost": "G"}
    combined_refinement = []
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            group = [r for r in rows if (r["boundary"], r["closure"]) == (boundary, closure)]
            records = [[r["N"], f"$1/{r['N']}$", sci(r["l2"]), decimal(r["order_l2"]),
                        sci(r["linf"]), decimal(r["order_linf"]), sci(r["residual_componentwise"], 2)]
                       for r in group]
            combined_refinement.extend([[boundary.title(), labels[closure], *line] for line in records])
            table(destination / f"refinement_{boundary}_{closure}.tex", "rrrrrrr",
                  r"$N$ & $h$ & $E_{2,h}$ & $p_2$ & $E_{\infty,h}$ & $p_\infty$ & $\eta_{\rm cw}$",
                  records, f"{boundary.title()} data, {labels[closure]} closure: four-mesh errors and pairwise orders. "
                  r"The direct SuperLU solver has no stopping tolerance; $\eta_{\rm cw}$ is the original-system componentwise backward residual.",
                  f"tab:refinement-{boundary}-{closure}")
        group = [r for r in rows if r["boundary"] == boundary]
        records = [[labels[r["closure"]], r["N"], sci(r["boundary_linf"]), decimal(r["order_boundary_linf"]),
                    sci(r["interior_linf"]), decimal(r["order_interior_linf"])] for r in group]
        table(destination / f"spatial_{boundary}.tex", "lrrrrr",
              r"Closure & $N$ & $E_{\infty,B}$ & $p_B$ & $E_{\infty,I}$ & $p_I$", records,
              f"{boundary.title()} data: error maxima restricted to the open right edge and the interior. "
              "This is a partition by node location, not an additive decomposition by error source.",
              f"tab:spatial-{boundary}")
    table(destination / "refinement_combined.tex", "llrrrrrrr",
          r"Boundary & Closure & $N$ & $h$ & $E_{2,h}$ & $p_2$ & $E_{\infty,h}$ & $p_\infty$ & $\eta_{\rm cw}$",
          combined_refinement,
          r"Global refinement errors for both boundary data and both closures; pairwise orders use adjacent grids.",
          "tab:refinement-combined", "scriptsize")
    table(destination / "spatial_combined.tex", "llrrrrr",
          r"Boundary & Closure & $N$ & $E_{\infty,B}$ & $p_B$ & $E_{\infty,I}$ & $p_I$",
          [[r["boundary"].title(), labels[r["closure"]], r["N"], sci(r["boundary_linf"]),
            decimal(r["order_boundary_linf"]), sci(r["interior_linf"]),
            decimal(r["order_interior_linf"])] for r in rows],
          r"Spatially restricted maximum errors; this is not an additive source decomposition.",
          "tab:spatial-combined", "scriptsize")
    table(destination / "fitted_global.tex", "llrrrrrr",
          r"Boundary & Closure & $p_2$ & $p_\infty$ & $p_B$ & $p_I$ & $p_{R,2}$ & $p_{R,\infty}$",
          [[r["boundary"].title(), labels[r["closure"]], decimal(r["fitted_order_l2"]),
            decimal(r["fitted_order_linf"]), decimal(r["fitted_order_boundary_linf"]),
            decimal(r["fitted_order_interior_linf"]), decimal(r["richardson_order_l2"]),
            decimal(r["richardson_order_linf"])] for r in fitted],
          r"Least-squares slopes and Richardson difference-based estimates on $N=128,256,512$. Three points do not independently establish an asymptotic regime.",
          "tab:fitted-global", "footnotesize")
    table(destination / "matrix_structure.tex", "llrrrr",
          r"Boundary & Closure & $\delta_A$ & $\delta_K$ & $\lambda_{\min}(K)$ & Energy defect",
          [[r["boundary"].title(), labels[r["closure"]], decimal(r["raw_asymmetry_frobenius"], 5),
            sci(r["weighted_asymmetry_frobenius"], 1), decimal(r["smallest_eigenvalue"], 4),
            sci(r["energy_identity_relative_defect"], 2)] for r in weighted],
          r"Raw and weighted matrix checks at $N=32$: $\delta_A=\|A-A^T\|_F/\|A\|_F$ and $\delta_K=\|K-K^T\|_F/\|K\|_F$. Eigenvalues are numerical estimates; positive definiteness for all stated grids is proved in Proposition~\ref{prop:energy}.",
          "tab:matrix-structure", "footnotesize")
    top_energy = max(r["N"] for r in energy)
    energy_grids = sorted({r["N"] for r in energy})[-3:]
    finest_energy = [r for r in energy if r["N"] == top_energy]
    table(destination / "energy_summary.tex", "llrrr",
          rf"Boundary & Closure & $E_{{E,h}}$ at $N={top_energy}$ & Fitted $p_E$ & $R_E^2$",
          [[r["boundary"].title(), labels[r["closure"]], sci(r["energy_error"]),
            decimal(r["fitted_energy_order_finest_three"]), decimal(r["r_squared_energy_finest_three"], 6)]
           for r in finest_energy],
          "Measured energy errors and fine-mesh fits on $N=" + ",".join(str(n) for n in energy_grids) + r"$. The energy norm is closure dependent; it is not the shared cost-comparison norm.",
          "tab:energy-summary")
    table(destination / "gmres_summary.tex", "llrrrr",
          r"Boundary & Closure & rtol & Iterations & $\eta_{\rm cw}$ & $\rho_{\rm alg}$",
          [[r["boundary"].title(), labels[r["closure"]], sci(r["rtol"], 0), r["iterations"],
            sci(r["componentwise_backward_error"], 2), sci(r["algebraic_to_discretisation_ratio"], 2)] for r in algebraic],
          r"GMRES--ILU cross-check at $N=64$, with $\mathrm{atol}=0$. Residuals are evaluated on the original raw system. The ratio $\rho_{\rm alg}$ is the iterative/direct maximum difference divided by the measured direct-solution error; the acceptance target is $10^{-2}$.",
          "tab:gmres-summary", "footnotesize")
    table(destination / "cg_summary.tex", "llrrrr",
          r"Boundary & Closure & Iterations & $\eta_{K,\infty}$ & $\eta_{\rm cw}$ & $\rho_{\rm alg}$",
          [[r["boundary"].title(), labels[r["closure"]], r["iterations"], sci(r["weighted_relative_inf_residual"], 2),
            sci(r["original_componentwise_backward_error"], 2), sci(r["algebraic_to_discretisation_ratio"], 2)] for r in cg_rows],
          r"Jacobi-preconditioned CG on $K U=Wb$ at $N=64$, rtol $10^{-10}$, atol zero. All return status zero. The two residual columns distinguish weighted normwise and original componentwise measures.",
          "tab:cg-summary", "footnotesize")
    if weak_gmres:
        table(destination / "weak_gmres_summary.tex", "llrrrrrl",
              r"Boundary & Closure & rtol & Iterations & $\rho_{\rm alg}$ & $\|r\|_\infty/\|b\|_\infty$ & Required & Rule met",
              [[r["boundary"].title(), labels[r["closure"]], sci(r["rtol"], 0),
                r["iterations"], sci(r["algebraic_to_discretisation_ratio"], 2),
                sci(r["relative_residual_inf"], 2), sci(r["required_relative_residual"], 2),
                "yes" if r["residual_rule_satisfied"] in (True, "True", 1) else "no"]
               for r in weak_gmres],
              r"Jacobi-preconditioned GMRES on the original matrix at $N=64$, without ILU factorisation. "
              r"``Required'' is $10^{-2}E_{\infty,h}/(\|A^{-1}\|_\infty\|b\|_\infty)$, the relative residual that guarantees $\rho_{\rm alg}<10^{-2}$; "
              r"the rule is sufficient, not necessary.",
              "tab:weak-gmres", "scriptsize")
    if fem_rows:
        def pick(boundary, closure, N):
            return next(r for r in fem_rows if (r["boundary"], r["closure"], r["N"]) == (boundary, closure, N))
        records = []
        for boundary in ("neumann", "robin"):
            for N in sorted({r["N"] for r in fem_rows}):
                fo, gh = pick(boundary, "first", N), pick(boundary, "ghost", N)
                records.append([boundary.title(), N, sci(fo["fem_linf_error"], 2),
                                decimal(fo.get("fem_pairwise_order", float("nan"))),
                                sci(fo["fdm_fem_linf_difference"], 2),
                                decimal(fo.get("fdm_fem_pairwise_order", float("nan"))),
                                sci(gh["fdm_fem_linf_difference"], 2),
                                decimal(gh.get("fdm_fem_pairwise_order", float("nan")))])
        table(destination / "fem_oracle_summary.tex", "lrrrrrrr",
              r"Boundary & $N$ & FEM error & $p_{\rm FEM}$ & FO--FEM & $p$ & G--FEM & $p$",
              records,
              r"Independent P1 triangular FEM solution of the weak form in Section~\ref{sec:formulation}, compared with both FDM closures at common grid vertices. "
              r"FEM error: nodal maximum against the manufactured solution; FO--FEM and G--FEM: nodal maximum difference between the FDM and FEM solutions. "
              r"Orders $p$ are pairwise between adjacent grids; the differences are not error bounds.",
              "tab:fem-oracle", "scriptsize")
    if pure_rows:
        table(destination / "pure_neumann_summary.tex", "llrrrrrr",
              r"Closure & $N$ & $\|A\boldsymbol 1\|_\infty$ & Defect $w^Tb$ & Trapezoid defect & Gauge error (uniform) & Gauge error (row-scaled) & Residual",
              [[labels[r["closure"]], r["N"], sci(r["null_vector_defect_inf"], 1),
                sci(r["discrete_compatibility_defect"], 2),
                sci(r["continuum_quadrature_defect"], 2),
                sci(r["gauge_solution_linf_error"], 2),
                sci(r["scaled_correction_solution_linf_error"], 2),
                sci(r["bordered_residual_inf"], 1)] for r in pure_rows],
              r"Pure-Neumann extension with all boundary vertices unknown. The defect uses a unit-sum left null vector $w$; "
              r"``Trapezoid defect'' is the trapezoidal quadrature of the continuous compatibility integral. "
              r"``Uniform'' adds one scalar correction to every row (bordered solve); ``row-scaled'' removes the same defect with a correction proportional to $|\mathrm{diag}\,A|$. "
              r"Both fix the arithmetic mean to zero. This is a distinct depth experiment, not the mixed problem's matrix.",
              "tab:pure-neumann", "scriptsize")
    if condition_rows:
        table(destination / "condition_summary.tex", "llrrrrrr",
              r"Boundary & Closure & $N$ & $\|A^{-1}\|_\infty$ & $\kappa_\infty$ & $\|A^{-1}\|_\infty\|r\|_\infty$ & Bound$/E_{\infty,h}$ & Round-off scale",
              [[r["boundary"].title(), labels[r["closure"]], r["N"],
                decimal(r["inverse_norm_inf_estimate"], 4), sci(r["condition_inf_estimate"], 2),
                sci(r["forward_error_bound_inf"], 2), sci(r["bound_to_discretisation_ratio"], 2),
                sci(r["roundoff_scale_estimate"], 2)] for r in condition_rows],
              r"Conditioning of the direct solve. $\|A^{-1}\|_\infty$ is estimated with Higham's one-norm estimator on the SuperLU factors; "
              r"$\kappa_\infty=\|A\|_\infty\|A^{-1}\|_\infty$. The forward-error bound $\|U-U_h\|_\infty\le\|A^{-1}\|_\infty\|r\|_\infty$ "
              r"is compared with the discretisation error. The round-off scale is $\kappa_\infty\,\varepsilon_{\rm mach}\|U\|_\infty$.",
              "tab:condition", "scriptsize")
    costs = []
    for r in matched:
        passed = r["status"] == "attained"
        costs.append([r["boundary"].title(), sci(r["target_linf"], 0), labels[r["closure"]],
                      r["selected_N"] if passed else "--", sci(r["achieved_linf"]) if passed else "not reached",
                      sig2(1000*r['assembly_seconds_median']) if passed else "--",
                      sig2(1000*r['solve_seconds_median']) if passed else "--",
                      sig2(1000*r['total_seconds_median']) if passed else "--"])
    table(destination / "matched_accuracy.tex", "lllrrrrr",
          r"Boundary & Target & Closure & $N$ & $E_{\infty,h}$ & Assembly (ms) & Solve (ms) & Total (ms)", costs,
          r"Coarsest successful grid in $\{4,8,16,32,64,128,256,512\}$. Separate assembly, solve, and complete-total medians are shown; their medians need not sum. Each median follows one warm-up run and uses 25 samples for $N\le32$ and 5 samples otherwise. Small-grid timings are noise sensitive.",
          "tab:matched-accuracy", "footnotesize")
    findings = []
    targets = sorted({r["target_linf"] for r in matched}, reverse=True)
    for boundary in ("neumann", "robin"):
        for target in targets:
            fo = next(r for r in matched if r["boundary"] == boundary and r["closure"] == "first" and r["target_linf"] == target)
            gh = next(r for r in matched if r["boundary"] == boundary and r["closure"] == "ghost" and r["target_linf"] == target)
            fo_ok, gh_ok = fo["status"] == "attained", gh["status"] == "attained"
            if fo_ok and gh_ok:
                ratio = fo["total_seconds_median"] / gh["total_seconds_median"]
                findings.append(f"For {boundary.title()} data at target {sci(target, 0)}, the selected FO/G grids are "
                                f"$N={fo['selected_N']}$ and $N={gh['selected_N']}$; their measured median cost ratio "
                                f"$T_{{\\rm FO}}/T_{{\\rm G}}$ is {ratio_text(ratio)}. ")
            elif gh_ok:
                findings.append(f"For {boundary.title()} data at target {sci(target, 0)}, FO does not meet the target "
                                f"on any tested grid up to $N=512$, while G meets it at $N={gh['selected_N']}$. "
                                "No finite FO/G cost ratio is reported for this unattained case. ")
            else:
                findings.append(f"For {boundary.title()} data at target {sci(target, 0)}, neither closure meets the target "
                                "on any tested grid up to $N=512$. ")
    (destination / "cost_findings.tex").write_text("\n\n".join(findings), encoding="utf-8")
    top = max(r["N"] for r in rows)
    def at_top(boundary, closure):
        return next(r for r in rows if (r["boundary"], r["closure"], r["N"]) == (boundary, closure, top))
    headline = {}
    for boundary, tag in (("neumann", "Neumann"), ("robin", "Robin")):
        fo, gh = at_top(boundary, "first"), at_top(boundary, "ghost")
        headline[f"ErrFO{tag}Top"] = sci(fo["linf"], 3)
        headline[f"ErrG{tag}Top"] = sci(gh["linf"], 3)
        headline[f"ErrRatio{tag}"] = f"{round(fo['linf'] / gh['linf'], -1):,.0f}".replace(",", "\\,")
        headline[f"NnzExcess{tag}"] = decimal(100.0 * (gh["nnz"] / fo["nnz"] - 1.0), 2)
    headline["TopGrid"] = str(top)
    if condition_rows:
        fo_inv = [r["inverse_norm_inf_estimate"] for r in condition_rows if r["closure"] == "first"]
        g_inv = [r["inverse_norm_inf_estimate"] for r in condition_rows if r["closure"] == "ghost"]
        n64 = [r for r in condition_rows if r["N"] == 64]
        headline["InvNormFOMin"], headline["InvNormFOMax"] = decimal(min(fo_inv), 3), decimal(max(fo_inv), 3)
        headline["InvNormGMin"], headline["InvNormGMax"] = decimal(min(g_inv), 3), decimal(max(g_inv), 3)
        headline["MaxForwardBoundRatio"] = sci(max(r["bound_to_discretisation_ratio"] for r in condition_rows), 2)
        headline["KappaSixtyFourMin"] = sci(min(r["condition_inf_estimate"] for r in n64), 2)
        headline["KappaSixtyFourMax"] = sci(max(r["condition_inf_estimate"] for r in n64), 2)
        headline["RoundoffScaleSixtyFourMax"] = sci(max(r["roundoff_scale_estimate"] for r in n64), 2)
    stats = {
        "RunPython": metadata["python"], "RunNumpy": metadata["numpy"],
        "RunScipy": metadata["scipy"], "RunMatplotlib": metadata["matplotlib"],
        "RunScikitFem": metadata["scikit_fem"],
        "MaxDirectBackward": sci(max(r["residual_componentwise"] for r in rows), 2),
        "MaxGMRESLooseRatio": sci(max(r["algebraic_to_discretisation_ratio"] for r in algebraic if r["rtol"] == 1e-8), 2),
        "MaxGMRESTightRatio": sci(max(r["algebraic_to_discretisation_ratio"] for r in algebraic if r["rtol"] == 1e-12), 2),
        "MaxCGRatio": sci(max(r["algebraic_to_discretisation_ratio"] for r in cg_rows), 2),
        "MaxTightMeanDifference": sci(max(r["right_boundary_mean_difference"] for r in algebraic if r["rtol"] == 1e-12), 2),
        "MaxEnergyIdentityDefect": sci(max(r["energy_identity_relative_defect"] for r in energy), 2),
        "NeumannInteriorGhostOrder": decimal(next(r["fitted_order_interior_linf"] for r in fitted if r["boundary"] == "neumann" and r["closure"] == "ghost")),
        "RobinInteriorGhostOrder": decimal(next(r["fitted_order_interior_linf"] for r in fitted if r["boundary"] == "robin" and r["closure"] == "ghost")),
    }
    stats.update(headline)
    (destination / "report_values.tex").write_text("\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in stats.items()) + "\n", encoding="utf-8")
    plot_fixed_targets(matched, destination)
    plot_solution_errors(destination)
    write_error_locations(destination)
    write_summary(rows, fitted, matched, metadata, findings, destination.parent)


def plot_fixed_targets(matched, destination):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    for ax, boundary in zip(axes, ("neumann", "robin")):
        for k, r in enumerate([r for r in matched if r["boundary"] == boundary]):
            label = f"{r['target_linf']:g}\n{r['closure']}"
            if r["status"] == "attained":
                y = 1000*r["total_seconds_median"]
                ax.bar(k, y, color="tab:blue" if r["closure"] == "first" else "tab:orange")
                ax.errorbar(k, y, yerr=[[y-1000*r["total_seconds_min"]],
                                       [1000*r["total_seconds_max"]-y]], fmt="none", color="black", capsize=3)
                ax.annotate(f"N={r['selected_N']}", (k,y), xytext=(0,6), textcoords="offset points", ha="center", fontsize=11)
            else:
                ax.text(k, 0.025, "not\nattained", transform=ax.get_xaxis_transform(), ha="center", fontsize=11)
            group = [q for q in matched if q["boundary"] == boundary]
            ax.set_xticks(range(len(group)), [f"{q['target_linf']:g}\n{q['closure']}" for q in group])
        ax.set_title(boundary.title()); ax.set_ylabel("assembly + solve (ms)")
        ax.set_yscale("log")
        ax.set_xlabel("fixed error threshold / closure"); ax.grid(True, axis="y", alpha=.25)
        ax.margins(y=.18)
    fig.tight_layout(); fig.savefig(destination / "matched_accuracy.png", dpi=220); plt.close(fig)


def plot_solution_errors(destination):
    N = 64
    first, ghost = run_case(N, "robin", "first"), run_case(N, "robin", "ghost")
    x = np.linspace(0,1,N+1); X,Y=np.meshgrid(x,x)
    exact = np.exp(X+Y)
    fig = plt.figure(figsize=(9.8,4))
    ax = fig.add_subplot(131, projection="3d")
    ax.plot_surface(X,Y,ghost.full_solution,cmap="viridis",linewidth=0,antialiased=True)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("U")
    ax.set_title("Robin G solution", fontsize=12)
    ax.set_xticks((0,0.5,1)); ax.set_yticks((0,0.5,1))
    for k, case in enumerate((first,ghost), start=2):
        ax = fig.add_subplot(1,3,k)
        error = case.full_solution-exact
        scale = float(np.max(np.abs(error)))
        im = ax.pcolormesh(X,Y,error,shading="auto",cmap="coolwarm",norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale))
        ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_title("FO signed error" if case.assembly.closure == "first" else "G signed error", fontsize=12)
        ax.set_xticks((0,0.5,1)); ax.set_yticks((0,0.5,1))
        fig.colorbar(im, ax=ax, shrink=.72, format="%.1e")
    fig.tight_layout(); fig.savefig(destination / "solution_error.png", dpi=220); plt.close(fig)


def write_error_locations(destination):
    records=[]
    for boundary in ("neumann","robin"):
        for closure in ("first","ghost"):
            for N in (32,64,128,256,512):
                case=run_case(N,boundary,closure)
                t=np.linspace(0,1,N+1)
                error=np.abs(case.full_solution-np.exp(t[:,None]+t[None,:]))
                j,i=np.unravel_index(np.argmax(error[1:N,1:N]),(N-1,N-1))
                records.append({"boundary":boundary,"closure":closure,"N":N,
                                "interior_max_i":int(i+1),"interior_max_j":int(j+1),
                                "x":float((i+1)/N),"y":float((j+1)/N),
                                "interior_to_global_max_ratio":case.diagnostics.interior_linf/case.diagnostics.linf})
    with (destination/"error_locations.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader();writer.writerows(records)


def write_summary(rows, fitted, matched, metadata, findings, root):
    lines = ["# Reproduced numerical results", "",
             f"Generated by run_all.py with Python {metadata['python']}, NumPy {metadata['numpy']}, "
             f"SciPy {metadata['scipy']}, Matplotlib {metadata['matplotlib']} on {metadata['platform']}.", "",
             "The baseline is explicitly fixed to SuperLU (`use_umfpack=False`).", "",
             "| Boundary | Closure | N | global max | discrete L2 | fitted max order | fitted L2 order |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for f in fitted:
        r=next(r for r in rows if r['boundary']==f['boundary'] and r['closure']==f['closure'] and r['N']==128)
        lines.append(f"| {r['boundary']} | {r['closure']} | 128 | {r['linf']:.7e} | {r['l2']:.7e} | {f['fitted_order_linf']:.5f} | {f['fitted_order_l2']:.5f} |")
    lines += ["", "Fits use N=128,256,512; all six refinement grids are retained in data/refinement_results.csv.", "",
              "## Fixed-target cost comparison", "",
              "Candidates: N=4,8,16,32,64,128,256,512. Targets in the same nodal maximum norm: " + ", ".join(f"{t:g}" for t in sorted({r["target_linf"] for r in matched}, reverse=True)) + ".",
              "The coarsest tested successful grid is selected. Failure within this finite set is reported without extrapolation.", "",
              "Cost is the median of individually measured assembly-plus-solve totals after a warm-up (25 samples for N<=32, 5 otherwise); minimum and maximum are also stored.",
              "It includes sparse conversions, ordering, factorisation and solve; it excludes validation, diagnostics, plotting and I/O.", ""]
    lines += [s.replace("$", "").replace("\\times", " x ").replace("\\rm ", "") for s in findings]
    lines += ["", "These timings belong to this machine/run; regenerate them instead of treating the ratios as universal complexity results."]
    (root / "RESULTS_SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def refresh_from_saved_data(root):
    """Refresh typography/tables without repeating the benchmark timings."""
    from make_figures import make_plots, make_tolerance_plot, make_weighted_energy_plots, make_depth_plots
    destination=root/"generated"
    def convert(value):
        try:
            return int(value) if re.fullmatch(r"[+-]?\d+",value) else float(value)
        except ValueError:
            return value
    def read(name):
        with (root/'data'/name).open(newline="",encoding="utf-8") as f:
            return [{k:convert(v) for k,v in row.items()} for row in csv.DictReader(f)]
    rows=read("refinement_results.csv")
    fitted=read("fitted_orders.csv")
    weighted=read("weighted_matrix_diagnostics.csv")
    energy=read("energy_convergence.csv")
    cg_rows=read("cg_comparison.csv")
    algebraic=read("algebraic_error_separation.csv")
    weak_gmres=read("weak_gmres_comparison.csv")
    fem_rows=read("fem_oracle_comparison.csv")
    pure_rows=read("pure_neumann_depth.csv")
    matched=read("matched_accuracy.csv")
    condition_rows=read("condition_estimates.csv")
    metadata=json.loads((root/"data"/"run_metadata.json").read_text())
    make_plots(rows,destination,fitted)
    make_tolerance_plot(algebraic,destination)
    make_weighted_energy_plots(energy,cg_rows,destination)
    make_depth_plots(fem_rows,pure_rows,destination)
    write_report_material(rows,fitted,weighted,energy,cg_rows,algebraic,matched,metadata,
                          destination,weak_gmres,fem_rows,pure_rows,condition_rows)