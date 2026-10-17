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


def table(path, columns, header, rows, caption, label, size="small"):
    text = [r"\begin{table}[htbp]", r"\centering", "\\" + size,
            r"\setlength{\tabcolsep}{4pt}", f"\\begin{{tabular}}{{{columns}}}",
            r"\toprule", header + " " + chr(92)*2, r"\midrule"]
    text += [" & ".join(str(v) for v in row) + r" \\" for row in rows]
    text += [r"\bottomrule", r"\end{tabular}", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}", r"\end{table}"]
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def write_report_material(rows, fitted, weighted, energy, cg_rows, algebraic,
                          matched, metadata, destination: Path):
    labels = {"first": "FO", "ghost": "G"}
    for boundary in ("neumann", "robin"):
        for closure in ("first", "ghost"):
            group = [r for r in rows if (r["boundary"], r["closure"]) == (boundary, closure)]
            records = [[r["N"], f"$1/{r['N']}$", sci(r["l2"]), decimal(r["order_l2"]),
                        sci(r["linf"]), decimal(r["order_linf"]), sci(r["residual_componentwise"], 2)]
                       for r in group]
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
    table(destination / "fitted_global.tex", "llrrrrrr",
          r"Boundary & Closure & $p_2$ & $R_2^2$ & $p_\infty$ & $R_\infty^2$ & $p_B$ & $p_I$",
          [[r["boundary"].title(), labels[r["closure"]], decimal(r["fitted_order_l2"]), decimal(r["r_squared_l2"], 6),
            decimal(r["fitted_order_linf"]), decimal(r["r_squared_linf"], 6),
            decimal(r["fitted_order_boundary_linf"]), decimal(r["fitted_order_interior_linf"])] for r in fitted],
          r"Least-squares fits on the identified fine-mesh range $N=32,64,128$. Pairwise orders are retained in the preceding refinement tables; $R^2$ measures fit quality rather than a confidence interval.",
          "tab:fitted-global", "footnotesize")
    table(destination / "matrix_structure.tex", "llrrrr",
          r"Boundary & Closure & $\delta_A$ & $\delta_K$ & $\lambda_{\min}(K)$ & Energy defect",
          [[r["boundary"].title(), labels[r["closure"]], decimal(r["raw_asymmetry_frobenius"], 5),
            sci(r["weighted_asymmetry_frobenius"], 1), decimal(r["smallest_eigenvalue"], 4),
            sci(r["energy_identity_relative_defect"], 2)] for r in weighted],
          r"Raw and weighted matrix checks at $N=32$: $\delta_A=\|A-A^T\|_F/\|A\|_F$ and $\delta_K=\|K-K^T\|_F/\|K\|_F$. Eigenvalues are numerical estimates; positive definiteness for all stated grids is proved in Proposition~\ref{prop:energy}.",
          "tab:matrix-structure", "footnotesize")
    finest_energy = [r for r in energy if r["N"] == 128]
    table(destination / "energy_summary.tex", "llrrr",
          r"Boundary & Closure & $E_{E,h}$ at $N=128$ & Fitted $p_E$ & $R_E^2$",
          [[r["boundary"].title(), labels[r["closure"]], sci(r["energy_error"]),
            decimal(r["fitted_energy_order_finest_three"]), decimal(r["r_squared_energy_finest_three"], 6)]
           for r in finest_energy],
          r"Measured energy errors and fine-mesh fits on $N=32,64,128$. The energy norm is closure dependent; it is not the shared cost-comparison norm.",
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
    costs = []
    for r in matched:
        passed = r["status"] == "attained"
        costs.append([r["boundary"].title(), sci(r["target_linf"], 0), labels[r["closure"]],
                      r["selected_N"] if passed else "--", sci(r["achieved_linf"]) if passed else "not reached",
                      decimal(1000*r["total_seconds_median"]) if passed else "--",
                      f"{1000*r['total_seconds_min']:.3f}--{1000*r['total_seconds_max']:.3f}" if passed else "--"])
    table(destination / "matched_accuracy.tex", "lllr rrr".replace(" ", ""),
          r"Boundary & Target & Closure & $N$ & $E_{\infty,h}$ & Median (ms) & Range (ms)", costs,
          r"Cost at fixed nodal maximum-error thresholds, selecting the coarsest successful grid in $\{4,8,16,32,64,128\}$. Each range is the minimum--maximum of five measured assembly-plus-solve totals after one warm-up, not a confidence interval. A missing entry means the target was not attained within the candidate set.",
          "tab:matched-accuracy", "footnotesize")
    findings = []
    for boundary in ("neumann", "robin"):
        for target in (1e-2, 5e-3):
            fo = next(r for r in matched if r["boundary"] == boundary and r["closure"] == "first" and r["target_linf"] == target)
            gh = next(r for r in matched if r["boundary"] == boundary and r["closure"] == "ghost" and r["target_linf"] == target)
            if fo["status"] == gh["status"] == "attained":
                ratio = fo["total_seconds_median"] / gh["total_seconds_median"]
                findings.append(f"For {boundary.title()} data at target {sci(target, 0)}, the selected FO/G grids are "
                                f"$N={fo['selected_N']}$ and $N={gh['selected_N']}$; their measured median cost ratio "
                                f"$T_{{\\rm FO}}/T_{{\\rm G}}$ is {ratio:.2f}. ")
            else:
                findings.append(f"For {boundary.title()} data at target {sci(target, 0)}, FO does not meet the target "
                                f"on any tested grid up to $N=128$, while G meets it at $N={gh['selected_N']}$. "
                                "No finite FO/G cost ratio is reported for this unattained case. ")
    (destination / "cost_findings.tex").write_text("\n\n".join(findings), encoding="utf-8")
    stats = {
        "RunPython": metadata["python"], "RunNumpy": metadata["numpy"],
        "RunScipy": metadata["scipy"], "RunMatplotlib": metadata["matplotlib"],
        "MaxDirectBackward": sci(max(r["residual_componentwise"] for r in rows), 2),
        "MaxGMRESLooseRatio": sci(max(r["algebraic_to_discretisation_ratio"] for r in algebraic if r["rtol"] == 1e-8), 2),
        "MaxGMRESTightRatio": sci(max(r["algebraic_to_discretisation_ratio"] for r in algebraic if r["rtol"] == 1e-12), 2),
        "MaxCGRatio": sci(max(r["algebraic_to_discretisation_ratio"] for r in cg_rows), 2),
        "MaxTightMeanDifference": sci(max(r["right_boundary_mean_difference"] for r in algebraic if r["rtol"] == 1e-12), 2),
        "MaxEnergyIdentityDefect": sci(max(r["energy_identity_relative_defect"] for r in energy), 2),
        "NeumannInteriorGhostOrder": decimal(next(r["fitted_order_interior_linf"] for r in fitted if r["boundary"] == "neumann" and r["closure"] == "ghost")),
        "RobinInteriorGhostOrder": decimal(next(r["fitted_order_interior_linf"] for r in fitted if r["boundary"] == "robin" and r["closure"] == "ghost")),
    }
    (destination / "report_values.tex").write_text("\n".join(f"\\newcommand{{\\{key}}}{{{value}}}" for key, value in stats.items()) + "\n", encoding="utf-8")
    plot_fixed_targets(matched, destination)
    plot_solution_errors(destination)
    write_error_locations(destination)
    write_summary(rows, fitted, matched, metadata, findings, destination.parent)


def plot_fixed_targets(matched, destination):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
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
            ax.set_xticks(range(4), [f"{q['target_linf']:g}\n{q['closure']}" for q in matched if q["boundary"] == boundary])
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
            for N in (32,64,128):
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
    lines += ["", "Fits use N=32,64,128; all four-mesh values are retained in data/refinement_results.csv.", "",
              "## Fixed-target cost comparison", "",
              "Candidates: N=4,8,16,32,64,128. Targets in the same nodal maximum norm: 0.01 and 0.005.",
              "The coarsest tested successful grid is selected. Failure within this finite set is reported without extrapolation.", "",
              "Cost is the median of five individually measured assembly-plus-solve totals after a warm-up; minimum and maximum are also stored.",
              "It includes sparse conversions, ordering, factorisation and solve; it excludes validation, diagnostics, plotting and I/O.", ""]
    lines += [s.replace("$", "").replace("\\times", " x ").replace("\\rm ", "") for s in findings]
    lines += ["", "These timings belong to this machine/run; regenerate them instead of treating the ratios as universal complexity results.",
              "", "The retained report appendices concerning team activity/ICS/AI audit have not been revised in this task."]
    (root / "RESULTS_SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def refresh_from_saved_data(root):
    """Refresh typography/tables without repeating the benchmark timings."""
    from make_figures import make_plots, make_tolerance_plot, make_weighted_energy_plots
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
    matched=read("matched_accuracy.csv")
    metadata=json.loads((root/"data"/"run_metadata.json").read_text())
    make_plots(rows,destination,fitted)
    make_tolerance_plot(algebraic,destination)
    make_weighted_energy_plots(energy,cg_rows,destination)
    write_report_material(rows,fitted,weighted,energy,cg_rows,algebraic,matched,metadata,destination)


