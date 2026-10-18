"""Plot the refinement, matrix, solver, and energy diagnostics."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 13, "axes.labelsize": 13,
                     "xtick.labelsize": 11, "ytick.labelsize": 11})
import numpy as np
import scipy

from models import assemble_system


def make_depth_plots(fem_rows: list[dict], pure_rows: list[dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for boundary in ("neumann", "robin"):
        for closure, marker in (("first", "o-"), ("ghost", "s--")):
            group = [r for r in fem_rows if (r["boundary"], r["closure"]) == (boundary, closure)]
            ax.loglog([1 / r["N"] for r in group],
                      [r["fdm_fem_linf_difference"] for r in group], marker,
                      label=f"{boundary.title()} / {closure}")
    ax.invert_xaxis(); ax.set_xlabel("h"); ax.set_ylabel("shared-node FDM–FEM max difference")
    ax.set_xticks([1/16, 1/32, 1/64, 1/128, 1/256, 1/512], ["1/16", "1/32", "1/64", "1/128", "1/256", "1/512"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(output_dir / "fem_oracle.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for closure, marker in (("first", "o-"), ("ghost", "s--")):
        group = [r for r in pure_rows if r["closure"] == closure]
        h = [1 / r["N"] for r in group]
        axes[0].loglog(h, [abs(r["discrete_compatibility_defect"]) for r in group],
                       marker, label=closure)
        axes[1].loglog(h, [r["gauge_solution_linf_error"] for r in group],
                       marker, label=f"{closure}, uniform correction")
        axes[1].loglog(h, [r["scaled_correction_solution_linf_error"] for r in group],
                       marker[0] + ":", label=f"{closure}, row-scaled correction")
    for ax, title in zip(axes, ("Compatibility defect", "Mean-zero solution error")):
        ax.invert_xaxis(); ax.set_xlabel("h"); ax.set_title(title)
        ax.set_xticks([1/8, 1/16, 1/32, 1/64, 1/128], ["1/8", "1/16", "1/32", "1/64", "1/128"])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(output_dir / "pure_neumann_depth.png", dpi=200); plt.close(fig)

def make_plots(rows: list[dict], output_dir: Path, fitted: list[dict]) -> None:
    fit_map = {(r["boundary"], r["closure"]): r for r in fitted}
    styles = {("neumann", "first"): "o-", ("neumann", "ghost"): "s-",
              ("robin", "first"): "o--", ("robin", "ghost"): "s--"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, style in styles.items():
        subset = [r for r in rows if (r["boundary"], r["closure"]) == key]
        label = f"{key[0].title()} / {key[1]}"
        axes[0].loglog([r["h"] for r in subset], [r["linf"] for r in subset], style,
                      label=f"{label}, p={fit_map[key]['fitted_order_linf']:.3f}")
        axes[1].loglog([r["h"] for r in subset], [r["l2"] for r in subset], style,
                      label=f"{label}, p={fit_map[key]['fitted_order_l2']:.3f}")
    ref_h = np.array([r["h"] for r in rows if r["boundary"] == "neumann" and r["closure"] == "first"])
    for ax, metric in zip(axes, ("linf", "l2")):
        for closure, order in (("first", 1), ("ghost", 2)):
            anchor = next(r[metric] for r in rows if r["boundary"] == "neumann" and r["closure"] == closure and r["N"] == 64)
            ax.loglog(ref_h, 1.6 * anchor * (ref_h / (1/64))**order,
                      color="0.45", linestyle=":" if order == 1 else "--",
                      linewidth=1, label=f"reference h^{order}")
    for ax, title in zip(axes, ("Global nodal max error", "Discrete L2 error")):
        ax.set_xlabel("h"); ax.set_ylabel("error"); ax.set_title(title); ax.grid(True, which="both", alpha=.3)
        ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.invert_xaxis()
    fig.set_size_inches(14, 4.5)
    fig.tight_layout(); fig.savefig(output_dir / "convergence.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, closure in zip(axes, ("first", "ghost")):
        A = assemble_system(32, "robin", closure).A
        asym = scipy.sparse.linalg.norm(A - A.T, ord="fro") / scipy.sparse.linalg.norm(A, ord="fro")
        ax.spy(A, markersize=.35)
        ax.set_title(f"Robin / {closure}: {A.shape[0]} unknowns, nnz={A.nnz}\nasymmetry={asym:.4f}", fontsize=10)
    fig.tight_layout(); fig.savefig(output_dir / "matrix_spy.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, style in styles.items():
        subset = [r for r in rows if (r["boundary"], r["closure"]) == key]
        ax.loglog([r["linf"] for r in subset], [r["total_seconds_median"] for r in subset], style, label=f"{key[0].title()} / {key[1]}")
    ax.set_xlabel("global nodal max error"); ax.set_ylabel("median of assembly + solve time (s)")
    ax.set_title("Cost versus accuracy (median of repeated runs; warm-up excluded)"); ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=11)
    fig.tight_layout(); fig.savefig(output_dir / "cost_vs_accuracy.png", dpi=200); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for key, style in styles.items():
        subset = [r for r in rows if (r["boundary"], r["closure"]) == key]
        label = f"{key[0].title()} / {key[1]}"
        axes[0].loglog(
            [r["h"] for r in subset], [r["boundary_linf"] for r in subset],
            style, label=f"{label}, p={fit_map[key]['fitted_order_boundary_linf']:.3f}",
        )
        axes[1].loglog(
            [r["h"] for r in subset], [r["interior_linf"] for r in subset],
            style, label=f"{label}, p={fit_map[key]['fitted_order_interior_linf']:.3f}",
        )
    for ax, title in zip(axes, ("Right-boundary max error", "Interior max error")):
        ax.set_xlabel("h"); ax.set_ylabel("error"); ax.set_title(title)
        ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=11); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(output_dir / "boundary_vs_interior.png", dpi=200); plt.close(fig)


def make_tolerance_plot(records: list[dict], output_dir: Path) -> None:
    styles = {("neumann", "first"): "o-", ("neumann", "ghost"): "s-",
              ("robin", "first"): "o--", ("robin", "ghost"): "s--"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key, style in styles.items():
        subset = sorted(
            (r for r in records if (r["boundary"], r["closure"]) == key),
            key=lambda r: r["rtol"], reverse=True,
        )
        ax.loglog(
            [r["rtol"] for r in subset],
            [r["algebraic_to_discretisation_ratio"] for r in subset],
            style,
            label=f"{key[0].title()} / {key[1]}",
        )
    ax.axhline(1e-2, color="black", linestyle=":", linewidth=1,
               label="1% acceptance threshold")
    ax.set_xlabel("GMRES relative tolerance")
    ax.set_ylabel("algebraic difference / discretisation error")
    ax.set_title("Algebraic-error separation on N=64")
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=11); ax.invert_xaxis()
    fig.tight_layout(); fig.savefig(output_dir / "tolerance_sweep.png", dpi=200); plt.close(fig)


def make_weighted_energy_plots(
    energy_rows: list[dict], cg_rows: list[dict], output_dir: Path
) -> None:
    styles = {("neumann", "first"): "o-", ("neumann", "ghost"): "s-",
              ("robin", "first"): "o--", ("robin", "ghost"): "s--"}
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for key, style in styles.items():
        subset = [r for r in energy_rows if (r["boundary"], r["closure"]) == key]
        ax.loglog(
            [r["h"] for r in subset], [r["energy_error"] for r in subset],
            style, label=f"{key[0].title()} / {key[1]}, p={subset[-1]['fitted_energy_order_finest_three']:.3f}",
        )
    ref_h = np.array(sorted({r["h"] for r in energy_rows}, reverse=True))
    for closure, order in (("first", 1), ("ghost", 2)):
        anchor = next(r["energy_error"] for r in energy_rows if r["boundary"] == "neumann" and r["closure"] == closure and r["N"] == 64)
        ax.loglog(ref_h, 1.6 * anchor * (ref_h / (1/64))**order,
                  color="0.45", linestyle=":" if order == 1 else "--",
                  linewidth=1, label=f"reference h^{order}")
    ax.set_xlabel("h"); ax.set_ylabel("weighted energy error")
    ax.set_title(r"Convergence in $\sqrt{h^2 e^T K e}$")
    ax.grid(True, which="both", alpha=.3); ax.legend(fontsize=11); ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(output_dir / "weighted_energy_convergence.png", dpi=200)
    plt.close(fig)

    labels = [f"{r['boundary'].title()}\n{r['closure'].title()}" for r in cg_rows]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].bar(x, [r["iterations"] for r in cg_rows], color="tab:blue")
    axes[0].set_ylabel("CG iterations")
    axes[0].set_title("Jacobi-preconditioned CG on N=64")
    axes[1].bar(x, [r["solve_seconds"] for r in cg_rows], color="tab:orange")
    axes[1].set_ylabel("solve time (s)")
    axes[1].set_title("Measured CG solve time")
    for ax in axes:
        ax.set_xticks(x, labels); ax.grid(True, axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(output_dir / "cg_solver_comparison.png", dpi=200)
    plt.close(fig)