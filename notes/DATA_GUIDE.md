# Data and figure guide

All saved CSV files are in `data/`. The same raw values are available as
LaTeX tables in `generated/raw_tables/`. The more compact tables called by
`report/experiments.tex` are directly under `generated/`.

| Evidence | CSV | Figure or concise LaTeX table |
|---|---|---|
| Six-grid errors, fitted and Richardson orders | `data/refinement_results.csv`, `data/fitted_orders.csv` | `figures/convergence.png`, `generated/refinement_combined.tex`, `generated/fitted_global.tex` |
| Open-boundary and interior errors | `data/refinement_results.csv`, `data/error_locations.csv` | `figures/boundary_vs_interior.png`, `figures/solution_error.png`, `generated/spatial_*.tex` |
| Matrix structure and energy identity | `data/weighted_matrix_diagnostics.csv` | `figures/matrix_spy.png`, `generated/matrix_structure.tex` |
| Weighted energy convergence | `data/energy_convergence.csv` | `figures/weighted_energy_convergence.png`, `generated/energy_summary.tex` |
| GMRES and CG checks | `data/algebraic_error_separation.csv`, `data/weak_gmres_comparison.csv`, `data/cg_comparison.csv` | `figures/tolerance_sweep.png`, `figures/cg_solver_comparison.png`, `generated/gmres_summary.tex`, `generated/weak_gmres_summary.tex`, `generated/cg_summary.tex` |
| Cost at fixed accuracy | `data/cost_accuracy_candidates.csv`, `data/matched_accuracy.csv` | `figures/matched_accuracy.png`, `figures/cost_vs_accuracy.png`, `generated/matched_accuracy.tex` |
| Independent FEM oracle | `data/fem_oracle_comparison.csv` | `figures/fem_oracle.png`, `generated/fem_oracle_summary.tex` |
| Pure-Neumann depth | `data/pure_neumann_depth.csv` | `figures/pure_neumann_depth.png`, `generated/pure_neumann_summary.tex` |

Run `python code/run_all.py` to regenerate the whole set. Saved-run software
and timing details are in `data/run_metadata.json`.
