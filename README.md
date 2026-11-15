# MTH321 Project 2 - Mixed-Boundary Poisson Problem

Group N: Kainan Zhao, Rengerile Na and Suning Peng.

This repository contains the submitted report, presentation, Python implementation and saved numerical results for Direction 2. The main study compares first-order backward-derivative (FO) and ghost-point boundary PDE (G) closures for a Poisson problem on the unit square, with Dirichlet data on three sides and Neumann or Robin data on the open right edge. The smooth manufactured benchmark supports first- and second-order maximum-error convergence, respectively.

Later studies compare direct/iterative solvers, FDM/FEM, smooth/corner-singular solutions on an L-shaped domain, and face averaging for variable diffusion. These extensions have separate boundary conditions and protocols.

## Start here

- [Final report](report/Group%20N%20report.pdf)
- [Presentation](slides/GroupN.pdf)
- [Presentation LaTeX source](slides/GroupN_latex.tex)
- [Numerical entry point](code/run_all.py)
- [Pinned dependencies](requirements.txt)

The complete editable report is maintained in Overleaf. Its PDF is included; the complete report LaTeX project is not included.

## Repository layout

| Path | Contents |
| --- | --- |
| `report/` | Submitted group report PDF |
| `slides/` | Presentation PDF and editable LaTeX source |
| `code/` | Assembly, solvers, experiments, plots and numerical table generation |
| `code/tests/` | 22 automated test methods |
| `data/` | 25 supporting CSVs and two run-environment metadata JSON files |
| `figures/` | 18 numerical PNG figures used in the report body |
| `generated/` | 22 numerical LaTeX tables and `report_values.tex` numerical macros |
| `requirements.txt` | Pinned NumPy, SciPy, Matplotlib and scikit-fem versions |

No separate `output/` directory is required. Numerical outputs belong in `data/`, `figures/` and `generated/`. The former `OUTPUTS.md` index has been incorporated below.

## Reproduce the numerical results

Use Python 3.12 or 3.13 and run commands from the repository root. Pinned versions: NumPy 2.1.3, SciPy 1.15.3, Matplotlib 3.10.0 and scikit-fem 12.0.2.

### Windows

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s code/tests -v
.\.venv\Scripts\python.exe code/run_all.py
```

### macOS / Linux

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s code/tests -v
.venv/bin/python code/run_all.py
```

Using these executable paths does not require environment activation. The test command should end with `Ran 22 tests` and `OK`. The numerical command should finish with `All checks passed.` and counts of 18 figures, 22 tables and 25 CSVs.

The full run includes N=512 solves, repeated timing measurements, FEM comparisons and all later extensions. Allow it to finish even when the terminal is quiet. An optional dependency-only check is:

```bash
python code/run_all.py --check-inputs
```

Use the virtual environment's Python executable if applicable. After dependency installation, numerical inputs are built into Python: no external dataset, report template, Overleaf, TeX installation or further network access is required.

The command computes in a temporary `.mth321-run-*` directory and validates the inventory before publishing. A computation or validation failure leaves previous published results in place. A successful run replaces numerical outputs and removes specifically identified obsolete exports, including duplicate/raw tables and the old `RESULTS_SUMMARY.md`. `code/report_outputs.py` defines these inventories. The report PDF and presentation are not regenerated.

## Experiments and validation

| Study | Implementation in `code/` | Evidence |
| --- | --- | --- |
| Mixed-boundary FO/G comparison | `models.py`, `experiments.py` | Six grids N=16,32,64,128,256,512; errors, residuals, sparsity and energy |
| Accuracy versus cost | `experiments.py` | Eight candidate grids N=4,...,512 and five common maximum-error targets |
| Solver cross-checks | `experiments.py` | SuperLU, ILU/Jacobi GMRES and CG on the weighted SPD system |
| Independent FEM oracle | `fem_oracle.py` | P1 triangular FEM assembled from the weak formulation |
| Pure-Neumann extension | `pure_neumann.py` | Compatibility, null spaces and mean-zero gauge |
| Supplementary comparisons | `depth_benchmarks.py` | Direct/Jacobi, square FDM/FEM, L-shaped smooth/singular, quadrature and solver checks |
| Variable diffusion | `variable_diffusion.py` | Sine and smooth benchmarks, layered interface fluxes and contrast sweep |
| Report exports | `reporting.py`, `body_tables.py`, `make_figures.py` | Figures, numerical tables and macros |

The mixed-boundary exact solution is u(x,y)=exp(x+y), with source and boundary data derived independently of the discrete equations. Neumann uses beta=0; Robin uses beta=2. Main convergence fits use N=128,256,512. Variable-diffusion sine fits instead use N=16,32,64, with separately defined norms.

The 22 tests check boundary rows, corners, manufactured defects, independent assembly, convergence, weighted matrix/energy identities, inverse norms, FEM, pure-Neumann compatibility and variable-diffusion forcing, symmetry and interface fluxes. Two loose-tolerance G/Jacobi-GMRES cases intentionally fail the one-percent algebraic-error target: a successful solver exit alone does not establish accuracy.

Verification on 15 November 2026 used Python 3.13.5 with the pinned package versions. All 22 tests passed and the complete entry point finished successfully. Its 68 files matched the declared inventory with no missing or unexpected files; all 18 PNGs decoded successfully. Fitted maximum-error orders were 1.004914 (Neumann FO), 1.002529 (Robin FO), 1.999929 (Neumann G) and 1.999937 (Robin G). The fresh run was saved separately and the pre-verification dataset was restored as a complete set to retain the report's saved timing values.

## Figure-to-file index

Numbering follows the supplied report body.

| Report figure | File in `figures/` |
| --- | --- |
| Figure 2 | `fem_oracle.png` |
| Figure 3 | `tolerance_sweep.png` |
| Figure 4 | `convergence.png` |
| Figure 5 | `boundary_vs_interior.png` |
| Figure 6 | `solution_error.png` |
| Figure 7 | `matrix_spy.png` |
| Figure 8 | `weighted_energy_convergence.png` |
| Figure 9 | `matched_accuracy.png` |
| Figure 10 | `cost_vs_accuracy.png` |
| Figure 11 | `pure_neumann_depth.png` |
| Figure 12 | `square_fdm_fem.png` |
| Figure 13 | `lshape_rates.png` |
| Figure 14 | `lshape_vs_dof.png` |
| Figure 15 | `lshape_mesh_level1.png` |
| Figure 16 | `lshape_mesh_level5.png` |
| Figure 17 | `layered_solution_cut.png` |
| Figure 18 | `layered_interface_flux.png` |
| Figure 19 | `layered_contrast_iterations.png` |

Figure 1 is authored in the report as a boundary schematic. Figure 10 includes all eight measured candidate grids, including N=4 and N=8.

## Table-to-file index

| Report table | File in `generated/` |
| --- | --- |
| Table 1 | `fo_bounds.tex` |
| Table 3 | `fem_oracle_summary.tex` |
| Table 4 | `gmres_summary.tex` |
| Table 5 | `weak_gmres_summary.tex` |
| Table 6 | `cg_summary.tex` |
| Table 7 | `refinement_combined.tex` |
| Table 8 | `fitted_global.tex` |
| Table 9 | `spatial_combined.tex` |
| Table 10 | `matrix_structure.tex` |
| Table 11 | `energy_summary.tex` |
| Table 12 | `matched_accuracy.tex` |
| Table 13 | `pure_neumann_summary.tex` |
| Table 14 | `depth_direct_jacobi.tex` |
| Table 15 | `depth_square_fdm_fem.tex` |
| Table 16 | `depth_lshape_errors.tex` |
| Table 17 | `depth_lshape_corner.tex` |
| Table 18 | `variable_sine.tex` |
| Table 19 | `variable_smooth.tex` |
| Table 20 | `variable_layered.tex` |
| Table 21 | `variable_interface.tex` |
| Table 22 | `variable_contrast.tex` |
| Table 23 | `variable_sine_fits.tex` |

Table 2 is the written claim/evidence plan authored in the report. `generated/report_values.tex` supplies macros, not another table. Supporting CSVs remain even when not displayed as separate tables. This entry point reproduces the numerical report body; historical appendix worksheets, audit records and contribution statements are not generated.

For use in Overleaf, preserve relative paths and load the required packages, including `amsmath`, `booktabs`, `graphicx` and `xurl`. The generated fragments are not standalone report chapters.

## Saved outputs and fresh runs

The included numerical outputs are retained together as the saved report dataset. A fresh run remeasures timings and records its actual environment in `data/run_metadata.json` and `data/depth_extension/run_metadata.json`. These identify the corresponding dataset's environment; a later verification environment must not be confused with an earlier archived run.

Timing values and cost ratios depend on the machine and system load. Last digits of errors, residuals, eigenvalue estimates and iteration counts can vary across libraries. Compare numerical quantities with suitable tolerances, not identical file hashes. If adopting a fresh dataset for the submitted report, update its timings, displayed values and environment description consistently. Do not mix selected outputs from different runs.

The report appendices summarise review procedures, findings and limitations. Separate historical audit worksheets/scripts, terminal logs and editable proof sources are not distributed in this repository. The numerical implementation, current automated tests and supporting numerical outputs are supplied for reproduction. The command does not read report templates or update report sources; the editable report is maintained in Overleaf.
