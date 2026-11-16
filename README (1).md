# MTH321 Project 2 - Mixed-Boundary Poisson Problem

**Group N:** Kainan Zhao, Rengerile Na and Suning Peng.

This project studies Direction 2: a Poisson problem on the unit square with Dirichlet data on three sides and Neumann or Robin data on the open right edge. It compares a first-order backward-derivative closure (FO) with a ghost-point boundary PDE closure (G).

The supplementary studies cover direct/iterative solvers, square FDM/FEM comparisons, smooth and corner-singular solutions on an L-shaped domain, and variable diffusion. Each supplementary benchmark has its own boundary conditions and error definitions.

- [Report PDF](report/Group%20N%20report.pdf)
- [Numerical entry point](code/run_all.py)
- [Pinned dependencies](requirements.txt)
- [Presentation](slides/GroupN.pdf)

The complete editable report is maintained in Overleaf. The numerical workflow uses Python inputs and runs independently of the report's LaTeX source.

## 1. Run from a downloaded copy

1. On GitHub, select **Code > Download ZIP**, then extract the ZIP.
2. Open PowerShell or a terminal in the extracted project folder. This is the folder containing `requirements.txt` and `code/`.
3. Use **Python 3.12 or 3.13**. Confirm the version before creating the virtual environment, then follow the commands for your operating system below.

The environment isolates this project's dependencies from other Python projects. The explicit executable paths below do not require environment activation.

### Windows / PowerShell

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s code/tests -v
.\.venv\Scripts\python.exe code/run_all.py
```

### macOS / Linux

```bash
python3 --version
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m unittest discover -s code/tests -v
.venv/bin/python code/run_all.py
```

The dependency versions in `requirements.txt` are:

| Package | Version |
| --- | --- |
| NumPy | 2.1.3 |
| SciPy | 1.15.3 |
| Matplotlib | 3.10.0 |
| scikit-fem | 12.0.2 |

The test command should finish with `Ran 22 tests` and `OK`. The numerical command should finish with `All checks passed.` and report **18 PNG figures, 22 numerical LaTeX tables and 25 CSV files**.

The full calculation includes N=512 solves, repeated timing measurements, FEM comparisons and the supplementary benchmarks. Runtime depends on the machine; some stages produce no intermediate terminal output. Plots are saved to files, so no plot window is expected.

After dependency installation, the numerical workflow requires no external dataset, TeX installation, Overleaf access or further network connection. Existing `data/`, `figures/` and `generated/` outputs are not required as inputs.

### Optional dependency preflight

Before the full calculation, check that the required Python packages are discoverable:

```powershell
# Windows
.\.venv\Scripts\python.exe code/run_all.py --check-inputs
```

```bash
# macOS / Linux
.venv/bin/python code/run_all.py --check-inputs
```

This preflight checks package availability. The test suite and full numerical run perform the numerical checks.

## 2. Outputs and overwrite behaviour

A successful run publishes the following outputs in the project folder:

| Location | Generated contents |
| --- | --- |
| `figures/` | 18 numerical PNGs used in Figures 2-19 |
| `generated/` | 22 numerical table fragments and `report_values.tex` macros |
| `data/` | 25 CSVs, including the `depth_extension/` and `variable_diffusion/` subfolders |
| `data/run_metadata.json` | Main-run environment, solver and timing settings |
| `data/depth_extension/run_metadata.json` | Supplementary-run environment and settings |

There are **68 generated files** in total: 18 PNGs, 22 tables, one macro file, 25 CSVs and two metadata JSON files. The inventory is defined in [code/report_outputs.py](code/report_outputs.py).

The command computes in a temporary `.mth321-run-*` folder and checks the expected inventory before publishing. A computation or validation failure leaves the previously published outputs in place. Successful publication replaces the numerical outputs and removes the specifically listed obsolete exports, including `generated/raw_tables/` and the former `RESULTS_SUMMARY.md`.

**A fresh run replaces saved timings and records the new environment.** Keep a separate copy of the submitted dataset if you want to compare it with a new run.

The entry point generates the numerical material used in the report body. Figure 1 is an authored boundary schematic; Table 2 is the written claim/evidence plan. These are maintained in the report. The command does not rebuild the report PDF, presentation, historical appendix audit records, AI Transparency Log or Individual Contribution Statements.

## 3. Repository layout and implementation

| Path | Purpose |
| --- | --- |
| `code/models.py` | Main index map, sparse assembly, boundary closures, solve and diagnostics |
| `code/experiments.py` | Refinement, timing, accuracy targets, solver comparisons and conditioning checks |
| `code/fem_oracle.py` | Independent P1 FEM assembly of the mixed weak formulation |
| `code/pure_neumann.py` | All-flux extension, null-space checks, compatibility and mean-zero gauge |
| `code/depth_benchmarks.py` | Direct/Jacobi, square FDM/FEM, L-shaped and quadrature/solver checks |
| `code/variable_diffusion.py` | Sine, exponential and layered-material benchmarks |
| `code/make_figures.py`, `code/reporting.py`, `code/body_tables.py` | Plots, numerical tables and macros |
| `code/run_all.py`, `code/report_outputs.py` | Complete workflow and output inventory |
| `code/tests/` | 22 automated test methods |
| `report/` | Report PDF |
| `slides/` | Presentation materials |

The main manufactured solution is `u(x,y)=exp(x+y)`. Neumann data use beta=0 and Robin data use beta=2. Main refinement uses N=16,32,64,128,256,512; fits use the finest three levels. The cost study also measures N=4 and N=8 and selects the coarsest tested grid meeting each prescribed error threshold.

The square supplementary comparison is all-Dirichlet. Its rows do not replace the main mixed-boundary results. The variable-diffusion sine fits use N=16,32,64 and the report distinguishes the FDM discrete nodal L2 norm from the continuous P1 FEM L2 error.

## 4. Figure-to-file and evidence index

Running the single entry point above regenerates every numerical figure listed below. CSV paths in the last column are relative to `data/`. Some field and mesh plots are constructed directly during the run rather than read from a saved CSV.

| Report figure | File in `figures/` | Supporting data or construction |
| --- | --- | --- |
| 2 | `fem_oracle.png` | `fem_oracle_comparison.csv` |
| 3 | `tolerance_sweep.png` | `algebraic_error_separation.csv` |
| 4 | `convergence.png` | `refinement_results.csv`, `fitted_orders.csv` |
| 5 | `boundary_vs_interior.png` | `refinement_results.csv`, `fitted_orders.csv` |
| 6 | `solution_error.png` | Recomputed Robin solution and errors at N=64 in `reporting.py` |
| 7 | `matrix_spy.png` | Reassembled Robin FO/G matrices at N=32 in `make_figures.py` |
| 8 | `weighted_energy_convergence.png` | `energy_convergence.csv` |
| 9 | `matched_accuracy.png` | `matched_accuracy.csv` |
| 10 | `cost_vs_accuracy.png` | `cost_accuracy_candidates.csv`; all eight candidate grids |
| 11 | `pure_neumann_depth.png` | `pure_neumann_depth.csv` |
| 12 | `square_fdm_fem.png` | `depth_extension/square_fdm_fem.csv` |
| 13 | `lshape_rates.png` | `depth_extension/lshape_smooth_singular.csv` |
| 14 | `lshape_vs_dof.png` | `depth_extension/lshape_smooth_singular.csv` |
| 15 | `lshape_mesh_level1.png` | `MeshTri.init_lshaped().refined(1)` in `depth_benchmarks.py` |
| 16 | `lshape_mesh_level5.png` | `MeshTri.init_lshaped().refined(5)` in `depth_benchmarks.py` |
| 17 | `layered_solution_cut.png` | Recomputed layered solutions at N=63, contrast 10, in `variable_diffusion.py` |
| 18 | `layered_interface_flux.png` | Recomputed whole-interface fluxes; `variable_diffusion/interface_flux_profiles.csv` retains the profiles |
| 19 | `layered_contrast_iterations.png` | `variable_diffusion/contrast_sweep.csv`, N=63 |

## 5. Table-to-file index

Table numbers follow the report body. Files below are in `generated/`.

| Report table | Generated file |
| --- | --- |
| 1 | `fo_bounds.tex` |
| 3 | `fem_oracle_summary.tex` |
| 4 | `gmres_summary.tex` |
| 5 | `weak_gmres_summary.tex` |
| 6 | `cg_summary.tex` |
| 7 | `refinement_combined.tex` |
| 8 | `fitted_global.tex` |
| 9 | `spatial_combined.tex` |
| 10 | `matrix_structure.tex` |
| 11 | `energy_summary.tex` |
| 12 | `matched_accuracy.tex` |
| 13 | `pure_neumann_summary.tex` |
| 14 | `depth_direct_jacobi.tex` |
| 15 | `depth_square_fdm_fem.tex` |
| 16 | `depth_lshape_errors.tex` |
| 17 | `depth_lshape_corner.tex` |
| 18 | `variable_sine.tex` |
| 19 | `variable_smooth.tex` |
| 20 | `variable_layered.tex` |
| 21 | `variable_interface.tex` |
| 22 | `variable_contrast.tex` |
| 23 | `variable_sine_fits.tex` |

`generated/report_values.tex` contains numerical macros and is not another table. The supporting CSVs also retain quantities discussed in the text, such as residuals, condition estimates, quadrature sensitivity and solver cross-checks.

For Overleaf use, copy the numerical outputs while preserving relative paths. The generated fragments require packages such as `amsmath`, `booktabs`, `graphicx` and `xurl`; they are not standalone report chapters. Replacing a generated table environment can change its label or caption, so keep manuscript references consistent when integrating it.

## 6. Validation and interpretation

The 22 tests cover boundary rows and corners, manufactured defects, independent assembly, observed convergence, weighted matrix and energy identities, inverse-norm estimation, FEM, pure-Neumann compatibility, and variable-diffusion forcing, symmetry and interface reconstruction.

Two loose-tolerance G/Jacobi-GMRES cases deliberately demonstrate failure of the one-percent algebraic-error target despite successful solver exit. These recorded accuracy failures are part of the verification experiment. Tight settings and the other stated checks must pass for the complete entry point to finish successfully.

A separate clean validation of the numerical source snapshot completed on Linux with Python 3.12.14 and the pinned dependencies. All 22 tests passed; the complete entry point reported `All checks passed.` and generated the declared 68-file inventory. All 18 PNGs decoded successfully. The clean copy did not require saved numerical outputs or report source files.

The saved report dataset records Windows 11 and Python 3.12.14 in its metadata. That saved-data environment is distinct from the separate validation environment. The report's numerical code and saved-data freeze is [ce53d80579ac6c59cf992f6df4f13cb0f0e1dcf2](https://github.com/brisssa113/MTH321-project-02/tree/ce53d80579ac6c59cf992f6df4f13cb0f0e1dcf2); later report or documentation updates can have a different commit.

### Reproducibility scope

- The weighted-energy random-vector and eigenvalue checks in `experiments.py` use a fixed generator seed of 321 and an explicit starting vector.
- The auxiliary `onenormest` call does not explicitly seed its internal sampling. The spectral estimates in `variable_diffusion.py` use `eigsh` without explicitly fixing its starting vectors. The implementation therefore does not fix every random element used by auxiliary estimators.
- These estimators do not set the PDE coefficients, boundary data or prescribed solver tolerances. Conditioning estimates are supplementary diagnostics; the report distinguishes them from proved stability bounds.
- Fresh runs remeasure wall-clock costs. Floating-point errors, residuals, diagnostic estimates and iteration counts can also vary with the numerical environment. Reproduction should be assessed through the stated norms, convergence trends, residuals and accuracy checks, rather than identical file hashes or timings.

Keep a run's CSVs, figures, tables and metadata together. If a fresh dataset is adopted for the report, update the displayed values, timing discussion and environment description consistently.

The report appendices contain the AI Transparency Log, member contribution statements and summaries of historical review procedures. Separate historical audit scripts, worksheets, terminal logs and editable proof sources are not distributed in this numerical repository. The executable tests above verify the current implementation; they do not reconstruct a historical audit session.
