# Results summary — MTH321 Project 2

This summary describes the numerical CSV files retained in
the revised submission package. The complete numerical entry was run from a
clean extracted copy; its successful execution and all 22 unit tests are
recorded in `validation/`. This file is generated from the retained CSVs by
`python code/summarise_results.py`. All paths are relative to the package root.

## 1. Mixed-boundary Poisson: first-order and ghost-point closures

The finest reported grid is N=512. Fitted orders use N=128, 256 and 512.
The first-order closure gives approximately first-order global convergence;
the ghost-point closure gives approximately second-order convergence for both
Neumann and Robin right-boundary conditions.


| Boundary | Closure | N | Discrete L2 error | Nodal maximum error | Fitted L2 order | Fitted maximum order |
| --- | --- | --- | --- | --- | --- | --- |
| neumann | first | 512 | 0.0005101827 | 0.001701592 | 1.005942 | 1.004914 |
| neumann | ghost | 512 | 2.145891e-07 | 8.461432e-07 | 2.000044 | 1.999936 |
| robin | first | 512 | 0.0003132603 | 0.001036041 | 1.003579 | 1.002529 |
| robin | ghost | 512 | 1.232438e-07 | 5.212732e-07 | 2.000102 | 1.999943 |


Sources: `data/report_summary.csv`, `data/refinement_results.csv` and
`data/fitted_orders.csv`. Cost-at-matched-accuracy results are in
`data/matched_accuracy.csv`, with candidate grids and measurements in
`data/cost_accuracy_candidates.csv`. Timing ratios are machine- and run-dependent.

Weighted matrix diagnostics at N=32 report zero measured symmetry defect for
K=W A and a positive smallest eigenvalue in each tested configuration. This is
a finite set of numerical checks, not by itself a proof for every grid.
See `data/weighted_matrix_diagnostics.csv` and the report's energy argument.
Raw A need not be symmetric; do not confuse its symmetry defect with that of K.

The original suite also retains energy convergence, direct/GMRES comparisons,
weighted CG, independent FEM comparisons and pure-Neumann compatibility/gauge
checks; the complete CSV index is provided below.

## 2. Sparse direct versus Jacobi: square all-Dirichlet benchmark

This is a separate constant-coefficient benchmark. The residual target is
0.008 times the reference discretisation error; the discrete barrier bound
1/8 converts this to a 0.1% algebraic-error budget. This bound is not transferred
to the mixed-boundary or variable-coefficient systems.


| N | Jacobi iterations | Final residual (infinity norm) | Algebraic / discretisation error |
| --- | --- | --- | --- |
| 8 | 220 | 2.498243e-06 | 0.0002253327 |
| 16 | 940 | 1.096632e-06 | 0.0003936547 |
| 32 | 4070 | 2.6802e-07 | 0.000386186 |
| 64 | 17410 | 7.027679e-08 | 0.0004050647 |


Maximum measured algebraic/discretisation ratio: 0.0004050647 (multiply by 100 to express as a percentage). Source: `data/depth_extension/direct_jacobi.csv`.


## 3. FDM versus P1 FEM on the square

Both methods use the same problem and mesh scale. The L2 comparison uses a
common piecewise-linear reconstruction on the triangular mesh. At N=64:


| Method | L2 error | Nodal maximum error | Last-pair L2 order |
| --- | --- | --- | --- |
| FDM | 0.0002171471 | 8.785608e-06 | 1.999878 |
| FEM | 0.0002066167 | 1.757164e-05 | 2.00006 |

Source: `data/depth_extension/square_fdm_fem.csv`. These are approximately second-order L2 results for this benchmark.


## 4. Smooth versus corner-singular solutions on the L-shaped domain

These experiments use P1 FEM, whereas the original mixed-boundary study uses
FDM. The singular exact solution is continuous but has a singular gradient at
the re-entrant corner; it is not a discontinuous solution. Five refinement
levels are retained. At the finest level:


| Case | h max | Free DOF | L2 error | Energy error | Last-pair L2 order | Last-pair energy order |
| --- | --- | --- | --- | --- | --- | --- |
| smooth | 0.04419417 | 2945 | 0.0001876049 | 0.03378023 | 1.9996 | 0.9998144 |
| singular | 0.04419417 | 2945 | 0.001241667 | 0.05002623 | 1.357709 | 0.6539968 |


The smooth-case rates are observations on this mesh family. Smoothness of the
primal solution alone does not establish a general second-order L2 theorem on
a nonconvex domain. The singular case exhibits the expected reduced convergence.
Boundary values are imposed by nodal interpolation, u_h = I_h g.

Source: `data/depth_extension/lshape_smooth_singular.csv`.
The corner-energy fraction refers to whole triangles selected by their
centroids, not the exact integral over a circular neighbourhood.

### Error-integration and solver checks


| Quadrature order | Singular energy error | Centroid-selected patch fraction |
| --- | --- | --- |
| 12 | 0.05002623 | 0.8428698 |
| 16 | 0.05009378 | 0.8432933 |
| 19 | 0.05013744 | 0.8435661 |


The order-12 to order-19
energy value changes by 0.222%. This sensitivity check supports the
trend, but does not certify every printed digit or provide a rigorous
quadrature-error bound.

The finest-grid tight-CG/direct nodal difference is
7.205347e-14, with CG status 0.
Sources: `data/depth_extension/lshape_quadrature.csv` and
`data/depth_extension/lshape_solver_check.csv`.



## 5. Variable diffusion

### Prescribed smooth sine benchmark

On the unit square, a=1+x, u=sin(pi*x) sin(pi*y), g=0 and
f=2*pi^2*(1+x)*sin(pi*x)*sin(pi*y)-pi*cos(pi*x)*sin(pi*y).
Four grids N=8,16,32,64 are retained. Finest-grid results:


| Method | N | L2 error | Nodal maximum error | Last-pair L2 order | Last-pair maximum order |
| --- | --- | --- | --- | --- | --- |
| FDM | 64 | 9.966202e-05 | 0.0002002977 | 2.000541 | 1.99955 |
| FEM | 64 | 0.0003364855 | 0.0002020744 | 1.998365 | 1.999237 |


Source: `data/variable_diffusion/sine_required.csv`. The separate exponential
manufactured-solution check remains in `data/variable_diffusion/smooth.csv`.
The order columns above are adjacent-grid orders. Least-squares fits of
log(error) against log(h) on N=16,32,64 are saved in `sine_fitted_orders.csv`.
FDM and FEM use different L2 reconstruction conventions here; compare their
convergence orders, not absolute L2 errors to rank methods.

### Two-layer medium: harmonic versus arithmetic face coefficients

At material contrast 10 the exact physical x-flux is J=-a*u_x=-20/11.
At N=63, the whole-interface diagnostics are:


| Method | Fit grids | Fitted L2 order | Fitted maximum order |
| --- | --- | --- | --- |
| FDM | 16,32,64 | 2.001353 | 2.00006 |
| FEM | 16,32,64 | 1.995934 | 1.995687 |

| Mean | Crossing-flux maximum error | Adjacent-face difference D_h | Two-sided reconstructed jump |
| --- | --- | --- | --- |
| harmonic | 1.207923e-13 | 3.497203e-13 | 3.568257e-13 |
| arithmetic | 0.6618507 | 0.4279642 | 4.196643e-13 |

| Mean | Nodal maximum solution error |
| --- | --- |
| harmonic | 6.661338e-15 |
| arithmetic | 0.009599576 |


Use `data/variable_diffusion/interface_diagnostics.csv` for the full-interface
flux error, adjacent-face difference and separately reconstructed jump.
The two traces use the nearest two nodes wholly in each material, extrapolate
their constant slopes to x=1/2, and use the same +x orientation. A small jump
does not establish accurate flux magnitude. `interface_flux_profiles.csv`
retains every sampled trace. The arithmetic maximum is near an endpoint on
the tested grids; no limiting constant has been proved. The older `layered.csv`
`interface_flux` and `flux_error` fields are a single-location diagnostic
using positive a*u_x; they must not be presented as whole-interface physical
flux or as a flux jump. Continuity and accuracy are different checks.

Exact nodal values satisfy the harmonic discrete equations in exact arithmetic;
the computed values still contain floating-point and algebraic error. Arithmetic
averaging does not reproduce those exact nodal values. This is specific to the benchmark
and its interface placement, not a universal exactness statement.

### Material-contrast sweep

`data/variable_diffusion/contrast_sweep.csv` contains 48 records: contrasts
1,2,5,10,20,50; N=7,15,31,63; and both averaging rules. It records iteration
counts, solution extrema, nodal and whole-interface flux errors, direct/CG
differences, and residuals. `nodal_error_inf` is the direct-solve error;
`cg_nodal_error_inf` is the independently recorded CG error. The CG iteration
count must not be paired with the direct error as if they describe one solve.
Selected finest-grid harmonic results:


| Contrast | CG iterations | Solution min | Solution max | Whole-interface flux error |
| --- | --- | --- | --- | --- |
| 1 | 197 | 0 | 1 | 2.109424e-14 |
| 2 | 196 | 0 | 1 | 3.264056e-14 |
| 5 | 205 | 0 | 1 | 1.223466e-13 |
| 10 | 204 | 0 | 1 | 1.207923e-13 |
| 20 | 203 | 0 | 1 | 1.398881e-13 |
| 50 | 202 | 0 | 1 | 1.303402e-13 |


## 6. Reproduction and interpretation

From the extracted package root:

```text
python -m pip install -r requirements.txt
python -m unittest discover -s code/tests -v
python code/run_all.py
```

Environment records: `data/run_metadata.json` and
`data/depth_extension/run_metadata.json`. The modules record separate metadata; `validation/execution.json` records
the successful clean full-entry run that regenerated the current results.
Rerunning may change floating-point last digits and timings.

Raw residuals from differently scaled FDM and FEM systems are not directly
comparable. Use the associated scaled/backward-error diagnostics and solver
cross-checks. Inverse-norm estimates are diagnostic estimates rather than
automatically certified forward-error bounds. Historical five-gate audit
material is separate from these numerical experiments and is not regenerated
by `code/run_all.py`.

This file summarises saved results; it is not an assertion that every optional
brief requirement or every theoretical claim has received a new audit.

## 7. Complete archived CSV index


- `data/algebraic_error_separation.csv`

- `data/cg_comparison.csv`

- `data/condition_estimates.csv`

- `data/cost_accuracy_candidates.csv`

- `data/depth_extension/direct_jacobi.csv`

- `data/depth_extension/lshape_quadrature.csv`

- `data/depth_extension/lshape_smooth_singular.csv`

- `data/depth_extension/lshape_solver_check.csv`

- `data/depth_extension/square_fdm_fem.csv`

- `data/energy_convergence.csv`

- `data/error_locations.csv`

- `data/fem_oracle_comparison.csv`

- `data/fitted_orders.csv`

- `data/matched_accuracy.csv`

- `data/pure_neumann_depth.csv`

- `data/refinement_results.csv`

- `data/report_summary.csv`

- `data/variable_diffusion/contrast_sweep.csv`

- `data/variable_diffusion/interface_diagnostics.csv`

- `data/variable_diffusion/interface_flux_profiles.csv`

- `data/variable_diffusion/layered.csv`

- `data/variable_diffusion/sine_fitted_orders.csv`

- `data/variable_diffusion/sine_required.csv`

- `data/variable_diffusion/smooth.csv`

- `data/weak_gmres_comparison.csv`

- `data/weighted_matrix_diagnostics.csv`
