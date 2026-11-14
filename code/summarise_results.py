from pathlib import Path
import csv
import io
import zipfile

root = Path(__file__).resolve().parents[1]
def rows(name):
    with (root/name).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))
main = rows('data/report_summary.csv')
jacobi = rows('data/depth_extension/direct_jacobi.csv')
square = rows('data/depth_extension/square_fdm_fem.csv')
lshape = rows('data/depth_extension/lshape_smooth_singular.csv')
quad = rows('data/depth_extension/lshape_quadrature.csv')
solver = rows('data/depth_extension/lshape_solver_check.csv')[0]
sine = rows('data/variable_diffusion/sine_required.csv')
fits = rows('data/variable_diffusion/sine_fitted_orders.csv')
interface = rows('data/variable_diffusion/interface_diagnostics.csv')
contrast = rows('data/variable_diffusion/contrast_sweep.csv')
layered = rows('data/variable_diffusion/layered.csv')
csv_names = sorted(p.relative_to(root).as_posix() for p in (root/'data').rglob('*.csv'))

def fmt(value):
    try:
        return f'{float(value):.7g}'
    except (ValueError, TypeError):
        return str(value)

def table(headers, records):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join('---' for _ in headers) + ' |'] +
                     ['| ' + ' | '.join(fmt(v) for v in record) + ' |' for record in records])

parts = ['''# Results summary — MTH321 Project 2

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
''']
parts.append(table(['Boundary', 'Closure', 'N', 'Discrete L2 error', 'Nodal maximum error', 'Fitted L2 order', 'Fitted maximum order'],
                   [[r[k] for k in ['boundary','closure','N','l2','linf','fitted_order_l2','fitted_order_linf']] for r in main]))
parts.append('''
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
''')
parts.append(table(['N','Jacobi iterations','Final residual (infinity norm)','Algebraic / discretisation error'],
                   [[r[k] for k in ['N','iterations','jacobi_residual_final','algebraic_over_discretisation']] for r in jacobi]))
parts.append(f"\nMaximum measured algebraic/discretisation ratio: {max(float(r['algebraic_over_discretisation']) for r in jacobi):.7g} (multiply by 100 to express as a percentage). Source: `data/depth_extension/direct_jacobi.csv`.")
parts.append('''
## 3. FDM versus P1 FEM on the square

Both methods use the same problem and mesh scale. The L2 comparison uses a
common piecewise-linear reconstruction on the triangular mesh. At N=64:
''')
parts.append(table(['Method','L2 error','Nodal maximum error','Last-pair L2 order'],
                   [[r[k] for k in ['method','L2','nodal_Linf','L2_order']] for r in square if r['N']=='64']))
parts.append('Source: `data/depth_extension/square_fdm_fem.csv`. These are approximately second-order L2 results for this benchmark.')
parts.append('''
## 4. Smooth versus corner-singular solutions on the L-shaped domain

These experiments use P1 FEM, whereas the original mixed-boundary study uses
FDM. The singular exact solution is continuous but has a singular gradient at
the re-entrant corner; it is not a discontinuous solution. Five refinement
levels are retained. At the finest level:
''')
parts.append(table(['Case','h max','Free DOF','L2 error','Energy error','Last-pair L2 order','Last-pair energy order'],
                   [[r[k] for k in ['kind','h','dof_free','L2','energy','L2_order','energy_order']] for r in lshape if r['level']=='5']))
parts.append('''
The smooth-case rates are observations on this mesh family. Smoothness of the
primal solution alone does not establish a general second-order L2 theorem on
a nonconvex domain. The singular case exhibits the expected reduced convergence.
Boundary values are imposed by nodal interpolation, u_h = I_h g.

Source: `data/depth_extension/lshape_smooth_singular.csv`.
The corner-energy fraction refers to whole triangles selected by their
centroids, not the exact integral over a circular neighbourhood.

### Error-integration and solver checks
''')
parts.append(table(['Quadrature order','Singular energy error','Centroid-selected patch fraction'],
                   [[r[k] for k in ['quadrature_order','energy','corner_fraction']] for r in quad]))
change = abs(float(quad[-1]['energy']) / float(quad[0]['energy']) - 1) * 100
parts.append(f'''
The order-{quad[0]['quadrature_order']} to order-{quad[-1]['quadrature_order']}
energy value changes by {change:.3f}%. This sensitivity check supports the
trend, but does not certify every printed digit or provide a rigorous
quadrature-error bound.

The finest-grid tight-CG/direct nodal difference is
{float(solver['nodal_difference_inf']):.7g}, with CG status {solver['cg_info']}.
Sources: `data/depth_extension/lshape_quadrature.csv` and
`data/depth_extension/lshape_solver_check.csv`.
''')
parts.append('''
## 5. Variable diffusion

### Prescribed smooth sine benchmark

On the unit square, a=1+x, u=sin(pi*x) sin(pi*y), g=0 and
f=2*pi^2*(1+x)*sin(pi*x)*sin(pi*y)-pi*cos(pi*x)*sin(pi*y).
Four grids N=8,16,32,64 are retained. Finest-grid results:
''')
parts.append(table(['Method','N','L2 error','Nodal maximum error','Last-pair L2 order','Last-pair maximum order'],
                   [[r[k] for k in ['method','N','error_l2','error_inf','order_l2','order_inf']] for r in sine if r['N']=='64']))
parts.append('''
Source: `data/variable_diffusion/sine_required.csv`. The separate exponential
manufactured-solution check remains in `data/variable_diffusion/smooth.csv`.
The order columns above are adjacent-grid orders. Least-squares fits of
log(error) against log(h) on N=16,32,64 are saved in `sine_fitted_orders.csv`.
FDM and FEM use different L2 reconstruction conventions here; compare their
convergence orders, not absolute L2 errors to rank methods.

### Two-layer medium: harmonic versus arithmetic face coefficients

At material contrast 10 the exact physical x-flux is J=-a*u_x=-20/11.
At N=63, the whole-interface diagnostics are:
''')
parts.append(table(['Method','Fit grids','Fitted L2 order','Fitted maximum order'], [[r[k] for k in ['method','fit_N','fitted_order_l2','fitted_order_inf']] for r in fits]))
parts.append(table(['Mean','Crossing-flux maximum error','Adjacent-face difference D_h','Two-sided reconstructed jump'],
                   [[r[k] for k in ['mean','interface_flux_error_max','adjacent_flux_jump_max','reconstructed_flux_jump_max']] for r in interface if r['N']=='63']))
parts.append(table(['Mean','Nodal maximum solution error'],
                   [[r[k] for k in ['mean','error_inf']] for r in layered if r['N']=='63']))
parts.append('''
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
''')
parts.append(table(['Contrast','CG iterations','Solution min','Solution max','Whole-interface flux error'],
                   [[r[k] for k in ['contrast','cg_iterations','solution_min','solution_max','interface_flux_error_max']] for r in contrast if r['N']=='63' and r['mean']=='harmonic']))
parts.append('''
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
''')
parts.extend('- `' + n + '`' for n in csv_names)
content = '\n\n'.join(parts).strip() + '\n'
destination = root / 'RESULTS_SUMMARY.md'
destination.write_bytes(content.encode('utf-8'))
print(destination)
