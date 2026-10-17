# MTH321 Project 2 — Mixed-Boundary Poisson Numerical Study

This repository contains the numerical implementation and reproducible results for the mixed-boundary Poisson study. It provides the solver, numerical experiments, saved data, figures, and LaTeX material for the implementation and experiments sections. The complete group report is maintained separately in Overleaf.

## Repository layout

| Path | Purpose |
|---|---|
| `code/run_all.py` | Single entry point for regenerating the numerical results |
| `code/models.py` | Discretisation and linear-system assembly |
| `code/experiments.py` | Numerical experiments and checks |
| `code/fem_oracle.py` | Independent finite-element comparison |
| `code/pure_neumann.py` | Pure-Neumann extension |
| `code/make_figures.py`, `code/reporting.py` | Figure and LaTeX-table generation |
| `code/tests/` | Automated tests |
| `data/` | Saved CSV results and run metadata |
| `figures/` | Generated figures used in the report |
| `generated/` | Report-ready LaTeX tables and optional full CSV-to-LaTeX tables |
| `report/` | Implementation and experiments LaTeX sections |
| `requirements.txt` | Python dependencies |

## Reproduce the results

Run these commands from the repository root with Python 3.12 or 3.13:

```bash
python -m pip install -r requirements.txt
python code/run_all.py
python -m unittest discover -s code/tests -v
```

The run updates `data/`, `figures/`, `generated/`, and `RESULTS_SUMMARY.md`. The saved run's software and platform information is recorded in `data/run_metadata.json`. Measured timings depend on the machine and may differ when regenerated.

## Using the report material

`report/implementation.tex` and `report/experiments.tex` are sections of a larger report, not standalone LaTeX documents. They refer to equations and references defined in the group report. When integrating them into Overleaf, keep `data/`, `figures/`, and `generated/` next to the report's main `.tex` file so relative paths resolve. The concise tables used by the sections are in `generated/`; `generated/raw_tables/` contains full conversions of the CSV data for optional appendix use.

The implementation includes mixed-boundary finite-difference closures. The experiments cover refinement, error behaviour, matrix structure, solver performance, an independent finite-element comparison, and a pure-Neumann extension. Numerical observations and measured timings should be interpreted separately from theoretical proofs.
