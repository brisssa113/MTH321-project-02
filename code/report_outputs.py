"""Numerical outputs used in the report body, excluding every appendix export.

Numbers identify the supplied report PDF; LaTeX itself controls numbering.
Figure 1 and Table 2 are authored schematic/text material, not numerical output.
"""

FIGURES = {
    "fem_oracle.png": 2,
    "tolerance_sweep.png": 3,
    "convergence.png": 4,
    "boundary_vs_interior.png": 5,
    "solution_error.png": 6,
    "matrix_spy.png": 7,
    "weighted_energy_convergence.png": 8,
    "matched_accuracy.png": 9,
    "cost_vs_accuracy.png": 10,
    "pure_neumann_depth.png": 11,
    "square_fdm_fem.png": 12,
    "lshape_rates.png": 13,
    "lshape_vs_dof.png": 14,
    "lshape_mesh_level1.png": 15,
    "lshape_mesh_level5.png": 16,
    "layered_solution_cut.png": 17,
    "layered_interface_flux.png": 18,
    "layered_contrast_iterations.png": 19,
}

TABLES = {
    "fo_bounds.tex": 1,
    "fem_oracle_summary.tex": 3,
    "gmres_summary.tex": 4,
    "weak_gmres_summary.tex": 5,
    "cg_summary.tex": 6,
    "refinement_combined.tex": 7,
    "fitted_global.tex": 8,
    "spatial_combined.tex": 9,
    "matrix_structure.tex": 10,
    "energy_summary.tex": 11,
    "matched_accuracy.tex": 12,
    "pure_neumann_summary.tex": 13,
    "depth_direct_jacobi.tex": 14,
    "depth_square_fdm_fem.tex": 15,
    "depth_lshape_errors.tex": 16,
    "depth_lshape_corner.tex": 17,
    "variable_sine.tex": 18,
    "variable_smooth.tex": 19,
    "variable_layered.tex": 20,
    "variable_interface.tex": 21,
    "variable_contrast.tex": 22,
    "variable_sine_fits.tex": 23,
}

CSV_FILES = (
    "algebraic_error_separation.csv", "cg_comparison.csv",
    "condition_estimates.csv", "cost_accuracy_candidates.csv",
    "energy_convergence.csv", "error_locations.csv",
    "fem_oracle_comparison.csv", "fitted_orders.csv", "matched_accuracy.csv",
    "pure_neumann_depth.csv", "refinement_results.csv",
    "weak_gmres_comparison.csv", "weighted_matrix_diagnostics.csv",
    "depth_extension/direct_jacobi.csv",
    "depth_extension/lshape_quadrature.csv",
    "depth_extension/lshape_smooth_singular.csv",
    "depth_extension/lshape_solver_check.csv",
    "depth_extension/square_fdm_fem.csv",
    "variable_diffusion/contrast_sweep.csv",
    "variable_diffusion/interface_diagnostics.csv",
    "variable_diffusion/interface_flux_profiles.csv",
    "variable_diffusion/layered.csv",
    "variable_diffusion/sine_fitted_orders.csv",
    "variable_diffusion/sine_required.csv",
    "variable_diffusion/smooth.csv",
)

METADATA_FILES = ("run_metadata.json", "depth_extension/run_metadata.json")

# Remove only known obsolete generator outputs after a successful new run.
LEGACY_FILES = (
    "figures/cg_solver_comparison.png", "data/report_summary.csv",
    "RESULTS_SUMMARY.md", "generated/condition_summary.tex",
    "generated/cost_findings.tex", "generated/report_summary.tex",
    "generated/section5_summary.tex", "generated/spatial_neumann.tex",
    "generated/spatial_robin.tex", "generated/refinement_neumann_first.tex",
    "generated/refinement_neumann_ghost.tex",
    "generated/refinement_robin_first.tex",
    "generated/refinement_robin_ghost.tex",
)


def expected_paths():
    """Return the complete set of files a successful numerical run publishes."""
    return (
        {"figures/" + name for name in FIGURES}
        | {"generated/" + name for name in TABLES}
        | {"generated/report_values.tex"}
        | {"data/" + name for name in CSV_FILES + METADATA_FILES}
    )
