from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fem_oracle import solve_mixed_fem
from models import exact_solution, run_case
from pure_neumann import assemble_pure_neumann, continuum_compatibility_quadrature, gauge_solve


class DepthExperimentTests(unittest.TestCase):
    def test_fem_oracle_refines_independently(self):
        errors = []
        for N in (8, 16):
            solution = solve_mixed_fem(N, "robin")
            xx, yy = np.meshgrid(np.linspace(0, 1, N + 1), np.linspace(0, 1, N + 1))
            errors.append(np.max(np.abs(solution - exact_solution(xx, yy))))
            self.assertLess(np.max(np.abs(solution - run_case(N, "robin", "ghost").full_solution)), 0.01)
        self.assertLess(errors[1], errors[0] / 2.0)

    def test_pure_neumann_nullspace_and_gauge(self):
        for closure in ("first", "ghost"):
            with self.subTest(closure=closure):
                A, b = assemble_pure_neumann(8, closure)
                np.testing.assert_allclose(A @ np.ones(A.shape[0]), 0, atol=1e-12)
                u, defect, lagrange, left_defect = gauge_solve(A, b)
                self.assertAlmostEqual(float(np.mean(u)), 0.0, places=12)
                self.assertAlmostEqual(defect, lagrange, places=10)
                self.assertLess(left_defect, 1e-10)
                self.assertLess(np.linalg.norm(A @ u + lagrange - b, ord=np.inf), 1e-9)

    def test_ghost_compatibility_matches_trapezoid(self):
        for N in (8, 16):
            A, b = assemble_pure_neumann(N, "ghost")
            _, defect, _, _ = gauge_solve(A, b)
            self.assertAlmostEqual(defect, continuum_compatibility_quadrature(N), places=10)
