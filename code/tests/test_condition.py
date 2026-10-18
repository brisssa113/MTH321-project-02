from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import inverse_inf_norm_estimate
from models import assemble_system
from pure_neumann import assemble_pure_neumann, left_null_vector, scaled_correction_solve


class ConditionEstimateTests(unittest.TestCase):
    def test_inverse_norm_matches_dense_inverse(self):
        for boundary in ("neumann", "robin"):
            for closure in ("first", "ghost"):
                A = assemble_system(12, boundary, closure).A
                dense = np.abs(np.linalg.inv(A.toarray())).sum(axis=1).max()
                self.assertAlmostEqual(inverse_inf_norm_estimate(A) / dense, 1.0, places=8)

    def test_inverse_norm_is_bounded_uniformly_in_h(self):
        # The discrete maximum principle bounds ||A^{-1}||_inf independently of N.
        for closure in ("first", "ghost"):
            values = [inverse_inf_norm_estimate(assemble_system(N, "neumann", closure).A)
                      for N in (8, 16, 32, 64)]
            self.assertLess(max(values), 1.0)
            self.assertLess(max(values) / min(values), 1.2)


class ScaledCorrectionTests(unittest.TestCase):
    def test_scaled_correction_gives_consistent_mean_zero_solution(self):
        for closure in ("first", "ghost"):
            A, b = assemble_pure_neumann(8, closure)
            w = left_null_vector(A)
            self.assertLess(np.linalg.norm(A.T @ w, ord=np.inf), 1e-10)
            u, mu, _ = scaled_correction_solve(A, b)
            self.assertAlmostEqual(float(np.mean(u)), 0.0, places=12)
            self.assertLess(abs(mu), 1e-9)


if __name__ == "__main__":
    unittest.main()