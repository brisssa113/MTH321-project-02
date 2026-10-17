from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

"""Additional mathematical cross-checks; no claim of human audit sign-off."""
import unittest
import numpy as np

from experiments import choose_accuracy_case
from models import assemble_system, run_case


def full_uneliminated_solution(N, boundary, closure):
    h = 1.0 / N
    beta = 0.0 if boundary == "neumann" else 2.0
    nodes = [(i, j) for j in range(N + 1) for i in range(N + 1)]
    if closure == "ghost":
        nodes += [(N + 1, j) for j in range(1, N)]
    numbering = {node: k for k, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)))
    rhs = np.zeros(len(nodes))
    for (i,j), k in numbering.items():
        if i == N + 1:
            matrix[k, numbering[N + 1,j]] = 1.0/(2*h)
            matrix[k, numbering[N - 1,j]] = -1.0/(2*h)
            matrix[k, numbering[N,j]] = beta
            rhs[k] = (1+beta)*np.exp(1+j*h)
        elif i == 0 or j in (0,N):
            matrix[k,k] = 1.0
            rhs[k] = np.exp((i+j)*h)
        elif i == N and closure == "first":
            matrix[k,k] = 1/h+beta
            matrix[k, numbering[i-1,j]] = -1/h
            rhs[k] = (1+beta)*np.exp(1+j*h)
        else:
            matrix[k,k] = 4/h**2
            for node in ((i-1,j),(i+1,j),(i,j-1),(i,j+1)):
                matrix[k, numbering[node]] = -1/h**2
            rhs[k] = -2*np.exp((i+j)*h)
    full = np.linalg.solve(matrix, rhs)
    return np.array([full[numbering[i,j]] for j in range(1,N) for i in range(1,N+1)])


class IndependentAssemblyTests(unittest.TestCase):
    def test_retained_Dirichlet_and_ghost_system(self):
        for N in (2,3,4,6):
            for boundary in ("neumann","robin"):
                for closure in ("first","ghost"):
                    with self.subTest(N=N,boundary=boundary,closure=closure):
                        full = full_uneliminated_solution(N,boundary,closure)
                        reduced = run_case(N,boundary,closure).solution
                        np.testing.assert_allclose(full,reduced,rtol=1e-11,atol=1e-11)

    def test_cubic_quartic_manufactured_defects(self):
        u = lambda x,y: 1+np.asarray(x)**3+np.asarray(y)**4
        f = lambda x,y: -6*np.asarray(x)-12*np.asarray(y)**2
        s = lambda y,b: 3+(0 if b=="neumann" else 2)*u(1,y)
        for N in (2,3,4,6):
            for boundary in ("neumann","robin"):
                for closure in ("first","ghost"):
                    with self.subTest(N=N,boundary=boundary,closure=closure):
                        a=assemble_system(N,boundary,closure,u_exact=u,f_source=f,s_boundary=s)
                        values=np.array([u(i/N,j/N) for j in range(1,N) for i in range(1,N+1)])
                        expected=np.array([
                            (-3/N+1/N**2 if closure=="first" else 2/N-2/N**2)
                            if i==N else -2/N**2
                            for j in range(1,N) for i in range(1,N+1)])
                        np.testing.assert_allclose(a.A@values-a.b,expected,rtol=1e-10,atol=1e-11)

    def test_fixed_target_selection_and_failure(self):
        rows=[{"N":8,"linf":0.02},{"N":32,"linf":0.003},{"N":16,"linf":0.007}]
        self.assertEqual(choose_accuracy_case(rows,0.01)["N"],16)
        self.assertIsNone(choose_accuracy_case(rows,0.001))
        with self.assertRaises(ValueError):
            choose_accuracy_case(rows,0)


if __name__ == "__main__":
    unittest.main()
