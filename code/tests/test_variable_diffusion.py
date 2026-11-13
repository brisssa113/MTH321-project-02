import unittest

import numpy as np
from scipy.sparse.linalg import spsolve

from code.variable_diffusion import assemble_fdm, assemble_fem, exact


class VariableDiffusionTests(unittest.TestCase):
    def test_shared_face_flux_and_material_interface(self):
        n = 15
        a, b = assemble_fdm(n, "layered", "harmonic")
        u = spsolve(a.tocsc(), b)
        x, y = np.meshgrid(np.arange(1,n)/n, np.arange(1,n)/n)
        self.assertLess(np.max(np.abs(u-exact(x,y,"layered").ravel())), 1e-11)
        arithmetic, arhs = assemble_fdm(n,"layered","arithmetic")
        bad = spsolve(arithmetic.tocsc(),arhs)
        self.assertGreater(np.max(np.abs(bad-exact(x,y,"layered").ravel())), 1e-3)

    def test_smooth_variable_fdm_fem_refine(self):
        errors = {}
        for method in ("fdm","fem"):
            errors[method] = []
            for n in (8,16):
                if method == "fdm":
                    a,b = assemble_fdm(n,"smooth")
                    x,y = np.meshgrid(np.arange(1,n)/n,np.arange(1,n)/n)
                    target = exact(x,y,"smooth").ravel()
                else:
                    mesh,_,a,b,_,free = assemble_fem(n,"smooth")
                    target = exact(mesh.p[0],mesh.p[1],"smooth")[free]
                errors[method].append(np.max(np.abs(spsolve(a.tocsc(),b)-target)))
            self.assertLess(errors[method][1],errors[method][0]/3)

    def test_contrast_sweep_harmonic_interface_and_bounds(self):
        n = 15
        for contrast in (2., 5., 10., 20., 50.):
            a, b = assemble_fdm(n, "layered", contrast=contrast)
            u = spsolve(a.tocsc(), b).reshape(n-1, n-1)
            x, y = np.meshgrid(np.arange(1,n)/n, np.arange(1,n)/n)
            self.assertLess(np.max(np.abs(u-exact(x,y,"layered",contrast))),1e-10)
            self.assertGreaterEqual(u.min(),-1e-12)
            self.assertLessEqual(u.max(),1+1e-12)
            k = n//2
            face = 2/(1+1/contrast)
            j = -face*n*(u[:,k]-u[:,k-1])
            self.assertLess(np.max(np.abs(j+1/(.5+.5/contrast))),1e-10)
        with self.assertRaises(ValueError):
            assemble_fdm(n,"layered",mean="physical")
