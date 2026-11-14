import unittest

import numpy as np
from scipy.sparse.linalg import spsolve

from code.variable_diffusion import (assemble_fdm, assemble_fem, exact,
    forcing, reconstructed_interface_fluxes, symmetry_defect)


class VariableDiffusionTests(unittest.TestCase):
    def test_prescribed_sine_forcing_by_independent_flux_derivative(self):
        # Differentiate independently specified a*grad(u), without using forcing.
        x=np.array([.13,.27,.51,.86]); y=np.array([.21,.44,.66,.79])
        eps=1e-25
        def qx(x,y): return (1+x)*np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)
        def qy(x,y): return (1+x)*np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)
        independent=-(qx(x+1j*eps,y).imag+qy(x,y+1j*eps).imag)/eps
        np.testing.assert_allclose(forcing(x,y,"smooth_sine"),independent,rtol=2e-14)
        np.testing.assert_allclose(exact(np.zeros_like(y),y,"smooth_sine"),0,atol=1e-15)

    def test_prescribed_sine_convergence_and_spd(self):
        for method in ("fdm","fem"):
            errors=[]
            for n in (8,16):
                if method=="fdm":
                    a,b=assemble_fdm(n,"smooth_sine")
                    x,y=np.meshgrid(np.arange(1,n)/n,np.arange(1,n)/n)
                    target=(np.sin(np.pi*x)*np.sin(np.pi*y)).ravel()
                else:
                    mesh,_,a,b,_,free=assemble_fem(n,"smooth_sine")
                    target=(np.sin(np.pi*mesh.p[0])*np.sin(np.pi*mesh.p[1]))[free]
                self.assertLess(symmetry_defect(a),1e-14)
                if n==8: self.assertGreater(np.linalg.eigvalsh(a.toarray())[0],0)
                errors.append(np.max(np.abs(spsolve(a.tocsc(),b)-target)))
            self.assertGreater(errors[0]/errors[1],3.5)
            self.assertLess(errors[0]/errors[1],4.5)

    def test_material_side_trace_reconstruction_has_common_orientation(self):
        n=15; x,y=np.meshgrid(np.arange(1,n)/n,np.arange(1,n)/n)
        # Deliberately unequal side slopes; independent expected flux jump=4.
        field=np.where(x<.5,2*x,1+3*(x-.5)) + .2*y
        left,right=reconstructed_interface_fluxes(field,n,contrast=2)
        np.testing.assert_allclose(left,-2,atol=1e-13)
        np.testing.assert_allclose(right,-6,atol=1e-13)
        self.assertAlmostEqual(float(np.max(np.abs(right-left))),4)

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
