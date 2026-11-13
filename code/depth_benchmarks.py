"""Reproducible matched comparisons and L-shaped regularity experiment."""
import csv
import json
import math
import platform
import time
from pathlib import Path
from importlib.metadata import version

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import cg, spsolve
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, MeshTri, asm, condense
from skfem.helpers import dot, grad


@BilinearForm
def laplace(u, v, w):
    return dot(grad(u), grad(v))


@LinearForm
def square_load(v, w):
    return -2.0 * np.exp(w.x[0] + w.x[1]) * v


def square_exact(x, y):
    return np.exp(x + y)


def square_fdm(n):
    """Five-point -Delta, exact Dirichlet values; n squares per side."""
    start = time.perf_counter()
    h = 1.0 / n
    m = n - 1
    axis = np.arange(1, n) * h
    x, y = np.meshgrid(axis, axis, indexing="xy")
    t = diags([-np.ones(m-1), 2*np.ones(m), -np.ones(m-1)], [-1, 0, 1], format="csr")
    a = (kron(eye(m, format="csr"), t) + kron(t, eye(m, format="csr"))).tocsr() / h**2
    b = -2.0 * square_exact(x, y)
    b[:, 0] += square_exact(0, y[:, 0]) / h**2
    b[:, -1] += square_exact(1, y[:, -1]) / h**2
    b[0, :] += square_exact(x[0, :], 0) / h**2
    b[-1, :] += square_exact(x[-1, :], 1) / h**2
    assembly = time.perf_counter() - start
    start = time.perf_counter()
    u = spsolve(a, b.ravel())
    solve_time = time.perf_counter() - start
    nodes = np.linspace(0, 1, n+1)
    xx, yy = np.meshgrid(nodes, nodes, indexing="xy")
    full = square_exact(xx, yy)
    full[1:-1, 1:-1] = u.reshape((m, m))
    return a, b.ravel(), u, full, assembly, solve_time


def square_fem(n):
    start = time.perf_counter()
    points = np.linspace(0, 1, n+1)
    mesh = MeshTri.init_tensor(points, points)
    basis = Basis(mesh, ElementTriP1(), intorder=10)
    a = asm(laplace, basis).tocsr()
    b = asm(square_load, basis)
    boundary = mesh.boundary_nodes()
    known = square_exact(mesh.p[0], mesh.p[1])
    ac, bc, x, free = condense(a, b, x=known.copy(), D=boundary)
    assembly = time.perf_counter() - start
    start = time.perf_counter()
    free_u = spsolve(ac.tocsc(), bc)
    solve_time = time.perf_counter() - start
    x[free] = free_u
    return mesh, basis, ac, bc, free_u, x, assembly, solve_time


def l2_error(basis, nodal, exact):
    coords = basis.global_coordinates()
    diff = basis.interpolate(nodal).value - exact(coords[0], coords[1])
    return float(np.sqrt(np.sum(diff**2 * basis.dx)))


def orders(rows, key):
    for i, row in enumerate(rows):
        if i == 0:
            row[key + "_order"] = ""
        else:
            prev = rows[i-1]
            row[key + "_order"] = math.log(prev[key] / row[key]) / math.log(prev["h"] / row["h"])


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def square_comparison(root):
    rows = []
    for n in (8, 16, 32, 64):
        a, b, u, full, fa, fs = square_fdm(n)
        mesh, basis, ac, bc, fu, fn, ea, es = square_fem(n)
        # MeshTri.init_tensor orders vertices by x-major grid index.
        fdm_nodal = full.T.ravel()
        assert np.max(np.abs(mesh.p[0] - np.repeat(np.linspace(0, 1, n+1), n+1))) < 1e-14
        assert np.max(np.abs(mesh.p[1] - np.tile(np.linspace(0, 1, n+1), n+1))) < 1e-14
        for method, nodal, mat, rhs, interior, assembly, solve in (
            ("FDM", fdm_nodal, a, b, u, fa, fs),
            ("FEM", fn, ac, bc, fu, ea, es),
        ):
            rows.append(dict(method=method, N=n, h=1/n,
                             dof_total=len(nodal), dof_free=len(interior), nnz=mat.nnz,
                             assembly_s=assembly, solve_s=solve,
                             residual_inf=float(np.linalg.norm(mat @ interior-rhs, np.inf)),
                             L2=l2_error(basis, nodal, square_exact),
                             nodal_Linf=float(np.max(np.abs(nodal-square_exact(mesh.p[0], mesh.p[1]))))))
    for method in ("FDM", "FEM"):
        selected = [r for r in rows if r["method"] == method]
        for key in ("L2", "nodal_Linf"):
            orders(selected, key)
    write_csv(root / "data" / "depth_extension" / "square_fdm_fem.csv", rows)
    return rows


def jacobi(a, b, target, maxiter=200000):
    x = np.zeros_like(b)
    invdiag = 1 / a.diagonal()
    initial = float(np.linalg.norm(b, np.inf))
    for k in range(1, maxiter+1):
        residual = b - a @ x
        x += invdiag * residual
        if k % 10 == 0 or k == 1:
            rn = float(np.linalg.norm(b-a @ x, np.inf))
            if rn <= target:
                return x, k, initial, rn
    raise RuntimeError(f"Jacobi did not meet residual target {target}")


def solver_comparison(root):
    rows = []
    for n in (8, 16, 32, 64):
        a, b, direct, full, assembly, direct_s = square_fdm(n)
        x = np.arange(1, n)/n
        xx, yy = np.meshgrid(x, x, indexing="xy")
        exact = square_exact(xx, yy).ravel()
        disc_error = float(np.linalg.norm(direct-exact, np.inf))
        # Discrete comparison principle: ||A^-1 r||_inf <= ||r||_inf/8
        # follows from the discrete barrier x(1-x)/2 on the unit square.
        target = 8e-3 * disc_error
        start = time.perf_counter()
        iterative, steps, r0, rn = jacobi(a, b, target)
        iterative_s = time.perf_counter()-start
        alg_error = float(np.linalg.norm(iterative-direct, np.inf))
        assert alg_error <= 1.001e-3 * disc_error
        rows.append(dict(N=n, h=1/n, free_dof=len(b), nnz=a.nnz,
                         assembly_s=assembly, direct_s=direct_s, jacobi_s=iterative_s,
                         direct_residual_inf=float(np.linalg.norm(b-a@direct, np.inf)),
                         jacobi_residual_initial=r0, residual_target=target,
                         jacobi_residual_final=rn, iterations=steps,
                         discretisation_error_inf=disc_error,
                         algebraic_error_inf=alg_error,
                         algebraic_over_discretisation=alg_error/disc_error,
                         jacobi_total_error_inf=float(np.linalg.norm(iterative-exact, np.inf))))
    write_csv(root / "data" / "depth_extension" / "direct_jacobi.csv", rows)
    return rows


def lshape_exact(x, y, kind):
    if kind == "smooth":
        return np.exp(x) * np.sin(y)
    theta = np.arctan2(y, x)
    theta = np.where(theta < np.pi/2 - 1e-14, theta + 2*np.pi, theta)
    phi = theta - np.pi/2
    r = np.hypot(x, y)
    return r**(2/3) * np.sin(2*phi/3)


def lshape_gradient(x, y, kind):
    if kind == "smooth":
        e = np.exp(x)
        return np.stack((e*np.sin(y), e*np.cos(y)))
    theta = np.arctan2(y, x)
    theta = np.where(theta < np.pi/2 - 1e-14, theta + 2*np.pi, theta)
    phi = theta - np.pi/2
    r = np.hypot(x, y)
    a = 2/3
    radial = a*r**(a-1)*np.sin(a*phi)
    angular = a*r**(a-1)*np.cos(a*phi)
    return np.stack((radial*np.cos(theta)-angular*np.sin(theta),
                     radial*np.sin(theta)+angular*np.cos(theta)))


def mesh_h(mesh):
    tri = mesh.p[:, mesh.t]
    return float(max(np.max(np.linalg.norm(tri[:, i]-tri[:, j], axis=0))
                     for i, j in ((0, 1), (1, 2), (2, 0))))


def lshape_comparison(root):
    rows = []
    for level in (1, 2, 3, 4, 5):
        mesh = MeshTri.init_lshaped().refined(level)
        basis = Basis(mesh, ElementTriP1(), intorder=12)
        start = time.perf_counter()
        a = asm(laplace, basis).tocsr()
        boundary = mesh.boundary_nodes()
        free = np.setdiff1d(np.arange(mesh.p.shape[1]), boundary)
        ac = a[free][:, free].tocsr()
        assembly = time.perf_counter()-start
        for kind in ("smooth", "singular"):
            known = lshape_exact(mesh.p[0], mesh.p[1], kind)
            start = time.perf_counter()
            rhs = -(a[free][:, boundary] @ known[boundary])
            u = known.copy()
            u[free] = spsolve(ac.tocsc(), rhs)
            solve = time.perf_counter()-start
            coords = basis.global_coordinates()
            diff = basis.interpolate(u).value-lshape_exact(coords[0], coords[1], kind)
            grad_diff = basis.interpolate(u).grad-lshape_gradient(coords[0], coords[1], kind)
            element_energy_sq = np.sum(np.sum(grad_diff**2, axis=0)*basis.dx, axis=1)
            centroids = np.mean(mesh.p[:, mesh.t], axis=1)
            corner = np.hypot(centroids[0], centroids[1]) <= 2*mesh_h(mesh)
            rows.append(dict(kind=kind, level=level, h=mesh_h(mesh),
                             dof_total=len(u), dof_free=len(free), nnz=ac.nnz,
                             assembly_s=assembly, solve_s=solve,
                             residual_inf=float(np.linalg.norm(ac@u[free]-rhs, np.inf)),
                             residual_scaled=float(np.max(np.abs(ac@u[free]-rhs) /
                                 np.maximum(np.asarray(abs(ac)@abs(u[free])+abs(rhs)).ravel(), 1e-300))),
                             L2=float(np.sqrt(np.sum(diff**2*basis.dx))),
                             energy=float(np.sqrt(np.sum(element_energy_sq))),
                             corner_energy_fraction=float(np.sum(element_energy_sq[corner])/np.sum(element_energy_sq)),
                             nodal_Linf=float(np.max(np.abs(u-known)))))
    for kind in ("smooth", "singular"):
        selected = [r for r in rows if r["kind"] == kind]
        for key in ("L2", "energy", "nodal_Linf"):
            orders(selected, key)
    write_csv(root / "data" / "depth_extension" / "lshape_smooth_singular.csv", rows)
    return rows


def lshape_quadrature_check(root):
    """Hold the finest singular FE solution fixed; vary only error quadrature."""
    mesh = MeshTri.init_lshaped().refined(5)
    low = Basis(mesh, ElementTriP1(), intorder=12)
    a = asm(laplace, low).tocsr()
    boundary = mesh.boundary_nodes()
    free = np.setdiff1d(np.arange(mesh.p.shape[1]), boundary)
    known = lshape_exact(*mesh.p, "singular")
    rhs = -(a[free][:, boundary] @ known[boundary])
    u = known.copy()
    u[free] = spsolve(a[free][:, free].tocsc(), rhs)
    centroids = np.mean(mesh.p[:, mesh.t], axis=1)
    corner = np.hypot(*centroids) <= 2 * mesh_h(mesh)
    rows = []
    for q in (12, 16, 19):
        basis = Basis(mesh, ElementTriP1(), intorder=q)
        xy = basis.global_coordinates()
        diff = basis.interpolate(u).grad - lshape_gradient(xy[0], xy[1], "singular")
        element = np.sum(np.sum(diff**2, axis=0) * basis.dx, axis=1)
        rows.append(dict(quadrature_order=q, energy=float(np.sqrt(element.sum())),
                         corner_fraction=float(element[corner].sum()/element.sum())))
    write_csv(root / "data" / "depth_extension" / "lshape_quadrature.csv", rows)
    return rows


def lshape_solver_check(root):
    """Independent tighter iterative solve on the finest singular system."""
    mesh = MeshTri.init_lshaped().refined(5)
    basis = Basis(mesh, ElementTriP1())
    a = asm(laplace, basis).tocsr()
    boundary = mesh.boundary_nodes()
    free = np.setdiff1d(np.arange(mesh.p.shape[1]), boundary)
    known = lshape_exact(*mesh.p, "singular")
    ac = a[free][:, free].tocsr()
    rhs = -(a[free][:, boundary] @ known[boundary])
    direct = spsolve(ac.tocsc(), rhs)
    iterative, info = cg(ac, rhs, rtol=1e-13, atol=0, maxiter=20000)
    assert info == 0
    rows = [dict(level=5, dof_free=len(free), cg_info=info,
                 nodal_difference_inf=float(np.max(np.abs(iterative-direct))),
                 direct_residual_inf=float(np.max(np.abs(rhs-ac@direct))),
                 cg_residual_inf=float(np.max(np.abs(rhs-ac@iterative))))]
    write_csv(root / "data" / "depth_extension" / "lshape_solver_check.csv", rows)
    return rows


def checks():
    y = np.array([0.1, 0.4, 0.9])
    assert np.max(np.abs(lshape_exact(np.zeros(3), y, "singular"))) < 1e-12
    assert np.max(np.abs(lshape_exact(y, np.zeros(3), "singular"))) < 1e-12
    assert abs(lshape_exact(-1.0, 0.5, "singular")) > 0.1
    assert abs(lshape_exact(0.5, -1.0, "singular")) > 0.1
    # Both smooth and singular exact solutions are harmonic at these points.
    for kind in ("smooth", "singular"):
        for x, yy in ((-0.7, 0.3), (-0.4, -0.6), (0.3, -0.8)):
            h = 1e-4
            f = lambda a,b: lshape_exact(a,b,kind)
            lap = (f(x+h,yy)+f(x-h,yy)+f(x,yy+h)+f(x,yy-h)-4*f(x,yy))/h**2
            assert abs(lap) < 1e-5, (kind, lap)


def make_figures(root, square, lshape):
    for filename, rows, group, ylabel, measures in (
        ("square_fdm_fem.png", square, "method", "L2 error", ("L2",)),
        ("lshape_rates.png", lshape, "kind", "Error", ("L2", "energy")),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in sorted({r[group] for r in rows}):
            sub = [r for r in rows if r[group] == name]
            for measure in measures:
                ax.loglog([r["h"] for r in sub], [r[measure] for r in sub], "o-", label=f"{name} {measure}")
        ax.set(xlabel="h (smaller is finer)", ylabel=ylabel)
        ax.grid(True, which="both", alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(root/"figures"/filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for kind in ("smooth", "singular"):
        sub = [r for r in lshape if r["kind"] == kind]
        for measure in ("L2", "energy"):
            ax.loglog([r["dof_free"] for r in sub], [r[measure] for r in sub], "o-", label=f"{kind} {measure}")
    ax.set(xlabel="Free degrees of freedom", ylabel="Error")
    ax.grid(True, which="both", alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root/"figures"/"lshape_vs_dof.png", dpi=180)
    plt.close(fig)
    for level in (1, 5):
        mesh = MeshTri.init_lshaped().refined(level)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.triplot(mesh.p[0], mesh.p[1], mesh.t.T, color="steelblue", lw=.7 if level == 1 else .12)
        ax.plot(0, 0, "ro", label="re-entrant corner")
        ax.plot([0, 1], [0, 0], "r--", lw=2, label="zero-trace rays")
        ax.plot([0, 0], [0, 1], "r--", lw=2)
        ax.set(xlabel="x", ylabel="y", title=f"L-shaped mesh: level {level}", aspect="equal")
        ax.legend(loc="lower left")
        fig.tight_layout()
        fig.savefig(root/"figures"/f"lshape_mesh_level{level}.png", dpi=180)
        plt.close(fig)


def make_report(root, square, solver, lshape):
    def fmt(x, digits=3):
        return f"{x:.{digits}g}"

    def order(x):
        return "--" if x == "" else f"{x:.2f}"

    def num(x):
        s = fmt(x)
        if "e" in s:
            mantissa, exponent = s.split("e")
            return f"{mantissa}\\times10^{{{int(exponent)}}}"
        return s

    def tms(seconds):
        return f"{1000*seconds:.2f}"

    lines = [r"\section{Supplementary depth comparisons}",
             r"\label{sec:depth-supplement}",
             r"The following experiments are separate from the mixed-boundary FO/G refinement study. They use new, explicitly stated protocols and do not change any archived mixed-boundary data. The direct--iterative and FDM--FEM comparisons use the same square manufactured problem; the regularity comparison uses an L-shaped all-Dirichlet problem.",
             "",
             r"\subsection{Direct versus iterative solve at matched accuracy}",
             r"On $\Omega=(0,1)^2$, set $u_*(x,y)=e^{x+y}$, $f=-2e^{x+y}$ and $u|_{\partial\Omega}=u_*$. The five-point FDM matrix is solved once by sparse direct factorisation and separately by unpreconditioned Jacobi iteration from zero. Both methods use the same assembled matrix and right-hand side. Jacobi stops on the true residual $\|b-Au^{(k)}\|_\infty$, not on the difference between successive iterates. The discrete barrier $\psi(x,y)=x(1-x)/2$ gives $\|A^{-1}r\|_\infty\le\|r\|_\infty/8$. We therefore set the residual threshold to $8\times10^{-3}E_{\mathrm{disc},\infty}$, where $E_{\mathrm{disc},\infty}=\|u_{\mathrm{direct}}-u_*\|_{\infty,\mathrm{nodes}}$. The direct solve and Jacobi solution are thus matched to a stated algebraic-error target of $10^{-3}$ times the discretisation error.",
             r"\begin{table}[htbp]\centering\small\resizebox{\linewidth}{!}{%",
             r"\begin{tabular}{rrrrrrr}\toprule",
             r"$N$ & free DOF & $E_{\rm disc,\infty}$ & Jacobi steps & final $\|r\|_\infty$ & $\|u_J-u_D\|_\infty/E_{\rm disc,\infty}$ & direct/Jacobi solve (ms)\\\midrule"]
    for r in solver:
        lines.append(f'{r["N"]} & {r["free_dof"]} & ${num(r["discretisation_error_inf"])}$ & {r["iterations"]} & ${num(r["jacobi_residual_final"])}$ & ${num(r["algebraic_over_discretisation"])}$ & {tms(r["direct_s"])}/{tms(r["jacobi_s"])} \\\\')
    lines += [r"\bottomrule\end{tabular}}",
              r"\caption{Matched-accuracy sparse direct and Jacobi comparison. Assembly time, the initial residual, target residual, final residual, and both final errors are retained in \texttt{data/depth\_extension/direct\_jacobi.csv}. Times are single local measurements and should not be interpreted as portable benchmarks.}",
              r"\end{table}",
              r"All four Jacobi solutions satisfy the target. No direct/iterative time crossover occurs for $N=8,16,32,64$; the sparse direct solve is faster throughout this tested range. This is a conclusion about unpreconditioned Jacobi under this protocol, not about every iterative method.",
              r"\paragraph{Why the residual target is sufficient.} Extend the error by zero to all Dirichlet nodes. On the unit-square five-point grid, $\psi(x,y)=x(1-x)/2$ has $-\Delta_h\psi=1$ at every free node and $0\leq\psi\leq1/8$ on the whole grid. Since the error vanishes on the boundary, the discrete comparison principle applied to $\pm e\leq\|r\|_\infty\psi$ gives $\|A^{-1}r\|_\infty\leq\|r\|_\infty/8$. This bound applies only to this all-Dirichlet constant-coefficient matrix. For the direct solution as a numerical reference, the observed difference is bounded by $\|r_J\|_\infty/8+\|r_D\|_\infty/8$; the reported residuals make the direct-solve term negligible at the stated precision.",
              "",
              r"\subsection{FDM versus FEM on identical square problem}",
              r"For the same $f$, exact solution, Dirichlet data and square grids, the FDM uses the standard five-point stencil. The FEM uses continuous piecewise-linear triangles, obtained by splitting each square in the same orientation, and high-order quadrature for the load. Both methods have $(N+1)^2$ total nodes and $(N-1)^2$ free degrees of freedom. Their algebraic systems are solved by sparse direct methods to residuals far below the discretisation errors. To compare the same $L^2$ quantity, the FDM nodal field is explicitly interpolated as a piecewise-linear function on the FEM triangles; this is a comparison convention, not the native bilinear FDM reconstruction. The nodal $L^\infty$ error is taken at the common nodes.",
              r"\begin{table}[htbp]\centering\small\resizebox{\linewidth}{!}{%",
              r"\begin{tabular}{llrrrrrr}\toprule",
              r"Method & $N$ & free DOF & nnz & $L^2$ error & order & nodal $L^\infty$ & assembly/solve (ms)\\\midrule"]
    for r in square:
        lines.append(f'{r["method"]} & {r["N"]} & {r["dof_free"]} & {r["nnz"]} & ${num(r["L2"])}$ & {order(r["L2_order"])} & ${num(r["nodal_Linf"])}$ & {tms(r["assembly_s"])}/{tms(r["solve_s"])} \\\\')
    lines += [r"\bottomrule\end{tabular}}",
              r"\caption{Square all-Dirichlet benchmark. Orders are pairwise log ratios under halving of $h=1/N$. The complete precision, residuals and timings are in \texttt{data/depth\_extension/square\_fdm\_fem.csv}.}",
              r"\end{table}",
              r"\begin{figure}[htbp]\centering\includegraphics[width=.68\linewidth]{figures/square_fdm_fem.png}\caption{Matched square-problem $L^2$ convergence. The horizontal axis uses ordinary increasing $h$ from left to right; refinement moves from right to left.}\end{figure}",
              r"Both methods show approximately second-order $L^2$ decay with this reconstruction. The equal nnz counts here follow from the aligned uniform triangulation and do not imply identical matrices or general equivalence of FDM and FEM. Assembly and solve times are reported separately; no solver or discretisation cost claim is made from one local timing alone.",
              "",
              r"\subsection{Smooth versus nonsmooth data on an L-shaped domain}",
              r"Let $\Omega=(-1,1)^2\setminus[0,1]^2$ and solve $-\Delta u=0$ with Dirichlet data from a manufactured exact solution. Both cases use the same P1 FEM code, the same nested quasi-uniform triangulations, the same direct solver, and the same quadrature. The smooth harmonic solution is $u_s(x,y)=e^x\sin y$. The singular harmonic solution is $u_c(r,\theta)=r^{2/3}\sin(2(\theta-\pi/2)/3)$, where $\theta=\operatorname{atan2}(y,x)$ and $2\pi$ is added if $\theta<\pi/2$. Both exact solutions are continuous. The second has an unbounded gradient near the re-entrant corner and is not $H^2$-regular. It is nonsmooth, not discontinuous. For $u_c$, $f=0$ away from the corner, both rays bounding the removed quadrant have zero trace, and the remaining outer-boundary trace is nonzero. These facts are checked in the code before the refinement runs.",
              r"\begin{table}[htbp]\centering\small\resizebox{\linewidth}{!}{%",
              r"\begin{tabular}{llrrrrrr}\toprule",
              r"Solution & level & $h_{\max}$ & free DOF & $L^2$ error & $L^2$ order & energy error & energy order\\\midrule"]
    for r in lshape:
        lines.append(f'{r["kind"]} & {r["level"]} & {fmt(r["h"])} & {r["dof_free"]} & ${num(r["L2"])}$ & {order(r["L2_order"])} & ${num(r["energy"])}$ & {order(r["energy_order"])} \\\\')
    lines += [r"\bottomrule\end{tabular}}",
              r"\caption{Five L-shaped uniform refinements. The energy error is $\|\nabla(u_*-u_h)\|_{L^2(\Omega)}$. Additional nodal maximum errors, nonzeros, residuals, and separate assembly/solve times are in \texttt{data/depth\_extension/lshape\_smooth\_singular.csv}.}",
              r"\end{table}",
              r"For a local refinement diagnostic, define $\eta_K^2=\int_K|\nabla(u_*-u_h)|^2$ using the known manufactured solution. The corner fraction below is $\sum_{K:\,|c_K|\le 2h_{\max}}\eta_K^2/\sum_K\eta_K^2$, with $c_K$ the triangle centroid. This is an exact-error indicator available in this manufactured test, not a computable a posteriori estimator for an unknown solution.",
              r"\begin{table}[htbp]\centering\small",
              r"\begin{tabular}{llrrrr}\toprule",
              r"Solution & level & free DOF & nnz & solve (ms) & corner error fraction\\\midrule"]
    for r in lshape:
        lines.append(f'{r["kind"]} & {r["level"]} & {r["dof_free"]} & {r["nnz"]} & {tms(r["solve_s"])} & {r["corner_energy_fraction"]:.2f} \\\\')
    lines += [r"\bottomrule\end{tabular}",
              r"\caption{L-shaped solve costs and the fraction of the exact energy error concentrated near the re-entrant corner. Assembly costs are listed separately in the CSV. Times are one local measurement.}",
              r"\end{table}",
              r"\begin{figure}[htbp]\centering\includegraphics[width=.68\linewidth]{figures/lshape_rates.png}\caption{Smooth and re-entrant-corner singular solutions, measured in the same $L^2$ and energy norms.}\end{figure}",
              r"\begin{figure}[htbp]\centering\includegraphics[width=.68\linewidth]{figures/lshape_vs_dof.png}\caption{The same L-shaped errors plotted against free degrees of freedom.}\end{figure}",
              r"The last two pairwise orders are approximately $2.00$ ($L^2$) and $1.00$ (energy) for the smooth case, versus $1.36$ and $0.65$ for the singular case. The latter agrees with the regularity-based P1 FEM expectations of approximately $h^{4/3}$ and $h^{2/3}$ up to small losses. On the finest mesh, about $84\%$ of the singular case's computed energy-error square is assigned to the union of whole triangles whose centroids lie within $2h_{\max}$ of the re-entrant corner, compared with less than $1\%$ for the smooth case. This centroid-selected patch is a mesh diagnostic, not the exact circular region. It supports corner-focused graded or adaptive refinement as a follow-up; no such refinement is claimed here. The measured nodal $L^\infty$ behaviour is retained as a diagnostic, not used as a proof of those norm estimates. The singular solution remains continuous; its reduced rate is caused by the corner gradient singularity. These FEM estimates are not silently transferred to an FDM scheme.",
              "",
              r"\paragraph{Reproduction and scope.} From the project root, install \texttt{requirements.txt} and run \texttt{python code/run\_all.py}. This command regenerates the main mixed-boundary results and the separate depth and variable-diffusion benchmarks. The square comparison is deliberately all-Dirichlet, so its rows and rates must not be substituted for the original mixed-boundary tables. The measured timing values depend on the execution machine; the error and convergence calculations are the substantive comparisons."]
    (root/"report"/"depth_extension.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main(root: Path):
    for name in ("data", "figures", "report", "data/depth_extension"):
        (root/name).mkdir(exist_ok=True)
    checks()
    square = square_comparison(root)
    solver = solver_comparison(root)
    lshape = lshape_comparison(root)
    quadrature = lshape_quadrature_check(root)
    solver_check = lshape_solver_check(root)
    make_figures(root, square, lshape)
    make_report(root, square, solver, lshape)
    with (root/"report"/"depth_extension.tex").open("a", encoding="utf-8") as f:
        sensitivity = abs(quadrature[-1]['energy']-quadrature[0]['energy'])/quadrature[-1]['energy']
        f.write("\n\\paragraph{Singular-error quadrature and algebraic checks.} "
                "On the fixed finest singular FE solution, error-only quadrature orders "
                "12, 16 and 19 yield energy errors "
                + ", ".join(f"{r['energy']:.8g}" for r in quadrature)
                + f". The order-12 versus order-19 change is {100*sensitivity:.3f} percent; "
                "the quadrature study supports the observed rate but does not certify all displayed digits. "
                "The finest-mesh tight-CG cross-check differs from the direct solution "
                + f"by {solver_check[0]['nodal_difference_inf']:.3g} in nodal maximum norm. "
                "Full values and residuals are in "
                "\\texttt{data/depth\\_extension/lshape\\_quadrature.csv} and "
                "\\texttt{data/depth\\_extension/lshape\\_solver\\_check.csv}. "
                "The L-shaped meshes are shown in Figures "
                "\\ref{fig:lshape-coarse} and \\ref{fig:lshape-fine}.\n"
                "\\begin{figure}[htbp]\\centering\\includegraphics[width=.45\\linewidth]{figures/lshape_mesh_level1.png}"
                "\\caption{Coarse L-shaped mesh, re-entrant corner and zero-trace rays.}\\label{fig:lshape-coarse}\\end{figure}\n"
                "\\begin{figure}[htbp]\\centering\\includegraphics[width=.55\\linewidth]{figures/lshape_mesh_level5.png}"
                "\\caption{Fine L-shaped mesh.}\\label{fig:lshape-fine}\\end{figure}\n")
        f.write((root/"report"/"lshape_theory.tex").read_text(encoding="utf-8"))
    metadata = dict(python=platform.python_version(), platform=platform.platform(),
                    numpy=version("numpy"), scipy=version("scipy"),
                    matplotlib=version("matplotlib"), scikit_fem=version("scikit-fem"),
                    source="Single fresh local run; timings are single measurements")
    (root/"data"/"depth_extension"/"run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    print("Checks passed. Wrote matched comparisons and L-shaped refinement tables.")
    for rows, key in ((square,"L2"),(lshape,"energy")):
        for r in rows[-4:]:
            print(r.get("method",r.get("kind")), r["h"], r[key], r.get(key+"_order"))
    print("Jacobi iterations:", [r["iterations"] for r in solver])
