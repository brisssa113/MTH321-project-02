"""Independent conservative variable-diffusion benchmarks on the unit square."""
import csv
import math
import time
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg, eigsh, spsolve
from skfem import Basis, BilinearForm, ElementTriP1, Functional, LinearForm, MeshTri, asm, condense
from skfem.helpers import dot, grad


def exact(x, y, case, contrast=10.0):
    if case == "smooth_sine":
        return np.sin(np.pi*x)*np.sin(np.pi*y)
    if case == "smooth":
        return np.exp(x+y)
    # Unit potential drop with continuous flux through x=1/2.
    length_resistance = .5 + .5/contrast
    return np.where(x <= .5, x, .5 + (x-.5)/contrast) / length_resistance


def forcing(x, y, case):
    if case == "smooth_sine":
        return (2*np.pi**2*(1+x)*np.sin(np.pi*x)-np.pi*np.cos(np.pi*x))*np.sin(np.pi*y)
    return -(3+2*x)*np.exp(x+y) if case == "smooth" else np.zeros_like(x)


def coefficient(x, case, contrast=10.0):
    return 1+x if case in ("smooth", "smooth_sine") else np.where(x < .5, 1., contrast)


def assemble_fdm(n, case, mean="harmonic", contrast=10.0):
    if case not in ("smooth", "smooth_sine", "layered") or mean not in ("harmonic", "arithmetic") or contrast <= 0:
        raise ValueError("invalid case or face mean")
    h = 1/n
    m = n-1
    a = lil_matrix((m*m, m*m))
    b = np.empty(m*m)
    def index(i, j):
        return (j-1)*m+i-1
    for j in range(1, n):
        y = j*h
        for i in range(1, n):
            x = i*h
            k = index(i, j)
            b[k] = forcing(x, y, case)
            for di, dj in ((-1,0),(1,0),(0,-1),(0,1)):
                nx, ny = (i+di)*h, (j+dj)*h
                fx = x + di*h/2
                if case == "layered" and di and abs(fx-.5) < 1e-12:
                    face = 2/(1+1/contrast) if mean == "harmonic" else (1+contrast)/2
                else:
                    face = float(coefficient(fx, case, contrast))
                c = face/h**2
                a[k,k] += c
                if 1 <= i+di < n and 1 <= j+dj < n:
                    a[k,index(i+di,j+dj)] -= c
                else:
                    b[k] += c*exact(nx, ny, case, contrast)
    return a.tocsr(), b


def assemble_fem(n, case):
    grid = np.linspace(0, 1, n+1)
    mesh = MeshTri.init_tensor(grid, grid)
    basis = Basis(mesh, ElementTriP1(), intorder=10)
    @BilinearForm
    def diffusion(u, v, w):
        return coefficient(w.x[0], case)*dot(grad(u), grad(v))
    @LinearForm
    def load(v, w):
        return forcing(w.x[0], w.x[1], case)*v
    a = asm(diffusion, basis).tocsr()
    b = asm(load, basis)
    known = exact(mesh.p[0], mesh.p[1], case)
    ac, bc, full, free = condense(a, b, x=known.copy(), D=mesh.boundary_nodes())
    return mesh, basis, ac.tocsr(), bc, full, free


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def symmetry_defect(a):
    """Relative Frobenius symmetry defect of the reduced matrix."""
    return float(np.linalg.norm((a-a.T).data) / np.linalg.norm(a.data))


def reconstructed_interface_fluxes(u, n, contrast=10.0):
    """Constant extrapolation of two-node material-side slopes to x=1/2.

    u[j-1,i-1] is U(i/n,j/n). Both fluxes use the +x orientation.
    For odd n, m=(n-1)/2: left nodes m-1,m; right nodes m+1,m+2.
    No shared crossing coefficient enters this independent trace diagnostic.
    """
    if n < 7 or n % 2 != 1 or u.shape != (n-1, n-1):
        raise ValueError("Interface reconstruction requires odd N>=7 and an interior grid")
    m = n//2
    left = -n*(u[:,m-1]-u[:,m-2])
    right = -contrast*n*(u[:,m+1]-u[:,m])
    return left, right


def run_required_direction3(root: Path):
    """Specified sine benchmark and complete layered-interface diagnostics."""
    out = root/"data"/"variable_diffusion"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (8, 16, 32, 64):
        for method in ("FDM", "FEM"):
            if method == "FDM":
                a, b = assemble_fdm(n, "smooth_sine")
            else:
                mesh, basis, a, b, full, free = assemble_fem(n, "smooth_sine")
            u = spsolve(a.tocsc(), b)
            if method == "FDM":
                xx, yy = np.meshgrid(np.arange(1,n)/n, np.arange(1,n)/n)
                err = u-exact(xx, yy, "smooth_sine").ravel()
                l2 = float(np.linalg.norm(err)/n)  # composite trapezoid; boundary error is zero
                linf = float(np.max(np.abs(err)))
            else:
                full[free] = u
                @Functional
                def error_sq(w):
                    return (w.u-exact(w.x[0],w.x[1],"smooth_sine"))**2
                l2 = math.sqrt(float(asm(error_sq,basis,u=full)))
                linf = float(np.max(np.abs(full-exact(mesh.p[0],mesh.p[1],"smooth_sine"))))
            rows.append(dict(method=method,N=n,h=1/n,dof=len(b),nnz=a.nnz,
                             error_l2=l2,error_inf=linf,
                             symmetry_defect_fro=symmetry_defect(a),
                             residual_inf=float(np.max(np.abs(b-a@u)))))
    for method in ("FDM","FEM"):
        sub = [r for r in rows if r["method"] == method]
        for k,r in enumerate(sub):
            r["order_l2"] = "" if k == 0 else math.log(sub[k-1]["error_l2"]/r["error_l2"],2)
            r["order_inf"] = "" if k == 0 else math.log(sub[k-1]["error_inf"]/r["error_inf"],2)
    write_csv(out/"sine_required.csv",rows)
    fitted = []
    for method in ("FDM", "FEM"):
        sub = [r for r in rows if r["method"] == method and r["N"] >= 16]
        xfit = np.log([r["h"] for r in sub])
        record = dict(method=method, fit_N="16,32,64", fit_h_min=1/64, fit_h_max=1/16)
        for key in ("l2", "inf"):
            yfit = np.log([r["error_"+key] for r in sub])
            slope, intercept = np.polyfit(xfit, yfit, 1)
            record["fitted_order_"+key] = float(slope)
            record["r_squared_"+key] = float(1-np.sum((yfit-slope*xfit-intercept)**2)/np.sum((yfit-yfit.mean())**2))
        fitted.append(record)
    write_csv(out/"sine_fitted_orders.csv", fitted)

    interface = []
    profiles = []
    for n in (7,15,31,63):
        for mean in ("harmonic","arithmetic"):
            a,b = assemble_fdm(n,"layered",mean)
            u = spsolve(a.tocsc(),b).reshape(n-1,n-1)
            face = 2/(1+1/10) if mean == "harmonic" else 5.5
            k = n//2
            # Negative sign follows Fourier's law J=-a*u_x.
            jf = -face*n*(u[:,k]-u[:,k-1])
            jl, jr = reconstructed_interface_fluxes(u, n)
            exact_j = -1/.55
            interface.append(dict(N=n,mean=mean,contrast=10,
                interface_flux_error_max=float(np.max(np.abs(jf-exact_j))),
                adjacent_flux_jump_max=float(np.max(np.maximum(np.abs(jf-jl),np.abs(jf-jr)))),
                reconstructed_flux_jump_max=float(np.max(np.abs(jr-jl))),
                reconstructed_left_error_max=float(np.max(np.abs(jl-exact_j))),
                reconstructed_right_error_max=float(np.max(np.abs(jr-exact_j))),
                max_flux_error_y=float((np.argmax(np.abs(jf-exact_j))+1)/n),
                flux_min=float(min(jf.min(),jl.min(),jr.min())),
                flux_max=float(max(jf.max(),jl.max(),jr.max())),
                residual_inf=float(np.max(np.abs(b-a@u.ravel())))))
            for j in range(n-1):
                profiles.append(dict(N=n,mean=mean,y=(j+1)/n,
                    crossing_flux=float(jf[j]),reconstructed_left_flux=float(jl[j]),
                    reconstructed_right_flux=float(jr[j]),exact_flux=exact_j))
    write_csv(out/"interface_diagnostics.csv",interface)
    write_csv(out/"interface_flux_profiles.csv",profiles)

    contrasts = []
    for n in (7, 15, 31, 63):
        for contrast in (1., 2., 5., 10., 20., 50.):
            for mean in ("harmonic", "arithmetic"):
                a, b = assemble_fdm(n, "layered", mean, contrast)
                u = spsolve(a.tocsc(), b)
                xx, yy = np.meshgrid(np.arange(1,n)/n, np.arange(1,n)/n)
                target = exact(xx, yy, "layered", contrast).ravel()
                steps = [0]
                def callback(_):
                    steps[0] += 1
                preconditioner = __import__("scipy").sparse.diags(1/a.diagonal())
                ucg, status = cg(a, b, M=preconditioner, rtol=1e-10, atol=0, callback=callback)
                if status != 0:
                    raise RuntimeError(f"CG failed for N={n}, contrast={contrast}, mean={mean}: {status}")
                field = u.reshape(n-1,n-1)
                k = n//2
                face = 2/(1+1/contrast) if mean == "harmonic" else (1+contrast)/2
                flux = -face*n*(field[:,k]-field[:,k-1])
                exact_flux = -1/(.5+.5/contrast)
                contrasts.append(dict(N=n,contrast=contrast,mean=mean,
                    dof=len(b),nnz=a.nnz,cg_iterations=steps[0],
                    nodal_error_inf=float(np.max(np.abs(u-target))),
                    solution_min=float(min(0.,u.min())),solution_max=float(max(1.,u.max())),
                    interface_flux_error_max=float(np.max(np.abs(flux-exact_flux))),
                    cg_direct_difference_inf=float(np.max(np.abs(ucg-u))),
                    cg_nodal_error_inf=float(np.max(np.abs(ucg-target))),
                    cg_residual_inf=float(np.max(np.abs(b-a@ucg))),
                    symmetry_defect_fro=symmetry_defect(a),
                    residual_inf=float(np.max(np.abs(b-a@u)))))
    write_csv(out/"contrast_sweep.csv",contrasts)

    from matplotlib import pyplot as plt
    fig_dir = root/"figures"
    fig_dir.mkdir(exist_ok=True)
    n = 63
    x = np.arange(n+1)/n
    fig, (ax, axerr) = plt.subplots(2,1,figsize=(6.1,5.0),sharex=True,
                                    gridspec_kw={"height_ratios":[2,1]})
    target_cut=exact(x,0,"layered")
    ax.plot(x,target_cut,"k-",label="exact")
    for mean,style in (("harmonic","o-"),("arithmetic","s--")):
        a,b=assemble_fdm(n,"layered",mean)
        u=spsolve(a.tocsc(),b).reshape(n-1,n-1)
        cut=np.r_[0,u[n//2-1,:],1]
        ax.plot(x,cut,style,markevery=6,markersize=3,label=mean)
        axerr.plot(x,np.abs(cut-target_cut),style,markevery=6,markersize=3,label=mean)
    ax.axvline(.5,color="0.5",linestyle=":",label="interface")
    axerr.axvline(.5,color="0.5",linestyle=":")
    ax.set(ylabel="u(x,y)",title="Layered solution cut, N=63, contrast 10")
    axerr.set(xlabel="x at y near 1/2",ylabel="absolute error")
    ax.legend(fontsize=8); axerr.legend(fontsize=8); fig.tight_layout()
    fig.savefig(fig_dir/"layered_solution_cut.png",dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.1,3.7))
    for mean,style in (("harmonic","o-"),("arithmetic","s--")):
        a,b=assemble_fdm(n,"layered",mean)
        u=spsolve(a.tocsc(),b).reshape(n-1,n-1)
        k=n//2
        face=2/(1+1/10) if mean=="harmonic" else 5.5
        ax.plot(np.arange(1,n)/n,-face*n*(u[:,k]-u[:,k-1]),style,
                markevery=6,markersize=3,label=mean)
    ax.axhline(-1/.55,color="k",label="exact Fourier flux")
    ax.set(xlabel="y on the x=1/2 interface",ylabel="J_x=-a u_x",
           title="Entire interface flux, N=63, contrast 10")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(fig_dir/"layered_interface_flux.png",dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.1,3.7))
    for mean,style in (("harmonic","o-"),("arithmetic","s--")):
        sub=[r for r in contrasts if r["N"]==63 and r["mean"]==mean]
        ax.plot([r["contrast"] for r in sub],[r["cg_iterations"] for r in sub],style,label=mean)
    ax.set(xscale="log",xlabel="material contrast a_R/a_L",ylabel="Jacobi-CG iterations",
           title="Fixed N=63 and rtol=1e-10")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(fig_dir/"layered_contrast_iterations.png",dpi=170); plt.close(fig)
    return rows, interface, contrasts


def run(root: Path):
    out = root/"data"/"variable_diffusion"
    out.mkdir(parents=True, exist_ok=True)
    smooth = []
    for n in (8, 16, 32, 64):
        for method in ("FDM", "FEM"):
            start = time.perf_counter()
            if method == "FDM":
                a, b = assemble_fdm(n, "smooth")
            else:
                mesh, basis, a, b, full, free = assemble_fem(n, "smooth")
            assembly_s = time.perf_counter()-start
            start = time.perf_counter()
            u = spsolve(a.tocsc(), b)
            solve_s = time.perf_counter()-start
            if method == "FDM":
                xx, yy = np.meshgrid(np.arange(1,n)/n, np.arange(1,n)/n)
                target = exact(xx, yy, "smooth").ravel()
            else:
                full[free] = u
                target = exact(mesh.p[0], mesh.p[1], "smooth")[free]
            error = float(np.max(np.abs(u-target)))
            count = [0]
            def callback(_):
                count[0] += 1
            precond = __import__("scipy").sparse.diags(1/a.diagonal())
            ucg, status = cg(a, b, M=precond, rtol=1e-10, atol=0, callback=callback)
            assert status == 0
            row = dict(case="smooth", method=method, N=n, h=1/n,
                       dof=len(b), nnz=a.nnz, error_inf=error,
                       residual_inf=float(np.max(np.abs(b-a@u))),
                       scaled_residual=float(np.max(np.abs(b-a@u)/np.maximum(np.asarray(abs(a)@abs(u)+abs(b)).ravel(),1e-300))),
                       cg_iterations=count[0], cg_direct_difference_inf=float(np.max(np.abs(ucg-u))),
                       dmp_offdiagonal_nonpositive=bool(np.max((a - __import__("scipy").sparse.diags(a.diagonal())).data) <= 1e-12),
                       assembly_s=assembly_s, solve_s=solve_s)
            if n <= 32:
                row["condition_2"] = float(eigsh(a,k=1,which="LM",return_eigenvectors=False)[0]/
                                           eigsh(a,k=1,which="SM",return_eigenvectors=False)[0])
            else:
                row["condition_2"] = ""
            smooth.append(row)
    for method in ("FDM", "FEM"):
        sub = [r for r in smooth if r["method"] == method]
        for i,r in enumerate(sub):
            r["order"] = "" if i == 0 else math.log(sub[i-1]["error_inf"]/r["error_inf"],2)
    write_csv(out/"smooth.csv", smooth)

    layered = []
    # Odd N puts the material interface exactly on a vertical face.
    for n in (7, 15, 31, 63):
        for mean in ("harmonic", "arithmetic"):
            a,b = assemble_fdm(n,"layered",mean)
            u = spsolve(a.tocsc(),b)
            xx, yy = np.meshgrid(np.arange(1,n)/n,np.arange(1,n)/n)
            err = float(np.max(np.abs(u-exact(xx,yy,"layered").ravel())))
            j = n//2
            left = u[j*(n-1)+j-1]
            right = u[j*(n-1)+j]
            face = 2/(1+1/10) if mean == "harmonic" else 5.5
            flux = face*(right-left)*n
            count = [0]
            def callback(_):
                count[0] += 1
            ucg, status = cg(a,b,M=__import__("scipy").sparse.diags(1/a.diagonal()),
                             rtol=1e-10,atol=0,callback=callback)
            assert status == 0
            layered.append(dict(N=n, mean=mean, dof=len(b), error_inf=err,
                                interface_flux=flux, flux_error=abs(flux-1/.55),
                                residual_inf=float(np.max(np.abs(b-a@u))),
                                scaled_residual=float(np.max(np.abs(b-a@u)/np.maximum(np.asarray(abs(a)@abs(u)+abs(b)).ravel(),1e-300))),
                                cg_iterations=count[0], cg_direct_difference_inf=float(np.max(np.abs(ucg-u))),
                                dmp_offdiagonal_nonpositive=bool(np.max((a-__import__("scipy").sparse.diags(a.diagonal())).data)<=1e-12),
                                condition_2=(float(eigsh(a,k=1,which="LM",return_eigenvectors=False)[0]/
                                                   eigsh(a,k=1,which="SM",return_eigenvectors=False)[0]) if n<=31 else "")))
    write_csv(out/"layered.csv",layered)
    sine, interface, contrasts = run_required_direction3(root)
    text = [r"\section{Variable diffusion benchmark}",
            r"\label{sec:variable-diffusion}",
            r"This is a separate all-Dirichlet benchmark for $-\nabla\cdot(a\nabla u)=f$ on $(0,1)^2$; it does not change the mixed-boundary model. The prescribed smooth case uses $a=1+x$, $u_*=\sin(\pi x)\sin(\pi y)$, $g=0$ and $f=2\pi^2(1+x)\sin(\pi x)\sin(\pi y)-\pi\cos(\pi x)\sin(\pi y)$. The FDM uses one shared positive coefficient per face in a conservative five-point flux balance. The P1 FEM assembles $\int a\nabla u_h\cdot\nabla v_h$ on the same square grid split into triangles. Both use exact Dirichlet data. The exponential example below is an additional check, not a substitute for this prescribed case.",
            r"\begin{table}[htbp]\centering\small\begin{tabular}{llrrrr}\toprule Method & $N$ & $L^2$ error & $L^2$ order & nodal $L^\infty$ & $L^\infty$ order\\\midrule"]
    for r in sine:
        ol2 = "--" if r["order_l2"] == "" else f'{r["order_l2"]:.2f}'
        oi = "--" if r["order_inf"] == "" else f'{r["order_inf"]:.2f}'
        text.append(f'{r["method"]} & {r["N"]} & {r["error_l2"]:.3g} & {ol2} & {r["error_inf"]:.3g} & {oi} \\\\')
    text += [r"\bottomrule\end{tabular}\caption{Prescribed sine benchmark on four matched meshes. The FDM $L^2$ value is the composite trapezoid norm of nodal errors with zero boundary error; the FEM $L^2$ value integrates the continuous P1 error. These are explicitly different reconstruction conventions. Free degrees of freedom, nonzeros and residuals are retained in \texttt{data/variable\_diffusion/sine\_required.csv}.}\end{table}",
             r"The reduced matrix symmetry defect is recorded for every sine case. Positive definiteness follows from the separate energy arguments in Section~\ref{sec:variable-theory}.",
             r"\paragraph{Additional exponential check.} For $a=1+x$, $u_*=e^{x+y}$ and $f=-(3+2x)e^{x+y}$, the following independent refinement provides a nonzero-Dirichlet test.",
            r"\begin{table}[htbp]\centering\small\begin{tabular}{llrrr}\toprule Method & $N$ & nodal $L^\infty$ error & order & CG steps\\\midrule"]
    for r in smooth:
        order = "--" if r["order"] == "" else f'{r["order"]:.2f}'
        text.append(f'{r["method"]} & {r["N"]} & {r["error_inf"]:.3g} & {order} & {r["cg_iterations"]} \\\\')
    text += [r"\bottomrule\end{tabular}\caption{Variable smooth coefficient, matched mesh and nodal error. Full residuals, timings, nonzeros and condition estimates are in \texttt{data/variable\_diffusion/smooth.csv}.}\end{table}",
             r"For a separate layered material, $a=1$ for $x<1/2$ and $a=10$ for $x>1/2$. The exact one-dimensional solution is $u_*(x)=[\min(x,1/2)+\max(x-1/2,0)/10]/0.55$, with positive gradient-weighted flux $a u_x=1/0.55$ and Fourier flux $J_x=-a u_x=-1/0.55$. Odd $N$ places the interface on a vertical face. Harmonic and arithmetic face means are compared on exactly the same grids; no FEM claim is made for this discontinuous case. The retained legacy column below is the absolute error in positive $a u_x$ at one sampled interface face; its sign convention does not affect the absolute error, but it does not represent the maximum over all interface faces.",
             r"\begin{table}[htbp]\centering\small\begin{tabular}{llrr}\toprule $N$ & face mean & maximum error & interface flux error\\\midrule"]
    for r in layered:
        text.append(f'{r["N"]} & {r["mean"]} & {r["error_inf"]:.3g} & {r["flux_error"]:.3g} \\\\')
    text += [r"\bottomrule\end{tabular}\caption{Layered interface benchmark; the interface-flux-error column is a single sampled face, not a whole-interface maximum. Complete values are in \texttt{data/variable\_diffusion/layered.csv}.}\end{table}",
             r"\begin{table}[htbp]\centering\small\begin{tabular}{llrrr}\toprule $N$ & mean & $E_{J,\infty}$ & $D_h$ & $\mathcal J_h$\\\midrule"]
    for r in interface:
        text.append(f'{r["N"]} & {r["mean"]} & {r["interface_flux_error_max"]:.3g} & {r["adjacent_flux_jump_max"]:.3g} & {r["reconstructed_flux_jump_max"]:.3g} \\\\')
    text += [r"\bottomrule\end{tabular}\caption{Three distinct Fourier-flux diagnostics: whole-interface error $E_{J,\infty}$, adjacent-face difference $D_h$, and material-side reconstructed trace jump $\mathcal J_h$. All use the common $+x$ orientation and the definitions below. Full precision and the maximum-error location are in \path{data/variable_diffusion/interface_diagnostics.csv}; all traces are in \path{data/variable_diffusion/interface_flux_profiles.csv}.}\end{table}",
             r"\begin{figure}[htbp]\centering\includegraphics[width=.70\linewidth]{figures/layered_solution_cut.png}\caption{Horizontal solution cut near $y=1/2$ on the $N=63$ grid. The vertical line marks the material interface; the exact piecewise-linear solution and both face-mean schemes are shown on identical nodes.}\end{figure}",
             r"\begin{figure}[htbp]\centering\includegraphics[width=.70\linewidth]{figures/layered_interface_flux.png}\caption{Fourier flux $J_x=-a u_x$ along every interior face crossing $x=1/2$. The harmonic reconstruction preserves the exact constant flux to round-off for this aligned one-dimensional benchmark; the arithmetic reconstruction has a position-dependent discrepancy.}\end{figure}",
             r"\begin{table}[htbp]\centering\small\begin{tabular}{lrrr}\toprule $a_R/a_L$ & mean & Jacobi-CG steps & Direct nodal error\\\midrule"]
    for r in contrasts:
        if r["N"]==63:
            text.append(f'{r["contrast"]:g} & {r["mean"]} & {r["cg_iterations"]} & {r["nodal_error_inf"]:.3g} \\\\')
    text += [r"\bottomrule\end{tabular}\caption{Material-contrast sweep at fixed $N=63$, where the nodal errors and solution extrema are from the sparse direct solve. CG contributes the iteration count; its nodal error, residual and difference from the direct solution are separately retained in the CSV. Jacobi-preconditioned CG uses relative tolerance $10^{-10}$ and zero absolute tolerance. All $N=7,15,31,63$ values, solver differences, residuals, solution extrema and whole-interface flux errors are in \texttt{data/variable\_diffusion/contrast\_sweep.csv}.}\end{table}",
             r"\begin{figure}[htbp]\centering\includegraphics[width=.70\linewidth]{figures/layered_contrast_iterations.png}\caption{Iteration count under a fixed stopping rule as material contrast changes. This is a measured solver diagnostic, not an asymptotic condition-number estimate.}\end{figure}",
             r"For all tested contrasts and both face means, direct-solve extrema including boundary nodes lie in $[0,1]$. Positive definiteness and a small reconstructed jump alone do not guarantee correct flux magnitude. The harmonic exactness statement is proved under its precise assumptions in Proposition~\ref{prop:harmonic-exact}; floating-point results retain algebraic error.",
             r"All assembled interior matrices have nonpositive off-diagonal entries and a positive Dirichlet boundary contribution, giving the discrete maximum-principle sign structure. The CSV records this check, the componentwise scaled residual, CG iteration count and direct--CG difference for each case. Estimated spectral condition numbers are reported through $N=32$ for smooth data and $N=31$ for layered data. They are properties of these specific unpreconditioned matrices and coefficient contrast; no extrapolated complexity claim is made."]

    text += [r"\FloatBarrier",
             r"\subsection{Definitions of interface diagnostics}",
             r"Let $m=(N-1)/2$, $h=1/N$ and $U_{i,j}=U(ih,jh)$. For each interior $j=1,\ldots,N-1$, define the crossing flux $J_{f,j}=-a_f(U_{m+1,j}-U_{m,j})/h$. The maximum error is $E_{J,\infty}=\max_j|J_{f,j}-J_*|$.",
             r"The left and right material-side reconstructions use the nearest two nodes wholly in each material, with constant extrapolation of their slope to the same interface $x=1/2$:",
             r"\[J_h^-(1/2,y_j)=-a_L\frac{U_{m,j}-U_{m-1,j}}h,\qquad J_h^+(1/2,y_j)=-a_R\frac{U_{m+2,j}-U_{m+1,j}}h.\]",
             r"Both are components in the common $+x$ direction, not opposing outward normals. Thus $\mathcal J_h=\max_j|J_h^+-J_h^-|$. The retained adjacent-face diagnostic is instead $D_h=\max_j\max(|J_{f,j}-J_h^-|,|J_{f,j}-J_h^+|)$. These definitions also specify the reconstruction for nonconstant computed fields. A small $\mathcal J_h$ measures agreement of the two reconstructed traces; either trace can still have an incorrect magnitude."]
    last = next(r for r in interface if r["N"] == 63 and r["mean"] == "arithmetic")
    text.append(f'For arithmetic averaging at $N=63$, $D_h={last["adjacent_flux_jump_max"]:.4g}$, $E_{{J,\\infty}}={last["interface_flux_error_max"]:.4g}$ and $\\mathcal J_h={last["reconstructed_flux_jump_max"]:.3g}$. The largest crossing-flux discrepancy occurs at $y={last["max_flux_error_y"]:.5g}$, near an interface endpoint. The four-grid observations do not establish a limiting constant.')
    with (out/"sine_fitted_orders.csv").open(newline="",encoding="utf8") as f:
        fits=list(csv.DictReader(f))
    text += [r"\FloatBarrier\subsection{Fitted sine orders and symmetry checks}",
             r"We fit $\log E=p\log h+c$ by ordinary least squares on the finest three existing grids $N=16,32,64$ ($h\in[1/64,1/16]$). Adjacent-grid orders in the earlier table and these fitted orders are distinct quantities. FDM uses the discrete nodal $L^2$ norm; FEM uses the continuous P1 error norm, so their absolute $L^2$ errors are not used to rank the methods.",
             r"\begin{table}[htbp]\centering\begin{tabular}{lrrr}\toprule Method & fitted $p_{L^2}$ & fitted $p_\infty$ & max $\delta_A$\\\midrule"]
    for r in fits:
        delta=max(v["symmetry_defect_fro"] for v in sine if v["method"]==r["method"])
        text.append(f'{r["method"]} & {float(r["fitted_order_l2"]):.4f} & {float(r["fitted_order_inf"]):.4f} & {delta:.3g} \\\\')
    text += [r"\bottomrule\end{tabular}\caption{Fitted sine orders on $N=16,32,64$ and maximum relative Frobenius symmetry defect over all four sine grids. Here $\delta_A=\|A-A^T\|_F/\|A\|_F$ for the reduced matrix after Dirichlet elimination. Fits and $R^2$ values are saved in \path{data/variable_diffusion/sine_fitted_orders.csv}; symmetry checks are in \path{data/variable_diffusion/sine_required.csv}.}\end{table}"]
    text.append((root/"report"/"variable_diffusion_theory.tex").read_text(encoding="utf-8"))

    (root/"report"/"variable_diffusion.tex").write_text("\n".join(text)+"\n",encoding="utf-8")
    return smooth,layered
