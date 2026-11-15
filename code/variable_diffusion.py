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
    """Compute all Section 9 data and figures without report-authoring inputs."""
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
    run_required_direction3(root)
    return smooth, layered
