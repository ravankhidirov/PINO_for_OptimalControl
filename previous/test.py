from fenics import *
from fenics_adjoint import *
from collections import OrderedDict
import os
import numpy as np
import random

random.seed(0)

tb = 3.0
te = 6.0

A  = random.uniform(-1.2, 1.2)
c0 = random.uniform(-1.0, 1.0)
c1 = random.uniform(-1.0, 1.0)
c2 = random.uniform(-1.0, 1.0)

random.seed(0)

b0 = random.uniform(-0.6, 0.6)
b1 = random.uniform(-1.0, 1.0)
b2 = random.uniform(-1.0, 1.0)
b3 = random.uniform(-1.0, 1.0)
b4 = random.uniform(-1.0, 1.0)
b5 = random.uniform(-1.0, 1.0)
b6 = random.uniform(-1.0, 1.0)

Asin  = 0.1
w     = 2.0*np.pi/2.5
phi   = 0.7

nu = Constant(1e-5)

mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, "CG", 1)

dt = Constant(0.1)
T = 10.0

data = Expression(
    "16*x[0]*(x[0]-1)*x[1]*(x[1]-1) * ("
    "B0 + B1*s + B2*pow(s,2) + B3*pow(s,3) + B4*pow(s,4) + B5*pow(s,5) + B6*pow(s,6)"
    " + I * A * pow(sb,2)*pow(1.0-sb,2) * (c0 + c1*sb + c2*pow(sb,2))"
    " + Asin * sin(w*t + phi)"
    ")",
    t=0.0,
    s=0.0,
    sb=0.0, I=0.0,
    B0=b0, B1=b1, B2=b2, B3=b3, B4=b4, B5=b5, B6=b6,
    tb=tb, te=te, A=A, c0=c0, c1=c1, c2=c2,
    Asin=Asin, w=w, phi=phi,
    degree=6
)

def make_ctrls():
    ctrls = OrderedDict()
    t = float(dt)
    while t <= T + 1e-12:
        ctrls[t] = Function(V)
        t += float(dt)
    return ctrls

def solve_heat(ctrls, store=False):
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Function(V, name="source")
    u_0 = Function(V, name="solution")
    d = Function(V, name="data")

    F = (((u - u_0) / dt) * v + nu * inner(grad(u), grad(v)) - f * v) * dx
    a, L = lhs(F), rhs(F)
    bc = DirichletBC(V, 0, "on_boundary")

    t = float(dt)

    j = 0.5 * float(dt) * assemble((u_0 - d) ** 2 * dx)

    times = []
    Us = []
    Ds = []
    Fs = []

    while t <= T + 1e-12:
        f.assign(ctrls[t])

        data.t = t
        data.s = t / T
        data.I  = 1.0 if (t >= tb and t <= te) else 0.0
        data.sb = (t - tb) / (te - tb) if data.I > 0.5 else 0.0

        d.assign(interpolate(data, V))

        solve(a == L, u_0, bc)

        if t > T - float(dt) + 1e-12:
            weight = 0.5
        else:
            weight = 1.0

        j += weight * float(dt) * assemble((u_0 - d) ** 2 * dx)

        if store:
            times.append(t)
            Us.append(u_0.copy(deepcopy=True))
            Ds.append(d.copy(deepcopy=True))
            Fs.append(f.copy(deepcopy=True))

        t += float(dt)

    if store:
        return u_0, d, j, times, Us, Ds, Fs
    return u_0, d, j



out_dir = "viz_time_control1_3"
os.makedirs(out_dir, exist_ok=True)

alpha_array = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]

from matplotlib import pyplot as plt, rc
rc("text", usetex=True)

all_times = None
all_d_vals = None
sol_by_alpha = {}
mae_by_alpha = {}

fig = plt.figure()
plot(mesh)
plt.scatter([0.5], [0.5], s=40)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.gca().set_aspect("equal")
plt.savefig(os.path.join(out_dir, "geometry_mesh_point.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

for a_val in alpha_array:
    alpha = Constant(a_val)

    a_dir = os.path.join(out_dir, f"alpha_{a_val:g}")
    os.makedirs(a_dir, exist_ok=True)

    ctrls = make_ctrls()

    u, d, j = solve_heat(ctrls)

    regularisation = alpha / 2 * sum(
        [1 / dt * (fb - fa) ** 2 * dx
         for fb, fa in zip(list(ctrls.values())[1:], list(ctrls.values())[:-1])]
    )

    J = j + assemble(regularisation)

    m = [Control(c) for c in ctrls.values()]
    rf = ReducedFunctional(J, m)
    opt_ctrls = minimize(rf, options={"maxiter": 50})

    opt_ctrls_dict = OrderedDict(zip(list(ctrls.keys()), opt_ctrls))
    u_opt, d_opt, j_opt, times, Us, Ds, Fs = solve_heat(opt_ctrls_dict, store=True)

    u_vals = np.array([ui((0.5, 0.5)) for ui in Us], dtype=float)
    d_vals = np.array([di((0.5, 0.5)) for di in Ds], dtype=float)

    mae = float(np.mean(np.abs(u_vals - d_vals)))
    mae_by_alpha[a_val] = mae
    sol_by_alpha[a_val] = u_vals

    if all_times is None:
        all_times = times
        all_d_vals = d_vals

    x = [c((0.5, 0.5)) for c in opt_ctrls]
    fig = plt.figure()
    plt.plot(times, x, marker="o")
    plt.xlabel("t")
    plt.ylabel("f(t) at (0.5,0.5)")
    plt.title(r"$\alpha={}$".format(a_val))
    plt.ylim([-3, 3])
    plt.savefig(os.path.join(a_dir, "control_point_trace.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


    src_l2 = [np.sqrt(assemble(fi * fi * dx)) for fi in Fs]
    mis_l2 = [np.sqrt(assemble((ui - di) * (ui - di) * dx)) for ui, di in zip(Us, Ds)]

    fig = plt.figure()
    plt.plot(times, src_l2, marker="o")
    plt.xlabel("t")
    plt.ylabel(r"$\|f(\cdot,t)\|_{L^2}$")
    plt.title(r"$\alpha={}$".format(a_val))
    plt.savefig(os.path.join(a_dir, "source_L2_over_time.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure()
    plt.plot(times, mis_l2, marker="o")
    plt.xlabel("t")
    plt.ylabel(r"$\|u(\cdot,t)-d(\cdot,t)\|_{L^2}$")
    plt.title(r"$\alpha={}$".format(a_val))
    plt.savefig(os.path.join(a_dir, "misfit_L2_over_time.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(times, d_vals, marker="o", markersize=5, linewidth=2.0, label="Data")
    ax.plot(times, u_vals, marker="o", markersize=5, linewidth=2.0, label="Solution")
    ax.set_xlabel(r"$t$", fontsize=12)
    ax.set_ylabel(r"Value at $(0.5,0.5)$", fontsize=12)
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(frameon=True, fontsize=11, loc="best")
    ax.set_title(r"$\alpha={}$, MAE={:.4f}".format(a_val, mae))
    fig.tight_layout()
    fig.savefig(os.path.join(a_dir, "solution_vs_data_point.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(all_times, all_d_vals, marker="o", markersize=4, linewidth=2.0, label="Data")

for a_val in alpha_array:
    u_vals = sol_by_alpha[a_val]
    ax.plot(all_times, u_vals, marker="o", markersize=3, linewidth=2.0,
            label=r"$\alpha={}$ (MAE={:.4f})".format(a_val, mae_by_alpha[a_val]))

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"Value at $(0.5,0.5)$")
ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.6)
ax.legend(frameon=True, fontsize=9, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "all_alphas_solution_vs_data_point.png"), dpi=300, bbox_inches="tight")
plt.close(fig)