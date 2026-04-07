import matplotlib as mpl
import ufl
import numpy as np
import pyvista
from petsc4py import PETSc
from mpi4py import MPI

from dolfinx import fem, mesh, io, plot
from dolfinx.fem.petsc import (
    assemble_vector,
    assemble_matrix,
    apply_lifting,
    set_bc,
)
import gmsh
import meshio

print("Imported!!")

gmsh.initialize()
gmsh.model.add("complex")

lc = 0.05

outer_pts = [
    (0, 0), (1, 0), (1, 0.3), (1.5, 0.3), (1.5, 0),
    (3, 0), (3, 1), (2.5, 1), (2.5, 1.5), (3, 1.5),
    (3, 3), (2, 3), (2, 2.5), (1.5, 2.5), (1.5, 3),
    (0, 3), (0, 2), (0.5, 2), (0.5, 1.5), (0, 1.5),
]

pts = [gmsh.model.geo.addPoint(x, y, 0, lc) for x, y in outer_pts]

lines = []
for i in range(len(pts)):
    lines.append(gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)]))

outer_loop = gmsh.model.geo.addCurveLoop(lines)


def make_circle_hole(cx, cy, r, lc_val):
    center = gmsh.model.geo.addPoint(cx, cy, 0, lc_val)
    p1 = gmsh.model.geo.addPoint(cx + r, cy, 0, lc_val)
    p2 = gmsh.model.geo.addPoint(cx, cy + r, 0, lc_val)
    p3 = gmsh.model.geo.addPoint(cx - r, cy, 0, lc_val)
    p4 = gmsh.model.geo.addPoint(cx, cy - r, 0, lc_val)
    a1 = gmsh.model.geo.addCircleArc(p1, center, p2)
    a2 = gmsh.model.geo.addCircleArc(p2, center, p3)
    a3 = gmsh.model.geo.addCircleArc(p3, center, p4)
    a4 = gmsh.model.geo.addCircleArc(p4, center, p1)
    loop = gmsh.model.geo.addCurveLoop([a1, a2, a3, a4])
    return loop, [a1, a2, a3, a4]


hole1_loop, hole1_arcs = make_circle_hole(1.5, 1.5, 0.3, lc)
hole2_loop, hole2_arcs = make_circle_hole(0.8, 0.8, 0.2, lc)
hole3_loop, hole3_arcs = make_circle_hole(2.3, 2.2, 0.15, lc)

surface = gmsh.model.geo.addPlaneSurface([outer_loop, hole1_loop, hole2_loop, hole3_loop])

gmsh.model.geo.synchronize()

gmsh.model.addPhysicalGroup(2, [surface], tag=1, name="domain")
gmsh.model.addPhysicalGroup(1, [lines[0], lines[4]], tag=10, name="hot_wall")
other_lines = [l for i, l in enumerate(lines) if i not in [0, 4]]
gmsh.model.addPhysicalGroup(1, other_lines, tag=20, name="cold_wall")
gmsh.model.addPhysicalGroup(1, hole1_arcs, tag=30, name="hole1")
gmsh.model.addPhysicalGroup(1, hole2_arcs, tag=31, name="hole2")
gmsh.model.addPhysicalGroup(1, hole3_arcs, tag=32, name="hole3")

gmsh.model.mesh.generate(2)
gmsh.write("complex.msh")
gmsh.finalize()

msh = meshio.read("complex.msh")

triangle_cells = meshio.Mesh(
    points=msh.points[:, :2],
    cells=[("triangle", c.data) for c in msh.cells if c.type == "triangle"],
)
meshio.write("complex.xdmf", triangle_cells)

with io.XDMFFile(MPI.COMM_WORLD, "complex.xdmf", "r") as xdmf_in:
    domain = xdmf_in.read_mesh(name="Grid")

V = fem.functionspace(domain, ("Lagrange", 1))

t = 0.0
T = 10.0
num_steps = 200
dt = T / num_steps


def initial_condition(x):
    return np.zeros(x.shape[1])


u_n = fem.Function(V)
u_n.name = "u_n"
u_n.interpolate(initial_condition)

fdim = domain.topology.dim - 1


def hot_wall(x):
    return np.isclose(x[1], 0.0, atol=1e-6)


def cold_wall(x):
    return np.isclose(x[1], 3.0, atol=1e-6)


hot_facets = mesh.locate_entities_boundary(domain, fdim, hot_wall)
cold_facets = mesh.locate_entities_boundary(domain, fdim, cold_wall)

bc_hot = fem.dirichletbc(
    PETSc.ScalarType(100.0),
    fem.locate_dofs_topological(V, fdim, hot_facets),
    V,
)
bc_cold = fem.dirichletbc(
    PETSc.ScalarType(0.0),
    fem.locate_dofs_topological(V, fdim, cold_facets),
    V,
)

bcs = [bc_hot, bc_cold]

uh = fem.Function(V)
uh.name = "uh"
uh.interpolate(initial_condition)

xdmf = io.XDMFFile(domain.comm, "diffusion2.xdmf", "w")
xdmf.write_mesh(domain)
xdmf.write_function(uh, t)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
f = fem.Constant(domain, PETSc.ScalarType(0))
a = u * v * ufl.dx + dt * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = (u_n + dt * f) * v * ufl.dx

bilinear_form = fem.form(a)
linear_form = fem.form(L)

A = assemble_matrix(bilinear_form, bcs=bcs)
A.assemble()
b = A.createVecRight()

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)

grid = pyvista.UnstructuredGrid(*plot.vtk_mesh(V))
grid.point_data["uh"] = uh.x.array
warped = grid.warp_by_scalar("uh", factor=0.01)

for i in range(num_steps):
    t += dt
    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b, linear_form)
    apply_lifting(b, [bilinear_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    solver.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()
    u_n.x.array[:] = uh.x.array
    xdmf.write_function(uh, t)
    new_warped = grid.warp_by_scalar("uh", factor=0.01)
    warped.points[:, :] = new_warped.points
    warped.point_data["uh"][:] = uh.x.array
    print(f"Step {i+1}/{num_steps}, t={t:.4f}, min(T)={uh.x.array.min():.2f}, max(T)={uh.x.array.max():.2f}")

xdmf.close()
A.destroy()
b.destroy()
solver.destroy()
print("Done!")