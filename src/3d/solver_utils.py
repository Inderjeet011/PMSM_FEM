"""Small helper functions for the 3D solver."""

import numpy as np
from pathlib import Path
from types import SimpleNamespace

from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl

from load_mesh import CURRENT_MAP, MAGNETS


def make_config():
    from mesh_3D import model_parameters

    freq = float(model_parameters["freq"])
    pole_pairs = 2
    steps_per_period = 3
    dt = (1.0 / freq) / steps_per_period

    omega_e = 2.0 * np.pi * freq
    omega_m = omega_e / max(pole_pairs, 1)

    root = Path(__file__).parents[2]
    return SimpleNamespace(
        dt=dt,
        # Keep this small for quick runs / debugging / visualization
        num_steps=8,
        degree_A=1,
        degree_V=1,
        mu0=float(model_parameters["mu_0"]),
        coil_current_peak=float(model_parameters["J"]),
        magnet_remanence=1.2,
        omega_e=omega_e,
        omega_m=omega_m,
        mesh_path=root / "meshes" / "3d" / "pmesh3D_ipm.xdmf",
        results_path=root / "results" / "3d" / "av_solver.xdmf",
        write_results=True,
        # PETSc iteration limits: bump these if you see non-convergence
        outer_max_it=100,
        ksp_A_max_it=5,
    )


def measure_over(dx, markers):
    measure = None
    for marker in markers:
        term = dx(marker)
        measure = term if measure is None else measure + term
    return measure


def make_ground_bc_V(V_space, cell_tags, conductor_markers):
    for m in conductor_markers:
        cells = cell_tags.find(m)
        if cells.size == 0:
            continue
        dofs = V_space.dofmap.cell_dofs(int(cells[0]))
        if len(dofs) == 0:
            continue
        u0 = fem.Function(V_space)
        u0.x.array[:] = 0.0
        return fem.dirichletbc(u0, np.array([int(dofs[0])], dtype=np.int32))
    u0 = fem.Function(V_space)
    u0.x.array[:] = 0.0
    return fem.dirichletbc(u0, np.array([], dtype=np.int32))


def setup_sources(mesh):
    DG0 = fem.functionspace(mesh, ("DG", 0))
    DG0_vec = fem.functionspace(mesh, ("DG", 0, (3,)))
    return fem.Function(DG0, name="Jz"), fem.Function(DG0_vec, name="M")


def initialise_magnetisation(mesh, cell_tags, M_vec, config):
    dofmap = mesh.geometry.dofmap
    coords = mesh.geometry.x
    vec_view = M_vec.x.array.reshape((-1, 3))
    magnitude = config.magnet_remanence / max(config.mu0, 1e-12)
    for marker in MAGNETS:
        cells = cell_tags.find(marker)
        for c in cells:
            geom_dofs = dofmap[c]
            cell_coords = coords[geom_dofs]
            cx = float(np.mean(cell_coords[:, 0]))
            cy = float(np.mean(cell_coords[:, 1]))
            norm = np.hypot(cx, cy)
            direction = np.array([1.0, 0.0, 0.0]) if norm < 1e-12 else np.array([cx / norm, cy / norm, 0.0])
            vec_view[c, :] = magnitude * direction
    M_vec.x.scatter_forward()


def rotate_magnetization(cell_tags, M_vec, config, t):
    theta_rot = config.omega_m * t
    vec_view = M_vec.x.array.reshape((-1, 3))
    magnitude = config.magnet_remanence / max(config.mu0, 1e-12)
    pm_spacing = (np.pi / 6) + (np.pi / 30)
    pm_angles = np.asarray([i * pm_spacing for i in range(10)])
    for marker in MAGNETS:
        cells = cell_tags.find(marker)
        sign = 1 if ((marker - 13) % 2 == 0) else -1
        theta_now = pm_angles[marker - 13] + theta_rot
        for c in cells:
            vec_view[c, 0] = sign * magnitude * np.cos(theta_now)
            vec_view[c, 1] = sign * magnitude * np.sin(theta_now)
            vec_view[c, 2] = 0.0
    M_vec.x.scatter_forward()


def update_currents(cell_tags, J_z, config, t):
    omega = config.omega_e
    J_peak = config.coil_current_peak
    J_z.x.array[:] = 0.0
    for marker, meta in CURRENT_MAP.items():
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        J_z.x.array[cells] = J_peak * meta["alpha"] * np.sin(omega * t + meta["beta"])


def assemble_rhs(a_blocks, L_blocks, block_bcs):
    bA = petsc.create_vector(L_blocks[0])
    petsc.assemble_vector(bA, L_blocks[0])
    petsc.apply_lifting(bA, [a_blocks[0][1]], bcs=[block_bcs[1]])
    petsc.set_bc(bA, block_bcs[0])
    bA.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bV = petsc.create_vector(L_blocks[1])
    petsc.assemble_vector(bV, L_blocks[1])
    petsc.apply_lifting(bV, [a_blocks[1][0]], bcs=[block_bcs[0]])
    petsc.set_bc(bV, block_bcs[1])
    bV.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    return PETSc.Vec().createNest([bA, bV])


def solve_one_step(mesh, A_space, V_space, cell_tags, config, ksp, mat_nest, a_blocks,
                   L_blocks, block_bcs, J_z, M_vec, A_prev, t):
    update_currents(cell_tags, J_z, config, t)
    rotate_magnetization(cell_tags, M_vec, config, t)

    rhs = assemble_rhs(a_blocks, L_blocks, block_bcs)
    sol = PETSc.Vec().createNest([petsc.create_vector(L_blocks[0]), petsc.create_vector(L_blocks[1])], comm=mesh.comm)
    sol.set(0.0)
    ksp.solve(rhs, sol)

    A_sol = fem.Function(A_space, name="A")
    V_sol = fem.Function(V_space, name="V")
    with sol.getNestSubVecs()[0].localForm() as src:
        A_sol.x.array[:] = src.array_r[:A_sol.x.array.size]
    A_sol.x.scatter_forward()
    with sol.getNestSubVecs()[1].localForm() as src:
        V_sol.x.array[:] = src.array_r[:V_sol.x.array.size]
    V_sol.x.scatter_forward()

    A_prev.x.array[:] = A_sol.x.array[:]
    return A_sol, V_sol


def compute_B_field(mesh, A_sol, B_space, B_magnitude_space):
    curlA = ufl.curl(A_sol)

    DG_vec = fem.functionspace(mesh, ("DG", 0, (3,)))
    B_dg = fem.Function(DG_vec, name="B_dg")
    B_dg.interpolate(fem.Expression(curlA, DG_vec.element.interpolation_points))
    B_dg.x.scatter_forward()

    B_sol = fem.Function(B_space, name="B")
    B_sol.interpolate(fem.Expression(curlA, B_space.element.interpolation_points))
    B_sol.x.scatter_forward()

    B_mag = fem.Function(B_magnitude_space, name="B_Magnitude")
    B_mag.interpolate(
        fem.Expression(
            ufl.sqrt(ufl.inner(curlA, curlA)),
            B_magnitude_space.element.interpolation_points,
        )
    )
    B_mag.x.scatter_forward()

    vals = B_mag.x.array
    return B_sol, B_mag, float(vals.max()), float(vals.min()), float(np.linalg.norm(vals)), B_dg


