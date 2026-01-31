"""Helper functions for 3D submesh-based solver."""

import numpy as np
from pathlib import Path
from types import SimpleNamespace
from mpi4py import MPI

from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "3d"))
from load_mesh import CURRENT_MAP, MAGNETS, AIR_GAP


def make_config():
    """Create configuration (same as original)."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "3d"))
    from mesh_3D import model_parameters

    freq = float(model_parameters["freq"])
    pole_pairs = 5
    steps_per_period = 40
    dt = (1.0 / freq) / steps_per_period
    omega_e = 2.0 * np.pi * freq
    rotation_direction = -1.0
    omega_m = rotation_direction * (omega_e / max(pole_pairs, 1))

    root = Path(__file__).parents[2]
    return SimpleNamespace(
        dt=dt,
        num_steps=1,
        degree_A=1,
        degree_V=1,
        mu0=float(model_parameters["mu_0"]),
        coil_current_peak=float(model_parameters["J"]),
        magnet_remanence=1.2,
        omega_e=omega_e,
        omega_m=omega_m,
        mesh_path=root / "meshes" / "3d" / "pmesh3D_ipm.xdmf",
        results_path=root / "results" / "3d_submesh" / "av_solver_submesh.xdmf",
        write_results=True,
        outer_max_it=100,
        ksp_A_max_it=5,
        epsilon_A=1e-8,
        epsilon_A_spd=1e-6,
    )


def measure_over(dx, markers):
    """Create measure over multiple markers."""
    measure = None
    for marker in markers:
        term = dx(marker)
        measure = term if measure is None else measure + term
    return measure


def setup_sources(mesh):
    """Setup source functions on parent mesh."""
    DG0 = fem.functionspace(mesh, ("DG", 0))
    DG0_vec = fem.functionspace(mesh, ("DG", 0, (3,)))
    return fem.Function(DG0, name="Jz"), fem.Function(DG0_vec, name="M")


def initialise_magnetisation(mesh, cell_tags, M_vec, config):
    """Initialize magnetization on parent mesh."""
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
    """Rotate magnetization on parent mesh."""
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
    """Update coil currents on parent mesh."""
    omega = config.omega_e
    J_peak = config.coil_current_peak
    J_z.x.array[:] = 0.0
    for marker, meta in CURRENT_MAP.items():
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        J_z.x.array[cells] = J_peak * meta["alpha"] * np.sin(omega * t + meta["beta"])


def compute_B_field(mesh, A_sol, B_space, B_magnitude_space):
    """Compute B field from A (on parent mesh)."""
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


def assemble_rhs_submesh(a_blocks, L_blocks, block_bcs, A_space, V_space,
                        mesh_parent=None, mesh_conductor=None, sigma_submesh=None,
                        A_prev=None, dof_mapper=None, config=None):
    """Assemble nested RHS vector for the submesh-based A–V system."""
    comm = A_space.mesh.comm
    nA = A_space.dofmap.index_map.size_global
    nV = V_space.dofmap.index_map.size_global

    tmp_bA = petsc.create_vector(A_space)
    petsc.assemble_vector(tmp_bA, L_blocks[0])
    petsc.set_bc(tmp_bA, block_bcs[0])
    tmp_bA.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    tmp_bV = petsc.create_vector(V_space)
    if (mesh_parent is not None and mesh_conductor is not None and sigma_submesh is not None
            and A_prev is not None and dof_mapper is not None and config is not None):
        from assemble_coupling_A10_quadrature_direct import assemble_L1_rhs_quadrature
        inv_dt = 1.0 / config.dt
        L1_vec = assemble_L1_rhs_quadrature(
            mesh_parent, mesh_conductor, A_space, V_space,
            sigma_submesh, A_prev, inv_dt, dof_mapper, config
        )
        L1_vec.copy(tmp_bV)
    else:
        tmp_bV.set(0.0)

    petsc.set_bc(tmp_bV, block_bcs[1])
    tmp_bV.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bA = PETSc.Vec().createMPI(nA, comm=comm)
    bV = PETSc.Vec().createMPI(nV, comm=comm)
    bA.set(0.0)
    bV.set(0.0)
    tmp_bA.copy(bA)
    tmp_bV.copy(bV)

    return PETSc.Vec().createNest([bA, bV], comm=comm)


def solve_one_step_submesh(mesh_parent, mesh_conductor, A_space, V_space,
                           cell_tags_parent, config, ksp, mat_nest,
                           a_blocks, L_blocks, block_bcs,
                           J_z, M_vec, A_prev, t,
                           sigma_submesh=None, dof_mapper=None):
    """Single time step solve for the submesh-based A–V system."""
    update_currents(cell_tags_parent, J_z, config, t)
    rotate_magnetization(cell_tags_parent, M_vec, config, t)

    comm = mesh_parent.comm
    nA = A_space.dofmap.index_map.size_global
    nV = V_space.dofmap.index_map.size_global
    rhs = assemble_rhs_submesh(
        a_blocks, L_blocks, block_bcs, A_space, V_space,
        mesh_parent=mesh_parent, mesh_conductor=mesh_conductor,
        sigma_submesh=sigma_submesh, A_prev=A_prev,
        dof_mapper=dof_mapper, config=config,
    )

    xA = PETSc.Vec().createMPI(nA, comm=comm)
    xV = PETSc.Vec().createMPI(nV, comm=comm)
    xA.set(0.0)
    xV.set(0.0)
    sol = PETSc.Vec().createNest([xA, xV], comm=comm)

    ksp.solve(rhs, sol)

    rhs_norm = rhs.norm(PETSc.NormType.NORM_2)
    Ax = rhs.duplicate()
    mat_nest.mult(sol, Ax)
    res = rhs.duplicate()
    res.copy(rhs)
    res.axpy(-1.0, Ax)
    residual_norm = res.norm(PETSc.NormType.NORM_2)
    relative_residual = residual_norm / rhs_norm if rhs_norm > 1e-30 else float("inf")

    A_sol = fem.Function(A_space, name="A")
    V_sol = fem.Function(V_space, name="V")

    tmp_xA = petsc.create_vector(A_space)
    tmp_xV = petsc.create_vector(V_space)
    xA_sub = sol.getNestSubVecs()[0]
    xV_sub = sol.getNestSubVecs()[1]
    xA_sub.copy(tmp_xA)
    xV_sub.copy(tmp_xV)

    A_sol.x.array[:] = tmp_xA.array[:A_sol.x.array.size]
    A_sol.x.scatter_forward()
    V_sol.x.array[:] = tmp_xV.array[:V_sol.x.array.size]
    V_sol.x.scatter_forward()

    A_prev.x.array[:] = A_sol.x.array[:]
    return A_sol, V_sol, residual_norm, rhs_norm, relative_residual
