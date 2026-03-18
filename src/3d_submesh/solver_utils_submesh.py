"""Helper functions for 3D submesh-based solver."""

import numpy as np
from pathlib import Path
from types import SimpleNamespace

from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "3d"))
from load_mesh import MAGNETS
sys.path.insert(0, str(Path(__file__).parent))
from load_mesh_submesh import COIL_DRIVE
from entity_map_utils import get_entity_map


def make_config():
    from mesh_3D import model_parameters

    freq = float(model_parameters["freq"])
    pole_pairs = 5
    steps_per_period = 10  # fewer steps → larger dt → smaller inv_dt, helps convergence with coupling 1.0
    dt = (1.0 / freq) / steps_per_period
    omega_e = 2.0 * np.pi * freq
    rotation_direction = -1.0
    omega_m = rotation_direction * (omega_e / max(pole_pairs, 1))

    root = Path(__file__).parents[2]
    return SimpleNamespace(
        dt=dt,
        num_steps=20,  # ~1 electrical period; rotor rotates ~72° for visible animation
        degree_A=1,
        degree_V=1,
        mu0=float(model_parameters["mu_0"]),
        # Restore permanent magnet excitation
        magnet_remanence=1.2,
        omega_e=omega_e,
        omega_m=omega_m,
        mesh_path=root / "meshes" / "3d" / "pmesh3D_ipm.xdmf",
        results_path=root / "results" / "3d_submesh" / "av_solver_submesh.xdmf",
        write_results=True,
        outer_max_it=500,
        outer_atol=9e-3,  # outer KSP: stop when ||r|| <= outer_atol (no relative tol)
        ksp_A_max_it=15,
        ksp_A_restart=35,
        ksp_A_rtol=2e-5,
        A_pc="boomeramg",  # options: "boomeramg", "gamg"
        # Restore applied coil voltage for field visualization
        V_amp=10,  # [V] peak voltage applied to drive coil: V(t) = V_amp * sin(omega_e * t)
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


def update_currents(J_z, mesh_parent, cell_tags_parent, config, t):
    """Set J_z: uniform J = I(t)/V_drive in drive coil (current source)."""
    J_z.x.array[:] = 0.0
    I_amp = float(getattr(config, "I_amp", 10.0))
    drive_volume = float(getattr(config, "drive_coil_volume", 1.0))
    if drive_volume <= 0:
        return
    I_t = I_amp * np.sin(config.omega_e * t)
    j_z_val = I_t / drive_volume
    coil_drive = COIL_DRIVE
    cells_drive = cell_tags_parent.find(coil_drive)
    if cells_drive.size > 0:
        J_z.x.array[cells_drive] = j_z_val
    J_z.x.scatter_forward()


def compute_B_field(mesh, A_sol, B_space, B_magnitude_space):
    """Compute B field from A (on parent mesh) and basic statistics."""
    curlA = ufl.curl(A_sol)
    DG_vec = fem.functionspace(mesh, ("DG", 0, (3,)))
    B_dg = fem.Function(DG_vec, name="B_dg")
    B_dg.interpolate(fem.Expression(curlA, DG_vec.element.interpolation_points))
    B_dg.x.scatter_forward()
    B_dg_vals = B_dg.x.array.reshape((-1, 3))
    B_dg_mag = np.linalg.norm(B_dg_vals, axis=1) if B_dg_vals.size else np.array([], dtype=float)

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

    max_B = float(B_dg_mag.max()) if B_dg_mag.size else 0.0
    avg_B = float(B_dg_mag.mean()) if B_dg_mag.size else 0.0
    return B_sol, B_mag, B_dg, max_B, avg_B


def compute_J_field_conductor(
    mesh_parent,
    mesh_conductor,
    entity_map,
    A_sol,
    A_prev,
    V_sol,
    sigma_parent,
    dt,
    degree=1,
):
    """Compute current density J = sigma*E on conductor submesh, with E = -grad(V) - dA/dt."""
    tdim = mesh_parent.topology.dim
    n_conductor_cells = mesh_conductor.topology.index_map(tdim).size_local + mesh_conductor.topology.index_map(tdim).num_ghosts
    parent_cells = get_entity_map(entity_map, inverse=False)
    conductor_cells = np.arange(n_conductor_cells, dtype=np.int32)

    # DG0 vector spaces
    DG0_vec_parent = fem.functionspace(mesh_parent, ("DG", 0, (mesh_parent.geometry.dim,)))
    DG0_vec_conductor = fem.functionspace(mesh_conductor, ("DG", 0, (mesh_conductor.geometry.dim,)))
    DG0_conductor = fem.functionspace(mesh_conductor, ("DG", 0))

    # A and A_prev on parent as DG0 vector (for cell-wise copy to conductor)
    A_vis = fem.Function(DG0_vec_parent, name="A_vis")
    A_prev_vis = fem.Function(DG0_vec_parent, name="A_prev_vis")
    A_vis.interpolate(fem.Expression(A_sol, DG0_vec_parent.element.interpolation_points))
    A_prev_vis.interpolate(fem.Expression(A_prev, DG0_vec_parent.element.interpolation_points))
    A_vis.x.scatter_forward()
    A_prev_vis.x.scatter_forward()

    A_conductor = fem.Function(DG0_vec_conductor, name="A_cond")
    A_prev_conductor = fem.Function(DG0_vec_conductor, name="A_prev_cond")
    A_conductor.interpolate(A_vis, cells0=parent_cells, cells1=conductor_cells)
    A_prev_conductor.interpolate(A_prev_vis, cells0=parent_cells, cells1=conductor_cells)
    A_conductor.x.scatter_forward()
    A_prev_conductor.x.scatter_forward()

    # dA/dt on conductor
    inv_dt = 1.0 / max(dt, 1e-30)
    dA_dt = fem.Function(DG0_vec_conductor, name="dA_dt")
    dA_dt.x.array[:] = (A_conductor.x.array - A_prev_conductor.x.array) * inv_dt
    dA_dt.x.scatter_forward()

    # -grad(V) on conductor
    grad_V = fem.Function(DG0_vec_conductor, name="grad_V")
    grad_V_expr = ufl.grad(V_sol)
    grad_V.interpolate(fem.Expression(grad_V_expr, DG0_vec_conductor.element.interpolation_points))
    grad_V.x.scatter_forward()

    # E = -grad(V) - dA/dt
    E_conductor = fem.Function(DG0_vec_conductor, name="E")
    E_conductor.x.array[:] = -grad_V.x.array - dA_dt.x.array
    E_conductor.x.scatter_forward()

    # sigma on conductor from parent (cell-wise copy via entity map)
    sigma_conductor = fem.Function(DG0_conductor, name="sigma_cond")
    for c in range(min(n_conductor_cells, parent_cells.size)):
        pc = int(parent_cells[c])
        if 0 <= pc < sigma_parent.x.array.size:
            sigma_conductor.x.array[c] = sigma_parent.x.array[pc]
    sigma_conductor.x.scatter_forward()

    # J = sigma * E
    J_conductor = fem.Function(DG0_vec_conductor, name="J")
    sig = sigma_conductor.x.array
    E_arr = E_conductor.x.array.reshape((-1, 3))
    J_conductor.x.array[:] = (sig[:, np.newaxis] * E_arr).ravel()
    J_conductor.x.scatter_forward()

    return J_conductor


def assemble_rhs_submesh(a_blocks, L_blocks, block_bcs, A_space, V_space):
    comm = A_space.mesh.comm
    nA = A_space.dofmap.index_map.size_global
    nV = V_space.dofmap.index_map.size_global

    tmp_bA = petsc.create_vector(A_space)
    petsc.assemble_vector(tmp_bA, L_blocks[0])
    petsc.apply_lifting(tmp_bA, [a_blocks[0][1]], bcs=[block_bcs[1]])
    petsc.set_bc(tmp_bA, block_bcs[0])
    tmp_bA.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    tmp_bV = petsc.create_vector(V_space)
    petsc.assemble_vector(tmp_bV, L_blocks[1])
    petsc.apply_lifting(tmp_bV, [a_blocks[1][0]], bcs=[block_bcs[0]])
    petsc.set_bc(tmp_bV, block_bcs[1])
    tmp_bV.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    bA = PETSc.Vec().createMPI(nA, comm=comm)
    bV = PETSc.Vec().createMPI(nV, comm=comm)
    bA.set(0.0)
    bV.set(0.0)
    tmp_bA.copy(bA)
    tmp_bV.copy(bV)

    return PETSc.Vec().createNest([bA, bV], comm=comm)


def solve_one_step_submesh(mesh_parent, A_space, V_space,
                           cell_tags_parent, config, ksp, mat_nest,
                           a_blocks, L_blocks, block_bcs,
                           J_z, M_vec, A_prev, t):
    rotate_magnetization(cell_tags_parent, M_vec, config, t)

    comm = mesh_parent.comm
    rhs = assemble_rhs_submesh(a_blocks, L_blocks, block_bcs, A_space, V_space)

    sol = rhs.duplicate()
    sol.set(0.0)

    ksp.solve(rhs, sol)

    A_mono = mat_nest.convert("aij")
    b_mono = A_mono.createVecRight()
    b_mono.set(0.0)
    x_mono = b_mono.duplicate()
    x_mono.set(0.0)

    try:
        isA_list, isV_list = mat_nest.getNestISs()
        isA = isA_list[0]
        isV = isV_list[1]
    except PETSc.Error:
        nA = A_space.dofmap.index_map.size_global
        nV = V_space.dofmap.index_map.size_global
        isA = PETSc.IS().createGeneral(np.arange(0, nA, dtype=np.int32), comm=comm)
        isV = PETSc.IS().createGeneral(np.arange(nA, nA + nV, dtype=np.int32), comm=comm)
    rhsA, rhsV = rhs.getNestSubVecs()
    xA, xV = sol.getNestSubVecs()
    A_arr = xA.getArray(readonly=True)
    V_arr = xV.getArray(readonly=True)
    PETSc.Scatter().create(rhsA, None, b_mono, isA).scatter(
        rhsA, b_mono, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    PETSc.Scatter().create(rhsV, None, b_mono, isV).scatter(
        rhsV, b_mono, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    PETSc.Scatter().create(xA, None, x_mono, isA).scatter(
        xA, x_mono, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    PETSc.Scatter().create(xV, None, x_mono, isV).scatter(
        xV, x_mono, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
    )
    b_mono.assemble()
    x_mono.assemble()

    r_mono = b_mono.duplicate()
    A_mono.mult(x_mono, r_mono)
    r_mono.aypx(-1.0, b_mono)
    rhs_norm = float(b_mono.norm(PETSc.NormType.NORM_2))
    residual_norm = float(r_mono.norm(PETSc.NormType.NORM_2))

    A_sol = fem.Function(A_space, name="A")
    V_sol = fem.Function(V_space, name="V")
    A_sol.x.array[:] = A_arr[:A_sol.x.array.size]
    A_sol.x.scatter_forward()
    V_sol.x.array[:] = V_arr[:V_sol.x.array.size]
    V_sol.x.scatter_forward()

    A_prev.x.array[:] = A_sol.x.array[:]
    A_prev.x.scatter_forward()
    return A_sol, V_sol, residual_norm, rhs_norm
