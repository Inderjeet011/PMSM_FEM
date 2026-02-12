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
        outer_max_it=300,
        outer_rtol=1e-4,
        outer_norm_type="unpreconditioned",  # "unpreconditioned"|"preconditioned"|"natural"|"none"
        ksp_A_max_it=5,
        epsilon_A=1e-8,
        epsilon_A_spd=1e-6,
        # Small sigma in air regions to help solver convergence (0 = use mesh_3D values only)
        sigma_air_min=1e-8,
        # Source: "current" = prescribed J in coils; "voltage" = potential difference (V on coils)
        source_type="voltage",
        voltage_amplitude=10.0,
        # --- Convergence postmortem: flip one at a time to see effect on rel_res ---
        use_schur=False,           # True = SCHUR fieldsplit + mat_nest as P; False = ADDITIVE + block diag P
        use_interior_nodes=True,   # True = pass interior nodes to AMS; False = skip
        use_motion_term=True,     # True = add -sigma*(u_rot × curl A)·v in a00 (dx_rpm); False = skip
        diagnostic_direct_solve=False,  # True = try one LU/MUMPS solve to check if system itself is OK
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
    """Update coil currents on parent mesh (current source) or zero J for voltage drive."""
    if getattr(config, "source_type", "current") == "voltage":
        J_z.x.array[:] = 0.0
        return
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


def assemble_rhs_submesh(a_blocks, L_blocks, block_bcs, A_space, V_space):
    """Assemble nested RHS vector from block forms (entity_maps)."""
    comm = A_space.mesh.comm
    nA = A_space.dofmap.index_map.size_global
    nV = V_space.dofmap.index_map.size_global

    tmp_bA = petsc.create_vector(A_space)
    petsc.assemble_vector(tmp_bA, L_blocks[0])
    petsc.set_bc(tmp_bA, block_bcs[0])
    tmp_bA.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    tmp_bV = petsc.create_vector(V_space)
    petsc.assemble_vector(tmp_bV, L_blocks[1])
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
                           voltage_update_data=None):
    """Single time step solve for the submesh-based A–V system."""
    update_currents(cell_tags_parent, J_z, config, t)
    rotate_magnetization(cell_tags_parent, M_vec, config, t)

    if voltage_update_data is not None:
        u_voltage, voltage_dofs, phase = voltage_update_data
        V_amp = float(getattr(config, "voltage_amplitude", 10.0))
        omega = config.omega_e
        val = V_amp * np.sin(omega * t + phase)
        imap = V_space.dofmap.index_map
        local_start, local_end = imap.local_range
        for g in voltage_dofs:
            if local_start <= g < local_end:
                u_voltage.x.array[g - local_start] = val
        u_voltage.x.scatter_forward()

    comm = mesh_parent.comm
    nA = A_space.dofmap.index_map.size_global
    nV = V_space.dofmap.index_map.size_global
    rhs = assemble_rhs_submesh(a_blocks, L_blocks, block_bcs, A_space, V_space)

    xA = PETSc.Vec().createMPI(nA, comm=comm)
    xV = PETSc.Vec().createMPI(nV, comm=comm)
    xA.set(0.0)
    xV.set(0.0)
    sol = PETSc.Vec().createNest([xA, xV], comm=comm)

    # Diagnostic: try a direct solve (LU) on monolithic AIJ to determine if the system itself is solvable.
    # If this yields a tiny residual, the formulation/assembly is fine and the issue is the iterative solver/PC.
    if bool(getattr(config, "diagnostic_direct_solve", False)):
        try:
            A_mono = mat_nest.convert("aij")
            # Build monolithic RHS b_mono from nested rhs using the nest index sets
            b_mono = A_mono.createVecRight()
            b_mono.set(0.0)
            isA, isV = mat_nest.getNestISs()
            rhsA, rhsV = rhs.getNestSubVecs()
            scatA = PETSc.Scatter().create(rhsA, None, b_mono, isA[0])
            scatV = PETSc.Scatter().create(rhsV, None, b_mono, isV[1])
            scatA.scatter(rhsA, b_mono, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            scatV.scatter(rhsV, b_mono, addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            b_mono.assemble()
            x_mono = b_mono.duplicate()
            x_mono.set(0.0)

            ksp_dir = PETSc.KSP().create(comm)
            ksp_dir.setOperators(A_mono)
            ksp_dir.setType("preonly")
            pc_dir = ksp_dir.getPC()
            pc_dir.setType("lu")
            try:
                pc_dir.setFactorSolverType("mumps")
            except Exception:
                pass
            ksp_dir.setFromOptions()
            ksp_dir.solve(b_mono, x_mono)

            r = b_mono.duplicate()
            A_mono.mult(x_mono, r)    # r = A x
            r.aypx(-1.0, b_mono)      # r = b - A x
            bnorm = b_mono.norm(PETSc.NormType.NORM_2)
            rnorm = r.norm(PETSc.NormType.NORM_2)
            rel = rnorm / bnorm if bnorm > 1e-30 else float("inf")
            if comm.rank == 0:
                print(f"  [direct LU diagnostic] ||b-Ax||={rnorm:.6e}, ||b||={bnorm:.6e}, rel={rel:.4e}, its={ksp_dir.getIterationNumber()}, reason={ksp_dir.getConvergedReason()}")
        except Exception as e:
            if comm.rank == 0:
                print(f"  [direct LU diagnostic] failed: {type(e).__name__}: {e}")

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
