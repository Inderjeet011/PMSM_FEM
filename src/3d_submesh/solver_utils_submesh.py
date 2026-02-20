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
        # Fast run: 100 its max, looser inner solve; stop when ||b-Ax|| < 0.5
        outer_max_it=100,
        outer_rtol=0.0,
        outer_atol=0.5,
        outer_norm_type="unpreconditioned",  # "unpreconditioned"|"preconditioned"|"natural"|"none"
        ksp_A_max_it=8,
        ksp_A_restart=35,
        ksp_A_rtol=2e-2,
        # Regularization (disabled per request)
        epsilon_A=0.0,
        epsilon_A_spd=0.0,
        gauge_alpha=0.0,
        # Small sigma in air regions to help solver convergence (0 = use mesh_3D values only)
        sigma_air_min=1e-8,
        # Set PM conductivity used in the solve. Use 0.0 to avoid sigma/dt damping in PMs.
        sigma_pm_override=0.0,
        # Disable rotor/aluminium eddy-current damping for B-field recovery (debug/default).
        sigma_rotor_override=0.0,
        sigma_al_override=0.0,
        # Make coils conductive so the V-block is well-posed (otherwise sigma_Cu=0 in mesh_3D).
        sigma_cu_override=5.96e7,
        # Source: "current" = prescribed J in coils; "voltage" = potential difference (V on coils)
        source_type="current",
        voltage_amplitude=10.0,
        # --- Convergence postmortem: flip one at a time to see effect on rel_res ---
        use_schur=True,           # True = SCHUR fieldsplit + mat_nest as P; False = ADDITIVE + block diag P
        schur_pre_type="selfp",   # "a11"|"selfp" (Schur preconditioning strategy)
        schur_fact_type="lower",  # "lower"|"full" (Schur factorization)
        use_interior_nodes=True,   # True = pass interior nodes to AMS; False = skip
        use_motion_term=True,     # True = add -sigma*(u_rot × curl A)·v in a00 (dx_rpm); False = skip
        diagnostic_direct_solve=False,  # True = try one LU/MUMPS solve to check if system itself is OK
        ams_use_spd_pc=False,     # Use A00_full (not SPD approx) for AMS PC
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
    # Robust statistics: curl(A) is naturally DG0 for N1curl; compute max|B| from DG0.
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

    # Use DG0-derived magnitude for min/max/norm (more reliable than CG interpolation for diagnostics).
    if B_dg_mag.size:
        max_B = float(B_dg_mag.max())
        min_B = float(B_dg_mag.min())
        norm_B = float(np.linalg.norm(B_dg_mag))
    else:
        max_B = 0.0
        min_B = 0.0
        norm_B = 0.0
    return B_sol, B_mag, max_B, min_B, norm_B, B_dg


def assemble_rhs_submesh(a_blocks, L_blocks, block_bcs, A_space, V_space):
    """Assemble nested RHS vector from block forms (entity_maps)."""
    comm = A_space.mesh.comm
    nA = A_space.dofmap.index_map.size_global
    nV = V_space.dofmap.index_map.size_global

    tmp_bA = petsc.create_vector(A_space)
    petsc.assemble_vector(tmp_bA, L_blocks[0])
    # Apply lifting for coupling block with Dirichlet BCs on V.
    # Without this, the coupled RHS is inconsistent with the BC-eliminated matrix,
    # and the solve can severely distort A (and thus B).
    petsc.apply_lifting(tmp_bA, [a_blocks[0][1]], bcs=[block_bcs[1]])
    petsc.set_bc(tmp_bA, block_bcs[0])
    tmp_bA.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    tmp_bV = petsc.create_vector(V_space)
    petsc.assemble_vector(tmp_bV, L_blocks[1])
    # Apply lifting for coupling block with Dirichlet BCs on A.
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
    rhs = assemble_rhs_submesh(a_blocks, L_blocks, block_bcs, A_space, V_space)
    # IMPORTANT: use rhs.duplicate() to guarantee the nest layout matches mat_nest.
    # Manually creating sub-vectors can lead to VecNest layout mismatches that break MatNest.mult()
    # (and thus corrupt the reported true residual).
    sol = rhs.duplicate()
    sol.set(0.0)

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

    # Robust residual diagnostics: compute ||b-Ax|| and ||b|| on a monolithic AIJ system.
    # (VecNest norms / MatNest mult outputs are not consistently supported across PETSc builds.)
    A_mono = mat_nest.convert("aij")
    b_mono = A_mono.createVecRight()
    b_mono.set(0.0)
    x_mono = b_mono.duplicate()
    x_mono.set(0.0)

    # Build monolithic block index sets.
    # Prefer MatNest-provided IS (correct even if the monolithic ordering is not contiguous).
    try:
        isA_list, isV_list = mat_nest.getNestISs()
        isA = isA_list[0]
        isV = isV_list[1]
    except Exception:
        nA = A_space.dofmap.index_map.size_global
        nV = V_space.dofmap.index_map.size_global
        isA = PETSc.IS().createGeneral(np.arange(0, nA, dtype=np.int32), comm=comm)
        isV = PETSc.IS().createGeneral(np.arange(nA, nA + nV, dtype=np.int32), comm=comm)
    rhsA, rhsV = rhs.getNestSubVecs()
    xA, xV = sol.getNestSubVecs()
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
    r_mono.aypx(-1.0, b_mono)  # r = b - A x
    rhs_norm = float(b_mono.norm(PETSc.NormType.NORM_2))
    residual_norm = float(r_mono.norm(PETSc.NormType.NORM_2))
    relative_residual = residual_norm / rhs_norm if rhs_norm > 1e-30 else float("inf")

    A_sol = fem.Function(A_space, name="A")
    V_sol = fem.Function(V_space, name="V")

    xA_sub = sol.getNestSubVecs()[0]
    xV_sub = sol.getNestSubVecs()[1]

    # IMPORTANT: copy the solved sub-vectors into DOLFINx Functions directly.
    # Using intermediate vectors can silently drop/permute entries (and corrupt B diagnostics).
    A_arr = xA_sub.getArray(readonly=True)
    V_arr = xV_sub.getArray(readonly=True)
    A_sol.x.array[:] = A_arr[:A_sol.x.array.size]
    A_sol.x.scatter_forward()
    V_sol.x.array[:] = V_arr[:V_sol.x.array.size]
    V_sol.x.scatter_forward()

    A_prev.x.array[:] = A_sol.x.array[:]
    A_prev.x.scatter_forward()
    return A_sol, V_sol, residual_norm, rhs_norm, relative_residual
