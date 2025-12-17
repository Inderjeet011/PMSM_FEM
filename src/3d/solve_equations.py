"""3D A-V eddy-current solver for PMSM.

This module builds the variational forms, assembles matrices, and configures
the linear solver for the coupled A-V (magnetic vector potential and scalar
potential) eddy-current problem with rotating permanent magnets and time-varying
coil currents.
"""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from petsc4py import PETSc
from mpi4py import MPI
import ufl

from load_mesh import CURRENT_MAP, MAGNETS, DomainTags3D

# Module-level dictionary to keep AMS objects alive (prevent garbage collection)
_ams_object_refs = {}


def make_ground_bc_V(mesh, V_space, cell_tags, conductor_markers):
    """Ground V at one point to remove the constant null space.
    
    The scalar potential V is only defined up to a constant. Setting V=0
    at one point makes the solution unique.
    """
    # Robust approach in parallel:
    # - pick any V dof from the conductor region on each rank
    # - select the smallest global dof id across MPI
    # - constrain that dof on its owning rank
    tdim = mesh.topology.dim

    local_best_gdof = np.iinfo(np.int64).max
    local_best_ldof = -1

    # Gather conductor cells on this rank
    local_cells = []
    for m in conductor_markers:
        cc = cell_tags.find(m)
        if cc.size > 0:
            local_cells.append(cc.astype(np.int32))

    if local_cells:
        cells = np.unique(np.concatenate(local_cells))

        # Map local dof -> global dof
        imap = V_space.dofmap.index_map
        size_local = imap.size_local

        # Pick the smallest owned global dof among dofs touching conductor cells
        for c in cells[: min(len(cells), 50)]:  # sample a few cells is enough
            dofs = V_space.dofmap.cell_dofs(int(c))
            for ldof in dofs:
                ldof = int(ldof)
                # Only allow owned dofs, not ghosts
                if ldof >= size_local:
                    continue
                gdof = int(imap.local_to_global(np.array([ldof], dtype=np.int32))[0])
                if gdof < local_best_gdof:
                    local_best_gdof = gdof
                    local_best_ldof = ldof

    global_best_gdof = mesh.comm.allreduce(local_best_gdof, op=MPI.MIN)

    # Only the owning rank sets the BC dof
    if local_best_gdof == global_best_gdof and local_best_ldof >= 0:
        dofs = np.array([local_best_ldof], dtype=np.int32)
    else:
        dofs = np.array([], dtype=np.int32)

    zero = fem.Function(V_space)
    zero.x.array[:] = 0.0
    bc = fem.dirichletbc(zero, dofs)

    if mesh.comm.rank == 0:
        owner = mesh.comm.allreduce(1 if dofs.size > 0 else 0, op=MPI.SUM)
        print(f"[DIAG] Ground V dof: global_dof={int(global_best_gdof)}, owners={owner}")

    return bc


def setup_sources(mesh, cell_tags):
    """Create function spaces for coil current density and magnet magnetization.
    
    Returns DG0 (piecewise constant) functions that will hold the source values.
    """
    DG0 = fem.functionspace(mesh, ("DG", 0))
    DG0_vec = fem.functionspace(mesh, ("DG", 0, (3,)))
    J_z = fem.Function(DG0, name="Jz")
    M_vec = fem.Function(DG0_vec, name="M")
    return J_z, M_vec


def initialise_magnetisation(mesh, cell_tags, M_vec, config):
    """Set initial magnetization direction for all permanent magnets.
    
    Magnets point radially outward. Magnitude is Br/μ₀ where Br is remanence.
    """
    dofmap = mesh.geometry.dofmap
    coords = mesh.geometry.x
    vec_view = M_vec.x.array.reshape((-1, 3))
    magnitude = config.magnet_remanence / max(config.mu0, 1e-12)
    
    for marker in MAGNETS:
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        for c in cells:
            geom_dofs = dofmap[c]
            cell_coords = coords[geom_dofs]
            cx = float(np.mean(cell_coords[:, 0]))
            cy = float(np.mean(cell_coords[:, 1]))
            norm = np.hypot(cx, cy)
            if norm < 1e-12:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = np.array([cx / norm, cy / norm, 0.0])
            vec_view[c, :] = magnitude * direction
    
    if mesh.comm.rank == 0:
        print("Magnetization initialized")


def rotate_magnetization(mesh, cell_tags, M_vec, config, t):
    """Update magnet directions as rotor spins.
    
    Each magnet rotates by angle ω_m * t while keeping alternating poles.
    """
    theta_rot = config.omega_m * t
    dofmap = mesh.geometry.dofmap
    coords = mesh.geometry.x
    vec_view = M_vec.x.array.reshape((-1, 3))
    magnitude = config.magnet_remanence / max(config.mu0, 1e-12)
    
    pm_spacing = (np.pi / 6) + (np.pi / 30)
    pm_angles = np.asarray([i * pm_spacing for i in range(10)])
    
    def get_pm_sign(marker):
        idx = marker - 13
        return 1 if (idx % 2 == 0) else -1
    
    for marker in MAGNETS:
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        
        sign = get_pm_sign(marker)
        pm_idx = marker - 13
        theta_pole_center = pm_angles[pm_idx]
        
        for c in cells:
            geom_dofs = dofmap[c]
            cell_coords = coords[geom_dofs]
            cx = float(np.mean(cell_coords[:, 0]))
            cy = float(np.mean(cell_coords[:, 1]))
            theta = np.arctan2(cy, cx)
            if theta < 0:
                theta += 2 * np.pi
            
            theta_now = theta_pole_center + theta_rot
            vec_view[c, 0] = sign * magnitude * np.cos(theta_now)
            vec_view[c, 1] = sign * magnitude * np.sin(theta_now)
            vec_view[c, 2] = 0.0
    
    M_vec.x.scatter_forward()


def update_currents(mesh, cell_tags, J_z, config, t):
    """Update coil currents with time-dependent 3-phase drive.
    
    Each coil follows J = J_peak * α * sin(ω_e*t + β) with phase shifts
    to create a rotating magnetic field.
    """
    omega = config.omega_e
    J_peak = config.coil_current_peak
    J_z.x.array[:] = 0.0
    for marker, meta in CURRENT_MAP.items():
        cells = cell_tags.find(marker)
        if cells.size == 0:
            continue
        drive = meta["alpha"] * np.sin(omega * t + meta["beta"])
        J_z.x.array[cells] = J_peak * drive


def current_stats(J_z):
    """Return maximum absolute current density for diagnostics."""
    if J_z is None:
        return 0.0
    return float(np.max(np.abs(J_z.x.array)))





def build_forms(mesh, A_space, V_space, sigma, nu, J_z, M_vec, A_prev,
                dx, dx_conductors, dx_magnets, ds, config, exterior_facet_tag=None):
    """Build all variational forms for the A-V eddy-current system.
    
    Returns bilinear forms (a00, a01, a10, a11), linear forms (L0, L1),
    and preconditioner forms (a00_spd, a00_motional). Uses backward Euler
    time-stepping with proper 1/dt scaling.
    """
    dt = fem.Constant(mesh, PETSc.ScalarType(config.dt))
    mu0 = config.mu0
    xcoord = ufl.SpatialCoordinate(mesh)
    omega = fem.Constant(mesh, PETSc.ScalarType(config.omega_m))
    u_rot = ufl.as_vector((-omega * xcoord[1], omega * xcoord[0], 0.0))
    
    A = ufl.TrialFunction(A_space)
    v = ufl.TestFunction(A_space)
    S = ufl.TrialFunction(V_space)
    q = ufl.TestFunction(V_space)
    
    curlA = ufl.curl(A)
    curlv = ufl.curl(v)
    
    # Weak boundary condition: A → 0 at exterior (penalty method)
    alpha = fem.Constant(mesh, PETSc.ScalarType(1e6))
    if exterior_facet_tag is not None:
        ds_exterior = ds(exterior_facet_tag)
    else:
        ds_exterior = ds
    
    # Backward Euler: time derivative terms get 1/dt factor
    inv_dt = fem.Constant(mesh, PETSc.ScalarType(1.0 / config.dt))

    # A00: curl-curl (no dt) + mass term (1/dt) + motional EMF (1/dt, nonsymmetric)
    a00 = nu * ufl.inner(curlA, curlv) * dx
    a00 += (mu0 * sigma * inv_dt) * ufl.inner(A, v) * dx
    # Motional EMF: u_rot × curl(A) from rotor motion
    a00_motional_expr = ufl.inner(ufl.cross(u_rot, curlA), v)
    a00 += (mu0 * sigma * inv_dt) * a00_motional_expr * dx_conductors
    a00 += alpha * ufl.inner(A, v) * ds_exterior
    
    # A00_spd: SPD version for preconditioner (different scaling for AMS)
    epsilon = fem.Constant(mesh, PETSc.ScalarType(1e-6))  # Small mass shift for zero-sigma regions
    a00_spd = dt * nu * ufl.inner(curlA, curlv) * dx  # dt factor for robustness
    a00_spd += sigma * mu0 * ufl.inner(A, v) * dx
    a00_spd += epsilon * ufl.inner(A, v) * dx
    # Scaled-down boundary penalty for AMS
    try:
        alpha_value = float(alpha.value)
    except Exception:
        alpha_value = 1e6  # Fallback to the nominal alpha used above
    alpha_spd_factor = getattr(config, "alpha_spd_factor", 1e-3)
    alpha_spd = fem.Constant(
        mesh, PETSc.ScalarType(alpha_spd_factor) * alpha_value
    )
    a00_spd += alpha_spd * ufl.inner(A, v) * ds_exterior
    if mesh.comm.rank == 0:
        print(
            f"[DIAG] A00_spd weak BC: alpha_spd_factor={alpha_spd_factor:.3e}, "
            f"alpha_spd={float(alpha_spd.value):.3e}"
        )
    
    # A-V coupling blocks (only in conductors where sigma > 0)
    a01 = mu0 * sigma * ufl.inner(v, ufl.grad(S)) * dx_conductors
    a10 = (mu0 * sigma * inv_dt) * ufl.inner(A, ufl.grad(q)) * dx_conductors
    a11_core = mu0 * sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx_conductors
    # Small stabilization for zero-sigma regions
    epsV = fem.Constant(mesh, PETSc.ScalarType(1e-8 * mu0))
    a11 = a11_core + epsV * ufl.inner(S, q) * dx

    # Right-hand side: coil current + previous step + permanent magnets
    J_term = J_z * v[2] * dx
    lagging = (mu0 * sigma * inv_dt) * ufl.inner(A_prev, v) * dx
    pm_term = -ufl.inner(mu0 * M_vec, curlv) * dx_magnets
    L0 = J_term + lagging + pm_term
    
    zero_scalar = fem.Constant(mesh, PETSc.ScalarType(0))
    L1 = zero_scalar * q * dx
    
    a_blocks = ((fem.form(a00), fem.form(a01)), (fem.form(a10), fem.form(a11)))
    a00_spd_form = fem.form(a00_spd)
    # Motional term for enriched preconditioner
    a00_motional_form = fem.form((mu0 * sigma * inv_dt) * a00_motional_expr * dx_conductors)
    L_blocks = (fem.form(L0), fem.form(L1))
    
    if mesh.comm.rank == 0:
        print("Forms built")
    
    return a_blocks, L_blocks, a00_spd_form, a00_motional_form





def rebuild_linear_forms(mesh, A_space, V_space, sigma, J_z, M_vec, A_prev,
                         dx, dx_magnets, config):
    """Rebuild linear forms when sources change."""
    mu0 = config.mu0
    dt = fem.Constant(mesh, PETSc.ScalarType(config.dt))
    inv_dt = fem.Constant(mesh, PETSc.ScalarType(1.0 / config.dt))
    
    v = ufl.TestFunction(A_space)
    q = ufl.TestFunction(V_space)
    curlv = ufl.curl(v)
    
    J_term = J_z * v[2] * dx
    lagging = (mu0 * sigma * inv_dt) * ufl.inner(A_prev, v) * dx
    pm_term = -ufl.inner(mu0 * M_vec, curlv) * dx_magnets
    L0 = J_term + lagging + pm_term
    
    zero_scalar = fem.Constant(mesh, PETSc.ScalarType(0))
    L1 = zero_scalar * q * dx
    
    return (fem.form(L0), fem.form(L1))


def assemble_system_matrix(mesh, a_blocks, block_bcs, a00_spd_form=None, a00_motional_form=None, beta_pc=0.3):
    """Assemble system matrix and A-block preconditioners."""
    mats = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            bcs_for_block = []
            if block_bcs[i]:
                bcs_for_block.extend(block_bcs[i])
            mat = petsc.assemble_matrix(a_blocks[i][j], bcs=bcs_for_block if bcs_for_block else None)
            mat.assemble()
            mats[i][j] = mat
    
    A00_standalone = mats[0][0].copy()
    A00_standalone.assemble()
    mat_nest = PETSc.Mat().createNest(mats, comm=mesh.comm)
    mat_nest.assemble()
    
    # Assemble SPD version for AMS preconditioner (no BCs - AMS needs full edge space)
    A00_spd = None
    if a00_spd_form is not None:
        A00_spd = petsc.assemble_matrix(a00_spd_form, bcs=None)
        A00_spd.assemble()
        A00_spd.setOption(PETSc.Mat.Option.SPD, True)
        if mesh.comm.rank == 0:
            print(
                f"A00_spd assembled (no BCs): size={A00_spd.getSize()}, "
                f"norm={A00_spd.norm(PETSc.NormType.NORM_FROBENIUS):.6e}"
            )
    else:
        if mesh.comm.rank == 0:
            print("WARNING: a00_spd_form is None, using full A00 for preconditioner")

    # Assemble motional term for enriched preconditioner
    A00_motional = None
    if a00_motional_form is not None:
        A00_motional = petsc.assemble_matrix(a00_motional_form, bcs=None)
        A00_motional.assemble()
        if mesh.comm.rank == 0:
            print(
                f"A00_motional assembled (no BCs): size={A00_motional.getSize()}, "
                f"norm={A00_motional.norm(PETSc.NormType.NORM_FROBENIUS):.6e}"
            )

    # Enriched preconditioner: SPD + some motional term
    A00_pc = None
    if A00_spd is not None and A00_motional is not None:
        A00_pc = A00_spd.copy()
        A00_pc.axpy(beta_pc, A00_motional, structure=PETSc.Mat.Structure.SUBSET_NONZERO_PATTERN)
        A00_pc.assemble()
        if mesh.comm.rank == 0:
            print(
                f"A00_pc (SPD + beta*motional) assembled: size={A00_pc.getSize()}, "
                f"norm={A00_pc.norm(PETSc.NormType.NORM_FROBENIUS):.6e}, beta={beta_pc}"
            )
    else:
        if mesh.comm.rank == 0:
            print("WARNING: A00_pc not built (missing A00_spd or A00_motional); falling back to A00_spd")
    
    if mesh.comm.rank == 0:
        print("Matrix assembled")
    
    return mats, mat_nest, A00_standalone, A00_spd, A00_pc



def configure_solver(mesh, mat_nest, mat_blocks, A_space, V_space, A00_spd=None, A00_pc=None, degree_A=None):
    """Configure linear solver for the 3D A–V system.

    Global structure:
      - Global KSP: FGMRES + fieldsplit
      - A-block sub-KSP: preonly + Hypre AMS, using
          * A00_full as the true operator block (nonsymmetric)
          * A00_spd as the SPD preconditioning block for AMS
      - V-block sub-KSP: Krylov on the Schur system + BoomerAMG/GAMG

    Physics/forms are unchanged: A01/A11 on dx_conductors, V grounded, A10≈0,
    A00_full includes weak BC; A00_spd is curl–curl+mass, no BCs.
    """
    import basix.ufl
    
    A = mat_nest
    A00_full = mat_blocks[0][0]  # Full operator (may include nonsymmetric terms)
    A01 = mat_blocks[0][1]
    A10 = mat_blocks[1][0]
    A11 = mat_blocks[1][1]
    A11.setOption(PETSc.Mat.Option.SPD, True)
    
    # Diagnostics for A-block preconditioner matrices
    if A00_spd is not None and mesh.comm.rank == 0:
        print("Using A00_spd as SPD core for AMS preconditioner (curl-curl + mass + weak BC)")
    
    # Diagnostics: report block norms to catch scaling / NaN issues early
    if mesh.comm.rank == 0:
        def _block_norm(label, mat):
            try:
                n = mat.norm(PETSc.NormType.NORM_FROBENIUS)
                print(f"{label} Frobenius norm = {n:.6e}")
            except Exception as exc:
                print(f"{label} norm failed: {exc}")

        _block_norm("A00_full", A00_full)
        if A01 is not None:
            _block_norm("A01", A01)
        if A10 is not None:
            _block_norm("A10", A10)
        _block_norm("A11", A11)

    # Diagnostics for A–V coupling
    diagnose_coupling(mesh, mat_blocks)

    # Create main KSP
    ksp = PETSc.KSP().create(comm=mesh.comm)
    # Use full nested operator for both operator and preconditioner.
    # FieldSplit will extract blocks from A; we override the A‑block
    # sub‑KSP preconditioner below to use A00_pc.
    ksp.setOperators(A, A)
    ksp.setType("fgmres")
    # Convergence based on relative tolerance (true residual), with a
    # reasonably high cap on the iteration count for coupled A–V.
    # Use a more achievable tolerance for this difficult coupled problem
    # rtol=1e-4 requires ~1 order of magnitude reduction from initial residual
    # Keep max_it small for quick iteration during setup.
    ksp.setTolerances(rtol=1e-4, atol=0.0, max_it=50)
    ksp.setGMRESRestart(120)  # Increased restart for better convergence
    # Python monitor (outer) – rank 0 only, print all iterations for steps 1-3
    def _outer_mon(ksp_obj, its, rnorm):
        # Print all iterations to see convergence history
        PETSc.Sys.Print(f"[OUTER] its={its:3d}, true_rnorm={rnorm:.6e}")
    ksp.setMonitor(_outer_mon)
    ksp.setConvergenceHistory()
    # Enable outer true-residual + reason via PETSc options
    opts_outer = PETSc.Options()
    opts_outer.setValue("-ksp_monitor_true_residual", "")
    opts_outer.setValue("-ksp_converged_reason", "")
    
    # Schur complement fieldsplit: solve A-V coupling by eliminating one block
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.LOWER)
    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.A11)  # Use A11 as Schur preconditioner
    schur_pre_type = "A11"
    if mesh.comm.rank == 0:
        print(f"Schur factorization: LOWER, Schur preconditioner: {schur_pre_type}")
    
    # Define field splits using nested IS extracted from the operator
    nested_IS = A.getNestISs()
    pc.setFieldSplitIS(("A", nested_IS[0][0]), ("V", nested_IS[1][1]))
    
    # AMS preconditioner setup: needs discrete gradient and coordinate vectors
    V_space_ams = fem.functionspace(mesh, ("Lagrange", 1))
    if mesh.comm.rank == 0:
        print(f"Created unconstrained V_space_ams (P1): {V_space_ams.dofmap.index_map.size_global} DOFs")

    # Discrete gradient: maps scalar functions to edge functions
    G = discrete_gradient(V_space_ams._cpp_object, A_space._cpp_object)
    G.assemble()

    # Coordinate vectors: x, y, z coordinates as functions
    coord_funcs = []
    vertex_coord_vecs = []
    xcoord = ufl.SpatialCoordinate(mesh)
    for dim in range(mesh.geometry.dim):
        coord_func = fem.Function(V_space_ams)
        coord_expr = fem.Expression(
            xcoord[dim], V_space_ams.element.interpolation_points
        )
        coord_func.interpolate(coord_expr)
        coord_func.x.scatter_forward()
        coord_funcs.append(coord_func)
        coord_vec = coord_func.x.petsc_vec
        vertex_coord_vecs.append(coord_vec)
        if mesh.comm.rank == 0:
            with coord_vec.localForm() as local:
                print(
                    f"Vertex coordinate vector {dim}: "
                    f"size={coord_vec.getSize()}, norm={np.linalg.norm(local.array_r):.6e}"
                )

    # Edge constant vectors: gradient of coordinates (needed for AMS)
    edge_const_vecs = []
    if mesh.comm.rank == 0:
        print("[DIAG] Computing edge constant vectors via G.createVecLeft() for AMS...")
    for dim in range(mesh.geometry.dim):
        edge_vec = G.createVecLeft()
        G.mult(vertex_coord_vecs[dim], edge_vec)  # G * coordinate = edge constant
        edge_const_vecs.append(edge_vec)
        if mesh.comm.rank == 0:
            with edge_vec.localForm() as local:
                print(
                    f"Edge constant vector {dim}: size={edge_vec.getSize()}, "
                    f"norm={np.linalg.norm(local.array_r):.6e}"
                )

    # Configure sub-solvers for A and V blocks
    pc.setUp()
    ksp_A, ksp_V = pc.getFieldSplitSubKSP()

    # A-block: use true operator (A00_full) but SPD preconditioner (A00_spd) for AMS
    if mesh.comm.rank == 0:
        print("[DIAG] Configuring AMS preconditioner on A-block sub-KSP...")
    ksp_A.setType("fgmres")
    ksp_A.setTolerances(rtol=0.0, atol=0.0, max_it=2)
    ksp_A.setGMRESRestart(10)
    if A00_spd is not None:
        ksp_A.setOperators(A00_full, A00_spd)
        if mesh.comm.rank == 0:
            print("A-block operators: A00_full (operator), A00_spd (Pmat)")
    else:
        ksp_A.setOperators(A00_full, A00_full)
        if mesh.comm.rank == 0:
            print("WARNING: A00_spd not available, using A00_full for both")
    
    pc_A = ksp_A.getPC()
    pc_A.setType("hypre")
    pc_A.setHYPREType("ams")
    pc_A.setHYPREDiscreteGradient(G)
    
    # Set coordinate vectors for AMS
    try:
        if hasattr(pc_A, "setHYPRECoordinateVectors"):
            if mesh.geometry.dim == 3:
                pc_A.setHYPRECoordinateVectors(
                    vertex_coord_vecs[0],
                    vertex_coord_vecs[1],
                    vertex_coord_vecs[2],
                )
                if mesh.comm.rank == 0:
                    print("AMS coordinate vectors set via setHYPRECoordinateVectors (3D)")
            elif mesh.geometry.dim == 2:
                pc_A.setHYPRECoordinateVectors(
                    vertex_coord_vecs[0],
                    vertex_coord_vecs[1],
                    None,
                )
                if mesh.comm.rank == 0:
                    print("AMS coordinate vectors set via setHYPRECoordinateVectors (2D)")
            else:
                pc_A.setHYPRECoordinateVectors(
                    vertex_coord_vecs[0],
                    None,
                    None,
                )
        else:
            raise AttributeError("setHYPRECoordinateVectors not available")
    except (AttributeError, TypeError):
        if mesh.geometry.dim == 3:
            pc_A.setHYPRESetEdgeConstantVectors(
                edge_const_vecs[0], edge_const_vecs[1], edge_const_vecs[2]
            )
            if mesh.comm.rank == 0:
                print("AMS edge constant vectors set (3D, via G*x, G*y, G*z)")
        elif mesh.geometry.dim == 2:
            pc_A.setHYPRESetEdgeConstantVectors(
                edge_const_vecs[0], edge_const_vecs[1], None
            )
            if mesh.comm.rank == 0:
                print("AMS edge constant vectors set (2D, via G*x, G*y)")
        else:
            pc_A.setHYPRESetEdgeConstantVectors(edge_const_vecs[0], None, None)
            if mesh.comm.rank == 0:
                print("AMS edge constant vectors set (1D, via G*x)")
    
    # Project gradients in zero-sigma regions (helps AMS)
    PETSc.Options().setValue("-fieldsplit_A_pc_hypre_ams_project_frequency", "1")
    pc_A.setFromOptions()
    
    if mesh.comm.rank == 0:
        print("A-block: FGMRES(2) + Hypre AMS with discrete gradient G")

    # V-block: just apply preconditioner (no Krylov iterations)
    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    
    # Use robust preconditioner for V-block (BoomerAMG or GAMG for SPD, ILU/Jacobi fallback)
    pc_V.setType("hypre")
    try:
        pc_V.setHYPREType("boomeramg")
        if mesh.comm.rank == 0:
            print("V-block preconditioner: Hypre BoomerAMG")
    except Exception:
        try:
            pc_V.setType("gamg")
            if mesh.comm.rank == 0:
                print("V-block preconditioner: GAMG (BoomerAMG not available)")
        except Exception:
            # Fallback to ILU or Jacobi if AMG not available
            try:
                pc_V.setType("ilu")
                if mesh.comm.rank == 0:
                    print("V-block preconditioner: ILU (AMG not available)")
            except Exception:
                pc_V.setType("jacobi")
                if mesh.comm.rank == 0:
                    print("V-block preconditioner: Jacobi (fallback)")
    pc_V.setFromOptions()

    # PETSc options for V-block AMG and outer KSP
    opts = PETSc.Options()
    opts.setValue("-fieldsplit_V_pc_hypre_boomeramg_coarsen_type", "HMIS")
    opts.setValue("-fieldsplit_V_pc_hypre_boomeramg_interp_type", "ext+i")
    opts.setValue("-fieldsplit_V_pc_hypre_boomeramg_relax_type_all", "symmetric-SOR/Jacobi")
    opts.setValue("-fieldsplit_V_pc_hypre_boomeramg_max_iter", "1")
    opts.setValue("-ksp_monitor_true_residual", "")
    opts.setValue("-ksp_converged_reason", "")
    opts.setValue("-fieldsplit_A_pc_hypre_ams_print_level", "0")
    opts.setValue("-fieldsplit_A_pc_hypre_ams_relax_type", "2")

    if mesh.comm.rank == 0:
        print("Solver configured: Schur fieldsplit")
        print("  - A-block: FGMRES(2) + Hypre AMS (with discrete gradient)")
        print("  - V-block: Preonly + BoomerAMG/GAMG")
        print(f"  - Schur factorization: LOWER, Schur pre: {schur_pre_type}")

    # Keep AMS-related objects alive to avoid GC issues
    _ams_object_refs[id(ksp)] = {
        "G": G,
        "V_space_ams": V_space_ams,
        "coord_funcs": coord_funcs,
        "edge_vecs": edge_const_vecs,
        "A00_spd": A00_spd,
    }

    return ksp




def diagnose_coupling(mesh, mat_blocks):
    """Diagnostics: check A–V coupling strength."""
    A00_full = mat_blocks[0][0]
    A01 = mat_blocks[0][1]
    A11 = mat_blocks[1][1]

    if mesh.comm.rank == 0:
        print("=== COUPLING DIAGNOSTICS ===")
        try:
            print(f"A00_full Frobenius norm = {A00_full.norm(PETSc.NormType.NORM_FROBENIUS):.6e}")
            print(f"A01 Frobenius norm      = {A01.norm(PETSc.NormType.NORM_FROBENIUS):.6e}")
            print(f"A11 Frobenius norm      = {A11.norm(PETSc.NormType.NORM_FROBENIUS):.6e}")
        except Exception as exc:
            print(f"[DIAG] Block norm computation failed: {exc}")

    # Random matvec tests on V-block space
    try:
        v_rand_A01 = A01.getVecRight()
        v_rand_A01.setRandom()
        v_norm = v_rand_A01.norm()
        y_A01 = A01.getVecLeft()
        A01.mult(v_rand_A01, y_A01)
        yA01_norm = y_A01.norm()

        v_rand_A11 = A11.getVecRight()
        v_rand_A11.setRandom()
        v11_norm = v_rand_A11.norm()
        y_A11 = A11.getVecLeft()
        A11.mult(v_rand_A11, y_A11)
        yA11_norm = y_A11.norm()

        if mesh.comm.rank == 0:
            print(f"||v_rand|| (A01)   = {v_norm:.6e}")
            print(f"||A01*v_rand||     = {yA01_norm:.6e}")
            print(f"||v_rand|| (A11)   = {v11_norm:.6e}")
            print(f"||A11*v_rand||     = {yA11_norm:.6e}")

            if v_norm > 0 and yA01_norm / v_norm < 1e-6:
                print("[WARN] A01 coupling appears numerically negligible – "
                      "check dx_conductors and sigma in conductors.")
        # Clean up
        v_rand_A01.destroy()
        y_A01.destroy()
        v_rand_A11.destroy()
        y_A11.destroy()
    except Exception as exc:
        if mesh.comm.rank == 0:
            print(f"[DIAG] Matvec coupling diagnostics failed: {exc}")
##############################################

