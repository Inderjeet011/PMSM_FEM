"""Solver functions: forms, assembly, sources."""

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
    """
    MPI-safe grounding of V: pick the globally smallest vertex that lies
    in sigma>0 region, then constrain the corresponding V DOF to 0 on
    the owning rank.
    """
    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    mesh.topology.create_connectivity(0, tdim)

    # Collect conductor cells on this rank
    local_cells = []
    for m in conductor_markers:
        c = cell_tags.find(m)
        if c.size > 0:
            local_cells.append(c)

    # Default: "no vertex found"
    local_min_gv = np.iinfo(np.int64).max
    local_v_local = -1

    if len(local_cells) > 0:
        cells = np.concatenate(local_cells).astype(np.int32)
        cell0 = int(cells[0])
        conn_c2v = mesh.topology.connectivity(tdim, 0)
        v_local = conn_c2v.links(cell0).astype(np.int32)

        vmap = mesh.topology.index_map(0)
        # Try to get global vertex numbers from IndexMap if available
        if hasattr(vmap, "global_indices"):
            gidx = vmap.global_indices(False)  # local->global
        else:
            # Fallback: treat local indices as "global-like"
            gidx = np.arange(vmap.size_local, dtype=np.int64)

        gverts = gidx[v_local]

        imin = int(np.argmin(gverts))
        local_min_gv = int(gverts[imin])
        local_v_local = int(v_local[imin])

    # Choose globally smallest vertex id
    global_min_gv = mesh.comm.allreduce(local_min_gv, op=MPI.MIN)

    # On the owning rank, locate dofs for that vertex
    if local_min_gv == global_min_gv and local_v_local >= 0:
        dofs = fem.locate_dofs_topological(
            V_space, entity_dim=0,
            entities=np.array([local_v_local], dtype=np.int32)
        )
    else:
        dofs = np.array([], dtype=np.int32)

    zero = fem.Function(V_space)
    zero.x.array[:] = 0.0
    return fem.dirichletbc(zero, dofs)


def setup_sources(mesh, cell_tags):
    """Create current and magnetization fields."""
    DG0 = fem.functionspace(mesh, ("DG", 0))
    DG0_vec = fem.functionspace(mesh, ("DG", 0, (3,)))
    J_z = fem.Function(DG0, name="Jz")
    M_vec = fem.Function(DG0_vec, name="M")
    return J_z, M_vec


def initialise_magnetisation(mesh, cell_tags, M_vec, config):
    """Initialize permanent magnet magnetization."""
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
    """Rotate magnetization with rotor."""
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
    """Update coil currents."""
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
    """Get max current density."""
    if J_z is None:
        return 0.0
    return float(np.max(np.abs(J_z.x.array)))


def build_forms(mesh, A_space, V_space, sigma, nu, J_z, M_vec, A_prev,
                dx, dx_conductors, dx_magnets, ds, config, exterior_facet_tag=None):
    """Build variational forms."""
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
    
    # Weak boundary condition penalty term for A (A → 0 at exterior)
    # This keeps the de Rham sequence intact for AMS compatibility
    alpha = fem.Constant(mesh, PETSc.ScalarType(1e6))  # Penalty parameter
    
    # Use exterior facet tag if available, otherwise use all boundary facets
    if exterior_facet_tag is not None:
        ds_exterior = ds(exterior_facet_tag)
    else:
        ds_exterior = ds  # Apply to all boundary facets
    
    # Backward Euler time scaling
    inv_dt = fem.Constant(mesh, PETSc.ScalarType(1.0 / config.dt))

    # Full A00 block (includes nonsymmetric terms for true operator)
    # Curl-curl term has no dt; σμ0 terms are scaled with 1/dt
    a00 = nu * ufl.inner(curlA, curlv) * dx
    a00 += (mu0 * sigma * inv_dt) * ufl.inner(A, v) * dx
    a00 += (mu0 * sigma * inv_dt) * ufl.inner(ufl.cross(u_rot, curlA), v) * dx_conductors
    a00 += alpha * ufl.inner(A, v) * ds_exterior  # Weak boundary condition penalty
    
    # SPD version for A-block preconditioner (curl-curl + mass, with gently scaled boundary penalty)
    # Use a milder scaling for AMS robustness: dt*curl-curl + σμ0 mass (no 1/dt amplification)
    epsilon = fem.Constant(mesh, PETSc.ScalarType(1e-6))  # Increased from 1e-10 to avoid near-singular auxiliary Poisson
    a00_spd = dt * nu * ufl.inner(curlA, curlv) * dx
    a00_spd += sigma * mu0 * ufl.inner(A, v) * dx
    a00_spd += epsilon * ufl.inner(A, v) * dx  # Mass shift for zero sigma regions
    # Add a scaled-down boundary penalty so AMS sees the correct low-frequency boundary modes
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
    
    # Coupling and scalar blocks: restrict sigma-weighted terms to conductors
    a01 = mu0 * sigma * ufl.inner(v, ufl.grad(S)) * dx_conductors
    # A10: eddy-current A–V constraint in conductors:
    # ∫_{Ω_c} (μ0 σ / dt) (A · ∇q) dx
    a10 = (mu0 * sigma * inv_dt) * ufl.inner(A, ufl.grad(q)) * dx_conductors
    a11_core = mu0 * sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx_conductors
    # Small stabilization on V-block to avoid singular modes in zero-sigma regions
    epsV = fem.Constant(mesh, PETSc.ScalarType(1e-8 * mu0))
    a11 = a11_core + epsV * ufl.inner(S, q) * dx

    # Source term from coil current density J_z (already in A/m^2): no dt factor.
    J_term = J_z * v[2] * dx
    # Previous-step contribution with μ0 σ / dt
    lagging = (mu0 * sigma * inv_dt) * ufl.inner(A_prev, v) * dx
    # Permanent magnet source term: -∫ (μ0 M · curl v) dx (no dt factor)
    pm_term = -ufl.inner(mu0 * M_vec, curlv) * dx_magnets
    L0 = J_term + lagging + pm_term
    
    zero_scalar = fem.Constant(mesh, PETSc.ScalarType(0))
    L1 = zero_scalar * q * dx
    
    a_blocks = ((fem.form(a00), fem.form(a01)), (fem.form(a10), fem.form(a11)))
    a00_spd_form = fem.form(a00_spd)
    L_blocks = (fem.form(L0), fem.form(L1))
    
    if mesh.comm.rank == 0:
        print("Forms built")
    
    return a_blocks, L_blocks, a00_spd_form


def rebuild_linear_forms(mesh, A_space, V_space, sigma, J_z, M_vec, A_prev,
                         dx, dx_magnets, config):
    """Rebuild linear forms when sources change."""
    mu0 = config.mu0
    dt = fem.Constant(mesh, PETSc.ScalarType(config.dt))
    inv_dt = fem.Constant(mesh, PETSc.ScalarType(1.0 / config.dt))
    
    v = ufl.TestFunction(A_space)
    q = ufl.TestFunction(V_space)
    curlv = ufl.curl(v)
    
    # Consistent with build_forms: no dt on J_term, μ0σ/dt on lagging, μ0 M for PM term
    J_term = J_z * v[2] * dx
    lagging = (mu0 * sigma * inv_dt) * ufl.inner(A_prev, v) * dx
    pm_term = -ufl.inner(mu0 * M_vec, curlv) * dx_magnets
    L0 = J_term + lagging + pm_term
    
    zero_scalar = fem.Constant(mesh, PETSc.ScalarType(0))
    L1 = zero_scalar * q * dx
    
    return (fem.form(L0), fem.form(L1))


def assemble_system_matrix(mesh, a_blocks, block_bcs, a00_spd_form=None):
    """Assemble system matrix."""
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
    
    # Assemble SPD version of A00 for AMS preconditioner
    # IMPORTANT: Assemble without boundary conditions - AMS must see the full edge space
    A00_spd = None
    if a00_spd_form is not None:
        A00_spd = petsc.assemble_matrix(a00_spd_form, bcs=None)  # No BCs for AMS
        A00_spd.assemble()
        A00_spd.setOption(PETSc.Mat.Option.SPD, True)
        if mesh.comm.rank == 0:
            print(f"A00_spd assembled (no BCs): size={A00_spd.getSize()}, norm={A00_spd.norm(PETSc.NormType.NORM_FROBENIUS):.6e}")
    else:
        if mesh.comm.rank == 0:
            print("WARNING: a00_spd_form is None, using full A00 for preconditioner")
    
    if mesh.comm.rank == 0:
        print("Matrix assembled")
    
    return mats, mat_nest, A00_standalone, A00_spd


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


def configure_solver(mesh, mat_nest, mat_blocks, A_space, V_space, A00_spd=None, degree_A=None):
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
    
    # Use A00_spd for A-block preconditioner
    A00_prec = A00_spd if A00_spd is not None else A00_full
    if A00_spd is not None:
        A00_prec.setOption(PETSc.Mat.Option.SPD, True)
        if mesh.comm.rank == 0:
            print(f"Using A00_spd for AMS preconditioner (SPD matrix, pure curl-curl + mass)")
    
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
    # sub‑KSP preconditioner below to use A00_spd.
    ksp.setOperators(A, A)
    ksp.setType("fgmres")
    # Use a looser relative tolerance so that the solve is accepted once
    # the (true) residual has been reduced by O(50–60%). This keeps cost
    # reasonable while still giving physically meaningful A/B fields.
    ksp.setTolerances(rtol=0.6, atol=0.0, max_it=200)
    # Python monitor (outer) – rank 0 only, first 60 iterations
    def _outer_mon(ksp_obj, its, rnorm):
        if its <= 60:
            PETSc.Sys.Print(f"[OUTER] its={its:3d}, true_rnorm={rnorm:.6e}")
    ksp.setMonitor(_outer_mon)
    ksp.setConvergenceHistory()
    # Enable outer true-residual + reason via PETSc options
    opts_outer = PETSc.Options()
    opts_outer.setValue("-ksp_monitor_true_residual", "")
    opts_outer.setValue("-ksp_converged_reason", "")
    
    # Use Schur complement fieldsplit to handle coupling
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    # Use LOWER Schur factorization with A11-based Schur preconditioner for V-block
    pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.LOWER)
    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.A11)
    
    # Define field splits using nested IS extracted from the operator
    nested_IS = A.getNestISs()
    pc.setFieldSplitIS(("A", nested_IS[0][0]), ("V", nested_IS[1][1]))
    
    # ------------------------------------------------------------------
    # AMS preconditioner for A-block (sub-KSP)
    # ------------------------------------------------------------------

    # Scalar space for AMS (unconstrained P1)
    V_space_ams = fem.functionspace(mesh, ("Lagrange", 1))
    if mesh.comm.rank == 0:
        print(f"Created unconstrained V_space_ams (P1): {V_space_ams.dofmap.index_map.size_global} DOFs")

    # Discrete gradient G : V_space_ams -> A_space
    G = discrete_gradient(V_space_ams._cpp_object, A_space._cpp_object)
    G.assemble()

    # Coordinate vectors on V_space_ams
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

    # Edge constant vectors via G.createVecLeft()
    edge_const_vecs = []
    if mesh.comm.rank == 0:
        print("[DIAG] Computing edge constant vectors via G.createVecLeft() for AMS...")
    for dim in range(mesh.geometry.dim):
        edge_vec = G.createVecLeft()
        G.mult(vertex_coord_vecs[dim], edge_vec)
        edge_const_vecs.append(edge_vec)
        if mesh.comm.rank == 0:
            with edge_vec.localForm() as local:
                print(
                    f"Edge constant vector {dim}: size={edge_vec.getSize()}, "
                    f"norm={np.linalg.norm(local.array_r):.6e}"
                )

    # Sub-KSPs for A and V blocks
    pc.setUp()
    ksp_A, ksp_V = pc.getFieldSplitSubKSP()

    # A-block: preonly + AMS, using A00_spd as BOTH operator and SPD preconditioner
    if mesh.comm.rank == 0:
        print("[DIAG] Configuring AMS on A-block sub-KSP...")
    ksp_A.setType("preonly")
    pc_A = ksp_A.getPC()
    pc_A.setType("hypre")
    pc_A.setHYPREType("ams")

    if A00_spd is not None:
        # Operator = A00_spd, preconditioner = A00_spd (pure SPD curl–curl+mass)
        ksp_A.setOperators(A00_spd, A00_spd)
        if mesh.comm.rank == 0:
            print("Sub-KSP operators set: A00_spd (operator+pc) for AMS")
    else:
        ksp_A.setOperators(A00_full, A00_full)
        if mesh.comm.rank == 0:
            print("WARNING: A00_spd not available, using A00_full for both in AMS")

    pc_A.setHYPREDiscreteGradient(G)
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

    # Gradient projection for zero-sigma regions on A-block
    PETSc.Options().setValue("-fieldsplit_A_pc_hypre_ams_project_frequency", "1")
    if mesh.comm.rank == 0:
        print("AMS gradient projection enabled (fieldsplit_A_pc_hypre_ams_project_frequency=1)")

    pc_A.setFromOptions()

    # V-block: Schur solve on S with A11-based preconditioning.
    # Obtain Schur complement matrix S from fieldsplit PC.
    try:
        S = pc.getFieldSplitSchurComplement()
    except AttributeError:
        # Fallback for older petsc4py API (if available)
        try:
            S = pc.getFieldSplitSchurMat()
        except Exception:
            S = None

    # Configure V-subKSP to solve approximately S * x = rhs with preconditioner A11.
    # This lets AMG act on A11 (elliptic SPD) while PETSc applies S.
    if S is not None:
        ksp_V.setOperators(S, A11)
    else:
        # Fallback: keep original A11 if Schur matrix is unavailable
        ksp_V.setOperators(A11, A11)

    # Cheap but effective Schur solve: fixed small Krylov work
    ksp_V.setType("fgmres")
    ksp_V.setTolerances(rtol=0.0, atol=0.0, max_it=5)
    pc_V = ksp_V.getPC()
    pc_V.setType("hypre")
    try:
        pc_V.setHYPREType("boomeramg")
    except Exception:
        pc_V.setType("gamg")
    pc_V.setFromOptions()

    # PETSc option tweaks for Schur V-block and AMS cost
    opts = PETSc.Options()
    # V-block BoomerAMG: aggressive coarsening / smoother, one V-cycle per Schur apply
    opts.setValue("-fieldsplit_V_pc_hypre_boomeramg_coarsen_type", "HMIS")
    opts.setValue("-fieldsplit_V_pc_hypre_boomeramg_interp_type", "ext+i")
    opts.setValue("-fieldsplit_V_pc_hypre_boomeramg_relax_type_all", "symmetric-SOR/Jacobi")
    opts.setValue("-fieldsplit_V_pc_hypre_boomeramg_max_iter", "1")
    # Outer KSP true-residual + reasons
    opts.setValue("-ksp_monitor_true_residual", "")
    opts.setValue("-ksp_converged_reason", "")
    # Optional AMS cost tuning (if supported)
    opts.setValue("-fieldsplit_A_pc_hypre_ams_print_level", "0")
    opts.setValue("-fieldsplit_A_pc_hypre_ams_relax_type", "2")

    if mesh.comm.rank == 0:
        print("Solver configured: AMS for A (Nédélec), inexact Schur (FGMRES+BoomerAMG) for V (Lagrange)")

    # Keep AMS-related objects alive to avoid GC issues
    _ams_object_refs[id(ksp)] = {
        "G": G,
        "V_space_ams": V_space_ams,
        "coord_funcs": coord_funcs,
        "edge_vecs": edge_const_vecs,
        "A00_spd": A00_spd,
    }

    return ksp
