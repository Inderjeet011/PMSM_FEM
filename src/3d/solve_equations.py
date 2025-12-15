"""Solver functions: forms, assembly, sources."""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from dolfinx.cpp.fem.petsc import discrete_gradient, interpolation_matrix
from petsc4py import PETSc
import ufl

from load_mesh import CURRENT_MAP, MAGNETS

# Module-level dictionary to keep AMS objects alive (prevent garbage collection)
_ams_object_refs = {}


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
    
    # Full A00 block (includes nonsymmetric terms for true operator)
    a00 = dt * nu * ufl.inner(curlA, curlv) * dx
    a00 += sigma * mu0 * ufl.inner(A, v) * dx
    a00 += sigma * mu0 * ufl.inner(ufl.cross(u_rot, curlA), v) * dx_conductors
    a00 += alpha * ufl.inner(A, v) * ds_exterior  # Weak boundary condition penalty
    
    # SPD version for AMS preconditioner (curl-curl + mass only, no nonsymmetric terms, no boundary penalty)
    # AMS requires pure curl-curl + mass operator without boundary terms
    epsilon = fem.Constant(mesh, PETSc.ScalarType(1e-6))  # Increased from 1e-10 to avoid near-singular auxiliary Poisson
    a00_spd = dt * nu * ufl.inner(curlA, curlv) * dx
    a00_spd += sigma * mu0 * ufl.inner(A, v) * dx
    a00_spd += epsilon * ufl.inner(A, v) * dx  # Mass shift for zero sigma regions
    # NO boundary penalty terms in a00_spd - AMS needs pure curl-curl + mass
    
    a01 = mu0 * sigma * ufl.inner(v, ufl.grad(S)) * dx
    gauge = fem.Constant(mesh, PETSc.ScalarType(1e-10))
    a10 = gauge * ufl.div(A) * q * dx
    a11 = mu0 * sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx
    
    J_term = dt * mu0 * J_z * v[2] * dx
    lagging = sigma * mu0 * ufl.inner(A_prev, v) * dx
    pm_term = -ufl.inner(M_vec, curlv) * dx_magnets
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
    
    v = ufl.TestFunction(A_space)
    q = ufl.TestFunction(V_space)
    curlv = ufl.curl(v)
    
    J_term = dt * mu0 * J_z * v[2] * dx
    lagging = sigma * mu0 * ufl.inner(A_prev, v) * dx
    pm_term = -ufl.inner(M_vec, curlv) * dx_magnets
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


def configure_solver(mesh, mat_nest, mat_blocks, A_space, V_space, A00_spd=None, degree_A=None):
    """Configure linear solver with Hypre AMS for Nédélec block.
    
    Requirements:
    - Remove boundary penalty from a00_spd (pure curl-curl + mass)
    - Use unconstrained V_space_ams for discrete gradient and coordinate vectors
    - Build coordinate vectors from vertex DOFs → mesh coordinates
    - Use KSP=CG for AMS (not preonly)
    - Add AMS gradient projection (projection_frequency=1)
    """
    import basix.ufl
    
    A = mat_nest
    A00_full = mat_blocks[0][0]  # Full operator (may include nonsymmetric terms)
    A11 = mat_blocks[1][1]
    A11.setOption(PETSc.Mat.Option.SPD, True)
    
    # Use A00_spd for AMS preconditioner
    A00_prec = A00_spd if A00_spd is not None else A00_full
    if A00_spd is not None:
        A00_prec.setOption(PETSc.Mat.Option.SPD, True)
        if mesh.comm.rank == 0:
            print(f"Using A00_spd for AMS preconditioner (SPD matrix, pure curl-curl + mass)")
    
    # Create block-diagonal preconditioner
    P = PETSc.Mat().createNest([[A00_prec, None], [None, A11]], comm=mesh.comm)
    P.assemble()
    
    # Create main KSP
    ksp = PETSc.KSP().create(comm=mesh.comm)
    ksp.setOperators(A, P)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=1e-3, atol=1e-6, max_it=500)  # Increased max_it and relaxed tolerances for better convergence
    
    # Use fieldsplit for block structure
    # Use Schur complement to handle coupling (A01, A10 ≠ 0)
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.FULL)
    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.A11)
    
    # Get index sets from P
    nested_IS = P.getNestISs()
    pc.setFieldSplitIS(("A", nested_IS[0][0]), ("V", nested_IS[1][1]))
    
    # ===== AMS CONFIGURATION =====
    # Step 1: Create unconstrained CG space for AMS (no Dirichlet BCs)
    # Force P1 for AMS scalar space
    V_space_ams = fem.functionspace(mesh, ("Lagrange", 1))
    if mesh.comm.rank == 0:
        print(f"Created unconstrained V_space_ams (P1): {V_space_ams.dofmap.index_map.size_global} DOFs")
    
    # Step 2: Build discrete gradient using unconstrained space
    G = discrete_gradient(V_space_ams._cpp_object, A_space._cpp_object)
    G.assemble()
    
    # Step 3: Build coordinate vectors from vertex DOFs → mesh coordinates
    # Create coordinate functions in V_space_ams and extract their vectors
    # IMPORTANT: Store Function objects to keep them alive (prevent garbage collection)
    coord_funcs = []  # Keep owning objects alive
    vertex_coord_vecs = []
    xcoord = ufl.SpatialCoordinate(mesh)
    
    for dim in range(mesh.geometry.dim):
        # Create a function in V_space_ams representing the coordinate component
        coord_func = fem.Function(V_space_ams)
        
        # Interpolate the coordinate component (xcoord[dim]) into V_space_ams
        coord_expr = fem.Expression(xcoord[dim], V_space_ams.element.interpolation_points)
        coord_func.interpolate(coord_expr)
        coord_func.x.scatter_forward()
        
        # Store the Function object to prevent garbage collection
        coord_funcs.append(coord_func)
        
        # Extract the PETSc vector (vertex coordinates)
        coord_vec = coord_func.x.petsc_vec
        vertex_coord_vecs.append(coord_vec)
        
        if mesh.comm.rank == 0:
            with coord_vec.localForm() as local:
                print(f"Vertex coordinate vector {dim}: size={coord_vec.getSize()}, norm={np.linalg.norm(local.array_r):.6e}")
    
    # Step 4: Configure AMS fully BEFORE any setUp() calls
    # Build all AMS components first, then configure on temporary KSP
    # This ensures all AMS components are ready before setup
    if mesh.comm.rank == 0:
        print("[DIAG] Pre-configuring AMS components...")
    import time
    t0 = time.time()
    
    # Step 5: Compute edge constant vectors for AMS
    # IMPORTANT: Use G.createVecLeft() for edge vectors, don't manually size them
    # We'll set these on the actual PC after getting sub-KSPs
    edge_const_vecs = []  # Keep edge vectors alive
    
    if mesh.comm.rank == 0:
        print("[DIAG] Computing edge constant vectors via G.createVecLeft()...")
    for dim in range(mesh.geometry.dim):
        # Create edge vector using G.createVecLeft() - this ensures correct size
        edge_vec = G.createVecLeft()
        G.mult(vertex_coord_vecs[dim], edge_vec)
        edge_const_vecs.append(edge_vec)  # Keep alive
        
        if mesh.comm.rank == 0:
            with edge_vec.localForm() as local:
                print(f"Edge constant vector {dim}: size={edge_vec.getSize()}, norm={np.linalg.norm(local.array_r):.6e}")
    
    t1 = time.time()
    if mesh.comm.rank == 0:
        print(f"[DIAG] AMS components prepared in {t1-t0:.3f} seconds")
    
    # Step 6: Now get sub-KSPs (after all AMS components are ready)
    # For Schur complement, we need setUp() to get sub-KSPs
    if mesh.comm.rank == 0:
        print("[DIAG] About to call pc.setUp() for Schur complement...")
    t0 = time.time()
    pc.setUp()
    t1 = time.time()
    if mesh.comm.rank == 0:
        print(f"[DIAG] pc.setUp() completed in {t1-t0:.3f} seconds")
    
    if mesh.comm.rank == 0:
        print("[DIAG] Getting sub-KSPs...")
    ksp_A, ksp_V = pc.getFieldSplitSubKSP()
    if mesh.comm.rank == 0:
        print("[DIAG] Sub-KSPs retrieved")
    
    # Step 7: Configure AMS on actual sub-KSP
    # For Schur complement, use preonly + AMS (AMS approximates A00^(-1))
    # IMPORTANT: For AMS, use SPD operator for BOTH operator and preconditioner
    if mesh.comm.rank == 0:
        print("[DIAG] Configuring AMS on sub-KSP...")
    t0 = time.time()
    ksp_A.setType("preonly")
    pc_A = ksp_A.getPC()
    pc_A.setType("hypre")
    pc_A.setHYPREType("ams")
    
    # Set operators
    if A00_spd is not None:
        ksp_A.setOperators(A00_spd, A00_spd)
        if mesh.comm.rank == 0:
            print("Sub-KSP operators set: A00_spd (both operator and preconditioner for AMS)")
    else:
        ksp_A.setOperators(A00_full, A00_full)
        if mesh.comm.rank == 0:
            print("WARNING: A00_spd not available, using A00_full for both")
    
    # Set discrete gradient
    pc_A.setHYPREDiscreteGradient(G)
    if mesh.comm.rank == 0:
        print("Discrete gradient set (from unconstrained V_space_ams)")
    
    # Set coordinate/edge constant vectors
    try:
        if hasattr(pc_A, 'setHYPRECoordinateVectors'):
            # Use coordinate vectors directly (preferred method)
            if mesh.geometry.dim == 3:
                pc_A.setHYPRECoordinateVectors(vertex_coord_vecs[0], vertex_coord_vecs[1], vertex_coord_vecs[2])
                if mesh.comm.rank == 0:
                    print("AMS coordinate vectors set via setHYPRECoordinateVectors (3D)")
            elif mesh.geometry.dim == 2:
                pc_A.setHYPRECoordinateVectors(vertex_coord_vecs[0], vertex_coord_vecs[1], None)
                if mesh.comm.rank == 0:
                    print("AMS coordinate vectors set via setHYPRECoordinateVectors (2D)")
            else:
                pc_A.setHYPRECoordinateVectors(vertex_coord_vecs[0], None, None)
                if mesh.comm.rank == 0:
                    print("AMS coordinate vectors set via setHYPRECoordinateVectors (1D)")
        else:
            raise AttributeError("setHYPRECoordinateVectors not available")
    except (AttributeError, TypeError):
        # Use edge constant vectors (computed via G.createVecLeft())
        if mesh.geometry.dim == 3:
            pc_A.setHYPRESetEdgeConstantVectors(edge_const_vecs[0], edge_const_vecs[1], edge_const_vecs[2])
            if mesh.comm.rank == 0:
                print("AMS edge constant vectors set (3D, computed via G*x, G*y, G*z)")
        elif mesh.geometry.dim == 2:
            pc_A.setHYPRESetEdgeConstantVectors(edge_const_vecs[0], edge_const_vecs[1], None)
            if mesh.comm.rank == 0:
                print("AMS edge constant vectors set (2D, computed via G*x, G*y)")
        else:
            pc_A.setHYPRESetEdgeConstantVectors(edge_const_vecs[0], None, None)
            if mesh.comm.rank == 0:
                print("AMS edge constant vectors set (1D, computed via G*x)")
    
    # Step 8: Add gradient projection for zero-sigma regions
    # Set projection frequency to 1 (project every iteration)
    # IMPORTANT: Use fieldsplit prefix for sub-KSP options
    if mesh.comm.rank == 0:
        print("[DIAG] Setting AMS projection options...")
    PETSc.Options().setValue("-fieldsplit_A_pc_hypre_ams_project_frequency", "1")
    if mesh.comm.rank == 0:
        print("AMS gradient projection enabled (fieldsplit_A_pc_hypre_ams_project_frequency=1)")
    
    # Step 9: setFromOptions() must be called AFTER all setup
    if mesh.comm.rank == 0:
        print("[DIAG] Calling pc_A.setFromOptions()...")
    t0 = time.time()
    pc_A.setFromOptions()
    t1 = time.time()
    if mesh.comm.rank == 0:
        print(f"[DIAG] pc_A.setFromOptions() completed in {t1-t0:.3f} seconds")
    
    if mesh.comm.rank == 0:
        print("AMS configured: discrete gradient + coordinate/edge constant vectors + gradient projection")
    
    # Keep GAMG for scalar potential block
    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    pc_V.setType("gamg")
    pc_V.setFromOptions()
    
    # Step 10: Pin AMS objects to prevent garbage collection
    # Store references in module-level dictionary (PETSc objects don't allow arbitrary attributes)
    # Use id(ksp) as key to keep references alive
    _ams_object_refs[id(ksp)] = {
        "G": G,
        "V_space_ams": V_space_ams,
        "coord_funcs": coord_funcs,  # Function objects (not just vectors) - KEEP ALIVE
        "edge_vecs": edge_const_vecs,  # Edge constant vectors - KEEP ALIVE
        "A00_spd": A00_spd,
    }
    
    if mesh.comm.rank == 0:
        print("Solver configured: Hypre AMS (preonly) for A (Nédélec), GAMG for V (Lagrange)")
        print("[DIAG] AMS objects pinned in module-level dict to prevent garbage collection")
    
    return ksp
