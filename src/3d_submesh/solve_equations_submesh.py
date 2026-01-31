"""3D A-V solver with submesh: forms, assembly, and PETSc solver setup.
A lives on parent mesh, V lives on conductor submesh.

Overall execution order (main_submesh.py):
  1. load_mesh_submesh: load mesh, extract submesh, setup materials, BCs
  2. interpolate_materials: sigma_submesh from parent sigma
  3. dof_mapping: create_dof_mapper for coupling
  4. solve_equations_submesh: build forms, assemble matrix, configure solver
  5. Time loop: solver_utils_submesh.solve_one_step_submesh

Order of execution within this module (called from main_submesh.py):
  1. build_forms_submesh() - define UFL forms (a00, a01, a10, a11, L0, L1)
  2. assemble_system_matrix_submesh() - assemble blocks in order:
     (a) A00 (petsc.assemble_matrix)
     (b) A01 (assemble_A01_block_quadrature_direct)
     (c) A10 (assemble_A10_block_quadrature_direct)
     (d) A11 (petsc.assemble_matrix)
  3. configure_solver_submesh() - set up PETSc KSP, fieldsplit PC, AMS for A-block
"""

from dolfinx import fem
from dolfinx.fem import petsc
from petsc4py import PETSc
import ufl
import numpy as np

_keepalive = []  # Keep PETSc objects alive (AMS gradient, vectors, etc.)


def build_forms_submesh(mesh_parent, mesh_conductor, A_space, V_space, 
                        sigma, nu, J_z, M_vec, A_prev,
                        dx_parent, dx_rs, dx_rpm, dx_c, dx_pm, 
                        dx_conductor, config, exterior_facet_tag=None):
    """
    Build forms for A-V system with A on parent mesh and V on conductor submesh.
    
    Parameters:
    -----------
    mesh_parent : dolfinx.mesh.Mesh
        Full parent mesh (for A)
    mesh_conductor : dolfinx.mesh.Mesh
        Conductor submesh (for V)
    A_space : dolfinx.fem.FunctionSpace
        Function space for A on parent mesh
    V_space : dolfinx.fem.FunctionSpace
        Function space for V on conductor submesh
    sigma, nu : dolfinx.fem.Function
        Material properties on parent mesh
    J_z, M_vec : dolfinx.fem.Function
        Sources on parent mesh
    A_prev : dolfinx.fem.Function
        Previous timestep A on parent mesh
    dx_parent : ufl.Measure
        Integration measure on parent mesh
    dx_rs, dx_rpm, dx_c, dx_pm : ufl.Measure
        Restricted measures on parent mesh
    dx_conductor : ufl.Measure
        Integration measure on conductor submesh
    config : SimpleNamespace
        Configuration parameters
    """
    dt = fem.Constant(mesh_parent, PETSc.ScalarType(config.dt))
    mu0 = config.mu0
    xcoord_parent = ufl.SpatialCoordinate(mesh_parent)
    omega = fem.Constant(mesh_parent, PETSc.ScalarType(config.omega_m))
    omega_vec = ufl.as_vector((0.0, 0.0, omega))
    u_rot = ufl.cross(omega_vec, xcoord_parent)
    
    # Trial and test functions
    A = ufl.TrialFunction(A_space)  # On parent mesh
    v = ufl.TestFunction(A_space)   # On parent mesh
    S = ufl.TrialFunction(V_space)  # On conductor submesh
    q = ufl.TestFunction(V_space)    # On conductor submesh
    
    curlA = ufl.curl(A)
    curlv = ufl.curl(v)
    
    inv_dt = fem.Constant(mesh_parent, PETSc.ScalarType(1.0 / config.dt))
    
    epsilon_A = float(getattr(config, "epsilon_A", 0.0))
    epsilon_A_spd = float(getattr(config, "epsilon_A_spd", 1e-6))
    eps_A = fem.Constant(mesh_parent, PETSc.ScalarType(epsilon_A))
    eps_spd = fem.Constant(mesh_parent, PETSc.ScalarType(epsilon_A_spd))
    
    # A-equation (on parent mesh): nu*curl(A)·curl(v) + (sigma/dt)*A·v + regularization
    a00 = (
        nu * ufl.inner(curlA, curlv) * dx_parent
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_rs
        - sigma * ufl.inner(ufl.cross(u_rot, curlA), v) * dx_rpm
        + eps_A * ufl.inner(A, v) * dx_parent
    )
    
    # SPD approximation for preconditioner
    dx_cond_all = dx_rs + dx_rpm
    a00_spd = (
        dt * nu * ufl.inner(curlA, curlv) * dx_parent
        + sigma * ufl.inner(A, v) * dx_cond_all
        + eps_spd * ufl.inner(A, v) * dx_parent
    )
    
    # A–V coupling terms
    #
    # In this submesh-based implementation, the true A01 and A10 coupling blocks
    # are assembled via custom quadrature kernels that use the DOF mapper and
    # entity maps. To avoid cross-mesh UFL complications (which require
    # entity_maps when compiling forms), we keep the UFL-level a01 and a10
    # identically zero and rely entirely on the custom PETSc assembly for
    # these blocks.
    #
    # This preserves the correct block structure for lifting/boundary handling
    # while delegating all physical coupling to the quadrature routines.
    # Use trivial scalar forms (no A–V coupling at UFL level); the actual
    # A01 and A10 blocks are provided by custom quadrature assembly.
    a01 = 0 * ufl.inner(ufl.grad(S), ufl.grad(S)) * dx_parent
    a10 = 0 * ufl.inner(ufl.grad(S), ufl.grad(S)) * dx_conductor
    
    # a11: sigma * grad(V)·grad(q) on conductor submesh
    # Need sigma on submesh - will be provided via interpolation_data
    # Create placeholder for now (will be replaced with interpolated sigma)
    DG0_submesh = fem.functionspace(mesh_conductor, ("DG", 0))
    sigma_submesh_placeholder = fem.Function(DG0_submesh)
    a11 = sigma_submesh_placeholder * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx_conductor
    
    # Right-hand side
    L0 = (
        J_z * v[2] * dx_c 
        + (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_rs 
        + ufl.inner(nu * mu0 * M_vec, curlv) * dx_pm
    )
    
    # L1: keep RHS for V-equation zero at UFL level; any coupling from
    # previous-step A can be added later via custom assembly if needed.
    L1 = 0 * S * dx_conductor
    
    # Store interpolation functions for later use
    interpolation_data = {
        'V_space_parent': None,
        'V_parent': None,
        'A_space_submesh': None,
        'A_submesh': None,
        'sigma_submesh': None,  # Will be set externally via interpolate_materials
        'sigma_submesh_placeholder': sigma_submesh_placeholder,  # For form building
    }
    
    a_blocks = ((fem.form(a00), fem.form(a01)), (fem.form(a10), fem.form(a11)))
    a00_spd_form = fem.form(a00_spd)
    L_blocks = (fem.form(L0), fem.form(L1))
    
    return a_blocks, L_blocks, a00_spd_form, interpolation_data


def assemble_system_matrix_submesh(mesh_parent, mesh_conductor, a_blocks, block_bcs, 
                                    a00_spd_form, interpolation_data, A_space_parent, V_space_submesh,
                                    dof_mapper=None, sigma_parent=None, config=None):
    """
    Assemble system matrix for mixed parent/submesh system.
    
    Note: This is a simplified assembly. A full implementation would:
    1. Use entity maps to properly couple parent and submesh DOFs
    2. Handle the interpolation between meshes during assembly
    """
    comm = mesh_parent.comm

    # Global DOF counts
    n_A_dofs = A_space_parent.dofmap.index_map.size_global
    n_V_dofs = V_space_submesh.dofmap.index_map.size_global

    # Assemble blocks separately
    mats = [[None, None], [None, None]]
    
    # A00 block (A-A coupling on parent): (nA x nA)
    bcs_A = block_bcs[0] if block_bcs else []
    mats[0][0] = petsc.assemble_matrix(a_blocks[0][0], bcs=bcs_A)
    mats[0][0].assemble()
    if dof_mapper is not None and config is not None:
        from assemble_coupling_A10_quadrature_direct import assemble_A01_block_quadrature_direct

        # Use same sigma_submesh as for A10
        sigma_submesh = interpolation_data.get('sigma_submesh')
        if sigma_submesh is None:
            from dolfinx import fem as _fem
            DG0_submesh = _fem.functionspace(mesh_conductor, ("DG", 0))
            sigma_submesh = _fem.Function(DG0_submesh)
            sigma_submesh.x.array[:] = 1e6

        # dx_conductor was passed in via build_forms_submesh and is not available
        # here; recreate it locally.
        import ufl as _ufl
        dx_c_local = _ufl.Measure("dx", domain=mesh_conductor)

        mats[0][1] = assemble_A01_block_quadrature_direct(
            mesh_parent, mesh_conductor, A_space_parent, V_space_submesh,
            sigma_submesh, dx_c_local, dof_mapper, config, sigma_parent=sigma_parent
        )
    else:
        # Fallback: zero matrix if no DOF mapper/config
        A01_mat = PETSc.Mat().create(comm)
        A01_mat.setType(PETSc.Mat.Type.AIJ)
        A01_mat.setSizes([(n_A_dofs, None), (n_V_dofs, None)])
        A01_mat.setUp()
        A01_mat.assemble()
        mats[0][1] = A01_mat
    
    # A10 block (A→V coupling): A on parent affects V on submesh
    # This is the one-way coupling we're implementing
    if dof_mapper is not None and config is not None:
        from assemble_coupling_A10_quadrature_direct import assemble_A10_block_quadrature_direct
        from dolfinx import fem as _fem
        import ufl
        from petsc4py import PETSc as PETSc_type
        
        # Get sigma_submesh from interpolation_data
        sigma_submesh = interpolation_data.get('sigma_submesh')
        if sigma_submesh is None:
            # Fallback
            from dolfinx import fem
            DG0_submesh = fem.functionspace(mesh_conductor, ("DG", 0))
            sigma_submesh = fem.Function(DG0_submesh)
            sigma_submesh.x.array[:] = 1e6
        
        # Create constants for form
        inv_dt = _fem.Constant(mesh_parent, PETSc_type.ScalarType(1.0 / config.dt))
        omega = _fem.Constant(mesh_parent, PETSc_type.ScalarType(config.omega_m))
        
        # Get dx_conductor measure (should be passed or created)
        dx_conductor = ufl.Measure("dx", domain=mesh_conductor)
        
        # Assemble A10 block using direct quadrature evaluation
        mats[1][0] = assemble_A10_block_quadrature_direct(
            mesh_parent, mesh_conductor, A_space_parent, V_space_submesh,
            sigma_submesh, inv_dt, omega, dx_conductor, dof_mapper, config
        )
    else:
        # Fallback: create zero matrix
        A10_mat = PETSc.Mat().create(comm)
        A10_mat.setType(PETSc.Mat.Type.AIJ)
        A10_mat.setSizes([(n_V_dofs, None), (n_A_dofs, None)])
        A10_mat.setUp()
        A10_mat.assemble()
        mats[1][0] = A10_mat
    
    # A11 block (V-V coupling on submesh)
    bcs_V = block_bcs[1] if block_bcs else []
    mats[1][1] = petsc.assemble_matrix(a_blocks[1][1], bcs=bcs_V)
    mats[1][1].assemble()
    
    # Use interpolated sigma from interpolation_data
    # This should have been set externally via interpolate_materials module
    sigma_submesh = interpolation_data.get('sigma_submesh', None)
    if sigma_submesh is None:
        # Fallback: create placeholder (should not happen if called correctly)
        from dolfinx import fem
        DG0_submesh = fem.functionspace(mesh_conductor, ("DG", 0))
        sigma_submesh = fem.Function(DG0_submesh)
        sigma_submesh.x.array[:] = 1e6  # Approximate conductor conductivity
        interpolation_data['sigma_submesh'] = sigma_submesh
        if mesh_parent.comm.rank == 0:
            print("Warning: sigma_submesh not provided, using fallback constant value")
    
    # Sanity checks: block sizes and communicators
    A00 = mats[0][0]
    A01 = mats[0][1]
    A10 = mats[1][0]
    A11 = mats[1][1]

    m00, n00 = A00.getSize()
    m01, n01 = A01.getSize()
    m10, n10 = A10.getSize()
    m11, n11 = A11.getSize()

    assert m00 == n_A_dofs and n00 == n_A_dofs, f"A00 shape {m00}x{n00} != {n_A_dofs}x{n_A_dofs}"
    assert m01 == n_A_dofs and n01 == n_V_dofs, f"A01 shape {m01}x{n01} != {n_A_dofs}x{n_V_dofs}"
    assert m10 == n_V_dofs and n10 == n_A_dofs, f"A10 shape {m10}x{n10} != {n_V_dofs}x{n_A_dofs}"
    assert m11 == n_V_dofs and n11 == n_V_dofs, f"A11 shape {m11}x{n11} != {n_V_dofs}x{n_V_dofs}"

    # Check communicators are compatible with mesh_parent.comm
    assert A00.getComm().size == comm.size, "A00 communicator size mismatch"
    assert A01.getComm().size == comm.size, "A01 communicator size mismatch"
    assert A10.getComm().size == comm.size, "A10 communicator size mismatch"
    assert A11.getComm().size == comm.size, "A11 communicator size mismatch"

    # Standalone A00 for AMS preconditioner
    A00_standalone = A00.copy()
    A00_standalone.assemble()
    
    # Create nested matrix with strict [[A00, A01], [A10, A11]] ordering
    mat_nest = PETSc.Mat().createNest(mats, comm=comm)
    mat_nest.assemble()

    # Quick MatVec sanity check: A * x = b with consistent VecNest layouts
    xA = PETSc.Vec().createMPI(n_A_dofs, comm=comm)
    xV = PETSc.Vec().createMPI(n_V_dofs, comm=comm)
    xA.setRandom()
    xV.setRandom()
    x_nest = PETSc.Vec().createNest([xA, xV], comm=comm)

    bA = PETSc.Vec().createMPI(n_A_dofs, comm=comm)
    bV = PETSc.Vec().createMPI(n_V_dofs, comm=comm)
    b_nest = PETSc.Vec().createNest([bA, bV], comm=comm)

    mat_nest.mult(x_nest, b_nest)

    # SPD approximation for A-block
    A00_spd = petsc.assemble_matrix(a00_spd_form, bcs=None)
    A00_spd.assemble()
    A00_spd.setOption(PETSc.Mat.Option.SPD, True)
    
    return mats, mat_nest, A00_standalone, A00_spd, interpolation_data


def configure_solver_submesh(mesh_parent, mat_nest, mat_blocks, A_space, V_space, 
                             A00_spd, config):
    """
    Configure PETSc solver for the mixed parent/submesh system using AMS for H(curl).

    This mirrors the original solver configuration:
      - Outer KSP: fgmres
      - PC: fieldsplit with SCHUR factorization
      - A-block: fgmres with AMS (HYPRE-AMS) preconditioner for H(curl)
      - V-block: preonly with BoomerAMG for H1
    """
    from dolfinx.cpp.fem.petsc import discrete_gradient
    
    comm = mesh_parent.comm
    A00_full = mat_blocks[0][0]

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(mat_nest, mat_nest)
    ksp.setType("fgmres")
    ksp.setTolerances(
        rtol=float(getattr(config, "outer_rtol", 1e-4)),
        atol=0.0,
        max_it=int(getattr(config, "outer_max_it", 50)),
    )

    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.LOWER)
    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.A11)

    isA, isV = mat_nest.getNestISs()
    pc.setFieldSplitIS(("A", isA[0]), ("V", isV[1]))
    pc.setUp()
    ksp_A, ksp_V = pc.getFieldSplitSubKSP()

    # A-block: AMS preconditioner for H(curl) / N1curl space
    V_ams = fem.functionspace(mesh_parent, ("Lagrange", 1))
    G = discrete_gradient(V_ams._cpp_object, A_space._cpp_object)
    G.assemble()

    xcoord = ufl.SpatialCoordinate(mesh_parent)
    verts = []
    for dim in range(3):
        f = fem.Function(V_ams)
        f.interpolate(fem.Expression(xcoord[dim], V_ams.element.interpolation_points))
        f.x.scatter_forward()
        verts.append(f.x.petsc_vec)

    e0 = G.createVecLeft()
    e1 = G.createVecLeft()
    e2 = G.createVecLeft()
    G.mult(verts[0], e0)
    G.mult(verts[1], e1)
    G.mult(verts[2], e2)

    ksp_A.setType("fgmres")
    ksp_A.setTolerances(rtol=0.0, atol=0.0, max_it=int(getattr(config, "ksp_A_max_it", 5)))
    ksp_A.setGMRESRestart(int(getattr(config, "ksp_A_restart", 30)))
    ksp_A.setOperators(A00_full, A00_spd)

    pc_A = ksp_A.getPC()
    pc_A.setType("hypre")
    pc_A.setHYPREType("ams")
    pc_A.setHYPREDiscreteGradient(G)
    pc_A.setHYPRESetEdgeConstantVectors(e0, e1, e2)

    # V-block: BoomerAMG for H1 space
    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    pc_V.setType("hypre")
    pc_V.setHYPREType("boomeramg")

    # Keep objects alive (prevent garbage collection)
    _keepalive.append((G, V_ams, (e0, e1, e2)))
    
    return ksp
