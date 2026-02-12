"""3D A-V solver with submesh: forms, assembly, and PETSc solver setup.
A lives on parent mesh, V lives on conductor submesh.
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
                        dx_conductor, config, entity_map, dx_cond_parent,
                        dx_air=None, exterior_facet_tag=None):
    """
    Build forms for A-V system with A on parent mesh and V on conductor submesh.
    Uses entity_maps for automatic cross-mesh coupling (no manual quadrature).

    Parameters:
    -----------
    entity_map : dolfinx.mesh.EntityMap
        Cell entity map submesh -> parent from create_submesh.
    dx_cond_parent : ufl.Measure
        Integration over conductor region on parent (e.g. dx_rs + dx_rpm + dx_c + dx_pm).
    (Other parameters as before: mesh_parent, mesh_conductor, A_space, V_space,
     sigma, nu, J_z, M_vec, A_prev, dx_*, config.)
    """
    dt = fem.Constant(mesh_parent, PETSc.ScalarType(config.dt))
    mu0 = config.mu0

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
    # Optional motion term: -sigma * (u_rot × curl A)·v on dx_rpm (config.use_motion_term).
    a00 = (
        nu * ufl.inner(curlA, curlv) * dx_parent
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_rs
        + eps_A * ufl.inner(A, v) * dx_parent
    )
    if dx_air is not None:
        a00 += (sigma * inv_dt) * ufl.inner(A, v) * dx_air
    if getattr(config, "use_motion_term", False):
        # u_rot = omega_m × r with rotation about z: (0,0,omega_m) × (x,y,z) = (-omega_m*y, omega_m*x, 0)
        x = ufl.SpatialCoordinate(mesh_parent)
        omega_m = float(config.omega_m)
        u_rot = ufl.as_vector((-omega_m * x[1], omega_m * x[0], 0.0))
        a00 += -sigma * ufl.inner(ufl.cross(u_rot, curlA), v) * dx_rpm
    
    # SPD approximation for preconditioner
    dx_cond_all = dx_rs + dx_rpm
    a00_spd = (
        dt * nu * ufl.inner(curlA, curlv) * dx_parent
        + sigma * ufl.inner(A, v) * dx_cond_all
        + eps_spd * ufl.inner(A, v) * dx_parent
    )
    
    # A–V coupling: conductor integrals on parent measure; entity_maps at form compile
    a01 = dt * sigma * ufl.inner(ufl.grad(S), v) * dx_cond_parent
    a10 = sigma * ufl.inner(ufl.grad(q), A) * dx_cond_parent
    a11 = dt * sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx_cond_parent

    L0 = (
        J_z * v[2] * dx_c
        + (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_rs
        + ufl.inner(nu * mu0 * M_vec, curlv) * dx_pm
    )
    if dx_air is not None:
        L0 += (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_air
    L1 = ufl.inner(ufl.grad(q), sigma * A_prev) * dx_cond_parent

    interpolation_data = {
        'V_space_parent': None,
        'V_parent': None,
        'A_space_submesh': None,
        'A_submesh': None,
        'sigma_submesh': None,
    }

    em = [entity_map]
    a_blocks = (
        (fem.form(a00), fem.form(a01, entity_maps=em)),
        (fem.form(a10, entity_maps=em), fem.form(a11, entity_maps=em)),
    )
    L_blocks = (fem.form(L0), fem.form(L1, entity_maps=em))
    a00_spd_form = fem.form(a00_spd)
    a_block_form = fem.form([[a00, a01], [a10, a11]], entity_maps=em)
    L_block_form = fem.form([L0, L1], entity_maps=em)
    return a_blocks, L_blocks, a00_spd_form, interpolation_data, a_block_form, L_block_form


def assemble_system_matrix_submesh(mesh_parent, a_blocks, block_bcs,
                                    a00_spd_form, interpolation_data, A_space_parent, V_space_submesh,
                                    a_block_form):
    """
    Assemble system matrix from block form (entity_maps); extract blocks for nested solver.
    """
    comm = mesh_parent.comm
    n_A_dofs = A_space_parent.dofmap.index_map.size_global
    n_V_dofs = V_space_submesh.dofmap.index_map.size_global

    bcs_flat = [bc for bclist in (block_bcs or [[], []]) for bc in bclist]
    A_mono = petsc.assemble_matrix(a_block_form, bcs=bcs_flat)
    A_mono.assemble()
    is_A = PETSc.IS().createGeneral(np.arange(0, n_A_dofs, dtype=np.int32), comm=comm)
    is_V = PETSc.IS().createGeneral(
        np.arange(n_A_dofs, n_A_dofs + n_V_dofs, dtype=np.int32), comm=comm
    )
    mats = [
        [A_mono.createSubMatrix(is_A, is_A), A_mono.createSubMatrix(is_A, is_V)],
        [A_mono.createSubMatrix(is_V, is_A), A_mono.createSubMatrix(is_V, is_V)],
    ]
    is_A.destroy()
    is_V.destroy()

    A00_standalone = mats[0][0].copy()
    A00_standalone.assemble()

    mat_nest = PETSc.Mat().createNest(mats, comm=comm)
    mat_nest.assemble()

    # SPD approximation for A-block
    A00_spd = petsc.assemble_matrix(a00_spd_form, bcs=None)
    A00_spd.assemble()
    A00_spd.setOption(PETSc.Mat.Option.SPD, True)
    
    return mats, mat_nest, A00_standalone, A00_spd, interpolation_data


def configure_solver_submesh(mesh_parent, mat_nest, mat_blocks, A_space, V_space,
                             A00_spd, config, cell_tags_parent=None, conductor_markers=()):
    """
    Configure PETSc solver for the mixed parent/submesh system using AMS for H(curl).

    Aligned with Arshad's setup for better convergence:
      - Outer KSP: GMRES with block-diagonal P = diag(A00, A11)
      - PC: fieldsplit ADDITIVE (solve A- and V-blocks in parallel)
      - A-block: preonly with AMS (HYPRE-AMS) for H(curl)
      - V-block: preonly with BoomerAMG for H1
      - Hypre AMS options (cycle type, AMG, relax) set explicitly

    If cell_tags_parent and conductor_markers are provided, builds an interior-nodes
    array (1 = air, 0 = conductor) and supplies it to AMS via setHYPREAMSSetInteriorNodes.
    """
    from dolfinx.cpp.fem.petsc import discrete_gradient

    comm = mesh_parent.comm
    A00_full = mat_blocks[0][0]
    A11_blk = mat_blocks[1][1]
    use_schur = bool(getattr(config, "use_schur", False))
    norm_name = str(getattr(config, "outer_norm_type", "unpreconditioned")).lower()
    norm_map = {
        "unpreconditioned": PETSc.KSP.NormType.UNPRECONDITIONED,
        "preconditioned": PETSc.KSP.NormType.PRECONDITIONED,
        "natural": PETSc.KSP.NormType.NATURAL,
        "none": PETSc.KSP.NormType.NONE,
    }
    norm_type = norm_map.get(norm_name, PETSc.KSP.NormType.UNPRECONDITIONED)

    if use_schur:
        # SCHUR fieldsplit: use full mat_nest as operator and preconditioner
        P_nest = None
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(mat_nest, mat_nest)
        ksp.setType("gmres")
        ksp.setGMRESRestart(100)
        ksp.setNormType(norm_type)
        ksp.setTolerances(
            rtol=float(getattr(config, "outer_rtol", 1e-4)),
            atol=0.0,
            max_it=int(getattr(config, "outer_max_it", 200)),
        )
        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
        pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.LOWER)
        pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELF)  # use A11 for Schur approx
        isA, isV = mat_nest.getNestISs()
        pc.setFieldSplitIS(("A", isA[0]), ("V", isV[1]))
        pc.setUp()
    else:
        # ADDITIVE fieldsplit + block-diagonal P = diag(A00, A11) (Arshad-style)
        A00_blk = mat_blocks[0][0]
        P_nest = PETSc.Mat().createNest(
            [[A00_blk, None], [None, A11_blk]], comm=comm
        )
        P_nest.assemble()
        ksp = PETSc.KSP().create(comm)
        ksp.setOperators(mat_nest, P_nest)
        ksp.setType("gmres")
        ksp.setGMRESRestart(100)
        ksp.setNormType(norm_type)
        ksp.setTolerances(
            rtol=float(getattr(config, "outer_rtol", 1e-4)),
            atol=0.0,
            max_it=int(getattr(config, "outer_max_it", 200)),
        )
        pc = ksp.getPC()
        pc.setType("fieldsplit")
        pc.setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
        isA, isV = mat_nest.getNestISs()
        pc.setFieldSplitIS(("A", isA[0]), ("V", isV[1]))
        pc.setUp()

    ksp_A, ksp_V = pc.getFieldSplitSubKSP()

    # A-block: AMS preconditioner for H(curl) / N1curl space
    # Use actual A-block (A00) for both operator and preconditioner (Arshad's style), not SPD approx.
    degree_A = int(getattr(config, "degree_A", 1))
    V_ams = fem.functionspace(mesh_parent, ("Lagrange", degree_A))
    G = discrete_gradient(V_ams._cpp_object, A_space._cpp_object)
    G.assemble()

    # Edge constant vectors for AMS: (1,0,0), (0,1,0), (0,0,1) in N1curl (Arshad's style)
    if degree_A == 1:
        cvec_0 = fem.Function(A_space)
        cvec_0.interpolate(
            lambda x: np.vstack(
                (np.ones_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0]))
            )
        )
        cvec_1 = fem.Function(A_space)
        cvec_1.interpolate(
            lambda x: np.vstack(
                (np.zeros_like(x[0]), np.ones_like(x[0]), np.zeros_like(x[0]))
            )
        )
        cvec_2 = fem.Function(A_space)
        cvec_2.interpolate(
            lambda x: np.vstack(
                (np.zeros_like(x[0]), np.zeros_like(x[0]), np.ones_like(x[0]))
            )
        )
        edge_vecs = (cvec_0, cvec_1, cvec_2)
    else:
        from dolfinx.cpp.fem.petsc import interpolation_matrix
        shape = (mesh_parent.geometry.dim,)
        Q = fem.functionspace(mesh_parent, ("Lagrange", degree_A, shape))
        Pi = interpolation_matrix(Q._cpp_object, A_space._cpp_object)
        Pi.assemble()
        edge_vecs = None  # use setInterpolations below

    ksp_A.setType("preonly")
    ksp_A.setOperators(A00_full, A00_full)

    pc_A = ksp_A.getPC()
    pc_A.setType("hypre")
    pc_A.setHYPREType("ams")
    pc_A.setHYPREDiscreteGradient(G)
    if degree_A == 1:
        pc_A.setHYPRESetEdgeConstantVectors(
            cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
        )
    else:
        pc_A.setHYPRESetInterpolations(dim=mesh_parent.geometry.dim, ND_Pi_Full=Pi)

    # Interior nodes for AMS: 1 = interior (air), 0 = non-interior (conductor)
    if cell_tags_parent is not None and conductor_markers:
        W = V_ams
        interior_nodes_array = fem.Function(W)
        interior_nodes_array.x.array[:] = 1.0
        interior_nodes_array.x.scatter_forward()

        dofmap = W.dofmap
        num_dofs_per_cell = dofmap.dof_layout.num_dofs
        cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

        tagged_cell_dofs_list = []
        for marker in conductor_markers:
            tagged_cells = cell_tags_parent.find(marker)
            if tagged_cells.size > 0:
                tagged_cell_dofs_list.append(cell_dofs[tagged_cells].flatten())
        if tagged_cell_dofs_list:
            unique_dofs = np.unique(np.concatenate(tagged_cell_dofs_list))
            interior_nodes_array.x.array[unique_dofs] = 0.0
            interior_nodes_array.x.scatter_forward()

        pc_A.setHYPREAMSSetInteriorNodes(interior_nodes_array.x.petsc_vec)
        _keepalive.append(interior_nodes_array)

    # Hypre AMS options (Arshad's settings for better convergence)
    opts = PETSc.Options()
    opts[f"{ksp_A.prefix}pc_hypre_ams_cycle_type"] = 13
    opts[f"{ksp_A.prefix}pc_hypre_ams_tol"] = 0
    opts[f"{ksp_A.prefix}pc_hypre_ams_max_iter"] = 1
    opts[f"{ksp_A.prefix}pc_hypre_ams_amg_beta_theta"] = 0.25
    opts[f"{ksp_A.prefix}pc_hypre_ams_print_level"] = 1
    opts[f"{ksp_A.prefix}pc_hypre_ams_amg_alpha_options"] = "10,1,6,6,4"
    opts[f"{ksp_A.prefix}pc_hypre_ams_amg_beta_options"] = "10,1,6,6,4"
    opts[f"{ksp_A.prefix}pc_hypre_ams_relax_type"] = 2
    opts[f"{ksp_A.prefix}pc_hypre_ams_relax_weight"] = 1.0
    opts[f"{ksp_A.prefix}pc_hypre_ams_relax_times"] = 1
    opts[f"{ksp_A.prefix}pc_hypre_ams_omega"] = 1.0
    opts[f"{ksp_A.prefix}pc_hypre_ams_projection_frequency"] = 50
    ksp_A.setFromOptions()

    # V-block: BoomerAMG for H1 space
    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    pc_V.setType("hypre")
    pc_V.setHYPREType("boomeramg")
    ksp_V.setFromOptions()

    # Keep objects alive (prevent garbage collection)
    if P_nest is not None:
        _keepalive.append(P_nest)
    if degree_A == 1:
        _keepalive.append((G, V_ams, edge_vecs))
    else:
        _keepalive.append((G, V_ams, Pi))

    return ksp
