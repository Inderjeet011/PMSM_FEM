"""3D A-V solver with submesh: forms, assembly, and PETSc solver setup.
A lives on parent mesh, V lives on conductor submesh.
"""

from dolfinx import fem  # type: ignore
from dolfinx.fem import petsc  # type: ignore
from petsc4py import PETSc  # type: ignore
import ufl  # type: ignore
import numpy as np  # type: ignore

_keepalive = []


def build_forms_submesh(mesh_parent, A_space, V_space,
                        sigma, nu, J_z, M_vec, A_prev,
                        dx_parent, dx_rs, dx_rpm, dx_c, dx_pm,
                        config, entity_map, dx_cond_parent, dx_air=None):
    dt = fem.Constant(mesh_parent, PETSc.ScalarType(config.dt))
    mu0 = config.mu0

    A = ufl.TrialFunction(A_space)
    v = ufl.TestFunction(A_space)
    S = ufl.TrialFunction(V_space)
    q = ufl.TestFunction(V_space)
    
    curlA = ufl.curl(A)
    curlv = ufl.curl(v)
    x_coord = ufl.SpatialCoordinate(mesh_parent)
    omega_m = fem.Constant(mesh_parent, PETSc.ScalarType(config.omega_m))
    u_rot = ufl.as_vector((-omega_m * x_coord[1], omega_m * x_coord[0], 0.0))
    
    inv_dt = fem.Constant(mesh_parent, PETSc.ScalarType(1.0 / config.dt))
    
    a00 = (
        nu * ufl.inner(curlA, curlv) * dx_parent
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_rs
    )
    if dx_air is not None:
        a00 += (sigma * inv_dt) * ufl.inner(A, v) * dx_air
    a00 += (sigma * inv_dt) * ufl.inner(A, v) * dx_cond_parent
    a00 += -sigma * ufl.inner(ufl.cross(u_rot, curlA), v) * dx_rpm

    a01 = dt * sigma * ufl.inner(ufl.grad(S), v) * dx_cond_parent
    a10 = -sigma * ufl.inner(ufl.grad(q), A) * dx_cond_parent
    a11 = dt * sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx_cond_parent

    L0 = (
        J_z * v[2] * dx_c
        + (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_rs
        + ufl.inner(nu * mu0 * M_vec, curlv) * dx_pm
    )
    if dx_air is not None:
        L0 += (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_air
    L0 += (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_cond_parent
    L1 = ufl.inner(ufl.grad(q), sigma * A_prev) * dx_cond_parent

    em = [entity_map]
    a_blocks = (
        (fem.form(a00), fem.form(a01, entity_maps=em)),
        (fem.form(a10, entity_maps=em), fem.form(a11, entity_maps=em)),
    )
    L_blocks = (fem.form(L0), fem.form(L1, entity_maps=em))
    a_block_form = fem.form([[a00, a01], [a10, a11]], entity_maps=em)
    return a_blocks, L_blocks, a_block_form


def assemble_system_matrix_submesh(mesh_parent, a_blocks, block_bcs,
                                    A_space_parent, V_space_submesh, a_block_form):
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

    mat_nest = PETSc.Mat().createNest(mats, comm=comm)
    mat_nest.assemble()
    return mats, mat_nest


def configure_solver_submesh(mesh_parent, mat_nest, mat_blocks, A_space, V_space,
                             config, cell_tags_parent=None, conductor_markers=()):
    from dolfinx.cpp.fem.petsc import discrete_gradient  # type: ignore

    comm = mesh_parent.comm
    A00_full = mat_blocks[0][0]

    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(mat_nest, mat_nest)
    ksp.setType("fgmres")
    ksp.setGMRESRestart(150)
    ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
    ksp.setTolerances(
        rtol=float(getattr(config, "outer_rtol", 1e-4)),
        atol=float(getattr(config, "outer_atol", 0.0)),
        max_it=int(getattr(config, "outer_max_it", 200)),
    )
    pc = ksp.getPC()
    pc.setType("fieldsplit")
    pc.setFieldSplitType(PETSc.PC.CompositeType.SCHUR)
    pc.setFieldSplitSchurFactType(PETSc.PC.SchurFactType.LOWER)
    pc.setFieldSplitSchurPreType(PETSc.PC.SchurPreType.SELFP)
    isA, isV = mat_nest.getNestISs()
    pc.setFieldSplitIS(("A", isA[0]), ("V", isV[1]))
    pc.setUp()

    ksp_A, ksp_V = pc.getFieldSplitSubKSP()

    V_ams = fem.functionspace(mesh_parent, ("Lagrange", 1))
    G = discrete_gradient(V_ams._cpp_object, A_space._cpp_object)
    G.assemble()

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
    ams_keepalive = (G, V_ams, (cvec_0, cvec_1, cvec_2))

    ksp_A.setType("fgmres")
    ksp_A.setGMRESRestart(int(getattr(config, "ksp_A_restart", 50)))
    ksp_A.setTolerances(
        rtol=float(getattr(config, "ksp_A_rtol", 1e-2)),
        atol=0.0,
        max_it=int(getattr(config, "ksp_A_max_it", 5)),
    )
    ksp_A.setOperators(A00_full, A00_full)

    pc_A = ksp_A.getPC()
    pc_A.setType("hypre")
    pc_A.setHYPREType("ams")
    pc_A.setHYPREDiscreteGradient(G)
    pc_A.setHYPRESetEdgeConstantVectors(
        cvec_0.x.petsc_vec, cvec_1.x.petsc_vec, cvec_2.x.petsc_vec
    )

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

    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    pc_V.setType("lu")
    try:
        pc_V.setFactorSolverType("mumps")
    except Exception:
        pass
    ksp_V.setFromOptions()

    _keepalive.append(ams_keepalive)

    ksp.setFromOptions()
    return ksp
