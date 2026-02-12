"""3D A-V solver helpers: forms, assembly, and PETSc solver setup."""

import numpy as np
from dolfinx import fem
from dolfinx.fem import petsc
from dolfinx.cpp.fem.petsc import discrete_gradient
from petsc4py import PETSc
import ufl

_keepalive = []   #PETSc object created in Python is passed to PETSc but not stored in Python, it WILL be destroyed


def _build_interior_nodes_for_ams(mesh, cell_tags, conductor_markers, degree=1):
    """
    Build CG(degree) vector for HYPRE AMS setHYPREAMSSetInteriorNodes.
    Values 1.0 = exterior (non-conductor), 0.0 = interior (conductor).
    Same convention as copper-rod: nodes in conductor cells get 0.0.
    """
    W = fem.functionspace(mesh, ("Lagrange", degree))
    interior_nodes_array = fem.Function(W)
    interior_nodes_array.x.array[:] = 1.0
    interior_nodes_array.x.scatter_forward()

    dofmap = W.dofmap
    num_dofs_per_cell = dofmap.dof_layout.num_dofs
    cell_dofs = dofmap.list.reshape(-1, num_dofs_per_cell)

    conductor_cells = np.concatenate([cell_tags.find(tag) for tag in conductor_markers])
    conductor_cells = np.unique(conductor_cells)
    if conductor_cells.size > 0:
        tagged_cell_dofs = cell_dofs[conductor_cells].flatten()
        unique_dofs = np.unique(tagged_cell_dofs)
        interior_nodes_array.x.array[unique_dofs] = 0.0
        interior_nodes_array.x.scatter_forward()

    return interior_nodes_array





def build_forms(mesh, A_space, V_space, sigma, nu, J_z, M_vec, A_prev,
                dx, dx_rs, dx_rpm, dx_c, dx_pm, ds, config, exterior_facet_tag=None):
    dt = fem.Constant(mesh, PETSc.ScalarType(config.dt))
    mu0 = config.mu0
    xcoord = ufl.SpatialCoordinate(mesh)
    omega = fem.Constant(mesh, PETSc.ScalarType(config.omega_m))
    #rigid rotation: u = ω × r with ω=(0,0,ω) and r=(x,y,z)
    omega_vec = ufl.as_vector((0.0, 0.0, omega))
    u_rot = ufl.cross(omega_vec, xcoord)
    
    A = ufl.TrialFunction(A_space)
    v = ufl.TestFunction(A_space)
    S = ufl.TrialFunction(V_space)
    q = ufl.TestFunction(V_space)
    
    curlA = ufl.curl(A)
    curlv = ufl.curl(v)
    
    inv_dt = fem.Constant(mesh, PETSc.ScalarType(1.0 / config.dt))

    epsilon_A = float(getattr(config, "epsilon_A", 0.0))
    epsilon_A_spd = float(getattr(config, "epsilon_A_spd", 1e-6))
    eps_A = fem.Constant(mesh, PETSc.ScalarType(epsilon_A))
    eps_spd = fem.Constant(mesh, PETSc.ScalarType(epsilon_A_spd))

    # A-equation (A–V formulation) + optional small mass regularization:
    a00 = (
        nu * ufl.inner(curlA, curlv) * dx
        + (sigma * inv_dt) * ufl.inner(A, v) * dx_rs
        - sigma * ufl.inner(ufl.cross(u_rot, curlA), v) * dx_rpm
        + eps_A * ufl.inner(A, v) * dx
    )
    
    # AMS-friendly SPD approximation of (dt * a00): dt*nu*curlcurl + sigma*mass (conductors only) + regularization
    dx_cond_all = dx_rs + dx_rpm
    a00_spd = (
        dt * nu * ufl.inner(curlA, curlv) * dx
        + sigma * ufl.inner(A, v) * dx_cond_all
        + eps_spd * ufl.inner(A, v) * dx
    )
    
    # A–V coupling terms (conductors only, no μ0):
    # - a01: ∫_{cond} sigma ∇V^{n+1}·v
    # - a10/a11: V-equation ∫_{cond} sigma ∇V^{n+1}·∇q + ∫_{cond} (sigma/dt) A^{n+1}·∇q = RHS(A^n)
    a01 = sigma * ufl.inner(ufl.grad(S), v) * dx_rs
    a10 = (
        -(sigma * inv_dt) * ufl.inner(A, ufl.grad(q)) * dx_rs
        + sigma * ufl.inner(ufl.cross(u_rot, curlA), ufl.grad(q)) * dx_rpm
    )
    #integrates σ∇V·∇q over Ω; σ=0 elsewhere applies restriction implicitly.
    a11 = sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx
    
    #Right hand side
    L0 = J_z * v[2] * dx_c + (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_rs +  ufl.inner(nu * mu0 * M_vec, curlv) * dx_pm
    L1 = (sigma * inv_dt) * ufl.inner(A_prev, ufl.grad(q)) * dx_rs
    
    a_blocks = ((fem.form(a00), fem.form(a01)), (fem.form(a10), fem.form(a11)))
    a00_spd_form = fem.form(a00_spd)
    L_blocks = (fem.form(L0), fem.form(L1))

    return a_blocks, L_blocks, a00_spd_form





def assemble_system_matrix(mesh, a_blocks, block_bcs, a00_spd_form):
    mats = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            bcs_for_block = []
            bcs_for_block.extend(block_bcs[i])
            mat = petsc.assemble_matrix(a_blocks[i][j], bcs=bcs_for_block)
            mat.assemble()
            mats[i][j] = mat
    
    A00_standalone = mats[0][0].copy()
    A00_standalone.assemble()
    mat_nest = PETSc.Mat().createNest(mats, comm=mesh.comm)
    mat_nest.assemble()
    
    A00_spd = petsc.assemble_matrix(a00_spd_form, bcs=None)
    A00_spd.assemble()
    A00_spd.setOption(PETSc.Mat.Option.SPD, True)

    return mats, mat_nest, A00_standalone, A00_spd, None



def configure_solver(mesh, mat_nest, mat_blocks, A_space, V_space, A00_spd, config,
                     cell_tags=None, conductor_markers=None):
    A00_full = mat_blocks[0][0]

    ksp = PETSc.KSP().create(mesh.comm)
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

    # A-block AMS (discrete gradient from CG1 to Nédélec)
    V_ams = fem.functionspace(mesh, ("Lagrange", 1))
    G = discrete_gradient(V_ams._cpp_object, A_space._cpp_object)
    G.assemble()

    xcoord = ufl.SpatialCoordinate(mesh)
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
    ksp_A.setTolerances(rtol=0.0, atol=0.0, max_it=int(getattr(config, "ksp_A_max_it", 10)))
    ksp_A.setGMRESRestart(int(getattr(config, "ksp_A_restart", 30)))
    ksp_A.setOperators(A00_full, A00_spd)

    pc_A = ksp_A.getPC()
    pc_A.setType("hypre")
    pc_A.setHYPREType("ams")
    pc_A.setHYPREDiscreteGradient(G)
    pc_A.setHYPRESetEdgeConstantVectors(e0, e1, e2)

    # Interior nodes for AMS (copper-rod style: 0 = conductor, 1 = non-conductor)
    if cell_tags is not None and conductor_markers is not None and len(conductor_markers) > 0:
        interior_nodes_array = _build_interior_nodes_for_ams(
            mesh, cell_tags, conductor_markers, degree=1
        )
        pc_A.setHYPREAMSSetInteriorNodes(interior_nodes_array.x.petsc_vec)
        _keepalive.append(interior_nodes_array)

    # AMS tuning similar to copper-rod setup
    opts = PETSc.Options()
    opts["pc_hypre_ams_cycle_type"] = 13
    opts["pc_hypre_ams_tol"] = 0.0
    opts["pc_hypre_ams_max_iter"] = 1
    opts["pc_hypre_ams_amg_beta_theta"] = 0.25
    opts["pc_hypre_ams_print_level"] = 1
    opts["pc_hypre_ams_amg_alpha_options"] = "10,1,6,6,4"
    opts["pc_hypre_ams_amg_beta_options"] = "10,1,6,6,4"
    opts["pc_hypre_ams_relax_type"] = 2
    opts["pc_hypre_ams_relax_weight"] = 1.0
    opts["pc_hypre_ams_relax_times"] = 1
    opts["pc_hypre_ams_omega"] = 1.0
    opts["pc_hypre_ams_projection_frequency"] = 25

    ksp_A.setFromOptions()

    ksp_V.setType("preonly")
    pc_V = ksp_V.getPC()
    pc_V.setType("hypre")
    pc_V.setHYPREType("boomeramg")

    _keepalive.append((G, V_ams, (e0, e1, e2)))
    return ksp




 