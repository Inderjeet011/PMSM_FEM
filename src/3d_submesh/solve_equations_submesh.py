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
                        exterior_facet_tag=None):
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
    
    # A–V coupling: conductor integrals on parent measure; entity_maps at form compile
    a01 = dt * sigma * ufl.inner(ufl.grad(S), v) * dx_cond_parent
    a10 = sigma * ufl.inner(ufl.grad(q), A) * dx_cond_parent
    a11 = dt * sigma * ufl.inner(ufl.grad(S), ufl.grad(q)) * dx_cond_parent

    L0 = (
        J_z * v[2] * dx_c
        + (sigma * inv_dt) * ufl.inner(A_prev, v) * dx_rs
        + ufl.inner(nu * mu0 * M_vec, curlv) * dx_pm
    )
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
